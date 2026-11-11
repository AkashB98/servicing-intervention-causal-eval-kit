"""CLI: generate | estimate | honesty | design | simulate | report |
check | demo | serve.

SYNTHETIC DATA ONLY - Northstar Lending is a fictional company and
every dataset handled here is fabricated sample data.

Exit codes:
  check : 0 = pass (no honesty flag), 1 = breach (the honesty check
          flags at least one intervention - EXPECTED on the seeded
          data, by design: the generator plants an unobserved
          confounder in the observational book, and this gate exists
          to catch it), 2 = usage / IO / data-quality error
  others: 0 = success, 2 = usage / IO error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from design import pilot_design  # noqa: E402
from estimator import estimate_all  # noqa: E402
from generator import (DEFAULT_N_HOLDOUT, DEFAULT_N_OBSERVATIONAL,
                       DEFAULT_N_PILOT, generate_dataset, load_rows,
                       rows_for_period, write_rows)  # noqa: E402
from honesty import honesty_check  # noqa: E402
from policy import simulate_policies  # noqa: E402
from report import build_report  # noqa: E402
from validation import validate_dataset  # noqa: E402

EXIT_OK, EXIT_FAIL, EXIT_ERROR = 0, 1, 2


def canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True)


def _load_or_error(path: str) -> Optional[List[Dict]]:
    try:
        return load_rows(path)
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
    except (ValueError, OSError) as exc:
        print(f"error: cannot read {path}: {exc}", file=sys.stderr)
    return None


def _validated_or_error(path: str):
    rows = _load_or_error(path)
    if rows is None:
        return None, EXIT_ERROR
    issues = validate_dataset(rows)
    if issues:
        for issue in issues[:20]:
            print(f"data-quality: {issue}", file=sys.stderr)
        return None, EXIT_ERROR
    return rows, EXIT_OK


def _write_out(args, text: str, summary: str) -> int:
    if getattr(args, "out", None):
        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError as exc:
            print(f"error: cannot write {args.out}: {exc}", file=sys.stderr)
            return EXIT_ERROR
        print(summary)
    else:
        print(text, end="")
    return EXIT_OK


def cmd_generate(args) -> int:
    rows = generate_dataset(seed=args.seed, n_pilot=args.n_pilot,
                            n_observational=args.n_observational,
                            n_holdout=args.n_holdout)
    fmt = "csv" if (args.out and args.out.endswith(".csv")) else "jsonl"
    if args.out:
        try:
            write_rows(rows, args.out, fmt=fmt)
        except OSError as exc:
            print(f"error: cannot write {args.out}: {exc}", file=sys.stderr)
            return EXIT_ERROR
        print(f"generated {len(rows)} synthetic rows (seed={args.seed}) -> {args.out}")
        return EXIT_OK
    from generator import rows_to_csv, rows_to_jsonl
    print(rows_to_csv(rows) if fmt == "csv" else rows_to_jsonl(rows), end="")
    return EXIT_OK


def cmd_estimate(args) -> int:
    rows, code = _validated_or_error(args.input)
    if rows is None:
        return code
    period_rows = rows_for_period(rows, args.period)
    if not period_rows:
        print(f"error: no {args.period} rows", file=sys.stderr)
        return EXIT_ERROR
    try:
        payload = {"period": args.period,
                   "estimates": estimate_all(period_rows)}
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return _write_out(args, canonical(payload) + "\n", f"estimate -> {args.out}")


def cmd_honesty(args) -> int:
    rows, code = _validated_or_error(args.input)
    if rows is None:
        return code
    try:
        payload = honesty_check(rows)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return _write_out(args, canonical(payload) + "\n", f"honesty -> {args.out}")


def cmd_design(args) -> int:
    rows, code = _validated_or_error(args.input)
    if rows is None:
        return code
    pilot = rows_for_period(rows, "pilot")
    try:
        payload = pilot_design(pilot)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return _write_out(args, canonical(payload) + "\n", f"design -> {args.out}")


def cmd_simulate(args) -> int:
    rows, code = _validated_or_error(args.input)
    if rows is None:
        return code
    try:
        payload = simulate_policies(rows)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return _write_out(args, canonical(payload) + "\n", f"simulate -> {args.out}")


def cmd_report(args) -> int:
    rows, code = _validated_or_error(args.input)
    if rows is None:
        return code
    try:
        report = build_report(rows)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return _write_out(args, canonical(report) + "\n", f"report -> {args.out}")


def cmd_check(args) -> int:
    rows = _load_or_error(args.input)
    if rows is None:
        return EXIT_ERROR
    issues = validate_dataset(rows)
    if issues:
        for issue in issues[:20]:
            print(f"data-quality: {issue}")
        print(f"check FAIL (data quality): {len(issues)} issue(s)")
        return EXIT_ERROR
    try:
        result = honesty_check(rows)
    except ValueError as exc:
        print(f"error: {exc}")
        return EXIT_ERROR
    for name, entry in result["per_intervention"].items():
        print(f"{name}: pilot_ate={entry['pilot_ate']:.4f} "
              f"observational_ate={entry['observational_ate']:.4f} "
              f"difference={entry['difference']:+.4f} "
              f"flag_threshold={entry['flag_threshold']:.4f} "
              f"flagged={entry['flagged']}")
    if result["any_flagged"]:
        print(f"check FAIL (honesty): observational estimates disagree "
              f"with the randomised pilot beyond noise for: "
              f"{', '.join(result['flagged'])} - do not use the "
              f"observational numbers as causal estimates")
        return EXIT_FAIL
    print("check PASS: observational and randomised estimates agree "
          "within noise")
    return EXIT_OK


def cmd_demo(args) -> int:
    import demo as demo_mod
    try:
        demo_mod.main()
    except AssertionError as exc:
        print(f"demo FAILED: {exc}", file=sys.stderr)
        return EXIT_FAIL
    return EXIT_OK


# ------------------------------------------------------------------- serve

class _State:
    def __init__(self):
        self.rows: List[Dict] = generate_dataset()


_STATE = _State()


def get_state() -> _State:
    return _STATE


def reset_state(rows: Optional[List[Dict]] = None) -> None:
    _STATE.rows = list(rows) if rows is not None else generate_dataset()


class _Handler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: Any) -> None:
        body = (canonical(payload) + "\n").encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> Dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        body = json.loads(raw) if raw.strip() else {}
        if not isinstance(body, dict):
            raise ValueError("expected a JSON object")
        return body

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            return self._send_json(200, {"status": "ok"})
        if self.path == "/report":
            try:
                return self._send_json(200, build_report(_STATE.rows))
            except ValueError as exc:
                return self._send_json(400, {"error": str(exc)})
        return self._send_json(404, {"error": "unknown endpoint"})

    def do_POST(self):  # noqa: N802
        try:
            body = self._body()
            rows = body.get("rows", _STATE.rows)
            if self.path == "/estimate":
                period = body.get("period", "pilot")
                period_rows = rows_for_period(rows, period)
                return self._send_json(200, {"period": period,
                                             "estimates": estimate_all(period_rows)})
            if self.path == "/honesty":
                return self._send_json(200, honesty_check(rows))
            if self.path == "/simulate":
                return self._send_json(200, simulate_policies(rows))
            return self._send_json(404, {"error": "unknown endpoint"})
        except (ValueError, TypeError, KeyError) as exc:
            return self._send_json(400, {"error": f"bad request: {exc}"})

    def log_message(self, *a: Any) -> None:
        pass


def make_server(port: int) -> HTTPServer:
    return HTTPServer(("127.0.0.1", port), _Handler)


def cmd_serve(args) -> int:
    if args.input:
        rows = _load_or_error(args.input)
        if rows is None:
            return EXIT_ERROR
        reset_state(rows)
    server = make_server(args.port)
    print(f"serving on http://127.0.0.1:{args.port}  "
          f"(POST /estimate, /honesty, /simulate; GET /report, /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="servicing-intervention-causal-eval-kit",
        description="Stratified causal evaluation of servicing "
                    "interventions (Northstar Lending sample data).")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate the synthetic borrower datasets")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--n-pilot", type=int, default=DEFAULT_N_PILOT)
    g.add_argument("--n-observational", type=int, default=DEFAULT_N_OBSERVATIONAL)
    g.add_argument("--n-holdout", type=int, default=DEFAULT_N_HOLDOUT)
    g.add_argument("--out", default=None)
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("estimate", help="stratified ATE per intervention, one period")
    e.add_argument("--input", required=True)
    e.add_argument("--period", default="pilot",
                   choices=["pilot", "observational", "holdout"])
    e.add_argument("--out", default=None)
    e.set_defaults(func=cmd_estimate)

    for name, fn, helptext in (
            ("honesty", cmd_honesty, "randomised vs observational honesty check"),
            ("design", cmd_design, "two-proportion sizing + pilot guardrails"),
            ("simulate", cmd_simulate, "score decision policies on the holdout"),
            ("report", cmd_report, "estimates + honesty + design + policies")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--input", required=True)
        s.add_argument("--out", default=None)
        s.set_defaults(func=fn)

    c = sub.add_parser("check", help="CI gate: exit 0 pass / 1 honesty breach / 2 error")
    c.add_argument("--input", required=True)
    c.set_defaults(func=cmd_check)

    sub.add_parser("demo", help="run the end-to-end demo").set_defaults(func=cmd_demo)

    s = sub.add_parser("serve", help="localhost JSON API")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--input", default=None)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
