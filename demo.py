"""End-to-end demo: generate -> THE HONESTY CHECK -> estimate ->
design -> simulate -> check -> serve.

SYNTHETIC DATA ONLY (fictional Northstar Lending datasets). Runs
fully offline; the serve round-trip is loopback-only on an ephemeral
port. Every stage asserts.

The demo leads with the honesty check because that is the point of
the kit: the observational book *looks* like evidence, and on the
seeded data its modification estimate is inflated far beyond noise
by a planted unobserved confounder. The randomised pilot is the
number to believe.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import EXIT_FAIL, EXIT_OK, main as cli_main, make_server, reset_state  # noqa: E402
from design import pilot_design  # noqa: E402
from estimator import estimate_all  # noqa: E402
from generator import generate_dataset, rows_for_period, rows_to_jsonl  # noqa: E402
from honesty import honesty_check  # noqa: E402
from policy import simulate_policies  # noqa: E402
from report import build_report  # noqa: E402
from validation import validate_dataset  # noqa: E402

# Seed whose unconfounded variant raises no flag - used to show the
# check gate passing when there is nothing to catch.
CLEAN_UNCONFOUNDED_SEED = 5000


def _request(port: int, path: str, payload=None, method: str = "GET"):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read())


def main() -> None:
    print("== 1. generate synthetic Northstar Lending datasets (seed=42) ==")
    rows = generate_dataset()
    again = generate_dataset()
    assert rows == again, "generator must be deterministic"
    assert len(rows) == 5200, len(rows)
    assert validate_dataset(rows) == [], validate_dataset(rows)[:3]
    counts = {p: len(rows_for_period(rows, p))
              for p in ("pilot", "observational", "holdout")}
    print(f"   {len(rows)} borrowers: pilot={counts['pilot']} "
          f"observational={counts['observational']} "
          f"holdout={counts['holdout']}, deterministic, clean")

    print("\n== 2. THE HONESTY CHECK: randomised pilot vs observational book ==")
    print("   The observational assignment depends on a planted latent")
    print("   willingness trait that also raises cure rates and is NOT in")
    print("   the analysis covariates - stratification cannot remove it.")
    result = honesty_check(rows)
    for name, entry in result["per_intervention"].items():
        print(f"   {name}: pilot ATE {entry['pilot_ate']:+.2%} vs "
              f"observational ATE {entry['observational_ate']:+.2%} "
              f"(difference {entry['difference']:+.2%}, "
              f"flag threshold {entry['flag_threshold']:.2%}) "
              f"flagged={entry['flagged']}")
    assert "modification" in result["flagged"], result
    print(f"   FLAGGED: {', '.join(result['flagged'])} - the "
          f"observational numbers for these interventions must not "
          f"be used as causal estimates.")

    print("\n== 3. stratified CATE estimates from the randomised pilot ==")
    pilot = rows_for_period(rows, "pilot")
    for name, est in estimate_all(pilot).items():
        print(f"   {name}: ATE {est['ate']:+.2%} "
              f"95% CI [{est['ci_95'][0]:+.2%}, {est['ci_95'][1]:+.2%}]")

    print("\n== 4. design: two-proportion sizing + guardrails (pilot) ==")
    design = pilot_design(pilot)
    print(f"   control cure rate {design['baseline_cure_rate']:.2%}; "
          f"MDE at pilot size {design['mde_at_pilot_size']:.2%}")
    for entry in design["table"]:
        print(f"   lift {entry['delta']:.0%}: n/arm = {entry['n_per_arm']}")
    for name, g in design["guardrails"].items():
        print(f"   {name}: complaints {g['complaint_rate']:.2%}, "
              f"re-delinquency {g['re_delinquency_rate']:.2%}")

    print("\n== 5. policy simulation - scored ONLY on the randomised holdout ==")
    sim = simulate_policies(rows)
    for score in sim["policies"]:
        print(f"   {score['policy']}: expected cures "
              f"{score['expected_cures']:.1f}, net value "
              f"${score['net_value']:,.0f}")
    print(f"   ranking by net value: {' > '.join(sim['ranking'])}")

    print("\n== 6. CLI check gate (exit 1 on seeded data BY DESIGN) ==")
    import tempfile
    tmp = tempfile.mkdtemp(prefix="servicing-demo-")
    seeded_path = os.path.join(tmp, "seeded.jsonl")
    with open(seeded_path, "w", encoding="utf-8") as fh:
        fh.write(rows_to_jsonl(rows))
    clean_path = os.path.join(tmp, "clean.jsonl")
    with open(clean_path, "w", encoding="utf-8") as fh:
        fh.write(rows_to_jsonl(generate_dataset(
            seed=CLEAN_UNCONFOUNDED_SEED, confounded=False)))
    code_seeded = cli_main(["check", "--input", seeded_path])
    code_clean = cli_main(["check", "--input", clean_path])
    code_err = cli_main(["check", "--input", "/tmp/no-such-servicing-xyz.jsonl"])
    print(f"   seeded={code_seeded} (1 expected: planted confounder "
          f"flagged) unconfounded={code_clean} missing-file={code_err}")
    assert code_seeded == EXIT_FAIL and code_clean == EXIT_OK and code_err == 2

    print("\n== 7. live localhost serve round-trip ==")
    reset_state(rows)
    server = make_server(0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, health = _request(port, "/health")
        assert status == 200 and health == {"status": "ok"}
        status, served_est = _request(port, "/estimate",
                                      payload={}, method="POST")
        assert status == 200
        assert served_est["estimates"]["modification"]["ate"] == \
            estimate_all(pilot)["modification"]["ate"]
        status, served_report = _request(port, "/report")
        assert status == 200 and served_report == build_report(rows)
        print("   POST /estimate and GET /report match local computation")
    finally:
        server.shutdown()
        server.server_close()
        reset_state()

    print("\nDemo OK: generate, honesty check, estimate, design, "
          "simulate, check and serve verified.")


if __name__ == "__main__":
    main()
