"""CLI: commands, exit codes, serve round-trips."""

import io
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import EXIT_ERROR, EXIT_FAIL, EXIT_OK, main as cli_main, make_server, reset_state  # noqa: E402
from generator import generate_dataset, rows_to_jsonl  # noqa: E402
from report import build_report  # noqa: E402

CLEAN_UNCONFOUNDED_SEED = 5000


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main(argv)
    return code, out.getvalue(), err.getvalue()


class CliBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="servicing-test-")
        self.good = os.path.join(self.tmp, "good.jsonl")
        with open(self.good, "w", encoding="utf-8") as fh:
            fh.write(rows_to_jsonl(generate_dataset()))


class TestGenerate(CliBase):
    def test_generate_to_file(self):
        out_path = os.path.join(self.tmp, "gen.jsonl")
        code, _, _ = run_cli(["generate", "--out", out_path])
        self.assertEqual(code, EXIT_OK)
        with open(out_path, encoding="utf-8") as fh:
            self.assertEqual(len(fh.read().strip().splitlines()), 5200)

    def test_generate_csv(self):
        out_path = os.path.join(self.tmp, "gen.csv")
        code, _, _ = run_cli(["generate", "--n-pilot", "4",
                              "--n-observational", "4", "--n-holdout", "4",
                              "--out", out_path])
        self.assertEqual(code, EXIT_OK)
        with open(out_path, encoding="utf-8") as fh:
            lines = fh.read().strip().splitlines()
        self.assertEqual(len(lines), 12 + 1)
        self.assertTrue(lines[0].startswith("borrower_id,period"))


class TestCommands(CliBase):
    def test_estimate_command(self):
        code, out, _ = run_cli(["estimate", "--input", self.good])
        self.assertEqual(code, EXIT_OK)
        payload = json.loads(out)
        self.assertEqual(payload["period"], "pilot")
        self.assertIn("modification", payload["estimates"])

    def test_estimate_observational_period(self):
        code, out, _ = run_cli(["estimate", "--input", self.good,
                                "--period", "observational"])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(json.loads(out)["period"], "observational")

    def test_honesty_command(self):
        code, out, _ = run_cli(["honesty", "--input", self.good])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("modification", json.loads(out)["flagged"])

    def test_design_command(self):
        code, out, _ = run_cli(["design", "--input", self.good])
        self.assertEqual(code, EXIT_OK)
        payload = json.loads(out)
        self.assertEqual(len(payload["table"]), 3)
        self.assertIn("guardrails", payload)

    def test_simulate_command(self):
        code, out, _ = run_cli(["simulate", "--input", self.good])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(json.loads(out)["best_policy"],
                         "targeted_modification")

    def test_report_command_matches_local(self):
        code, out, _ = run_cli(["report", "--input", self.good])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(json.loads(out), build_report(generate_dataset()))

    def test_missing_file_is_error(self):
        code, _, _ = run_cli(["estimate", "--input", "/tmp/nope-servicing.jsonl"])
        self.assertEqual(code, EXIT_ERROR)


class TestCheck(CliBase):
    def test_check_breach_on_seeded_by_design(self):
        code, out, _ = run_cli(["check", "--input", self.good])
        self.assertEqual(code, EXIT_FAIL)
        self.assertIn("check FAIL (honesty)", out)
        self.assertIn("modification", out)

    def test_check_passes_on_unconfounded(self):
        clean = os.path.join(self.tmp, "clean.jsonl")
        with open(clean, "w", encoding="utf-8") as fh:
            fh.write(rows_to_jsonl(generate_dataset(
                seed=CLEAN_UNCONFOUNDED_SEED, confounded=False)))
        code, out, _ = run_cli(["check", "--input", clean])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("check PASS", out)

    def test_check_error_on_missing_file(self):
        code, _, _ = run_cli(["check", "--input", "/tmp/nope-servicing.jsonl"])
        self.assertEqual(code, EXIT_ERROR)

    def test_check_error_on_bad_data(self):
        bad = os.path.join(self.tmp, "bad.jsonl")
        rows = generate_dataset()
        rows[0] = dict(rows[0], cured=7)
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write(rows_to_jsonl(rows))
        code, _, _ = run_cli(["check", "--input", bad])
        self.assertEqual(code, EXIT_ERROR)


class TestServe(CliBase):
    def _serve(self):
        reset_state(generate_dataset())
        server = make_server(0)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, port

    def _get(self, port, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}",
                                    timeout=10) as resp:
            return resp.status, json.loads(resp.read())

    def _post(self, port, path, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())

    def test_health(self):
        server, port = self._serve()
        try:
            self.assertEqual(self._get(port, "/health"),
                             (200, {"status": "ok"}))
        finally:
            server.shutdown(); server.server_close(); reset_state()

    def test_report_round_trip(self):
        server, port = self._serve()
        try:
            status, payload = self._get(port, "/report")
            self.assertEqual(status, 200)
            self.assertEqual(payload, build_report(generate_dataset()))
        finally:
            server.shutdown(); server.server_close(); reset_state()

    def test_estimate_post_round_trip(self):
        server, port = self._serve()
        try:
            status, payload = self._post(port, "/estimate", {})
            self.assertEqual(status, 200)
            self.assertEqual(payload["period"], "pilot")
        finally:
            server.shutdown(); server.server_close(); reset_state()

    def test_honesty_and_simulate_posts(self):
        server, port = self._serve()
        try:
            status, payload = self._post(port, "/honesty", {})
            self.assertEqual(status, 200)
            self.assertIn("modification", payload["flagged"])
            status, payload = self._post(port, "/simulate", {})
            self.assertEqual(status, 200)
            self.assertEqual(payload["best_policy"], "targeted_modification")
        finally:
            server.shutdown(); server.server_close(); reset_state()

    def test_unknown_endpoint_404(self):
        server, port = self._serve()
        try:
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                self._get(port, "/nope")
            self.assertEqual(ctx.exception.code, 404)
        finally:
            server.shutdown(); server.server_close(); reset_state()


if __name__ == "__main__":
    unittest.main()
