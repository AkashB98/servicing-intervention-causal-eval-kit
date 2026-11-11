"""Golden evals for servicing-intervention-causal-eval-kit.

SYNTHETIC DATA ONLY (fictional Northstar Lending datasets). Every
eval is deterministic: run_evals.py writes eval_report.json and the
file must be byte-identical across runs. No timestamps, ports, or
temp paths enter the report - only names, booleans, pinned counts
and hashes.

E1-E12 pin the seeded pipeline. E13-E15 are calibration evals across
fresh seeds: the pilot estimator's 95% CI coverage of the planted
effects, and the honesty check's firing rate on confounded vs
unconfounded observational data (it must fire on the first and
rarely on the second - a check that flags everything proves
nothing).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import threading
import urllib.request
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import EXIT_ERROR, EXIT_FAIL, EXIT_OK, main as cli_main, make_server, reset_state  # noqa: E402
from design import pilot_design, sample_size_per_arm  # noqa: E402
from estimator import estimate_all, estimate_ate  # noqa: E402
from generator import (generate_dataset, load_rows, rows_for_period,
                       rows_to_jsonl, true_effect)  # noqa: E402
from honesty import honesty_check  # noqa: E402
from policy import simulate_policies  # noqa: E402
from report import build_report  # noqa: E402
from validation import validate_dataset, validate_row  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXTURE = os.path.join(ROOT, "fixtures", "sample_borrowers.jsonl")
CLEAN_UNCONFOUNDED_SEED = 5000
RESULTS = []


def check(name, passed, detail=""):
    RESULTS.append({"name": name, "passed": bool(passed), "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return passed


def _run_cli(argv):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return cli_main(argv)


def main():
    print("== golden evals: servicing-intervention-causal-eval-kit ==")
    rows = generate_dataset()
    pilot = rows_for_period(rows, "pilot")
    observational = rows_for_period(rows, "observational")

    # E1: generator determinism (bytes pinned)
    text1 = rows_to_jsonl(generate_dataset())
    text2 = rows_to_jsonl(generate_dataset())
    md5 = hashlib.md5(text1.encode()).hexdigest()
    check("E1 generator determinism (seed=42, 5200 borrowers)",
          text1 == text2 and len(rows) == 5200
          and md5 == "a813e256c94d9555fa68bf869952a527",
          f"jsonl_md5={md5}")

    # E2: period counts pinned
    check("E2 period counts pinned (pilot 2400, observational 1200, holdout 1600)",
          len(pilot) == 2400 and len(observational) == 1200
          and len(rows_for_period(rows, "holdout")) == 1600)

    # E3: pilot (randomised) ATEs pinned
    pe = estimate_all(pilot)
    check("E3 pilot ATEs pinned (comms .0360, plan .0508, modification .0739)",
          pe["comms_only"]["ate"] == 0.035991
          and pe["payment_plan"]["ate"] == 0.050758
          and pe["modification"]["ate"] == 0.073893
          and pe["modification"]["ci_95"] == [0.020411, 0.127376],
          f"modification_ate={pe['modification']['ate']}")

    # E4: observational ATEs pinned (inflated, by design)
    oe = estimate_all(observational)
    check("E4 observational ATEs pinned (comms .1527, plan .1075, modification .2582)",
          oe["comms_only"]["ate"] == 0.152745
          and oe["payment_plan"]["ate"] == 0.107527
          and oe["modification"]["ate"] == 0.258226,
          f"modification_ate={oe['modification']['ate']}")

    # E5: honesty check pinned - comms + modification flag,
    # modification's disagreement is the largest.
    honesty = honesty_check(rows)
    mod = honesty["per_intervention"]["modification"]
    check("E5 honesty pinned (flagged comms_only+modification, mod diff +0.184333)",
          honesty["flagged"] == ["comms_only", "modification"]
          and mod["difference"] == 0.184333 and mod["flagged"]
          and not honesty["per_intervention"]["payment_plan"]["flagged"],
          f"flagged={honesty['flagged']}")

    # E6: estimator hand-computed (single stratum: 3/10 vs 6/10)
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from helpers import arm  # noqa: E402
    fixture = arm("no_action", 10, 3) + arm("payment_plan", 10, 6)
    est = estimate_ate(fixture, "payment_plan")
    check("E6 estimator hand-computed (ATE 0.3, variance 0.045)",
          est["ate"] == 0.3 and abs(est["standard_error"] - 0.045 ** 0.5) < 1e-5,
          f"ate={est['ate']} se={est['standard_error']}")

    # E7: two-proportion sizing hand-computed
    check("E7 sample size hand-computed (baseline .40, delta .05 -> 1531/arm)",
          sample_size_per_arm(0.40, 0.05) == 1531
          and sample_size_per_arm(0.40, 0.03) > sample_size_per_arm(0.40, 0.08))

    # E8: pilot design pinned (baseline, table, MDE, guardrails)
    design = pilot_design(pilot)
    check("E8 pilot design pinned (baseline .478896, MDE .080552, n 4356/1566/609)",
          design["baseline_cure_rate"] == 0.478896
          and design["mde_at_pilot_size"] == 0.080552
          and [t["n_per_arm"] for t in design["table"]] == [4356, 1566, 609]
          and design["guardrails"]["comms_only"]["complaint_rate"] == 0.054313
          and design["guardrails"]["no_action"]["re_delinquency_rate"] == 0.094156,
          f"baseline={design['baseline_cure_rate']}")

    # E9: policy simulation pinned (ranking + net values, holdout only)
    sim = simulate_policies(rows)
    by_name = {p["policy"]: p for p in sim["policies"]}
    check("E9 policies pinned (targeted wins, net $721,629.05; best-per-stratum 3rd)",
          sim["ranking"] == ["targeted_modification", "comms_first",
                             "best_per_stratum", "no_action"]
          and by_name["targeted_modification"]["net_value"] == 721629.05
          and by_name["best_per_stratum"]["net_value"] == 631859.48
          and by_name["no_action"]["expected_cures"] == 711.1408,
          f"best={sim['best_policy']}")

    # E10: validation flags planted defects, generated book clean
    bad = dict(rows[0], period="during", cured=2)
    planted = validate_row(bad)
    check("E10 validation flags planted defects, generated book clean",
          len(planted) == 2 and validate_dataset(rows) == [],
          f"planted={planted}")

    # E11: CLI check exit codes 1 (seeded, by design) / 0 / 2
    import tempfile
    tmp = tempfile.mkdtemp(prefix="servicing-evals-")
    seeded_path = os.path.join(tmp, "seeded.jsonl")
    with open(seeded_path, "w", encoding="utf-8") as fh:
        fh.write(rows_to_jsonl(rows))
    clean_path = os.path.join(tmp, "clean.jsonl")
    with open(clean_path, "w", encoding="utf-8") as fh:
        fh.write(rows_to_jsonl(generate_dataset(
            seed=CLEAN_UNCONFOUNDED_SEED, confounded=False)))
    codes = (_run_cli(["check", "--input", seeded_path]),
             _run_cli(["check", "--input", clean_path]),
             _run_cli(["check", "--input", "/tmp/no-such-servicing-xyz.jsonl"]))
    check("E11 CLI check exit codes 1/0/2 (seeded breach is by design)",
          codes == (EXIT_FAIL, EXIT_OK, EXIT_ERROR), f"codes={codes}")

    # E12: serve round-trip matches local computation
    reset_state(rows)
    server = make_server(0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/honesty", data=b"{}",
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            served_honesty = json.loads(resp.read())
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/report",
                                    timeout=10) as resp:
            served_report = json.loads(resp.read())
        serve_ok = (served_honesty == honesty
                    and served_report == build_report(rows))
    finally:
        server.shutdown()
        server.server_close()
        reset_state()
    check("E12 serve round-trip matches local honesty + report", serve_ok)

    # E13: bundled fixture pinned (5200 rows, clean, same report)
    fixture_rows = load_rows(FIXTURE)
    report = build_report(rows)
    check("E13 fixture pinned (5200 rows, clean, report identical)",
          len(fixture_rows) == 5200 and validate_dataset(fixture_rows) == []
          and build_report(fixture_rows) == report)

    # ---- calibration evals across fresh seeds ----
    print("  .. calibration evals (fresh seeds) ..")

    # E14: CI coverage - for each seed, the sample's true ATE is the
    # mean planted effect over its pilot rows; the pilot 95% CI must
    # cover it about 95% of the time. Pinned counts over 40 seeds.
    covered = {"comms_only": 0, "payment_plan": 0, "modification": 0}
    for seed in range(7000, 7040):
        r = generate_dataset(seed=seed)
        p = rows_for_period(r, "pilot")
        for name in covered:
            truth = sum(true_effect(name, row["dpd_bucket"],
                                    row["balance_band"])
                        for row in p) / len(p)
            e = estimate_ate(p, name)
            covered[name] += e["ci_95"][0] <= truth <= e["ci_95"][1]
    check("E14 CI coverage (pinned 37/40 comms, 40/40 plan, 39/40 modification)",
          covered == {"comms_only": 37, "payment_plan": 40,
                      "modification": 39},
          f"covered={covered}")

    # E15: honesty firing rates - on confounded observational data
    # (30 seeds) modification must flag essentially always; on the
    # unconfounded variant (30 seeds) it must rarely flag.
    conf_flags = sum("modification" in honesty_check(
        generate_dataset(seed=s))["flagged"] for s in range(6000, 6030))
    unc_flags = sum("modification" in honesty_check(
        generate_dataset(seed=s, confounded=False))["flagged"]
        for s in range(5000, 5030))
    check("E15 honesty firing (confounded 30/30, unconfounded 4/30)",
          conf_flags == 30 and unc_flags == 4,
          f"confounded={conf_flags}/30 unconfounded={unc_flags}/30")

    passed = sum(1 for r in RESULTS if r["passed"])
    total_n = len(RESULTS)
    summary_doc = {"passed": passed, "total": total_n,
                   "all_passed": passed == total_n, "evals": RESULTS}
    with open(os.path.join(HERE, "eval_report.json"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary_doc, sort_keys=True) + "\n")
    print(f"\n{passed}/{total_n} evals passed -> evals/eval_report.json")
    return 0 if passed == total_n else 1


if __name__ == "__main__":
    sys.exit(main())
