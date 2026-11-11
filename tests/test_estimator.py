"""Estimator: hand-computed stratified difference-in-means."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from estimator import (estimate_all, estimate_ate, estimate_stratum,
                       stratum_key, stratum_label, z_value)
from helpers import arm, row


class TestStratumMath(unittest.TestCase):
    def test_single_stratum_hand_computed(self):
        rows = (arm("no_action", 10, 3) + arm("payment_plan", 10, 6))
        est = estimate_stratum(rows, "payment_plan")
        self.assertEqual(est["cate"], 0.3)
        # var = .6*.4/10 + .3*.7/10 = 0.045
        self.assertAlmostEqual(est["variance"], 0.045, places=12)

    def test_pooled_two_strata_hand_computed(self):
        # Stratum 1 (B|60|mid): 20 rows, CATE 0.3. Stratum 2
        # (D|90|low): 10 rows, CATE 0.0. Weights 2/3, 1/3 -> ATE 0.2.
        rows = (arm("no_action", 10, 3) + arm("payment_plan", 10, 6)
                + arm("no_action", 5, 1, risk_band="D", dpd_bucket=90,
                      income_band="low")
                + arm("payment_plan", 5, 1, risk_band="D", dpd_bucket=90,
                      income_band="low"))
        est = estimate_ate(rows, "payment_plan")
        self.assertAlmostEqual(est["ate"], 0.2, places=6)
        var = (2 / 3) ** 2 * 0.045 + (1 / 3) ** 2 * (0.2 * 0.8 / 5 * 2)
        self.assertAlmostEqual(est["standard_error"], math.sqrt(var),
                               places=5)
        self.assertEqual(est["coverage"], 1.0)
        self.assertEqual(len(est["strata"]), 2)

    def test_ci_uses_normal_quantile(self):
        rows = arm("no_action", 10, 3) + arm("payment_plan", 10, 6)
        est = estimate_ate(rows, "payment_plan")
        half = z_value(0.975) * est["standard_error"]
        self.assertAlmostEqual(est["ci_95"][1] - est["ate"], half, places=5)

    def test_missing_arm_stratum_excluded_and_reported(self):
        rows = (arm("no_action", 10, 3) + arm("payment_plan", 10, 6)
                + arm("no_action", 4, 1, risk_band="A", dpd_bucket=30,
                      income_band="high"))
        est = estimate_ate(rows, "payment_plan")
        self.assertEqual(est["n_rows"], 24)
        self.assertEqual(est["n_covered"], 20)
        self.assertAlmostEqual(est["coverage"], 20 / 24, places=6)
        self.assertEqual(est["ate"], 0.3)

    def test_control_against_itself_rejected(self):
        with self.assertRaises(ValueError):
            estimate_ate(arm("no_action", 4, 1), "no_action")

    def test_unknown_intervention_rejected(self):
        with self.assertRaises(ValueError):
            estimate_ate(arm("no_action", 4, 1), "foreclosure")

    def test_stratum_key_and_label(self):
        r = row(risk_band="C", dpd_bucket=90, income_band="low")
        self.assertEqual(stratum_key(r), ("C", 90, "low"))
        self.assertEqual(stratum_label(("C", 90, "low")), "C|90|low")

    def test_estimate_all_covers_three_interventions(self):
        from generator import generate_dataset, rows_for_period
        pilot = rows_for_period(generate_dataset(), "pilot")
        out = estimate_all(pilot)
        self.assertEqual(set(out),
                         {"comms_only", "payment_plan", "modification"})
        for est in out.values():
            self.assertLess(est["ci_95"][0], est["ate"])
            self.assertGreater(est["ci_95"][1], est["ate"])


if __name__ == "__main__":
    unittest.main()
