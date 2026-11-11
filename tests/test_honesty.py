"""Honesty check: flag logic on hand fixtures + seeded behaviour."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import generate_dataset
from honesty import compare_estimates, honesty_check


def _estimates(ate, se):
    return {name: {"ate": ate, "standard_error": se}
            for name in ("comms_only", "payment_plan", "modification")}


class TestCompareEstimates(unittest.TestCase):
    def test_big_difference_small_noise_flags(self):
        out = compare_estimates(_estimates(0.05, 0.01),
                                _estimates(0.20, 0.01))
        self.assertTrue(out["any_flagged"])
        self.assertEqual(len(out["flagged"]), 3)

    def test_same_difference_big_noise_does_not_flag(self):
        out = compare_estimates(_estimates(0.05, 0.10),
                                _estimates(0.20, 0.10))
        self.assertFalse(out["any_flagged"])

    def test_agreement_does_not_flag(self):
        out = compare_estimates(_estimates(0.07, 0.02),
                                _estimates(0.075, 0.02))
        self.assertFalse(out["any_flagged"])

    def test_threshold_is_z_times_se_difference(self):
        out = compare_estimates(_estimates(0.0, 0.03), _estimates(0.0, 0.04))
        entry = out["per_intervention"]["modification"]
        # se_diff = 0.05, threshold = 1.959964 * 0.05
        self.assertAlmostEqual(entry["se_difference"], 0.05, places=6)
        self.assertAlmostEqual(entry["flag_threshold"], 0.097998, places=5)


class TestSeededHonesty(unittest.TestCase):
    def test_modification_flags_on_seeded_data(self):
        out = honesty_check(generate_dataset())
        self.assertIn("modification", out["flagged"])
        mod = out["per_intervention"]["modification"]
        self.assertGreater(mod["difference"], 0.10)
        # Modification is the most inflated estimate on the seed.
        diffs = {k: abs(v["difference"])
                 for k, v in out["per_intervention"].items()}
        self.assertEqual(max(diffs, key=diffs.get), "modification")

    def test_missing_period_raises(self):
        rows = [r for r in generate_dataset() if r["period"] == "pilot"]
        with self.assertRaises(ValueError):
            honesty_check(rows)


if __name__ == "__main__":
    unittest.main()
