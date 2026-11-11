"""Policy simulator: rules, hand-computed scoring, seeded ranking."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import generate_dataset
from helpers import arm, row
from policy import (ACTION_COSTS, RECOVERY_PER_CURE,
                    build_best_per_stratum, policy_comms_first,
                    policy_no_action, policy_targeted, score_policy,
                    simulate_policies)


class TestRules(unittest.TestCase):
    def test_constant_policies(self):
        r = row()
        self.assertEqual(policy_no_action(r), "no_action")
        self.assertEqual(policy_comms_first(r), "comms_only")

    def test_targeted_rule(self):
        self.assertEqual(policy_targeted(
            row(dpd_bucket=90, balance_band="high")), "modification")
        self.assertEqual(policy_targeted(
            row(dpd_bucket=90, balance_band="low")), "payment_plan")
        self.assertEqual(policy_targeted(
            row(dpd_bucket=60, balance_band="high")), "payment_plan")
        self.assertEqual(policy_targeted(
            row(dpd_bucket=30, balance_band="high")), "comms_only")


class TestScoring(unittest.TestCase):
    def _holdout(self):
        # One stratum; control cures 2/10, comms cures 5/10.
        return (arm("no_action", 10, 2, period="holdout")
                + arm("comms_only", 10, 5, period="holdout"))

    def test_score_hand_computed(self):
        score = score_policy(self._holdout(), policy_comms_first, "comms")
        self.assertEqual(score["n_borrowers"], 20)
        self.assertAlmostEqual(score["expected_cures"], 20 * 0.5)
        self.assertAlmostEqual(score["total_cost"], 20 * ACTION_COSTS["comms_only"])
        self.assertAlmostEqual(
            score["net_value"],
            20 * 0.5 * RECOVERY_PER_CURE - 20 * ACTION_COSTS["comms_only"])

    def test_no_action_costs_nothing(self):
        score = score_policy(self._holdout(), policy_no_action, "none")
        self.assertEqual(score["total_cost"], 0.0)
        self.assertAlmostEqual(score["expected_cures"], 20 * 0.2)

    def test_empty_holdout_rejected(self):
        with self.assertRaises(ValueError):
            score_policy([], policy_no_action, "none")


class TestBestPerStratum(unittest.TestCase):
    def test_cost_hurdle_blocks_tiny_effects(self):
        # Stratum where modification's CATE is +0.10: net value
        # 0.10*800 - 120 < 0, so the rule must NOT buy it. Comms at
        # +0.02 nets 0.02*800 - 5 = +11 > 0 and clears its own much
        # lower hurdle - the hurdle is per-action, as it should be.
        pilot = (arm("no_action", 100, 30) + arm("modification", 100, 40)
                 + arm("comms_only", 100, 32) + arm("payment_plan", 100, 31))
        mapping = build_best_per_stratum(pilot)
        self.assertEqual(mapping["B|60|mid"], "comms_only")
        self.assertNotEqual(mapping["B|60|mid"], "modification")

    def test_large_effect_wins(self):
        pilot = (arm("no_action", 100, 20) + arm("modification", 100, 60)
                 + arm("comms_only", 100, 25) + arm("payment_plan", 100, 30))
        mapping = build_best_per_stratum(pilot)
        self.assertEqual(mapping["B|60|mid"], "modification")


class TestSeededSimulation(unittest.TestCase):
    def test_simulate_policies_shape_and_ranking(self):
        sim = simulate_policies(generate_dataset())
        self.assertEqual(sim["best_policy"], "targeted_modification")
        self.assertEqual(sim["ranking"][0], "targeted_modification")
        self.assertEqual(len(sim["policies"]), 4)
        by_name = {p["policy"]: p for p in sim["policies"]}
        self.assertGreater(by_name["targeted_modification"]["net_value"],
                           by_name["no_action"]["net_value"])
        self.assertEqual(by_name["no_action"]["total_cost"], 0.0)


if __name__ == "__main__":
    unittest.main()
