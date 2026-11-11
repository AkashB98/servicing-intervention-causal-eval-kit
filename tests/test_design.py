"""Design: two-proportion sizing hand checks + guardrails."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from design import (guardrail_readouts, mde_for_n, pilot_design,
                    power_for_effect, sample_size_per_arm)
from estimator import z_value
from helpers import arm, row


class TestTwoProportion(unittest.TestCase):
    def test_sample_size_hand_computed(self):
        # z = z_.975 + z_.80; n = z^2 (p1q1 + p2q2) / delta^2
        z = z_value(0.975) + z_value(0.80)
        expected = math.ceil(z ** 2 * (0.4 * 0.6 + 0.45 * 0.55) / 0.05 ** 2)
        self.assertEqual(sample_size_per_arm(0.4, 0.05), expected)
        self.assertEqual(expected, 1531)

    def test_bigger_delta_needs_fewer_borrowers(self):
        self.assertGreater(sample_size_per_arm(0.4, 0.03),
                           sample_size_per_arm(0.4, 0.08))

    def test_mde_inverts_sample_size(self):
        n = sample_size_per_arm(0.4, 0.05)
        mde = mde_for_n(0.4, n)
        self.assertAlmostEqual(mde, 0.05, delta=0.002)
        self.assertLessEqual(sample_size_per_arm(0.4, mde), n + 1)

    def test_power_at_sized_n_is_near_target(self):
        n = sample_size_per_arm(0.35, 0.06)
        self.assertAlmostEqual(power_for_effect(0.35, 0.06, n), 0.80,
                               delta=0.01)

    def test_power_increases_with_n(self):
        self.assertLess(power_for_effect(0.35, 0.06, 200),
                        power_for_effect(0.35, 0.06, 800))

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            sample_size_per_arm(0.0, 0.05)
        with self.assertRaises(ValueError):
            sample_size_per_arm(0.4, 0.0)
        with self.assertRaises(ValueError):
            sample_size_per_arm(0.95, 0.10)


class TestGuardrails(unittest.TestCase):
    def test_guardrail_rates_hand_computed(self):
        rows = []
        for i in range(10):
            rows.append(row(intervention="comms_only",
                            complaint=1 if i < 2 else 0,
                            re_delinquent=1 if i < 3 else 0))
        out = guardrail_readouts(rows)
        self.assertEqual(out["comms_only"]["complaint_rate"], 0.2)
        self.assertEqual(out["comms_only"]["re_delinquency_rate"], 0.3)
        self.assertIsNone(out["modification"]["complaint_rate"])

    def test_exactly_two_guardrails(self):
        out = guardrail_readouts(arm("no_action", 4, 1))
        for entry in out.values():
            self.assertEqual(set(entry),
                             {"n", "complaint_rate", "re_delinquency_rate"})

    def test_pilot_design_anchors_on_control_rate(self):
        rows = (arm("no_action", 10, 4) + arm("comms_only", 10, 5)
                + arm("payment_plan", 10, 5) + arm("modification", 10, 6))
        design = pilot_design(rows)
        self.assertEqual(design["baseline_cure_rate"], 0.4)
        self.assertEqual(len(design["table"]), 3)
        self.assertEqual(design["table"][1]["n_per_arm"],
                         sample_size_per_arm(0.4, 0.05))


if __name__ == "__main__":
    unittest.main()
