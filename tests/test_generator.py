"""Generator: determinism, shape, periods, confounding channel, IO."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import (INTERVENTIONS, generate_dataset, load_rows,
                       rows_for_period, rows_to_csv, rows_to_jsonl,
                       true_effect, write_rows)


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.rows = generate_dataset()

    def test_default_shape(self):
        self.assertEqual(len(self.rows), 5200)
        self.assertEqual(len(rows_for_period(self.rows, "pilot")), 2400)
        self.assertEqual(len(rows_for_period(self.rows, "observational")), 1200)
        self.assertEqual(len(rows_for_period(self.rows, "holdout")), 1600)

    def test_deterministic_same_seed(self):
        self.assertEqual(generate_dataset(), generate_dataset())

    def test_different_seed_differs(self):
        self.assertNotEqual(generate_dataset(seed=7), self.rows)

    def test_borrower_ids_unique(self):
        ids = [r["borrower_id"] for r in self.rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(i.startswith("NS-") for i in ids))

    def test_outcomes_binary_and_arms_complete(self):
        for field in ("cured", "complaint", "re_delinquent"):
            self.assertTrue(all(r[field] in (0, 1) for r in self.rows))
        for period in ("pilot", "observational", "holdout"):
            arms = {r["intervention"]
                    for r in rows_for_period(self.rows, period)}
            self.assertEqual(arms, set(INTERVENTIONS))

    def test_no_latent_trait_in_rows(self):
        # Willingness is planted inside the generator only; it must
        # never leak into the analysis data.
        for r in self.rows:
            self.assertNotIn("willingness", r)
            self.assertNotIn("latent_willingness", r)

    def test_confounding_touches_only_observational(self):
        plain = generate_dataset(confounded=False)
        self.assertEqual(rows_for_period(plain, "pilot"),
                         rows_for_period(self.rows, "pilot"))
        self.assertEqual(rows_for_period(plain, "holdout"),
                         rows_for_period(self.rows, "holdout"))
        self.assertNotEqual(rows_for_period(plain, "observational"),
                            rows_for_period(self.rows, "observational"))

    def test_true_effects_segment_pattern(self):
        self.assertGreater(true_effect("comms_only", 30, "mid"),
                           true_effect("comms_only", 90, "mid"))
        self.assertGreater(true_effect("modification", 90, "high"),
                           true_effect("modification", 90, "low"))
        self.assertEqual(true_effect("no_action", 90, "high"), 0.0)

    def test_negative_sizes_rejected(self):
        with self.assertRaises(ValueError):
            generate_dataset(n_pilot=-1)

    def test_jsonl_csv_round_trip(self):
        tmp = tempfile.mkdtemp(prefix="servicing-gen-")
        for name, fmt in (("d.jsonl", "jsonl"), ("d.csv", "csv")):
            path = os.path.join(tmp, name)
            write_rows(self.rows, path, fmt=fmt)
            self.assertEqual(load_rows(path), self.rows)
        self.assertEqual(rows_to_jsonl(self.rows).count("\n"), 5200)
        self.assertTrue(rows_to_csv(self.rows).startswith("borrower_id,"))


if __name__ == "__main__":
    unittest.main()
