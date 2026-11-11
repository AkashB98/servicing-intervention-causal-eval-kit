"""Validation: generated data clean, planted defects flagged."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import generate_dataset
from helpers import row
from validation import validate_dataset, validate_row


class TestValidation(unittest.TestCase):
    def test_generated_dataset_clean(self):
        self.assertEqual(validate_dataset(generate_dataset()), [])

    def test_missing_field(self):
        r = row()
        del r["cured"]
        issues = validate_row(r)
        self.assertEqual(len(issues), 1)
        self.assertIn("cured", issues[0])

    def test_bad_enums_and_types(self):
        r = row(period="during", risk_band="Z", dpd_bucket=True,
                intervention="foreclosure", cured=2)
        issues = validate_row(r)
        self.assertEqual(len(issues), 5)

    def test_bad_dpd_value(self):
        self.assertEqual(len(validate_row(row(dpd_bucket=45))), 1)

    def test_duplicate_borrower_flagged(self):
        rows = generate_dataset(n_pilot=4, n_observational=4, n_holdout=4)
        rows[1] = dict(rows[1], borrower_id=rows[0]["borrower_id"])
        issues = validate_dataset(rows)
        self.assertTrue(any("duplicate" in i for i in issues))

    def test_empty_dataset(self):
        self.assertEqual(validate_dataset([]), ["dataset is empty"])

    def test_missing_period_flagged(self):
        rows = [r for r in generate_dataset(n_pilot=4, n_observational=4,
                                            n_holdout=4)
                if r["period"] != "holdout"]
        issues = validate_dataset(rows)
        self.assertTrue(any("holdout" in i for i in issues))


if __name__ == "__main__":
    unittest.main()
