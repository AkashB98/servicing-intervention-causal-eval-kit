"""Data-quality validation for borrower rows.

Returns a list of human-readable issue strings; an empty list means
the dataset is clean. Validation never raises on bad *data* - it
reports - so the CLI ``check`` gate can map issues to exit code 2.
"""

from __future__ import annotations

from typing import Dict, List

from generator import (BALANCE_BANDS, DPD_BUCKETS, INCOME_BANDS,
                       INTERVENTIONS, PERIODS, RISK_BANDS, ROW_FIELDS)

REQUIRED_FIELDS = ROW_FIELDS
_BINARY = ("cured", "complaint", "re_delinquent")


def validate_row(row: Dict, index: int = 0) -> List[str]:
    issues: List[str] = []
    label = row.get("borrower_id", f"<row {index}>")
    for field in REQUIRED_FIELDS:
        if field not in row or row[field] is None or row[field] == "":
            issues.append(f"{label}: missing field '{field}'")
    if issues:
        return issues
    if not isinstance(row["borrower_id"], str) or not row["borrower_id"]:
        issues.append(f"{label}: borrower_id must be a non-empty string")
    if row.get("period") not in PERIODS:
        issues.append(f"{label}: period must be one of {PERIODS}")
    if row.get("risk_band") not in RISK_BANDS:
        issues.append(f"{label}: risk_band must be one of {RISK_BANDS}")
    dpd = row.get("dpd_bucket")
    if not isinstance(dpd, int) or isinstance(dpd, bool):
        issues.append(f"{label}: dpd_bucket must be an int")
    elif dpd not in DPD_BUCKETS:
        issues.append(f"{label}: dpd_bucket must be one of {DPD_BUCKETS}")
    if row.get("income_band") not in INCOME_BANDS:
        issues.append(f"{label}: income_band must be one of {INCOME_BANDS}")
    if row.get("balance_band") not in BALANCE_BANDS:
        issues.append(f"{label}: balance_band must be one of {BALANCE_BANDS}")
    if row.get("intervention") not in INTERVENTIONS:
        issues.append(f"{label}: intervention must be one of {INTERVENTIONS}")
    for field in _BINARY:
        v = row.get(field)
        if not isinstance(v, int) or isinstance(v, bool) or v not in (0, 1):
            issues.append(f"{label}: {field} must be 0 or 1")
    return issues


def validate_dataset(rows: List[Dict]) -> List[str]:
    issues: List[str] = []
    if not rows:
        return ["dataset is empty"]
    for i, row in enumerate(rows):
        issues.extend(validate_row(row, index=i))
    seen = set()
    for row in rows:
        key = row.get("borrower_id")
        if key in seen:
            issues.append(f"{key}: duplicate borrower_id")
        seen.add(key)
    periods = {r.get("period") for r in rows}
    for period in PERIODS:
        if period not in periods:
            issues.append(f"dataset has no {period} rows")
    return issues
