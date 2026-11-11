"""Shared row builders for the test suite (not a test module)."""

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_counter = itertools.count(1)


def row(period="pilot", risk_band="B", dpd_bucket=60, income_band="mid",
        balance_band="mid", intervention="no_action", cured=0,
        complaint=0, re_delinquent=0, borrower_id=None):
    return {"borrower_id": borrower_id or f"T-{next(_counter):05d}",
            "period": period, "risk_band": risk_band,
            "dpd_bucket": dpd_bucket, "income_band": income_band,
            "balance_band": balance_band, "intervention": intervention,
            "cured": cured, "complaint": complaint,
            "re_delinquent": re_delinquent}


def arm(intervention, n, cured_count, **kw):
    """``n`` rows for one intervention in one stratum, ``cured_count`` cured."""
    return [row(intervention=intervention, cured=1 if i < cured_count else 0,
                **kw) for i in range(n)]
