"""Full report: pilot estimates + honesty check + design + policies.

Assembles the four questions a servicing-intervention evaluation
must answer before anyone acts on it:

  1. What did the randomised pilot estimate?  -> ``pilot_estimates``
  2. Does the observational book agree?       -> ``honesty``
     (on the seeded data it does not - that disagreement, flagged
     by name, is the point of the kit)
  3. Was the pilot big enough, and what did  -> ``design``
     the guardrails do?
  4. Which decision policy earns most on      -> ``policies``
     randomised held-out data?
"""

from __future__ import annotations

from typing import Dict, List

from design import pilot_design
from estimator import estimate_all
from generator import rows_for_period
from honesty import honesty_check
from policy import simulate_policies


def build_report(rows: List[Dict]) -> Dict:
    pilot = rows_for_period(rows, "pilot")
    observational = rows_for_period(rows, "observational")
    holdout = rows_for_period(rows, "holdout")
    return {
        "dataset": {
            "rows": len(rows),
            "pilot_rows": len(pilot),
            "observational_rows": len(observational),
            "holdout_rows": len(holdout),
            "company": "Northstar Lending (fictional)",
            "synthetic": True,
        },
        "pilot_estimates": estimate_all(pilot),
        "observational_estimates": estimate_all(observational),
        "honesty": honesty_check(rows),
        "design": pilot_design(pilot),
        "policies": simulate_policies(rows),
    }
