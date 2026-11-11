"""The honesty check: randomised pilot vs observational book.

Both arms of the kit estimate the same stratified ATE — **ATE
(average treatment effect)** — the mean effect across the whole
population — for each intervention. On randomised data the estimator
is unbiased by design. On the observational book, assignment depends
on a planted latent willingness trait that also raises cure rates
and is *not* in the analysis covariates, so the observational
estimate is expected to disagree — most strongly for modification,
the intervention agents preferentially offer to borrowers who
"sound willing".

The check is a named, documented rule:

  ``honesty_check`` flags intervention ``t`` when the disagreement
  is larger than sampling noise alone would produce at the 5%
  level:

      | ATE_obs(t) - ATE_pilot(t) |  >  z_{0.975} * SE_difference

  where ``SE_difference = sqrt(SE_pilot^2 + SE_obs^2)`` from the
  two stratified estimators. A fixed percentage-point threshold
  was considered and rejected: at these sample sizes the standard
  error of the difference is itself several points, so a fixed
  threshold flags noise, not confounding (the calibration evals
  verify the firing rate on unconfounded data).

A flag is not an estimate and not a verdict on the observational
*data*; it is a warning label: *this observational number must not
be used as if it were causal.* The observational arm exists in this
kit as that warning, demonstrated on data where the truth is known.
"""

from __future__ import annotations

import math
from typing import Dict, List

from estimator import estimate_all, z_value
from generator import CONTROL, INTERVENTIONS, rows_for_period

FLAG_ALPHA = 0.05


def compare_estimates(pilot_ate: Dict, observational_ate: Dict,
                      alpha: float = FLAG_ALPHA) -> Dict:
    """Pure comparison logic (hand-testable): per-intervention
    disagreement, its standard error, and the flag, from two
    ``estimate_all`` results."""
    z_crit = z_value(1 - alpha / 2)
    per = {}
    flagged = []
    for name in INTERVENTIONS:
        if name == CONTROL:
            continue
        p = pilot_ate[name]["ate"]
        o = observational_ate[name]["ate"]
        se_diff = math.sqrt(pilot_ate[name]["standard_error"] ** 2
                            + observational_ate[name]["standard_error"] ** 2)
        diff = round(o - p, 6)
        threshold = round(z_crit * se_diff, 6)
        is_flag = abs(diff) > threshold
        per[name] = {"pilot_ate": p, "observational_ate": o,
                     "difference": diff,
                     "se_difference": round(se_diff, 6),
                     "flag_threshold": threshold,
                     "flagged": bool(is_flag)}
        if is_flag:
            flagged.append(name)
    return {"alpha": alpha, "per_intervention": per,
            "flagged": flagged, "any_flagged": bool(flagged)}


def honesty_check(rows: List[Dict], alpha: float = FLAG_ALPHA) -> Dict:
    """Estimate both arms from a combined dataset and compare them."""
    pilot = rows_for_period(rows, "pilot")
    observational = rows_for_period(rows, "observational")
    if not pilot or not observational:
        raise ValueError("honesty check needs pilot and observational rows")
    return compare_estimates(estimate_all(pilot),
                             estimate_all(observational),
                             alpha=alpha)
