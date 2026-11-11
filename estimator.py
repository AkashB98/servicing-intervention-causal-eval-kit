"""Stratified CATE estimator: difference-in-means within exact strata.

The method is exactly what this module calls it: a **stratified
CATE** — **CATE (conditional average treatment effect)** — the
average effect inside one subgroup — estimated by
difference-in-means within each stratum, then pooled. It is *not*
uplift modelling and *not* causal machine learning: there is no
fitted outcome model, no propensity model, and no logistic
regression anywhere in this kit.

Strata are exact cells of the discrete analysis covariates only:

  risk band  x  days-past-due bucket  x  income band   (4 x 3 x 3 = 36)

For intervention ``t`` vs the ``no_action`` control, the stratum
estimate is ``mean(cured | t, s) - mean(cured | control, s)``, and
the pooled ATE — **ATE (average treatment effect)** — the mean
effect across the whole population — weights each stratum by its
share of the rows being analysed. Strata missing either arm are
excluded and the remaining weights renormalised; the share of rows
covered is reported so the exclusion is never silent.

Standard errors come from the two-proportion variance inside each
stratum, combined under the same weights:

  Var = sum_s  w_s^2 * ( p_t(1-p_t)/n_t + p_c(1-p_c)/n_c )

Confidence intervals use quantiles from ``statistics.NormalDist``
(never a hard-coded 1.96).
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Dict, List, Tuple

from generator import CONTROL, INTERVENTIONS

_NORM = NormalDist()


def z_value(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    return _NORM.inv_cdf(p)


def normal_cdf(x: float) -> float:
    return _NORM.cdf(x)


def stratum_key(row: Dict) -> Tuple[str, int, str]:
    return (row["risk_band"], int(row["dpd_bucket"]), row["income_band"])


def stratum_label(key: Tuple[str, int, str]) -> str:
    return f"{key[0]}|{key[1]}|{key[2]}"


def _rate(rows: List[Dict]) -> float:
    return sum(int(r["cured"]) for r in rows) / len(rows) if rows else 0.0


def estimate_stratum(rows: List[Dict], intervention: str) -> Dict:
    """Difference-in-means for one stratum's rows (both arms present)."""
    treated = [r for r in rows if r["intervention"] == intervention]
    control = [r for r in rows if r["intervention"] == CONTROL]
    if not treated or not control:
        raise ValueError("stratum needs both arms to estimate an effect")
    p_t, p_c = _rate(treated), _rate(control)
    var = (p_t * (1 - p_t) / len(treated)
           + p_c * (1 - p_c) / len(control))
    return {"n_treated": len(treated), "n_control": len(control),
            "treated_rate": p_t, "control_rate": p_c,
            "cate": p_t - p_c, "variance": var}


def estimate_ate(rows: List[Dict], intervention: str,
                 alpha: float = 0.05) -> Dict:
    """Pooled stratified ATE of ``intervention`` vs control on ``rows``."""
    if intervention == CONTROL:
        raise ValueError("cannot estimate the control against itself")
    if intervention not in INTERVENTIONS:
        raise ValueError(f"unknown intervention: {intervention}")
    by_stratum: Dict[Tuple, List[Dict]] = {}
    for row in rows:
        by_stratum.setdefault(stratum_key(row), []).append(row)
    total = len(rows)
    ate = 0.0
    var = 0.0
    covered = 0
    strata_out = []
    raw_weights = {}
    estimable = {}
    for key in sorted(by_stratum):
        srows = by_stratum[key]
        arms = {r["intervention"] for r in srows}
        if intervention in arms and CONTROL in arms:
            est = estimate_stratum(srows, intervention)
            estimable[key] = (srows, est)
            raw_weights[key] = len(srows)
    covered = sum(raw_weights.values())
    if not estimable or total == 0:
        raise ValueError("no stratum has both arms for this intervention")
    for key in sorted(estimable):
        srows, est = estimable[key]
        w = raw_weights[key] / covered
        ate += w * est["cate"]
        var += (w ** 2) * est["variance"]
        strata_out.append({
            "stratum": stratum_label(key),
            "n": len(srows),
            "n_treated": est["n_treated"],
            "n_control": est["n_control"],
            "weight": round(w, 6),
            "treated_rate": round(est["treated_rate"], 6),
            "control_rate": round(est["control_rate"], 6),
            "cate": round(est["cate"], 6),
        })
    se = math.sqrt(var)
    z = z_value(1 - alpha / 2)
    return {
        "intervention": intervention,
        "control": CONTROL,
        "ate": round(ate, 6),
        "standard_error": round(se, 6),
        "ci_95": [round(ate - z * se, 6), round(ate + z * se, 6)],
        "alpha": alpha,
        "n_rows": total,
        "n_covered": covered,
        "coverage": round(covered / total, 6) if total else 0.0,
        "strata": strata_out,
    }


def estimate_all(rows: List[Dict], alpha: float = 0.05) -> Dict:
    """ATE per non-control intervention on ``rows``."""
    return {name: estimate_ate(rows, name, alpha=alpha)
            for name in INTERVENTIONS if name != CONTROL}
