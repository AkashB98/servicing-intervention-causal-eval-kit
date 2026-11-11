"""Experiment design: two-proportion sample size / MDE, plus guardrails.

Sizing uses the classic two-proportion formula only — no simulation,
no fitted model. For a baseline cure rate ``p1``, a target lift
``delta`` (so ``p2 = p1 + delta``), two-sided ``alpha`` and
``power``:

  n per arm = (z_{1-alpha/2} + z_{power})^2
              * ( p1(1-p1) + p2(1-p2) ) / delta^2

**MDE (minimum detectable effect)** — the smallest lift a design
can see with the stated power — inverts the same formula for a
given ``n`` per arm (solved by bisection on the formula itself, so
the two functions agree exactly). Normal quantiles come from
``statistics.NormalDist``.

Guardrails are exactly two, read straight off the pilot per
intervention: the complaint rate and the re-delinquency rate.
"""

from __future__ import annotations

import math
from typing import Dict, List

from estimator import normal_cdf, z_value
from generator import CONTROL, INTERVENTIONS

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80


def sample_size_per_arm(baseline: float, delta: float,
                        alpha: float = DEFAULT_ALPHA,
                        power: float = DEFAULT_POWER) -> int:
    """Borrowers per arm needed to detect ``delta`` over ``baseline``."""
    if not 0.0 < baseline < 1.0:
        raise ValueError("baseline must be in (0, 1)")
    if delta == 0:
        raise ValueError("delta must be non-zero")
    p2 = baseline + delta
    if not 0.0 < p2 < 1.0:
        raise ValueError("baseline + delta must be in (0, 1)")
    z = z_value(1 - alpha / 2) + z_value(power)
    n = (z ** 2) * (baseline * (1 - baseline) + p2 * (1 - p2)) / (delta ** 2)
    return max(2, math.ceil(n))


def power_for_effect(baseline: float, delta: float, n_per_arm: int,
                     alpha: float = DEFAULT_ALPHA) -> float:
    """Detection probability for ``delta`` with ``n_per_arm`` per arm."""
    if n_per_arm < 1:
        raise ValueError("n_per_arm must be >= 1")
    p2 = baseline + delta
    var = baseline * (1 - baseline) + p2 * (1 - p2)
    se = math.sqrt(var / n_per_arm)
    if se == 0:
        return 1.0
    return normal_cdf(abs(delta) / se - z_value(1 - alpha / 2))


def mde_for_n(baseline: float, n_per_arm: int,
              alpha: float = DEFAULT_ALPHA,
              power: float = DEFAULT_POWER) -> float:
    """Smallest detectable lift at ``n_per_arm`` (bisection inverse of
    ``sample_size_per_arm``, so the two agree to rounding)."""
    if not 0.0 < baseline < 1.0:
        raise ValueError("baseline must be in (0, 1)")
    if n_per_arm < 2:
        raise ValueError("n_per_arm must be >= 2")
    lo, hi = 1e-6, 1.0 - baseline - 1e-6
    for _ in range(80):
        mid = (lo + hi) / 2
        if sample_size_per_arm(baseline, mid, alpha, power) > n_per_arm:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 6)


def design_table(baseline: float, deltas=(0.03, 0.05, 0.08),
                 alpha: float = DEFAULT_ALPHA,
                 power: float = DEFAULT_POWER) -> List[Dict]:
    return [{"delta": d,
             "treated_rate": round(baseline + d, 6),
             "n_per_arm": sample_size_per_arm(baseline, d, alpha, power),
             "power": power, "alpha": alpha} for d in deltas]


def guardrail_readouts(rows: List[Dict]) -> Dict:
    """Exactly two guardrails per intervention arm, from ``rows``:
    complaint rate and re-delinquency rate."""
    out = {}
    for name in INTERVENTIONS:
        arm = [r for r in rows if r["intervention"] == name]
        n = len(arm)
        out[name] = {
            "n": n,
            "complaint_rate": round(
                sum(int(r["complaint"]) for r in arm) / n, 6) if n else None,
            "re_delinquency_rate": round(
                sum(int(r["re_delinquent"]) for r in arm) / n, 6) if n else None,
        }
    return out


def pilot_design(rows: List[Dict], deltas=(0.03, 0.05, 0.08)) -> Dict:
    """Design summary anchored on the pilot's control cure rate."""
    control = [r for r in rows if r["intervention"] == CONTROL]
    if not control:
        raise ValueError("no control rows to anchor the design on")
    baseline = sum(int(r["cured"]) for r in control) / len(control)
    n_per_arm_avg = round(len(rows) / len(INTERVENTIONS))
    return {
        "baseline_cure_rate": round(baseline, 6),
        "alpha": DEFAULT_ALPHA, "power": DEFAULT_POWER,
        "table": design_table(baseline, deltas=deltas),
        "mde_at_pilot_size": mde_for_n(baseline, max(2, n_per_arm_avg)),
        "n_per_arm_at_pilot_size": n_per_arm_avg,
        "guardrails": guardrail_readouts(rows),
    }
