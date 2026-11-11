"""Decision policies scored ONLY on the randomised held-out period.

A policy is a rule mapping a borrower's characteristics to one
intervention. Policies are scored on the held-out randomised period
— never on observational data, and there is no off-policy evaluation
of observational data anywhere in this kit.

Scoring method (plug-in on randomised cells): because holdout
assignment is randomised within strata, the observed cure /
complaint / re-delinquency rates of each stratum x intervention
cell are unbiased for that cell. A policy's expected outcome for a
holdout borrower is the rate of the cell the policy sends them to;
averaging over all holdout borrowers scores the policy. Cells that
are empty fall back to the intervention's overall holdout rate.
This uses every holdout row and stays unbiased under randomisation.

Economics are flat and explicitly fictional: a flat cost per action
and a flat recovery value per cure. There is no loan-modification
NPV and no re-default modelling.

  costs (fictional):    no_action $0, comms_only $5,
                        payment_plan $25, modification $120
  recovery per cure:    $800 flat (fictional)
"""

from __future__ import annotations

from typing import Callable, Dict, List

from estimator import estimate_all, stratum_key
from generator import CONTROL, INTERVENTIONS, rows_for_period

ACTION_COSTS = {"no_action": 0.0, "comms_only": 5.0,
                "payment_plan": 25.0, "modification": 120.0}
RECOVERY_PER_CURE = 800.0


def policy_no_action(row: Dict) -> str:
    return "no_action"


def policy_comms_first(row: Dict) -> str:
    return "comms_only"


def policy_targeted(row: Dict) -> str:
    """Modification for 90-DPD high-balance, plan otherwise in late
    delinquency, comms first for early delinquency."""
    if int(row["dpd_bucket"]) == 90 and row["balance_band"] == "high":
        return "modification"
    if int(row["dpd_bucket"]) in (60, 90):
        return "payment_plan"
    return "comms_only"


def build_best_per_stratum(pilot_rows: List[Dict]) -> Dict:
    """Map each stratum to an intervention by pilot stratified
    CATE — **CATE (conditional average treatment effect)** — the
    average effect inside one subgroup — translated into a decision
    the way a servicing team would: an intervention is chosen only
    when its estimated net value in that stratum,
    ``CATE * recovery_per_cure - action_cost``, beats no action.
    That cost hurdle matters: stratum CATEs are noisy, and without
    it the rule buys the most expensive action wherever noise is
    kindest. Strata with no positive-net option get no action."""
    estimates = estimate_all(pilot_rows)
    best: Dict = {}
    all_strata = set()
    for est in estimates.values():
        for s in est["strata"]:
            all_strata.add(s["stratum"])
    cate_by: Dict[str, Dict[str, float]] = {}
    for name, est in estimates.items():
        for s in est["strata"]:
            cate_by.setdefault(s["stratum"], {})[name] = s["cate"]
    for label in all_strata:
        options = cate_by.get(label, {})
        net = {name: cate * RECOVERY_PER_CURE - ACTION_COSTS[name]
               for name, cate in options.items()}
        if not net:
            best[label] = CONTROL
            continue
        winner = max(net, key=lambda k: net[k])
        best[label] = winner if net[winner] > 0 else CONTROL
    return best


def make_best_per_stratum_policy(mapping: Dict) -> Callable[[Dict], str]:
    from estimator import stratum_label
    def _policy(row: Dict) -> str:
        return mapping.get(stratum_label(stratum_key(row)), CONTROL)
    return _policy


def _cell_rates(holdout_rows: List[Dict]) -> Dict:
    cells: Dict = {}
    overall: Dict[str, Dict[str, float]] = {}
    for name in INTERVENTIONS:
        arm = [r for r in holdout_rows if r["intervention"] == name]
        if arm:
            overall[name] = {
                "cured": sum(int(r["cured"]) for r in arm) / len(arm),
                "complaint": sum(int(r["complaint"]) for r in arm) / len(arm),
                "re_delinquent": sum(int(r["re_delinquent"]) for r in arm) / len(arm),
            }
    for row in holdout_rows:
        cells.setdefault((stratum_key(row), row["intervention"]), []).append(row)
    rates = {}
    for key, rs in cells.items():
        rates[key] = {
            "cured": sum(int(r["cured"]) for r in rs) / len(rs),
            "complaint": sum(int(r["complaint"]) for r in rs) / len(rs),
            "re_delinquent": sum(int(r["re_delinquent"]) for r in rs) / len(rs),
        }
    return {"cells": rates, "overall": overall}


def score_policy(holdout_rows: List[Dict], policy_fn: Callable,
                 name: str) -> Dict:
    if not holdout_rows:
        raise ValueError("no holdout rows to score on")
    rates = _cell_rates(holdout_rows)
    cures = cost = complaints = redelinq = 0.0
    assignments: Dict[str, int] = {k: 0 for k in INTERVENTIONS}
    for row in holdout_rows:
        action = policy_fn(row)
        if action not in INTERVENTIONS:
            raise ValueError(f"policy returned unknown action: {action}")
        assignments[action] += 1
        cell = rates["cells"].get((stratum_key(row), action),
                                  rates["overall"].get(action))
        if cell is None:
            raise ValueError(f"no holdout data at all for {action}")
        cures += cell["cured"]
        complaints += cell["complaint"]
        redelinq += cell["re_delinquent"]
        cost += ACTION_COSTS[action]
    n = len(holdout_rows)
    gross = cures * RECOVERY_PER_CURE
    return {
        "policy": name,
        "n_borrowers": n,
        "expected_cures": round(cures, 4),
        "cure_rate": round(cures / n, 6),
        "total_cost": round(cost, 2),
        "gross_recovery": round(gross, 2),
        "net_value": round(gross - cost, 2),
        "net_value_per_borrower": round((gross - cost) / n, 4),
        "complaint_rate": round(complaints / n, 6),
        "re_delinquency_rate": round(redelinq / n, 6),
        "assignments": assignments,
    }


def simulate_policies(rows: List[Dict]) -> Dict:
    """Score every named policy on the holdout period of ``rows``.
    The best-per-stratum policy is *defined* from the pilot, then
    scored — like the others — only on the holdout."""
    pilot = rows_for_period(rows, "pilot")
    holdout = rows_for_period(rows, "holdout")
    if not pilot or not holdout:
        raise ValueError("simulate needs pilot and holdout rows")
    mapping = build_best_per_stratum(pilot)
    policies = [
        ("no_action", policy_no_action),
        ("comms_first", policy_comms_first),
        ("targeted_modification", policy_targeted),
        ("best_per_stratum", make_best_per_stratum_policy(mapping)),
    ]
    scored = [score_policy(holdout, fn, name) for name, fn in policies]
    ranked = sorted(scored, key=lambda s: -s["net_value"])
    return {
        "scored_on": "holdout (randomised period only)",
        "costs_fictional": ACTION_COSTS,
        "recovery_per_cure_fictional": RECOVERY_PER_CURE,
        "best_per_stratum_mapping_size": len(mapping),
        "policies": scored,
        "ranking": [s["policy"] for s in ranked],
        "best_policy": ranked[0]["policy"],
    }
