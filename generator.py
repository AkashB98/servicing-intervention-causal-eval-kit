"""Deterministic synthetic delinquent-borrower generator - Northstar
Lending (fictional).

SYNTHETIC DATA ONLY. No real company, customer, or loan data is ever
used. Every row is fabricated from a seeded PRNG so the same seed
always yields the same datasets, byte-for-byte.

One row per borrower:

  borrower_id, period, risk_band, dpd_bucket, income_band,
  balance_band, intervention, cured, complaint, re_delinquent

Three periods are generated:

  pilot         - borrowers randomly assigned intervention vs control
                  within strata (the randomised pilot)
  observational - borrowers assigned by servicing agents: the choice
                  depends on observed covariates PLUS a planted latent
                  willingness trait (see below)
  holdout       - a second randomised period, reserved for scoring
                  decision policies; never used to estimate effects

The planted confounder (the centrepiece of this kit)
----------------------------------------------------
Each borrower has a latent *willingness* trait: some borrowers
"sound willing" on a call. Willingness is drawn inside the generator,
raises the probability of curing, and - in the observational period
only - makes a servicing agent much more likely to offer a loan
modification. Willingness is **not** written into any row and is
**not** among the analysis covariates, so no stratification on the
observed covariates can remove its effect. The observational estimate
for modification is therefore biased upward by construction, and the
honesty check in ``honesty.py`` exists to catch exactly that.

The risk band is just a covariate here. In practice it would come
from a delinquency-scoring system (the sibling build
``collections-risk-scorer``); this kit does not rebuild scoring.

True cure process
-----------------
Baseline cure rates differ by risk band, with small adjustments for
days-past-due (DPD), income band and balance band. Each intervention
adds a segment-dependent effect: comms-only helps early delinquency
most, a payment plan helps mid-stage most, and a modification helps
late-stage / high-balance borrowers most. Willingness adds a flat
lift. Guardrail outcomes: a complaint indicator and a
re-delinquency indicator.
"""

from __future__ import annotations

import random
from typing import Dict, List

DEFAULT_SEED = 42
DEFAULT_N_PILOT = 2400
DEFAULT_N_OBSERVATIONAL = 1200
DEFAULT_N_HOLDOUT = 1600

RISK_BANDS = ["A", "B", "C", "D"]
DPD_BUCKETS = [30, 60, 90]
INCOME_BANDS = ["low", "mid", "high"]
BALANCE_BANDS = ["low", "mid", "high"]
INTERVENTIONS = ["no_action", "comms_only", "payment_plan", "modification"]
CONTROL = "no_action"
PERIODS = ["pilot", "observational", "holdout"]

ROW_FIELDS = ["borrower_id", "period", "risk_band", "dpd_bucket",
              "income_band", "balance_band", "intervention", "cured",
              "complaint", "re_delinquent"]

_BASE_CURE = {"A": 0.52, "B": 0.40, "C": 0.28, "D": 0.17}
_DPD_ADJ = {30: 0.04, 60: 0.0, 90: -0.04}
_INCOME_ADJ = {"low": -0.03, "mid": 0.0, "high": 0.03}
_BALANCE_ADJ = {"low": 0.01, "mid": 0.0, "high": -0.02}
WILLINGNESS_PROB = 0.45
WILLINGNESS_EFFECT = 0.28
_REDELINQ_BASE = {"A": 0.08, "B": 0.12, "C": 0.17, "D": 0.23}


def true_effect(intervention: str, dpd_bucket: int, balance_band: str) -> float:
    """Planted cure-probability lift of ``intervention`` for a segment."""
    if intervention == "comms_only":
        return {30: 0.11, 60: 0.05, 90: 0.01}[dpd_bucket]
    if intervention == "payment_plan":
        return {30: 0.06, 60: 0.10, 90: 0.07}[dpd_bucket]
    if intervention == "modification":
        if dpd_bucket == 30:
            return 0.02
        if dpd_bucket == 60:
            return 0.07
        return {"low": 0.09, "mid": 0.13, "high": 0.18}[balance_band]
    return 0.0


def _clip(p: float) -> float:
    return min(0.95, max(0.01, p))


def _draw_covariates(rng: random.Random):
    risk = rng.choices(RISK_BANDS, weights=[0.20, 0.30, 0.30, 0.20])[0]
    dpd = rng.choices(DPD_BUCKETS, weights=[0.40, 0.35, 0.25])[0]
    income = rng.choices(INCOME_BANDS, weights=[0.35, 0.40, 0.25])[0]
    balance = rng.choices(BALANCE_BANDS, weights=[0.35, 0.35, 0.30])[0]
    willing = 1 if rng.random() < WILLINGNESS_PROB else 0
    return risk, dpd, income, balance, willing


def _random_assignment(rng: random.Random) -> str:
    return rng.choice(INTERVENTIONS)


def _observational_assignment(rng: random.Random, dpd: int, income: str,
                              balance: str, willing: int,
                              confounded: bool) -> str:
    """Agent-driven assignment: observed covariates, plus - when
    ``confounded`` - the latent willingness trait."""
    w = {name: 1.0 for name in INTERVENTIONS}
    if dpd == 90 and balance == "high":
        w["modification"] *= 2.2
    if dpd == 90:
        w["modification"] *= 1.5
        w["payment_plan"] *= 1.3
    if dpd == 30:
        w["comms_only"] *= 2.0
        w["modification"] *= 0.5
    if income == "low":
        w["payment_plan"] *= 1.4
    if confounded:
        if willing:
            w["modification"] *= 5.0
            w["payment_plan"] *= 1.5
            w["no_action"] *= 0.6
        else:
            w["modification"] *= 0.35
            w["no_action"] *= 1.8
    weights = [w[name] for name in INTERVENTIONS]
    return rng.choices(INTERVENTIONS, weights=weights)[0]


def _draw_outcomes(rng: random.Random, risk: str, dpd: int, income: str,
                   balance: str, willing: int, intervention: str):
    p_cure = _clip(_BASE_CURE[risk] + _DPD_ADJ[dpd] + _INCOME_ADJ[income]
                   + _BALANCE_ADJ[balance]
                   + true_effect(intervention, dpd, balance)
                   + WILLINGNESS_EFFECT * willing)
    cured = 1 if rng.random() < p_cure else 0
    p_complaint = 0.02 + {"no_action": 0.0, "comms_only": 0.015,
                          "payment_plan": 0.005,
                          "modification": 0.02}[intervention]
    if risk == "D":
        p_complaint += 0.01
    complaint = 1 if rng.random() < _clip(p_complaint) else 0
    p_redelinq = _REDELINQ_BASE[risk] + (0.03 if dpd == 90 else
                                         -0.02 if dpd == 30 else 0.0)
    p_redelinq += {"no_action": 0.0, "comms_only": -0.01,
                   "payment_plan": -0.03,
                   "modification": -0.05}[intervention]
    p_redelinq -= 0.04 * willing + 0.05 * cured
    re_delinquent = 1 if rng.random() < _clip(p_redelinq) else 0
    return cured, complaint, re_delinquent


def _generate_period(rng: random.Random, period: str, n: int,
                     id_start: int, confounded: bool) -> List[Dict]:
    rows: List[Dict] = []
    for i in range(n):
        risk, dpd, income, balance, willing = _draw_covariates(rng)
        if period == "observational":
            intervention = _observational_assignment(
                rng, dpd, income, balance, willing, confounded)
        else:
            intervention = _random_assignment(rng)
        cured, complaint, re_delinquent = _draw_outcomes(
            rng, risk, dpd, income, balance, willing, intervention)
        rows.append({
            "borrower_id": f"NS-{id_start + i:05d}",
            "period": period,
            "risk_band": risk,
            "dpd_bucket": dpd,
            "income_band": income,
            "balance_band": balance,
            "intervention": intervention,
            "cured": cured,
            "complaint": complaint,
            "re_delinquent": re_delinquent,
        })
    return rows


def generate_dataset(seed: int = DEFAULT_SEED,
                     n_pilot: int = DEFAULT_N_PILOT,
                     n_observational: int = DEFAULT_N_OBSERVATIONAL,
                     n_holdout: int = DEFAULT_N_HOLDOUT,
                     confounded: bool = True) -> List[Dict]:
    """All three periods, deterministic for ``seed``.

    ``confounded=False`` removes the willingness channel from the
    observational assignment (outcomes still depend on willingness,
    but assignment no longer does, so stratification is unbiased).
    The calibration evals use that variant to show the honesty check
    rarely fires when there is nothing to catch.
    """
    if min(n_pilot, n_observational, n_holdout) < 0:
        raise ValueError("period sizes must be >= 0")
    pilot = _generate_period(random.Random(seed), "pilot", n_pilot, 1,
                             confounded)
    obs = _generate_period(random.Random(seed + 100000), "observational",
                           n_observational, n_pilot + 1, confounded)
    hold = _generate_period(random.Random(seed + 200000), "holdout",
                            n_holdout, n_pilot + n_observational + 1,
                            confounded)
    return pilot + obs + hold


def rows_for_period(rows: List[Dict], period: str) -> List[Dict]:
    return [r for r in rows if r["period"] == period]


# ------------------------------------------------------------ serializers

def rows_to_jsonl(rows: List[Dict]) -> str:
    import json
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def rows_to_csv(rows: List[Dict]) -> str:
    import csv
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=ROW_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row[k] for k in ROW_FIELDS})
    return buf.getvalue()


# --------------------------------------------------------------- loaders

_INT_FIELDS = ("dpd_bucket", "cured", "complaint", "re_delinquent")


def _coerce(row: Dict) -> Dict:
    out = dict(row)
    for field in _INT_FIELDS:
        if isinstance(out.get(field), str):
            try:
                out[field] = int(out[field])
            except ValueError:
                pass
    return out


def load_rows(path: str) -> List[Dict]:
    import csv
    import io
    import json
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    stripped = text.strip()
    if not stripped:
        return []
    if path.endswith(".csv"):
        return [_coerce(dict(r)) for r in csv.DictReader(io.StringIO(text))]
    if stripped.startswith("["):
        return [_coerce(dict(x)) for x in json.loads(stripped)]
    return [_coerce(json.loads(line))
            for line in stripped.splitlines() if line.strip()]


def write_rows(rows: List[Dict], path: str, fmt: str = None) -> None:
    if fmt is None:
        fmt = "csv" if path.endswith(".csv") else "jsonl"
    text = rows_to_csv(rows) if fmt == "csv" else rows_to_jsonl(rows)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


# Aliases matching the sibling builds' naming.
load_dataset = load_rows
write_dataset = write_rows
