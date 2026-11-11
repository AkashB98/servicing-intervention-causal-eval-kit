# servicing-intervention-causal-eval-kit

**Stratified causal evaluation of servicing interventions whose centrepiece is an honesty check: a randomised pilot says a loan modification lifts cure rates by +7.4 points; the observational book says +25.8 — and the kit flags the observational number as confounded, by name, because a planted "sounds willing" trait that no covariate captures is doing the inflating.**

When a delinquent borrower falls behind, a servicing team chooses: send communications only, offer a payment plan, offer a loan modification, or do nothing. Each choice costs money and changes the chance the borrower **cures** — returns to current on the loan. Choosing well requires **causal** estimates — what each intervention *causes*, not what it merely correlates with — translated into a decision policy. This project builds that evaluation end to end on synthetic data where the truth is known: a randomised pilot, a stratified estimator, a design calculator, a policy simulator scored only on randomised held-out data — and an observational arm included deliberately as a warning, with a check that catches it lying. The build was inspired by a Senior Applied Scientist, Servicing job description (Upstart, 2026-11-10) centred on causal treatment-effect estimates translated into decision policies for servicing interventions; it is an original implementation, not a reproduction of any company's approach.

> **Synthetic data only.** Every figure is fabricated by a seeded generator for a fictional lender (**Northstar Lending**). No real company, customer, or loan data is used anywhere in this project.

Stdlib-only Python — deliberately. A causal claim is an audit artifact: anyone should be able to read every line of the statistics, re-run it offline, and get byte-identical outputs, with no dependency tree between them and the math. Zero network, zero API keys, zero external dependencies.

## The honesty check (the headline)

The generator plants a **confounder** — a variable that drives both who gets a treatment and the outcome, biasing any comparison that cannot account for it. Here it is a latent *willingness* trait: some borrowers "sound willing" on a call. Willingness raises a borrower's cure probability, and in the observational period it makes a servicing agent far more likely to offer a modification. Willingness is **not** written into any data row and is **not** among the analysis covariates, so no amount of stratification on the observed covariates can remove it. The check compares, per intervention, the observational **ATE** — **average treatment effect** — the mean effect across the whole population — against the randomised pilot's ATE, and flags the intervention when the disagreement exceeds what sampling noise alone would produce at the 5% level (`|difference| > 1.96 × SE of the difference`). On the seeded data:

| Intervention | Pilot ATE (randomised) | Observational ATE | Difference | Flagged? |
|---|---|---|---|---|
| comms-only | +3.60% | +15.27% | +11.68 pts | 🚩 Yes |
| payment plan | +5.08% | +10.75% | +5.68 pts | No (within noise) |
| modification | +7.39% | +25.82% | **+18.43 pts** | 🚩 Yes |

Modification — the intervention agents hand to the borrowers who sound most willing — is inflated the most, exactly as planted. Calibration evals show the check earns its flags: across 30 fresh confounded seeds it flags modification **30/30** times; across 30 seeds whose observational assignment ignores willingness, only **4/30**.

## Quickstart

```bash
# 1. Generate the synthetic Northstar Lending datasets
#    (2,400 pilot + 1,200 observational + 1,600 holdout, seed 42)
python3 cli.py generate --out dataset.jsonl

# 2. Stratified ATE per intervention from the randomised pilot
python3 cli.py estimate --input dataset.jsonl

# 3. The honesty check: pilot vs observational, per intervention
python3 cli.py honesty --input dataset.jsonl

# 4. Design: two-proportion sizing table + pilot guardrails
python3 cli.py design --input dataset.jsonl

# 5. Policies scored ONLY on the randomised holdout
python3 cli.py simulate --input dataset.jsonl

# 6. Full report: estimates + honesty + design + policies (JSON)
python3 cli.py report --input dataset.jsonl

# 7. CI gate: exit 0 = agreement, 1 = honesty breach (EXPECTED on
#    the seeded data - the planted confounder is flagged, by design),
#    2 = data-quality / IO error
python3 cli.py check --input dataset.jsonl

# 8. One-command end-to-end demo (honesty check leads)
python3 demo.py

# 9. Tiny localhost API: POST /estimate /honesty /simulate,
#    GET /report, GET /health on 127.0.0.1
python3 cli.py serve --port 8765
```

## The seeded evaluation at a glance (seed 42)

**Pilot estimates** (stratified, vs no action; 95% **CI** — **confidence interval** — the range that would cover the true value in 95% of repeated experiments): comms-only **+3.60%** [−1.78%, +8.98%], payment plan **+5.08%** [−0.37%, +10.52%], modification **+7.39%** [+2.04%, +12.74%]. Only modification's interval excludes zero at this pilot size — the design table below shows why: the pilot's **MDE** — **minimum detectable effect** — the smallest lift the design can reliably see — is **8.06 points** at 600 borrowers per arm, so the smaller true effects are honestly unresolved rather than dressed up.

**Design** (two-proportion formula, baseline cure rate 47.89%, α = 0.05, power = 0.80): detecting a 3-point lift needs **4,356** borrowers per arm; 5 points needs **1,566**; 8 points needs **609**. Guardrails from the pilot — exactly two, by design: complaint rate (no action 2.27%, comms 5.43%, plan 2.33%, modification 3.67%) and re-delinquency rate (9.42%, 9.90%, 7.51%, 7.68%).

**Policies on the randomised holdout** (flat fictional economics: $0 / $5 / $25 / $120 per action, $800 flat recovery per cure — labelled fictional, no modification NPV, no re-default modelling):

| Policy | Expected cures (of 1,600) | Net value | Complaints | Re-delinquency |
|---|---|---|---|---|
| targeted modification rule | **949.1** | **$721,629** | 3.09% | 7.76% |
| comms-first for all | 858.4 | $678,741 | 2.52% | 9.83% |
| best-per-stratum by pilot CATE | 832.3 | $631,859 | 3.30% | 9.05% |
| no action | 711.1 | $568,913 | 1.71% | 12.73% |

The hand-written targeted rule wins on the seed, and the learned best-per-stratum policy finishes third — reported as-is, not tuned away (see the development loop). A per-stratum CATE estimated from ~17 borrowers per arm per stratum is noisy, and the pilot's own MDE says so; the simple rule encodes the segment structure the generator planted, and at this sample size that beats learning it.

## How it works

**The generator.** Each borrower draws a risk band (A–D), a days-past-due bucket (30/60/90), an income band, and a balance band, plus the latent willingness trait. Baseline cure rates differ by risk band; each intervention adds a segment-dependent effect — comms-only helps early delinquency most (+11 pts at 30 DPD, +1 at 90), a payment plan helps mid-stage most, a modification helps late-stage high-balance borrowers most (+18 pts at 90 DPD / high balance). The risk band is just a covariate here: in practice it would come from a delinquency-scoring system (the sibling build `collections-risk-scorer`); this kit does not rebuild scoring, and there are no credit vintages (that is `credit-retrospective-kit`'s job).

**The estimator — stratified CATE, named honestly.** The method is a **stratified CATE** — **CATE (conditional average treatment effect)** — the average effect inside one subgroup — computed by difference-in-means within exact strata (risk band × DPD bucket × income band, 36 cells), pooled with stratum weights, with standard errors from the two-proportion variance. It is *not* uplift modelling and *not* causal machine learning: no fitted outcome model, no **propensity model** — a model of who gets treated — and no logistic regression anywhere. Strata missing either arm are excluded with weights renormalised, and the coverage share is reported so the exclusion is never silent.

**Policy scoring.** Policies are scored only on the randomised holdout — never on observational data, and there is no **off-policy evaluation** — estimating a policy's value from data collected under a different policy — of observational data anywhere in this kit. Because holdout assignment is randomised within strata, each stratum × intervention cell's observed rates are unbiased for that cell; a policy's score averages the cell rates of the actions it chooses. The best-per-stratum policy translates pilot CATEs into decisions through a cost hurdle: an action is chosen for a stratum only when `CATE × $800 − action cost` beats doing nothing.

**Data quality first.** `validation.py` runs before any estimate: required fields, enum membership, the int/bool trap on the DPD bucket and the binary outcomes, duplicate borrower IDs, and presence of all three periods. It reports issues and never raises on bad data, so the `check` gate can map them to exit code 2, distinct from an honesty breach (exit 1).

## Architecture

| Module | Job |
|---|---|
| `generator.py` | Deterministic synthetic Northstar Lending datasets: covariates, planted latent willingness, segment-dependent true effects, randomised pilot/holdout assignment, confounded observational assignment (plus an unconfounded variant for calibration); CSV / JSONL serialisers and loaders |
| `estimator.py` | Stratified CATE by difference-in-means over exact strata (risk × DPD × income), pooled ATE, two-proportion standard errors, NormalDist confidence intervals |
| `honesty.py` | The named honesty check: observational vs randomised ATE per intervention, flagged when the disagreement exceeds 1.96 × the standard error of the difference |
| `design.py` | Two-proportion sample-size / MDE calculator (and its bisection inverse), power for an effect, and the two pilot guardrail readouts (complaint, re-delinquency) |
| `policy.py` | Four named decision policies (no action, comms-first, targeted modification, best-per-stratum with a cost hurdle) scored by plug-in cell rates on the randomised holdout only, under flat fictional economics |
| `report.py` | Full report assembly: dataset summary + pilot and observational estimates + honesty + design + policies |
| `validation.py` | Data-quality gate: required fields, enums, binary outcomes, duplicates, period presence — reports issues, never raises on bad data |
| `cli.py` | `generate` / `estimate` / `honesty` / `design` / `simulate` / `report` / `check` (CI exit 0/1/2) / `demo` / `serve` (stdlib `http.server` on 127.0.0.1: `POST /estimate`, `POST /honesty`, `POST /simulate`, `GET /report`, `GET /health`); CSV, JSONL, and JSON-array inputs |
| `demo.py` | End-to-end narrative with asserts, led by the honesty check: determinism, flagged confounding, pilot estimates, design, holdout policy ranking, check exit codes (1 on seeded by design, 0 on an unconfounded variant, 2 on a missing file), live localhost serve round-trip |
| `fixtures/` | The full seeded datasets (`sample_borrowers.jsonl` + `.csv`) with pinned totals |

## Evals

`python3 evals/run_evals.py` → **15/15 golden evals** (~7 s), report committed at `evals/eval_report.json` (md5 `7590c0cff23e0c419dc9a479fffd8b34`, byte-identical across 3 verified runs):

- **E1** generator determinism — seed 42 JSONL bytes pinned by md5
- **E2** period counts pinned (2,400 / 1,200 / 1,600)
- **E3** pilot ATEs pinned (modification +7.39%, CI [+2.04%, +12.74%])
- **E4** observational ATEs pinned (modification +25.82% — inflated, by design)
- **E5** honesty pinned — flagged = comms-only + modification; modification difference +18.43 pts, payment plan not flagged
- **E6** estimator hand-computed (3/10 vs 6/10 → ATE 0.30, variance 0.045)
- **E7** two-proportion sizing hand-computed (baseline 0.40, δ = 0.05 → 1,531 per arm)
- **E8** pilot design pinned (baseline 47.89%, MDE 8.06 pts, sizing table, guardrail rates)
- **E9** policy simulation pinned (targeted rule wins at $721,629; best-per-stratum third at $631,859)
- **E10** validation flags planted defects; the generated datasets validate clean
- **E11** CLI `check` exit codes 1 / 0 / 2 (the seeded 1 is the planted confounder, by design)
- **E12** serve round-trip matches local honesty check + report exactly
- **E13** bundled fixture pinned (5,200 rows, validates clean, identical report)
- **E14** CI coverage across 40 fresh seeds — comms 37/40, payment plan 40/40, modification 39/40 against each seed's planted sample ATE (bands consistent with 95% nominal)
- **E15** honesty firing rates across 30 seeds each — confounded: modification flagged 30/30; unconfounded variant: 4/30

Unit tests: `python3 -m unittest discover -s tests` → **68 hermetic tests**, including a no-network enforcement test (sockets stubbed), hand-computed estimator and sample-size fixtures, honesty flag logic under small vs large noise, the per-action cost-hurdle cases, validation defects, CLI exit codes, JSON/CSV round-trips, and serve round-trips.

## Development loop — what the checks actually caught

1. **The first honesty threshold flagged noise, not confounding.** The original rule flagged any disagreement over a fixed 2 points. The standard error of the pilot-vs-observational difference at these sample sizes is itself ~5 points, so on *unconfounded* data the check fired on 40/40 seeds — a check that flags everything proves nothing. Fix: flag only disagreements beyond 1.96 × the standard error of the difference, and prove the behaviour with the firing-rate calibration eval (E15: 30/30 confounded, 4/30 unconfounded).
2. **The first confounder was too weak to see.** With the original willingness channel, the seeded modification disagreement was +0.6 points — real bias, but smaller than the pilot's own noise, so nothing flagged and the centrepiece was invisible. The willingness channel was strengthened (agents offer modifications to willing-sounding borrowers at 5× the base rate, and rarely to unwilling ones), bringing the seeded disagreement to +18.43 points, ~4 standard errors — a bias no honest reader can miss.
3. **The learned policy bought noise until it had to pay for it.** The first best-per-stratum rule (pick each stratum's largest CATE) assigned the $120 modification to 339 of 1,600 holdout borrowers on the strength of per-stratum estimates built from ~17 borrowers per arm — and finished *last*, behind doing nothing. Fix: the cost hurdle (`CATE × recovery − action cost` must beat zero). It now finishes third, ahead of no action — and still behind the hand-written targeted rule, which is the honest result at this pilot size and is reported as such rather than tuned away.
4. **A test I wrote was wrong; the code was right.** The cost-hurdle unit test expected "no action" in a stratum where comms-only's small effect (+0.02 × $800 − $5 = +$11) correctly clears its own low hurdle while modification's (+0.10 × $800 − $120 = −$40) correctly does not. The hurdle is per-action; the test now pins that asymmetry deliberately.

## Scope and honest limits

- **Stratified CATE is not uplift modelling.** Estimates are exact-cell difference-in-means over three discrete covariates. Anything continuous, high-dimensional, or sparse-celled would need a model — and a model's assumptions — which this kit deliberately does not introduce.
- **The observational arm is a warning, not an estimate.** Its numbers are printed next to the pilot's precisely so the disagreement is visible. Nothing in this kit adjusts, weights, or otherwise "rescues" the observational estimate, and policies are never scored on it.
- **Flat fictional economics.** Costs and the $800 recovery-per-cure are flat, invented constants for comparing policies on one scale. There is no loan-modification NPV, no re-default modelling, and no term structure — a real servicing decision would need all three, from real data.
- **The willingness story is one planted confounder, told cleanly.** Real observational servicing data carries many confounders, some observed and some not, plus selection into the book itself. The kit demonstrates the failure mode and the check, not a census of real-world bias.
- **Per-stratum CATEs are noisy at pilot scale.** With ~600 borrowers per arm spread over 36 strata, individual stratum effects carry wide intervals; the pooled ATEs and the MDE table are the reliable readouts, and the policy result (item 3 above) is what that noise costs when it is ignored.
- **The API is a demo surface.** It binds 127.0.0.1 only, holds its dataset in memory, and resets on restart.

## Why this project

It takes the author's target craft — applied causal inference for servicing decisions — and rebuilds it as an analytics-engineering artifact: estimates translated into decision policies, policies scored only where scoring is honest, an experiment sized before it is run, and an observational dataset treated the way a forward-deployed engineer should treat a customer's convenient-but-biased data source: measured against a randomised benchmark, flagged by name when it disagrees, and never quietly believed. Make the design inspectable, make the outputs reproducible, and make the failure modes exit codes.
