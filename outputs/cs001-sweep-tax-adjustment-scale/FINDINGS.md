# CS001 parameter sweeps (2026-09-01)

Two L2_local_batch one-dimensional sweeps over the two primitives flagged
`"provisional project choice"` in the primitive table's own provenance —
`tax_adjustment_scale` (κ_τ) and `automation_persistence_annual` — run at
the illustrative Farhi-based smoke calibration. Exploratory first CS001
tranche; not a parameter search, calibration exercise, or approved CS001
result. Every point reruns the full Stage 1 + Stage 2A pipeline and passed
every acceptance check (26/26 points across both sweeps); no infeasible or
failed point was silently dropped, per CS001's own standard for this lane.

## Sweep 1: tax-adjustment scale traces the oscillation transition exactly

15 points, `κ_τ ∈ [0.001, 2.0]`, crossing the closed-form monotone ↔
oscillatory threshold `χ* = γ(γ+ρ)²/(32ρ²) ≈ 0.010929` derived independently
in the local LQ system and computation plan (a pure function of `γ, ρ`,
**not** of `κ_τ` itself — confirmed: `χ*` is bit-identical across all 15
points). See [`oscillation_transition.png`](oscillation_transition.png).

| κ_τ | χ | Re(μ) | \|Im(μ)\| |
|---:|---:|---:|---:|
| 0.001 | 0.00282 | −0.0380 | 0 (real roots) |
| 0.00387 | 0.01093 | −0.0260 | 0 — within 0.07% of χ* |
| 0.005 | 0.01412 | −0.0269 | 0.00870 (complex pair) |
| 0.5 (baseline) | 1.4119 | −0.0760 | 0.07827 |
| 2.0 | 5.6476 | −0.1090 | 0.11354 |

The transition happens exactly where the closed-form predicts: real,
distinct roots below `χ*`, a complex-conjugate pair immediately above it.
This is independent numerical confirmation of the analytical result, not
just a smoke check that the solver runs. Economically: stronger/cheaper tax
feedback (`κ_τ`, hence `χ`, larger) produces faster convergence (`Re(μ)`
more negative) but increasingly oscillatory capital-tax dynamics — cheap
adjustment lets the planner react fast enough to overshoot.

## Sweep 2: automation persistence leaves the capital-tax loop untouched, but drives the hedge value roughly 15x

11 points, `automation_persistence_annual ∈ [0.3, 0.98]` (stationary sd held
at its calibrated target of 0.25, so lower persistence implies a larger
instantaneous diffusion `σ̂_x`, per the model's own annual-to-continuous-time
translation).

**`χ`, `χ*`, and the closed-loop roots are bit-identical across the entire
sweep.** This is not a bug — it is the triangular structure the proof plan
describes: the 2×2 real (capital-tax) Riccati block solves from `A_r, Q_rr`
alone, neither of which involves `κ_x`. Automation persistence only enters
through the exogenous/cross blocks and the portfolio.

The leading portfolio and its welfare value move a great deal, though:

| persistence | leading position | fiscal-hedge component | hedge value (%perm. consumption) |
|---:|---:|---:|---:|
| 0.30 | 0.105 | −12.464 | 34.27% |
| 0.50 | 0.173 | −12.396 | 20.20% |
| 0.81 (baseline) | 0.446 | −12.123 | 6.95% |
| 0.95 | 0.801 | −11.768 | 2.70% |
| 0.98 | 0.675 | −11.894 | 1.99% |

The leading-order hedge value falls roughly monotonically from ~34% to ~2%
of permanent consumption as assumed persistence rises from 0.3 to 0.95 —
lower persistence (faster mean reversion at a fixed long-run/stationary
variance) implies more instantaneous automation volatility, which is
exactly what raises the stakes of holding the correct hedge. Position and
hedge value are not perfectly monotonic at the extreme high-persistence end
(0.95→0.98 the leading position dips before the hedge value keeps falling);
this is a minor, second-order feature at the edge of the swept range that
was not chased further here. Boundary slack stays comfortably positive
throughout (min ≈0.0184–0.0196).

**Reading:** the qualitative conclusions in the baseline FINDINGS.md
(hedge value dominates access value; the fiscal hedge is a large short
position against the naive Merton demand) are robust to both swept
parameters. The *quantitative* size of the hedge welfare value is highly
sensitive to the automation-persistence assumption specifically — a ~15x
range across a plausible parameter band — which matters for any future
attempt to attach a magnitude to this mechanism, though not for this
project's current scope.

## Reproducing

```bash
uv run python -m tai_public_finance.cs001_lq_anchor.sweep_cli \
  --config configs/cs001/lq_farhi_smoke.json \
  --parameter-path parameters.tax_adjustment_scale \
  --values 0.001,0.002,0.00387,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.5,2.0 \
  --output-dir outputs/cs001-sweep-tax-adjustment-scale

uv run python -m tai_public_finance.cs001_lq_anchor.sweep_cli \
  --config configs/cs001/lq_farhi_smoke.json \
  --parameter-path parameters.automation_persistence_annual \
  --values 0.3,0.4,0.5,0.6,0.7,0.75,0.81,0.85,0.9,0.95,0.98 \
  --output-dir outputs/cs001-sweep-automation-persistence
```

Run records: `runs/RUN-20260901T195544Z-CS001-SWEEP-43f4dc95-01.yaml` (tax
adjustment) and `runs/RUN-20260901T195551Z-CS001-SWEEP-43f4dc95-01.yaml`
(automation persistence).
