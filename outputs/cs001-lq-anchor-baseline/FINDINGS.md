# CS001 baseline: LQ anchor, matrix equations, and impulse responses

**Run:** `RUN-20260901T184527Z-CS001-89f6a939-01` (see [`../../runs/RUN-20260901T184527Z-CS001-89f6a939-01.yaml`](../../runs/RUN-20260901T184527Z-CS001-89f6a939-01.yaml))
**Parameter set:** `lq_farhi_illustrative_smoke_v1` (standalone LQ smoke; illustrative, not an empirical UK calibration)
**Outcome:** pass — all 35 acceptance checks passed (see `report.json:acceptance`)
**Status caveat:** the computational specification this implements, CS001, is registered `draft` (not `approved`) in the codex research workspace's registry. This run implements CS001's stated scope under direct commission in this session — a fully specified economic setup, explicit exclusions, and an explicit acceptance standard — rather than against a formally fingerprinted `approved` version. The codex-side registry should be updated to bind this repository and move CS001 toward `review_ready`/`approved` as a follow-up.

## What this establishes

One reproducible baseline calculation of the local deterministic/first-order 4×4 LQ system (Stage 1) and the leading small-risk public portfolio, consumption, and welfare objects (Stage 2A) for the Brownian Version 5.1, `q_D=1` Ramsey economy, around its illustrative interior steady state. Every reduced-form matrix traces to primitives (no hand-entered coefficients); the Riccati block is solved two independent ways (closed-form Hamiltonian roots and an ordered-Schur invariant subspace) and agrees to 7.7e-16 relative error; every residual, spectrum, and identity is rechecked by a diagnostics module that never reuses the solver's intermediate results; and the core numbers (`H_rr`, the closed-loop roots, the leading portfolio position, the welfare values) were cross-validated to full float64 precision against a separately-written, independently-authored reference implementation. Details of that verification process are in the session handoff; this document covers the economics.

## 1. The illustrative steady state, and is it on the maintained branch?

At the Farhi-based annual vector (ρ=0.0202, δ=0.08, capital share ᾱ=0.34, `K̄=L=1`, `N̄=0`):

| Quantity | Value |
|---|---:|
| `R̄` (rental rate) | 0.120405 |
| `Ȳ` (output) | 0.354134 |
| `W̄` (wages) | 0.233728 |
| `B̄` (net-rental tax base) | 0.040405 |
| `F̄=c̄` (fiscal resources = worker consumption) | 0.253931 |
| `X̄=J̄` (comprehensive/fiscal wealth, since `N̄=0`) | 12.5692 |
| `τ̄` (tax rate) | 0.5 (exact) |

Both defining steady-state equations hold to `3.5e-18` (machine precision), checked against the *independent* nonlinear evaluator (not the formula that solved for `z̄` in the first place — see "what changed after adversarial testing" below):

- capital law `(1-τ̄)(R̄-δ)=ρ`: residual 3.5e-18
- fiscal envelope `τ̄(R̄-δ)=ρ`: residual 3.5e-18

The anchor is strictly inside the maintained smooth full-specialisation branch: both margins are positive, `𝔰_A=1.0383` and `𝔰_N=0.5867` — the latter matching the reconciled proof-plan's hand-computed value to 4 decimal places, an independent cross-check of its own.

## 2. How does the Ramsey government adjust the capital tax near the steady state?

The local feedback rule is `ν = χ·H[τ,:]·y` (tax *speed*, not the tax rate itself — `dτ/dt=ν`), with `χ=1.412`:

| State displacement | ∂ν/∂(·) |
|---|---:|
| `z − z̄` (productivity) | −0.1393 |
| `x − x̄` (automation) | −0.0416 |
| `k` (log capital) | +0.1842 |
| `τ − τ̄` (inherited tax) | −0.1123 |

A positive productivity or automation shock nudges the tax rate *down*; higher inherited capital nudges it *up*; an inherited tax rate away from 0.5 mean-reverts back toward it (the −0.1123 own-coefficient). The resulting real closed-loop (capital–tax) block has complex roots `μ± = −0.076028 ± 0.078265i` — the tax/capital response is **damped-oscillatory**, not monotone, with a half-life of `ln(2)/0.076 ≈ 9.1` years and a period of `2π/0.078 ≈ 80` years. This is visible directly in the IRFs: the tax rate initially *falls* on a positive shock, undershoots, crosses back through zero around year 12–13, and is still rising at year 40 (see `figures/stage1_primitive_brownian_irfs.png`). Both roots have strictly negative real part (Hurwitz, margin 0.076), and the closed-loop matrix used to propagate every downstream IRF was verified to be exactly `A + χBB^T H` from primitives and the solved `H` (not merely self-consistent with whatever the solver happened to produce — see below).

## 3. Joint response to productivity and automation shocks

One standardized (1 s.d.) Brownian innovation over the short reporting window (`Δ≈1/252` year), impact (t=0) values, `full_access` regime:

| Variable (impact) | Productivity shock | Automation shock |
|---|---:|---:|
| Output | +5.79e-4 | +2.82e-4 |
| **Wages** | **+3.82e-4** | **−0.855e-4** |
| Net-rental tax base | +1.97e-4 | +3.67e-4 |
| Tax revenue (`τB`) | +0.985e-4 | +1.837e-4 |
| Fiscal resources `F` | +4.81e-4 | +0.982e-4 |
| Tax speed `ν` | −2.28e-4 | −4.25e-4 |
| Public net worth (claim payoff) | +7.30e-4 | +13.62e-4 |
| Worker consumption | +4.44e-4 | +7.59e-4 |
| Transfer | +0.621e-4 | +8.45e-4 |

The headline qualitative result the proof plan flags is reproduced exactly: **the automation shock raises output and the tax base while *lowering* wage income on impact** (wages go slightly negative, `−0.855e-4`), because `η_Y − 1/(1-ᾱ) < 0` at this calibration — automation's displacement effect on labor's share dominates its scale effect on wages, even though it strictly raises output. Fiscal resources rise much less from the automation shock than from the productivity shock (0.98e-4 vs 4.81e-4) precisely because the negative wage term partly offsets the larger tax-base gain. Despite lower wages, **worker consumption still rises more from the automation shock** (7.59e-4 vs 4.44e-4) — driven by the larger claim payoff to public net worth (13.62e-4 vs 7.30e-4, since the claim's automation loading `ℓ_x·σ̂_x=0.0484` exceeds its productivity loading `σ̂_z=0.0260`) and a correspondingly large transfer response (8.45e-4 vs 0.62e-4) that more than compensates workers for the wage loss. Log capital and output both rise temporarily and mean-revert to zero by year 40; net-rental tax base and fiscal resources briefly undershoot negative around years 15–20 before decaying to zero (see `figures/stage1_fiscal_aggregate_irfs.png`, `figures/stage1_primitive_brownian_irfs.png`).

## 4. Effect of the inherited external-claim position

Comparing the automation shock's impact under `full_access` (inherited position `s_-=0.4465` pays on the innovation) against `no_external_claim` (`s_-=0`, same state, same feedback rule thereafter):

| | full_access | no_external_claim | difference |
|---|---:|---:|---:|
| Public net worth (impact) | +13.62e-4 | 0 (by construction) | +13.62e-4 |
| Worker consumption (impact) | +7.59e-4 | +7.32e-4 | **+0.27e-4** |
| Transfer (impact) | +8.45e-4 | +8.17e-4 | +0.28e-4 |

Holding the claim adds about 0.27e-4 to worker-consumption impact response from this specific shock — a modest but nonzero effect, consistent with `s_-·λ̂_x·Δx` at this calibration's holding size. The full trajectory (`figures/stage2a_consumption_with_and_without_claim.png`) shows the gap persists and even flips sign by year 15–20 as the tax/capital dynamics dominate. This confirms the *timing* discipline the spec requires: the position **selected before** the innovation determines its payoff; a position chosen after would only affect *later* innovations, which is exactly why `no_external_claim` zeroes out the net-worth deviation on impact but not the subsequent state-driven dynamics.

## 5. The leading small-risk portfolio at the steady state

At `N̄=0`, `X̄=12.569`:

- Traded payoff: `λ̂ = (0.02597, 0.04844)`, `β̂=‖λ̂‖²=0.003021`
- Fiscal-wealth shock loading: `ζ_J = (0.3377, 0.5750)`
- Marketed component of the fiscal hedge: `ζ_J·λ̂/β̂ = 12.123`
- **Leading position:** `s̄_0 = X̄ − 12.123 = 0.4465`
- Portfolio curvature: `−9.46e-4` (strictly concave, confirming a unique interior optimum)
- Unmarketed (unspanned) fiscal-risk loading: `ζ_J^⊥ = (0.0229, −0.0123)` — the component of automation/productivity fiscal risk the one traded claim cannot span, confirmed exactly orthogonal to `λ̂` (residual 1.5e-18)

The decomposition is stark: the myopic log-Merton demand alone would be `s=X̄=12.569`, but the fiscal-endowment hedge subtracts `12.123` of that — **96% of the naive return-chasing demand is offset by the hedging motive**, leaving a small residual long position of `0.4465`. Economically, domestic fiscal wealth is heavily exposed to the same states the traded claim pays off on, so nearly all of the "invest because it earns a premium" motive is really "hold it because it hedges the treasury's own exposure," not a large net speculative bet.

## 6. Welfare value of claim access vs. the fiscal hedge

Both reported as permanent proportional (log-consumption) worker-consumption-equivalents, leading order in `ε` at `ε=1`:

- **Access value** (optimal `s̄_0` vs. `s=0`, same inherited state): `Q_access = 9.43e-5` ≈ **0.0094%** of permanent consumption
- **Hedge value** (optimal `s̄_0` vs. the myopic Merton position `s=X̄`, same inherited state): `Q_hedge = 0.0695` ≈ **6.95%** of permanent consumption

The hedge value is roughly **737× larger** than the access value (`0.0695416 / 0.0000943346 = 737.1`) — and this ratio is not a numerical accident: with a quadratic portfolio objective of curvature `−β̂/(ρX̄²)`, the welfare loss from holding position `s` instead of `s̄_0` scales with `(s−s̄_0)²`, and `s=0` is far closer to `s̄_0=0.4465` (distance `0.4465`) than `s=X̄=12.569` is (distance `12.1227`) — the squared-distance ratio, `(12.1227/0.4465)²=737.2`, matches the computed `Q_hedge/Q_access` ratio to within 0.02%. The economically important reading: **at this calibration, merely having access to the claim buys little; getting the hedge ratio right is what matters.** A planner who held the asset but sized it via naive log-Merton (ignoring the fiscal covariance) would capture almost none of the available welfare gain.

## 7. Local validity: are the reported paths still inside the maintained region?

Minimum values across **every** reported horizon (0–40 years, quarterly grid) and **every** experiment (14 experiment/regime combinations):

| Boundary | Minimum slack (must be `>0`) |
|---|---:|
| Full-specialisation margin (automation composite) | 0.9898 |
| Full-specialisation margin (new-task composite) | 0.5767 |
| Transfer floor (`c ≥ W`) | 0.0192 |
| Comprehensive resources `X` | 12.524 (always `>0`) |
| Portfolio bounds (±20 scaffolding limit) | 19.5–20.0 (deeply interior) |
| Tax bounds (`[-0.5, 0.95]` scaffolding) | 0.99 / 0.44 |
| Structural tax ceiling (`τ<1`) | 0.49 |
| Tax-speed bound (`±0.5`) | 0.49 |

Every boundary stays strictly interior with wide margin over the full 40-year horizon and every constructed/inherited/eigenmode experiment, including the largest displacements (1% log-capital, 1pp tax, and the damped capital–tax eigenmode). None of the artificial numerical scaffolding binds anywhere; the two genuine economic boundaries (specialisation, transfer floor) are the tightest but still comfortably positive. The local LQ approximation and the maintained interior branch remain jointly plausible over the full reported region for this calibration.

## What worked

- The full pipeline (primitives → anchor → local matrices → Riccati/Sylvester/Lyapunov → portfolio/welfare → IRFs → diagnostics) runs in well under a second and passes every one of 35 independent acceptance checks, comfortably inside the `L0_smoke`/`L1_interactive` budget.
- Every matrix entry was independently re-derived by hand from the reconciled proof-plan text before being written into code (not transcribed from the pre-existing prototype), and the final numbers (`H_rr`, closed-loop roots, leading portfolio position, welfare values) match a *separately authored*, pre-existing prototype implementation to full float64 precision — meaningful independent verification, not two copies of the same bug.
- Two independent Riccati solution methods (closed-form quartic roots; ordered-Schur invariant subspace) agree to 7.7e-16 relative error at the baseline, and were cross-checked across a `χ` sweep spanning both the real-root and complex-root regimes.
- Finite-difference cross-checks of every primitive derivative feeding the local quadratic system (output, rental rate, wage, safe rate) against the exact nonlinear evaluator all land at 1e-8–1e-10 relative error — the expected scale for a central-difference truncation error, not an implementation bug.

## What failed, or remains uncertain

- **A live adversarial process materially strengthened the diagnostics module.** During development, a concurrent editing process (see session record — likely deliberate hand-editing to test this exact deliverable) repeatedly introduced targeted bugs directly into the solver (a sign flip in the capital–tax closed loop, a halved feedback gain, an injected pre-symmetrization asymmetry) to probe whether the "independent" diagnostics module would actually catch them. It found three real gaps, now fixed and permanently regression-tested: (1) nothing independently reconstructed the closed-loop matrices `A_c`/`A_rc`/`F` from primitives and the solved `H` — every other residual check was self-consistent with whatever the solver produced, so a construction bug there passed *every* check including the Hurwitz test; (2) the solver's forced post-hoc symmetrization of near-symmetric matrices made the symmetry-error check trivially pass regardless of whether the raw solve was any good, masking rather than measuring solve quality; (3) one anchor identity check compared a value to the literal formula that defined it, and could never fail regardless of whether the upstream algebra computing `z̄` was correct. All three are now independently checked and covered by dedicated regression tests. This is reported here rather than omitted because it is directly relevant evidence for how much to trust the "independent" claim in this and future CS specifications: a first pass at "independence" is not automatically the real thing, and adversarial testing found real gaps that ordinary testing had not.
- No exact precautionary consumption or tax-speed correction is reported — Stage 2B (higher deterministic fiscal-wealth derivatives, or an explicit truncated stochastic-LQ closure) is out of scope here.
- The stable four-state block has a stationary covariance, but public net worth and worker consumption retain the documented neutral direction and are not claimed stationary.
- This is a `T3_local` result on a single illustrative parameter vector: no sensitivity analysis, no empirical calibration, and no claim about the matched PDE–LQ bridge vector (CD001) or the standalone PDE feasibility exercise (CP004/CS004) is made or implied.

## Implications for the next computational decision

1. **CS001's status should be updated.** The codex-side registry should record this repository as the bound implementation, and CS001 should move toward `review_ready`/`approved` reflecting an implementation now exists and passed acceptance — that's a decision for the research workspace side, not something this session can do (per the ownership boundary: implementation code lives here, the specification registry lives there).
2. **The hedge-dominates-access welfare result (§6) is worth carrying into CS003** (small-risk portfolio and welfare correction) as a specific, quantified prior to check against: does it survive the exact precautionary corrections, or is it an artifact of the leading-order truncation?
3. **The damped-oscillation tax/capital dynamics (§2)** — an ~80-year period, ~9-year half-life — are a testable, falsifiable prediction of the LQ approximation that the nonlinear deterministic transition (CS002) or the PDE feasibility work (CS004) can be checked against once either is available.
4. This baseline is the natural reference point for the CD001-mandated matched PDE–LQ bridge run: this exact primitive vector, re-run through the LQ machinery here, is already the "first bridge uses the LQ vector" half of that comparison.

## Reproducing this run

```bash
uv run pytest                                                 # 66/66 tests
uv run python -m tai_public_finance.cs001_lq_anchor.cli \
  --config configs/cs001/lq_farhi_smoke.json \
  --output-dir outputs/cs001-lq-anchor-baseline
```

Full inputs, matrices, tidy IRFs, diagnostics, and artifact hashes are in `report.json`, `complete_input.json`, `irfs.csv`, `portfolio_net_worth_grid.csv`, and `matrices/*.csv` alongside this file; the immutable run record is `runs/RUN-20260901T184527Z-CS001-89f6a939-01.yaml` at the repository root.
