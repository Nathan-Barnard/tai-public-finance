# Public-net-worth (N/J) sensitivity exercise (2026-09-01)

**Based on:** `RUN-20260901T194938Z-CS001-d876a61e-01` (repaired baseline; unchanged). Same primitive calibration, q_D=1, worker-only welfare, maintained smooth full-specialisation branch. **New run:** `RUN-20260901T201333Z-CS001-NJGRID-8e4ceb25-01`, 18 points, 13 feasible, 5 retained-and-flagged infeasible.

**Derived thresholds** (from the anchor/portfolio objects, not hardcoded — see `transfer_boundary_net_worth_ratio`/`portfolio_sign_change_net_worth_ratio` and their tests):
- Transfer boundary `ρ(N̄+J̄)=W̄`: **N̄/J̄ = −0.079560** (task estimate: ≈−0.0796 ✓)
- Portfolio sign-change `N̄+J̄=(ζ_J·λ̂)/β̂`: **N̄/J̄ = −0.035523** (task estimate: ≈−0.0355 ✓)

Both match the stated estimates to 4 decimal places — no discrepancy to diagnose.

## The central structural result

**The fiscal-hedge component is exactly constant across the entire N̄/J̄ grid: −12.1227, for every feasible and infeasible point alike.** It is a fixed absolute quantity (`ζ_J`, `λ̂`, `β̂` are anchor-level objects that do not depend on `N̄`), not a share of resources. Consequently:

- **Return-demand component** = `X̄ = N̄+J̄` moves one-for-one with `N̄` (slope exactly 1).
- **Residual safe position** `N̄−s̄` = `−(J̄ + fiscal_hedge_component)` = **−0.4465, exactly constant** across every single point in the grid, feasible or not.
- **Optimal position** `s̄(N̄) = N̄ + 0.4465` exactly — a pure vertical shift of `N̄` by a fixed constant.

## Answers to the five questions

**1. Over what feasible range is the government long vs. short?** Short (`s̄<0`) for feasible `N̄/J̄ ∈ [−0.0796, −0.0355)`; long (`s̄>0`) for `N̄/J̄ ∈ (−0.0355, 0.5]` and beyond. The baseline `N̄=0` sits well inside the long region, with `s̄=0.4465`.

**2. Is "borrow safely to hold the claim" robust, or specific to N̄=0?** **Robust across the entire feasible family.** `N̄−s̄=−0.4465<0` holds at *every* feasible grid point, not just near `N̄=0` — the government always finances part of its risky position with additional safe borrowing beyond `N̄` itself, because that residual is a fixed constant independent of `N̄`. What *is* specific to a narrow band (`N̄/J̄∈[−0.0796,−0.0355)`, about 4.4 percentage points of the ratio wide) is the *sign of s̄ itself* — the government is short, not long, in that band. The "small long position" language in the original baseline findings describes the generic case (true for essentially the whole feasible range above the sign-change point, including all of 0/0.25/0.5), not a knife-edge artifact of `N̄=0`.

**3. How do return-demand and fiscal-hedge separately move with N̄?** Return-demand rises one-for-one with `N̄` (elasticity of the *level*, not the ratio, is exactly 1). Fiscal-hedge does not move at all. All of the sensitivity of `s̄` to `N̄` comes from the return-demand side; none comes from the hedge side.

**4. How does claim-access value change with fiscal capacity?** It is **essentially zero exactly at the sign-change point** (`1.1e-14` at `N̄/J̄=−0.035523`) — mechanically, because access is only valuable when the optimal position differs from the forced `s=0` comparator, and at that point it doesn't. Access value rises in *both* directions away from that point (`9.4e-5` at baseline `N̄=0`; `9.5e-3` at `N̄/J̄=0.5`; `1.5e-4` at the feasible boundary `N̄/J̄=−0.077`) — a U-shape in `N̄`, not a monotone function of fiscal capacity.

**5. Which apparent results disappear once the transfer constraint is enforced?** All five infeasible points (`N̄/J̄ = −0.15, −0.12, −0.085, −0.082, −0.07956`) show the government *short* the claim by economically large amounts (`s̄` from −1.44 to −0.55) — this could read as "a poor government sells the claim short at scale." **That result is not economically admissible**: every one of these points requires a negative transfer (workers receiving less than the wage bill), which violates the maintained `c≥W̄` condition. The genuinely admissible short region is narrow (`N̄/J̄∈[−0.0796,−0.0355)`, `s̄∈(−0.55,0)`), far smaller than the unconstrained numbers suggest.

## Interpretation discipline maintained

Each row is a different member of the deterministic fiscal-wealth family (see `[[cs001-lq-anchor-baseline]]`/proof plan) — the comparisons above are comparative statics across that family, not a claim about what happens *if* public net worth changed dynamically. Access and Merton-misallocation values are both evaluated at the same inherited state within each row. No empirical, global, finite-risk, or exact-precautionary-policy claim is made. See [`portfolio_vs_net_worth.png`](portfolio_vs_net_worth.png) and [`net_worth_grid.csv`](net_worth_grid.csv) for the full data.

## Reproducing

```bash
uv run pytest
uv run python -m tai_public_finance.cs001_lq_anchor.net_worth_grid_cli \
  --config configs/cs001/lq_farhi_smoke.json \
  --ratios="-0.15,-0.12,-0.085,-0.082,-0.07956,-0.077,-0.074,-0.06,-0.045,-0.040,-0.035523,-0.031,-0.026,-0.01,0.0,0.1,0.25,0.5" \
  --output-dir outputs/cs001-net-worth-sensitivity
```
