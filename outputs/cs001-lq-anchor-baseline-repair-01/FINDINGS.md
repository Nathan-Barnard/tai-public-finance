# CS001 baseline repair (2026-09-01)

**Run:** `RUN-20260901T194938Z-CS001-d876a61e-01` — supersedes
`RUN-20260901T184527Z-CS001-89f6a939-01`, which is preserved unchanged as
historical evidence at [`../cs001-lq-anchor-baseline/`](../cs001-lq-anchor-baseline/).
**Outcome:** pass — all acceptance checks passed.
**Scope:** this is an **exploratory first CS001 tranche**, not a completed
or approved CS001 result — CS001 remains `draft`, unfingerprinted, in the
codex registry.

The underlying computation (matrices, Riccati/Sylvester/Lyapunov solve,
IRF propagation) was already correct and is unchanged. This run corrects
five reporting/interpretation defects found in the first baseline. Only the
corrections are documented below; everything else in the original
[FINDINGS.md](../cs001-lq-anchor-baseline/FINDINGS.md) (steady state, tax
feedback, joint-shock IRFs, boundary slack) stands.

## 1. Portfolio decomposition, corrected

The leading position now decomposes into two explicitly-signed components:

- **Return-demand component** = `X̄` = **12.5692**
- **Fiscal-hedge component** = `−(ζ_J·λ̂)/β̂` = **−12.1227**
- **Optimal position** = 12.5692 + (−12.1227) = **0.4465**

The fiscal-hedge component is **negative** — it *reduces* exposure to the
claim, because domestic fiscal wealth already covaries positively with the
claim's payoff. This is a short/reduced position relative to the naive
log-Merton demand, not "holding more of the claim." (The original
FINDINGS.md described this correctly in prose but never exposed the signed
components in the data; both now agree.)

## 2. Welfare language, corrected

- **Access value** (`Q_access = 9.4335e-05`, ≈0.0094% of permanent
  consumption): optimal `s̄_0` vs. `s=0`, same inherited state.
- **"Hedge value"** (`Q_hedge = 0.069542`, ≈6.95% of permanent consumption)
  is the **leading-order loss avoided** by holding `s̄_0` instead of being
  forced into the myopic Merton position `s=X̄`. It is **not** a second,
  additive welfare gain stacked on top of `Q_access` — the two compare the
  optimum against two different baselines (`s=0` vs. `s=X̄`) and must not be
  summed or read as sequential stages ("first get access, then get an
  additional gain from hedging").
- Both are **leading small-risk coefficients evaluated at ε=1**, not exact
  finite-risk welfare magnitudes.

## 3. Claim/no-claim IRF gap, corrected

The original FINDINGS.md claimed the full-access/no-claim worker-consumption
gap "persists and even flips sign by year 15–20." **This was wrong** — it
described the individual absolute consumption paths (which do converge
toward zero as the shock decays) rather than the gap between them. Directly
measured from `irfs.csv`:

| Shock | Gap (constant across all 161 reported horizons) |
|---|---:|
| Productivity | `1.475534e-05` (identical to 1e-11) |
| Automation | `2.752428e-05` (identical to 1e-11) |

The gap is **exactly constant and strictly positive** over the full 40-year
horizon for both shocks — not merely same-signed. This is a structural
property of the linear system: comprehensive resources `X` has no
mean-reversion term (`Ẋ₁ = X̄·d_r'·y₁`, with no `X₁` self-term), so a pure
net-worth payoff carrying zero state displacement is a permanent, undecaying
addition to `X` and hence to consumption via `c=ρX`. `test_repair_2026_09_01.py`
now asserts this exactly (`max(gaps) - min(gaps) < 1e-12`).

## 4. Public-net-worth grid, repaired

Every row now reports comprehensive resources, worker consumption, wages,
transfer, transfer feasibility, portfolio-bound feasibility, and an overall
`feasible` flag with explicit `failure_reasons`. No row is dropped, and no
row can silently pass through as feasible:

| N/J | Transfer | Feasible | Failure reason |
|---:|---:|:---:|---|
| −0.50 | −0.1068 | **No** | `negative_transfer` |
| −0.25 | −0.0433 | **No** | `negative_transfer` |
| 0.00 | 0.0202 | Yes | — |
| 0.25 | 0.0837 | Yes | — |
| 0.50 | 0.1472 | Yes | — |

The `−0.5` and `−0.25` members of the deterministic fiscal-wealth family
violate the maintained non-negative-transfer condition (`c≥W`) and are
retained as failed experiments, never labelled feasible. Previously the code
did not compute transfer feasibility at all for the grid, and a
non-positive-`X` row would raise an exception rather than being recorded.

## 5. Provenance and fingerprints, corrected

The run record's `specification` object no longer reports the complete-input
hash as if it were CS001's own fingerprint. CS001 has no approved fingerprint
yet (`status: draft`, `fingerprint_sha256: null`). The primitive, experiment,
and complete-input fingerprints are now recorded separately under
`input_fingerprints` in the run record.

## What's unfinished

Nothing from the six priority fixes was skipped or weakened. Not attempted,
per the stated scope: parameter sweeps, alternative models, Stage 2B exact
precautionary corrections, or new economic derivations — all correctly out
of scope for this repair.

## Reproducing this run

```bash
uv run pytest
uv run python -m tai_public_finance.cs001_lq_anchor.cli \
  --config configs/cs001/lq_farhi_smoke.json \
  --output-dir outputs/cs001-lq-anchor-baseline-repair-01
```
