# CS002 D2 review-repair findings

**Supersedes:** [`RUN-20260902T190139Z-CS002-D2-0b721f21-01`](../../runs/RUN-20260902T190139Z-CS002-D2-0b721f21-01.yaml)
(`outputs/cs002-d2-mean-reversion-20260902T190139Z/`), left unmodified. That
run's headline `computational_pass` verdict is superseded by this run's
`numerical_failure` — not because the repair introduced a regression, but
because the repair corrects a check that was previously too weak to detect
a real problem the old evidence already contained.

**This run:** [`RUN-20260902T203314Z-CS002-D2-REPAIR-94332783-01`](../../runs/RUN-20260902T203314Z-CS002-D2-REPAIR-94332783-01.yaml),
code commit `9433278` (full: `94332783e8db9951bf446582c4af5875a0962f65`),
config `lq_farhi_d2_mean_reversion_v1` (fingerprint `af6b8d613e26...`,
identical to the old run — same primitives, same shocks, same horizons).

## Repair 1: physical-unit terminal map

**Defect:** `terminal.py`'s `lq_stable_manifold_costates` and
`lq_quadratic_value_tail` read CS001's NORMALIZED fiscal-wealth
coefficients (`j`, `H`) and used them directly (`j + H@y`,
`j@y + 0.5 y'Hy`) with no `K_bar` conversion, instead of the physical-unit
formula the D2 handoff itself specifies:

```
J_y^L = K_bar*(j + H@y)
J^L   = J_bar + K_bar*(j@y + 0.5 y'Hy)
ell_T = J_k^L / K_T
m_T   = J_tau^L
```

Since every calibration run so far has `K_bar=1` (baked into CS001's own
steady-state normalization, `anchor.py`), multiplying by the missing factor
was numerically invisible — a silent no-op — in all existing evidence, but
would have been silently wrong at any other `K_bar`. This terminal map is
not only a post-processing formula: it is also the actual BVP terminal
BOUNDARY CONDITION for `ell(T)`/`m(T)` under the `"lq_stable_manifold"`
convention (`bvp.py`'s `economic_bc`/`economic_bc_with_exogenous_path`), so
the defect would have affected the solved path itself, not only reported
values, at any `K_bar != 1`.

**Fix:** added the explicit `K_bar` multiplier to both functions.

**Verification:**
- New manufactured test (`K_bar=2.5`, normalized `j_k=1` per the review's
  explicit instruction — not `j_k=K_bar`, which is what the bug's own
  "cancellation" pattern would require to look right): confirms the anchor
  gives `ell=1`; the displaced-point gradient and value tail match the
  explicit `K_bar`-scaled formula; a central finite-difference derivative
  of the value tail (a completely independent check, not a re-derivation
  of the same closed form) agrees with the returned costates to `rel=1e-5`;
  and an inlined replica of the OLD (unscaled) formula disagrees with the
  correct result by a full factor of `K_bar` (~0.59 absolute on ell≈0.98,
  ~0.06 absolute on the value tail) — far beyond any numerical tolerance.
- An existing test (`test_terminal_generalizes_to_arbitrary_tau_bar_and_
  capital_bar`) had a fixture bug of its own: it set `j_k=capital_bar`
  (the UNNORMALIZED quantity the review's bug produces), which happened to
  make the anchor's `ell=1` check pass under the OLD buggy code by
  cancellation. Corrected to `j_k=1` (normalized) so it now validates the
  FIXED formula instead of accidentally re-validating the bug.
- K_bar=1 no-op, full precision: see the "K_bar=1 regression" section below
  — every reported quantity in both shock directions matches the prior run
  to floating-point noise (worst full-path difference ~1e-12).

## Repair 2: complete horizon and mesh comparisons

**Defect:** `experiment_d2.py`'s `_horizon_mesh_comparisons` compared only
the four raw BVP state variables (`k`, `tau`, `ell`, `m`) — not the other
14 reported quantities (`z`, `x`, `r0`, `capital`, `nu`, `output`,
`fiscal_resources`, `J`, `X`, `N`, `c`, `varpi`, and the two LQ comparator
paths) — and scored all four under ONE tolerance pooled from the LARGEST
peak response among them (`tolerance = max(1e-7, 1e-3 * max(peak_response
across k/tau/ell/m))`), rather than each variable's own effect-scaled
tolerance.

**Fix:**
- `recovery.py` gained `REPORTED_PATH_NAMES` (all 18 reported quantities),
  `anchor_reference_values` (each quantity's steady-state/anchor reference,
  read directly off CS001's own `SteadyState` fields — `output_bar`,
  `fiscal_resources_bar`, `comprehensive_resources_bar`,
  `worker_consumption_bar`, `public_net_worth_bar` — no new economic
  derivation), and `reconstruct_reported_paths` (evaluates every quantity
  at a given `t_grid` from one solved BVP result plus its recovery
  objects).
- `experiment_d2.py`'s rewritten `_horizon_mesh_comparisons` reconstructs
  the COMPLETE solution for both sides of every comparison on the full
  pairwise common interval (`_pairwise_common_grid`, unchanged from D1's
  own inherited fix — still correctly gives `[0,40]` for the 80-vs-40
  comparison, not `[0,20]`), and `_compare_reported_paths` scores each of
  the 18 paths against `max(1e-7, 1e-3 * that path's own peak response)`
  independently — never pooled.
- The comparison run's `J`/`X`/`N`/`varpi` are recovered FRESH at its own
  horizon-appropriate terminal tail (same convention as the primary run:
  quadratic LQ tail for `J`, `tail_value=0` for `varpi`, each evaluated at
  that run's own terminal state/horizon); the baseline side reuses the
  SAME already-recovered objects the final report itself uses, only
  interpolated onto each comparison's own grid.

**Verification:** new regression tests confirm (a) omitting a required
path from either side is rejected (`ValueError`); (b) perturbing any ONE
of the 18 paths fails only that path's own check, not the others (no
implicit pooling); (c) a path with a tiny effect (tight tolerance) fails a
perturbation that would have PASSED under a co-reported path's much larger,
pooled tolerance; (d) the 80-vs-40 comparison's common grid still reaches
`t=40` (direct check) and, end to end, several reported paths' actual
maximum-difference TIME reaches `t=40` in the real run (impossible under
the old `[0,20]`-capped construction).

## The repair surfaced a genuine, previously invisible finding

With the check now complete, `horizon_mesh_stability` **fails** for both
shock directions — not from a bug in either repair, but from a real
property of the model at the declared 20-year comparison horizon:

**`varpi`'s ODE (`varpi_dot = rho*varpi - (r0-rho)/rho`) has relaxation
timescale `1/rho ≈ 49.5 years`** (`rho ≈ 0.0202` in this calibration) —
far longer than the 20-year comparison window. Its terminal tail
(`varpi(T)=0` at whichever horizon a run uses) has not decayed to
negligible influence by `t=20`. Concretely, for the productivity direction:

| t (years) | baseline varpi (T=40 tail) | 20y-comparison varpi (T=20 tail) | diff |
|---:|---:|---:|---:|
| 0 | -0.451635 | -0.447222 | -0.004413 |
| 10 | -0.054860 | -0.049459 | -0.005401 |
| 18 | -0.010112 | -0.003764 | -0.006349 |
| 20 | -0.006610 | 0.000000 | -0.006610 |

The divergence is smooth and monotonically growing toward `t=20` (where the
20y run's terminal condition forces `varpi=0` exactly, while the baseline's
own `varpi(20)` is a genuine, non-trivial interior value) — the expected
shape of an artificially-short terminal tail's influence propagating
backward along a slowly-relaxing ODE, not a numerical artifact.

**Magnitude:** `max_diff=6.610e-03` against `tolerance=4.516e-04` for
productivity (ratio 14.6x over); `max_diff=2.634e-02` against
`tolerance=1.767e-03` for automation (ratio 14.9x over) — both maxima
occur exactly at `t=20.0`, the comparison's own endpoint.

**Secondary finding (automation direction, `tax_rate`):** `max_diff=
1.138e-05` against its own tolerance `9.593e-06` (ratio 1.19x over) — this
value is BIT-IDENTICAL to the old run's own recorded `max_diff_tau`
(`1.1378039747045321e-05`, see `outputs/cs002-d2-mean-reversion-
20260902T190139Z/horizon_mesh_comparisons.csv`), confirming this is purely
a tolerance-scoring difference, not a data difference: the OLD pooled
tolerance (`4.108e-05`, set by `ell`'s larger peak response) let this same
number pass; `tax_rate`'s own, correctly-scoped tolerance does not. This is
exactly the failure mode Repair 2 targets, directly demonstrated on real
data.

**80-year and mesh-refinement comparisons pass cleanly** in both
directions across all 18 paths — the instability is specific to the
20-year comparison horizon, not a general solver or model problem.

**Disposition:** per this task's explicit stop condition ("a recovered
path fails horizon or mesh stability"), this is reported as a genuine
finding rather than forced to a pass by adjusting tolerances, the terminal
convention, or the declared comparison horizons — any of which the task
explicitly forbids doing to obtain convergence. The declared 20-year
comparison horizon is not adequate to certify `varpi`'s horizon-stability
given this calibration's discount rate; a research-owner decision (e.g.
dropping 20y from `varpi`'s own comparison set, or accepting `varpi`
specifically as `horizon-sensitive` at 20y while still `stable` at ≥40y)
is needed before this can be re-evaluated, and is out of this task's scope
(changing the config or acceptance thresholds is explicitly prohibited
here).

## K_bar=1 regression (full-path, both directions)

Compared against the superseded run's own committed `{direction}_path.csv`
(the complete reported solution, all 18 columns, 201 points over
`[0,40]`) — see `regression_comparison.json` for the complete per-column
table. Worst full-path absolute difference across ALL 18 reported
quantities, both directions:

- productivity: `J`, `max_abs_diff=1.055e-12` (at `t=32.5`)
- automation: `J`, `max_abs_diff=1.634e-13` (at `t=17.5`)

Every other independently-computed scalar also matches to displayed
precision: `J(0)`, `varpi(0)` and both routes' disagreements, the
independent ODE/boundary residuals, budget-separation residual, the
wrong-`r0` detection margin, and the `X`/`N` route deviations are
unchanged (see `regression_comparison.json`). Every check EXCEPT
`horizon_mesh_stability` remains `True` in both directions. This confirms
Repair 1's `K_bar` multiplier is a true no-op at `K_bar=1`, and that
Repair 2's rewrite changed only what is compared and how tolerances are
scored — not any underlying recovered quantity.

## Tests

- Focused (`tests/cs002_nonlinear_transition/`): **96 passed**, 0 failed.
- Full repository suite: **174 passed**, 0 failed (168 in the pre-edit
  baseline, +6 net new: 1 Repair-1 regression test, 5 Repair-2 regression
  tests).
- 3 pre-existing tests (2 in `test_experiment_d2.py`, 1 in
  `test_cs002_d2_cli.py`) were updated in place — not weakened or deleted
  — to assert the corrected, honest `numerical_failure`/`horizon_mesh_
  stability=False` outcome instead of the `computational_pass` their
  assertions previously assumed; each documents why in its own docstring.
  No tolerance was weakened anywhere to obtain a pass.

## Runtime

Material run wall time: ~5.2s (well under the 10-minute `L1_interactive`
budget). Total implementation time: see the run record's
`task_elapsed_seconds` (an end-to-end estimate, not an instrumented
measurement, per its own documented caveat).
