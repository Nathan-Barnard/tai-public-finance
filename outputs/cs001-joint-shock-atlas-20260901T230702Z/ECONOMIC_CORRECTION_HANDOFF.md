# Handoff: economic correction of 2026-09-01 (wages are not government revenue)

Scope: this session (tai-public-finnace-claude-27, branch
`cs001/joint-shock-atlas`) applied the correction to its own new
deliverables and audited the shared code and committed findings on `main`
read-only. It did not edit shared modules or other sessions' outputs.

## Files changed (all new, on `cs001/joint-shock-atlas`)

- `src/tai_public_finance/cs001_lq_anchor/shock_atlas.py` (new)
- `src/tai_public_finance/cs001_lq_anchor/shock_atlas_cli.py` (new)
- `tests/cs001_lq_anchor/test_shock_atlas.py` (new)
- `outputs/cs001-joint-shock-atlas-20260901T230702Z/` (new run bundle)
- `STATUS.md` on `main` (one additive coordination row)

## Tests run

`uv run pytest`: 109 passed (78 pre-existing + 31 new) at commit `ee48aff`;
the full atlas passed all independent checks (`numerical_diagnostics.json`).

## Did any numerical result change?

No. The maintained equations, the Riccati/Sylvester/Lyapunov solution, the
leading portfolio and the IRF propagation are untouched; the atlas reuses
them and reproduces the repaired baseline (frozen regression snapshots pass;
the dα = +0.01 constructed experiments are reproduced to 5.6e-17). What
changed is naming, reported accounts, and claimed status.

## What was only relabelled or reclassified

- "fiscal wealth J" → planner-resource wealth (worker fiscal-endowment
  wealth; includes future worker wages); "comprehensive resources X" →
  worker comprehensive resources; "leading position" → unconstrained
  leading small-risk portfolio; "borrow safely to hold the claim" →
  unconstrained negative safe position / desired debt-financed risky
  holding, conditional on slack genuine transfer and solvency boundaries.
- Feasibility flags in the atlas rows are split into genuine economic
  conditions (`economic_conditions_ok`: specialisation branch, transfer floor,
  positive c and X, τ < 1) and numerical scaffolding slack
  (`numerical_scaffolding_ok`). Neither is called borrowing capacity. Every
  portfolio statement carries the classification *unconstrained local
  desired portfolio; genuine fiscal-capacity feasibility unverified*.
- The separate accounts (W, τB, T, Ψ, τB − T − Ψ, N, s, N − s, transfer-floor
  slack) are reported on every row; `F − c = τB − T` and the J = J_W + J_B
  split are verified on every row; the anchor decomposition
  W/ρ = 11.569, τB/ρ = 1.000, J = 12.569 is recorded in `models.json` with
  its constant-flow caveat.

## Did any existing feasibility check depend on treating J as collateral?

No. On `main`, J enters only (i) consumption c = ρ(N + J), (ii) the
leading-portfolio first-order condition through X, and (iii) the order-two
welfare objects. The checks that use J are: the transfer floor c ≥ W
(a genuine worker-side condition, `diagnostics.py:491`,
`portfolio.py:244-248`) and X > 0 (a prerequisite of the log-Merton
objects, `portfolio.py:81-82`). Neither asserts debt capacity. The
portfolio "feasibility" flags compare positions with the artificial ±20 caps:
`portfolio.py:115-117, 155-159` (`portfolio_lower/upper_slack`,
`zero_position_feasible`, `merton_comparator_feasible`),
`diagnostics.py:483-485, 492-503` (`portfolio_interior`,
`same_state_*_comparator_feasible`, `reported_paths_numerical_scaffolding_slack`),
and `portfolio.py:249-259` where `net_worth_grid` folds the artificial cap
into a row-level `feasible` flag (`portfolio_bound_feasible`). These are
numerical-validity checks whose *names* suggest economic feasibility; they
should be relabelled (e.g. `numerical_scaffolding_slack`) but no code
logic uses them as proof of borrowing capacity.

## Exact locations of remaining "fiscal wealth", "safe borrowing", "borrowing capacity" language on `main` (canonical updates pending)

Code identifiers (rename is a cross-module change; not done here to avoid
colliding with the three other active CS001 branches):

- `anchor.py:46-47, 88-90, 118-119` — `fiscal_wealth_normalized_bar`, `fiscal_wealth_bar`
- `equations.py:65, 141-150, 186` — `linear_fiscal_wealth` (j)
- `portfolio.py:4-5, 25, 31-37, 79-80, 89, 99, 140, 147, 180, 191, 200-218` — docstrings and fields `fiscal_wealth`, `marketed_fiscal_wealth_amount`, `net_worth_to_fiscal_wealth_ratios`, "fiscal-wealth shock loading", "deterministic fiscal-wealth family"
- `portfolio.py:46, 48, 157, 159, 237-259` and `diagnostics.py:483-485, 492-503` — `*_feasible` / `portfolio_interior` names applied to artificial caps (see above)
- `diagnostics.py:69-85, 378, 402, 452` — `linear_fiscal_wealth_identity_errors`
- `irfs.py:180, 279` — `linear_fiscal_wealth`
- `cli.py:59, 95` and `net_worth_grid_cli.py:51, 68, 111-120, 136` — "fiscal-wealth", `n_feasible`, "grid points are feasible (positive-X, non-negative transfer, portfolio-bound)"
- `reporting.py:355, 390` — "feasibility" wording in the run-record template
- `tests/cs001_lq_anchor/test_regression.py:25`, `test_net_worth_thresholds.py:19-81`, `test_portfolio.py:11-23, 45`, `test_anchor.py:34-36`, `test_equations.py:52-55`, `test_repair_2026_09_01.py:19-64` — tests bound to the current names/flags

Committed findings prose (interpretive, superseded by this correction;
edit or supersede, do not silently rewrite immutable run records):

- `outputs/cs001-net-worth-sensitivity/FINDINGS.md:23` — "Is 'borrow safely to hold the claim' robust ... the government always finances part of its risky position with additional safe borrowing"; `:27` — "How does claim-access value change with fiscal capacity?"
- `outputs/cs001-lq-anchor-baseline/FINDINGS.md` §5 — "Leading position", "fiscal-wealth shock loading", "Marketed component of the fiscal hedge"; §7 — portfolio/tax scaffolding slack listed alongside genuine boundaries under "Local validity"
- `outputs/cs001-lq-anchor-baseline-repair-01/FINDINGS.md` §1, §4 — "Fiscal-hedge component", "N/J grid ... feasible"
- `runs/RUN-*.yaml` records (immutable): `economic_quantities.portfolio_anchor.*`, `reliable_region` wording — leave as historical evidence; the correction applies to interpretation

Codex research workspace (read-only for this session; canonical owner must update):

- `research-notes/local-lq-system-computation-and-proof-plan.md` — "fiscal wealth" throughout; "the illustrative N̄=0 government borrows safely to hold the risky international claim"
- `hypotheses/hypothesis_map.md`, HYP005 — "Fiscal wealth can support debt-financed public risky investment ... nonfinancial wealth in the separated local problem"
- `computation/specifications/...--CS001.md` — output names (`fiscal wealth`, "portfolio/welfare objects")

## Next separate modelling task (recorded, not implemented)

An auxiliary government fiscal-capacity object C_G(S), excluding wages and
based on attainable government primary surpluses with T = 0, requires an
approved economic specification before implementation. It must not be
substituted into the worker welfare or portfolio equations: J and C_G
answer different questions.
