# Morning report: CS001 joint productivity-automation shock atlas

Run `outputs/cs001-joint-shock-atlas-20260901T230702Z`, commit `ee48aff`
(branch `cs001/joint-shock-atlas`, clean worktree), started 2026-09-01
23:07:03 UTC, computation finished 19 seconds later. Every independent
check passed; nothing was weakened to finish. This is exploratory
local-LQ evidence on the illustrative Farhi-based vector: a directional
mechanism map, not a probability statement about shocks, a global
solution, or a welfare ranking.

## The map in plain language

Think of a compass whose east (0°) is a good productivity surprise, north
(90°) is a good automation surprise, west a bad productivity surprise and
south a bad automation surprise. Every point on the circle is the same
size of surprise (one standard deviation of the joint Brownian innovation
over one trading day). Going counter-clockwise from east:

1. **East to 77°, "everyone gains"**: output, wages, capital-tax receipts,
   planner-resource wealth, the inherited claim, worker consumption and
   transfers all rise. Transfers rise faster than tax receipts almost
   everywhere in this arc.
2. **77° to 102°, "automation-dominant"** (pure automation is at 90°):
   output and rents rise, wages fall on impact, planner resources F still
   rise. Workers are compensated: consumption and transfers rise.
3. **102° to 116°, "growth with a poorer worker-planner"**: output still
   rises but F falls because the wage loss now exceeds the extra rents.
4. **116° to 150°, "domestic contraction, claim gain"**: productivity is
   falling while automation rises; output, wages and F all fall, yet
   capitalized future resources and the claim payoff rise, so consumption
   and transfers rise.
5. **152°, the blind spot**: the claim-neutral direction, which at this
   calibration is also the rental/tax-base-neutral direction. Nothing
   traded pays, capital and taxes never move, and the whole loss (output,
   wages, F, planner-resource wealth) is unspanned.
6. **152° to 176°, "cushioned contraction"**: everything falls, but wages
   fall faster than consumption so transfers rise.
7. **176° to 180°**: even transfers fall.

The southern half mirrors the northern half with every sign flipped (exact
to 7e-15). Exact boundary angles for every object are in
`zero_impact_thresholds.csv`; the arcs are in `impact_sign_regions.csv`.

## Where output, wages and fiscal resources disagree

- Output up, wages down: 77.4°–115.95° (a 38.6° arc containing pure
  automation), mirrored at 257.4°–295.95° for output down, wages up.
- Output up, planner resources F down: only 101.55°–115.95° (14.4°).
- Capital-tax receipts up while F down: 101.55°–151.8°.
- The government's primary cash flow τB − T falls on impact for every
  direction in 5°–180° except at the exact ends: transfers respond more
  than tax receipts to any good-technology surprise.

## Insurance by the inherited claim

- The claim pays in proportion to the technology mixture along 61.8°; it
  pays nothing at 151.8°/331.8°. Planner-resource wealth loads along 59.6°.
  The 2.2° gap is the unspanned part of fiscal risk (3.9% of the loading),
  and the claim-neutral direction is exactly the unspanned direction: a
  fiscal loss of 16.3e-4 there with zero payoff.
- Well insured: anything near ±62° (for pure automation, 98% of the
  planner-wealth change is marketed). Poorly insured: near ±152°.
- Payoff positive while F falls (101.55°–151.8°) and payoff negative while
  output rises (295.95°–331.8°) both happen: the claim prices technology
  states, not cash flows.
- With the optimal inherited position s* = 0.4465, the claim adds 3.4%–3.9%
  to the impact consumption response of every named direction and 3%–4% to
  transfers, except for the productivity direction where it is 31% of the
  (small) transfer response. It never flips the sign of consumption or
  transfers on the five-degree grid; it shifts the transfer-neutral
  direction by 0.9°. Over the whole path the payoff is paid out in full:
  discounted cumulative transfers minus discounted cumulative tax receipts
  equals the payoff on all 438 paths (error 1.5e-16).

## Neutral directions that unravel

All named "neutral" directions are neutral on impact only. At the
baseline (equal persistence 0.81 for both states) the claim/rental-neutral
line is an exact invariant of the capital-tax loop: it stays neutral for
capital, tax, rental base and claim loading for the full 40 years. The
output-neutral direction loses output neutrality through capital
accumulation (output positive within a quarter, 10% of its cancelling
components at 5.5 years, 12% at 10 years); the wage-neutral direction
loses it at 1.25 years; the resource-neutral direction at 7.5 years.

With unequal persistence the mixture rotates and everything unravels
faster (same initial displacement; automation persistence 0.50 / 0.81 /
0.95): the claim/rental line breaks at 0.75 / never / 1.75 years, the
output-neutral direction at 0.75 / 5.5 / 1.0 years, the resource-neutral
direction at 0.75 / 7.5 / 5.5 years. The state mixture rotates by up to
56° within five years at persistence 0.50 (toward the productivity axis)
and by up to 20° at 0.95 (toward the automation axis). At persistence
0.95 every automation-containing direction eventually raises output,
wages and F; at 0.50 the responses die within a few years and only the
impact classification survives.

## How capital and taxes change the incidence over time

Capital peaks at 10 years in essentially every direction (the loop's
damped oscillation, roots −0.076 ± 0.078i, is direction-independent; only
amplitudes differ: 4.1e-4 for productivity, 7.6e-4 for automation). The
tax rate falls on impact wherever the rental base rises, overshoots once
and reverses after 10–15 years. Impact wage losses turn into wage gains by
10 years for 77.4°–120° and by 20 years up to 140°, but the discounted
cumulative wage response stays negative beyond 101.5°, so recovery is
not a cumulative gain for automation-heavy mixtures with falling
productivity. Sign changes reported near 40 years for 45°–77° are 1e-8
tails, not reversals.

## Answers to the ten atlas questions

1. Output up, worker income down: 77.4°–115.95° (and 257.4°–295.95° for the reverse).
2. Output up, current planner resources down: 101.55°–115.95°.
3. Offsetting automation's impact wage loss needs dz = 0.477·dα, i.e. 0.224 standardized productivity innovations per standardized automation innovation (for the project's dα = +0.01 check: dz = +0.0048).
4. The claim pays over 331.8°–151.8°; the unspanned losses sit along 151.8° (productivity down, automation up) and are 3.9% of the fiscal loading, maximal exactly where the claim pays nothing.
5. The claim changes worker consumption by 3%–4% of its response and transfers by 3%–4% except in productivity-led directions (31%); it matters most near the transfer-neutral directions (~176°, ~356°), where it sets the small transfer response.
6. Yes to both: payoff > 0 with F < 0 over 101.55°–151.8°; payoff < 0 with output > 0 over 295.95°–331.8°.
7. Only the claim/rental-neutral line stays neutral, and only under equal persistence; every other neutral combination unravels within 1.25–7.5 years at baseline and within 0.5–2.5 years under unequal persistence.
8. Capital accumulation converts impact wage losses into gains for 77.4°–140° within 20 years but not in present value beyond 101.5°; tax cuts on impact reverse into tax rises after 10–15 years everywhere the rental base rose.
9. Consumption rises with falling wages over 77.4°–149.65°; for pure automation the transfer is financed 79% by the capitalized value of workers' own future wages, 8% by capitalized future capital-tax receipts, 3% by the inherited claim payoff and 10% is wage compensation. Timing runs through a persistent fall in public net worth.
10. Nothing in the atlas violates the maintained transfer, portfolio, tax, tax-speed or specialisation conditions (70,518 rows, 0 failures). The closest genuine boundary is the transfer floor under the one-year OU normalization in the negative-automation directions (transfer level 0.0086 against 0.0202 at the anchor): a displacement 1.7 times larger would drive transfers to zero. Every statement about the negative safe position and the persistent decline of N is an unconstrained result; fiscal-capacity feasibility is unverified.

## What is robust inside the maintained branch, and what surfaced

- Robust (exact for the linear system, verified to 1e-14): the sign map,
  the zero angles, the financing decomposition, the timing distinction
  between a realized innovation and an inherited state, and the
  present-value cash-flow identity.
- Surfaced: **no model problem and no implementation error**. Two
  **interpretation qualifications**: (i) impact neutrality is not path
  neutrality, and the one exception (the claim/rental line) is a
  calibration alignment (k_I = log K̄/L and κ_z = κ_x) that breaks under
  unequal persistence; (ii) the unconstrained negative safe position and
  the transfer-financing paths are not evidence of borrowing capacity. One
  **feasibility proximity**: the transfer floor under one-year OU negative
  automation displacements. One **numerical limitation**: the local
  linear map cannot say how region boundaries move for large shocks.
- The economic correction of 2026-09-01 (wages are worker income, not
  revenue) is applied throughout this bundle; the audit of shared code and
  earlier findings is in `ECONOMIC_CORRECTION_HANDOFF.md`. No numerical
  result changed; naming and claimed status did. No existing check used
  planner-resource wealth as collateral, but several `feasible`-named flags
  measure artificial caps and need relabelling in shared modules.

## Runtime and effort

| Phase | Time |
|---|---|
| Machine: atlas computation (4 models, 438 paths, 70,518 rows, all checks) | 10.4 s |
| Machine: figures | 1.2 s |
| Machine: full test suite (109 tests), each run | 7 s |
| Implementation and debugging (session 22:20–23:07 UTC, incl. peer coordination and a disk-full incident) | ~50 min |
| Verification runs (smoke, two full runs, labelling fix) | ~10 min |
| Analysis and reporting | ~45 min |

The two-hour numerical ceiling was never approached; the night was not
used. Two labelling defects were fixed between the first and the
definitive run (persistence-0.81 rows carried the baseline label; noise
crossings on an identically-zero path); no numerical value changed.

## Environment incident

At 22:45 UTC the machine's data volume hit 100% (203 MiB free) and the
test suite failed with "No space left on device". The largest recent
writers were two >500 MB CSVs in session fe's risky-asset-search worktree.
I pruned the uv package cache (4.0 GiB, regenerable), warned the peers,
and fe compressed its files; the definitive run started with 4.0 GiB free.
The disk is still at 98%.

## Reproducing and where things are

```bash
uv run pytest
uv run python -m tai_public_finance.cs001_lq_anchor.shock_atlas_cli \
  --config configs/cs001/lq_farhi_smoke.json \
  --output-dir outputs/cs001-joint-shock-atlas-<UTC timestamp> --mode full
```

Tables: `atlas_raw.csv` (13 key horizons; hash-referenced, not committed),
`atlas_raw_quarterly.csv.gz` (161 horizons), `path_features.csv`,
`named_directions.csv`, `zero_impact_thresholds.csv`,
`impact_sign_regions.csv`, `persistence_unravelling.csv`,
`persistence_named_paths.csv`, `failed_rows.csv` (empty),
`numerical_diagnostics.json`, `models.json`, `runtime.json`,
`manifest.json`, `state.json`, `events.log`, `figure_data/`, `figures/`.
Review packets: `NUMERICAL_REVIEW_PACKET.md`, `ECONOMIC_REVIEW_PACKET.md`,
`ECONOMIC_CORRECTION_HANDOFF.md`. Run record: `runs/RUN-20260901T230703Z-CS001-ATLAS-ee48affe-01.yaml`.

## Suggested next decisions

1. Merge `cs001/joint-shock-atlas` (new files only; touches no shared module).
2. Route the fiscal-capacity object C_G(S) to an economic specification before any borrowing-capacity claim.
3. Relabel the artificial-cap `feasible` flags in `portfolio.py`/`diagnostics.py` in a dedicated small PR once the other CS001 branches land.
4. Carry the impact-versus-path neutrality distinction and the invariant-line alignment into the CS002/CS004 comparisons.
