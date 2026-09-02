# Economic review packet: CS001 joint productivity-automation shock atlas

Read-only packet for the economic reviewer. Everything below is exploratory
local-LQ evidence on the illustrative Farhi-based vector (branch
`cs001/joint-shock-atlas`, commit `ee48aff`, run directory
`outputs/cs001-joint-shock-atlas-20260901T230702Z`). It is a directional
mechanism map: not a probability statement about shocks, not a global
constrained solution, not a welfare ranking of realized shocks.

Units: all responses are first-order deviations in the model's level units
(K̄ = L = 1; Ȳ = 0.354, W̄ = 0.234, τ̄B̄ = T̄ = 0.0202, J̄ = X̄ = 12.57,
c̄ = 0.254). "1e-4" below means 0.0001 in those units. The standardized
innovation is one Brownian standard deviation over the 1/252-year window,
so a pure productivity innovation moves z by 1.64e-3 and a pure automation
innovation moves the automation share by 7.67e-4 (x by 1.02e-2).

## 0. Names and status (economic correction of 2026-09-01)

- J is **planner-resource wealth** (worker fiscal-endowment wealth). It
  capitalizes future worker wages plus capital-tax resources. At the anchor
  J = W̄/ρ + τ̄B̄/ρ = 11.569 + 1.000 = 12.569. The first-order loading of J
  on the two technology states is 92% capitalized wages: j_W = (11.91, 3.22)
  against j_B = (1.09, 0.32) on (z, x). J is not government borrowing
  capacity or collateral.
- X = N + J is **worker comprehensive resources**; c = ρX; T = c − W.
- s* = 0.4465 is the **unconstrained leading small-risk portfolio**;
  N̄ − s* = −0.4465 is an unconstrained negative safe position. Its
  feasibility against a genuine government borrowing-capacity constraint is
  **unverified**; the ±20 portfolio/debt caps are numerical scaffolding and
  establish nothing about capacity. Classification of every portfolio
  statement below: *unconstrained local desired portfolio; genuine
  fiscal-capacity feasibility unverified.*
- F = τB + W is a worker-planner resource object containing wages. The
  government's primary cash flow is τB − T − Ψ; `F − c = τB − T` holds on
  every reported row (max error 2.6e-18).

## 1. The direction circle and its main regions (impact, s_- = s*)

Angle θ: 0° = +productivity, 90° = +automation, 180° = −productivity,
270° = −automation; θ and θ+180° are opposite realizations, and every
first-order response is exactly odd under that map (checked to 7e-15).
Named directions at this calibration (positive automation sign): worker-
income-neutral 77.40° (dz/dα = +0.477), pure automation 90°, primary-
resource-neutral 101.55° (dz/dα = −0.436), output-neutral 115.95°
(dz/dα = −1.038), claim-payoff-neutral 151.80° (dz/dα = −3.979). The
claim-neutral and rental/tax-base-neutral directions coincide **exactly**
because k_I − log(K̄/L) = 0 (so h_I = η_Y + 1/α); this is a calibration
alignment, not a general identity.

| Arc (θ) | Output Y | Wages W | Planner resources F | Capital-tax receipts τB | Claim payoff | Worker consumption c | Transfer T | Reading |
|---|---|---|---|---|---|---|---|---|
| 0°–77.4° | + | + | + | + | + | + | + | broad expansion: productivity-led, automation adds rents; transfers rise more than tax receipts (τB − T < 0 for θ ≥ 5°) |
| 77.4°–101.5° | + | − | + | + | + | + | + | automation-dominant: output and rents up, wages down, planner resources still up (contains pure automation) |
| 101.5°–115.95° | + | − | − | + | + | + | + | output up but planner resources down: the wage loss now outweighs the extra rents |
| 115.95°–149.6° | − | − | − | + | + | + | + | domestic contraction with a claim gain: productivity down, automation up; capitalized wealth J still rises because automation raises future rents |
| 149.6°–151.8° | − | − | − | + | + | − | + | thin sliver where J and c turn negative while the claim still pays |
| 151.8° | − | − | − | 0 | 0 | − | + | claim-neutral = rental-neutral = unspanned direction: no payoff, no rental-base change, no tax-speed response |
| 151.8°–175.8° | − | − | − | − | − | − | + | everything falls, but wages fall by more than consumption so transfers rise |
| 175.8°–180° | − | − | − | − | − | − | − | productivity-led contraction: transfers fall too |

The other half (180°–360°) is the mirror image with all signs reversed.
Exact zero angles of every impact object (from the linear functionals, not
from grid interpolation) are in `zero_impact_thresholds.csv`; the arcs
above come from `impact_sign_regions.csv`.

Where output, wages and planner resources disagree in sign:

- Y > 0 with W < 0: θ ∈ (77.4°, 115.95°) and its mirror (Y < 0, W > 0) at
  (257.4°, 295.95°). Pure automation sits in the middle of this arc.
- Y > 0 with F < 0: only the 14.4° arc (101.55°, 115.95°), and its mirror.
- τB and F disagree over (101.55°, 151.8°): rents and tax receipts rise while
  planner resources fall.

## 2. Financing of transfers (the separate accounts)

At impact, with s_- = s*, `ΔT = ρΔJ_W + ρΔJ_B + ρ·payoff − ΔW` on every
row. The decomposition for the named directions (1e-4 units):

| Direction | ΔT | ρΔJ_W (capitalized wages) | ρΔJ_B (capitalized tax) | ρ·payoff | −ΔW |
|---|---|---|---|---|---|
| pure productivity | 0.62 | 3.94 | 0.36 | 0.15 | −3.82 |
| worker-income-neutral | 8.38 | 7.35 | 0.73 | 0.30 | 0.00 |
| pure automation | 8.45 | 6.65 | 0.67 | 0.27 | 0.86 |
| primary-resource-neutral | 8.15 | 5.72 | 0.59 | 0.24 | 1.60 |
| output-neutral | 7.32 | 4.25 | 0.45 | 0.18 | 2.44 |
| claim-neutral | 3.44 | −0.33 | 0.00 | 0.00 | 3.77 |

Three readings:

1. Where consumption rises although wages fall, θ ∈ (77.4°, 149.65°), the
   transfer is financed overwhelmingly by the capitalized value of workers'
   own future wages (79% of ΔT for pure automation), then by wage
   compensation (10%), capitalized future capital-tax receipts (8%) and the
   inherited claim payoff (3%). The government smooths worker consumption
   against planner-resource wealth; it does not tax wages.
2. In the claim-neutral direction the transfer rise (3.44) is almost
   entirely wage compensation (3.77) net of a small fall in capitalized
   resources: consumption falls slightly (−0.33) while wages fall a lot.
3. Timing: the present value of the government's primary cash-flow response
   satisfies `∫e^{-ρt}(τB − T)dt = −payoff` on every one of the 438 paths
   (max error 1.5e-16, computed from `path_features.csv`). Without an
   inherited claim, cumulative transfers equal cumulative tax receipts in
   present value; with it, the payoff is entirely paid out. The
   intertemporal reallocation runs through public net worth, which falls
   persistently in the transfer-heavy directions (output-neutral direction:
   N = −25e-4 at 15 years for a one-sd innovation). Whether that path is
   feasible against a genuine borrowing constraint is unverified.

## 3. Marketed versus unspanned disturbances

- The inherited claim pays on `λ̂·u`, with λ̂ = (0.0260, 0.0484) pointing at
  61.8°. The payoff is positive over (331.8°, 151.8°) and zero exactly at
  the claim-neutral direction 151.8°/331.8°.
- Planner-resource wealth loads on ζ_J = (0.338, 0.575), pointing at 59.6°.
  The 2.2° misalignment is the unspanned component ζ_J^⊥ = (0.0229, −0.0123):
  3.9% of the fiscal loading. The claim-neutral direction is, by
  construction in two dimensions, the unspanned direction: along it J falls
  by 16.3e-4 with zero payoff (a "fiscal loss invisible to the traded
  claim"). Along its opposite (331.8°) J rises with no payoff.
- Well-insured directions: anything near ±61.8° (payoff and J move together;
  pure automation: 98% of the J change is marketed, unspanned −7.7e-4).
  Poorly insured: near ±151.8°. Productivity innovations carry the larger
  unspanned share (7%: unspanned +14.4e-4 against marketed 198e-4).
- Both `payoff > 0 with F < 0` (θ ∈ (101.55°, 151.8°)) and
  `payoff < 0 with Y > 0` (θ ∈ (295.95°, 331.8°)) occur: the claim prices the
  technology states, not output or fiscal cash flows.
- How much the claim changes worker consumption and transfers: with
  s_- = s* the extra impact consumption is ρ·payoff, which is 3.4%–3.9% of
  the no-claim consumption response for every named direction (grid range
  −16% to +6%, the extremes sitting where the no-claim response itself
  vanishes near 150°). For transfers the claim is 31% of the productivity
  direction's transfer response (because wages absorb most of a
  productivity gain) and 2.6%–3.7% elsewhere. On the five-degree grid the
  claim never flips the sign of Δc or ΔT; it moves the transfer-neutral
  direction from 176.7° to 175.8° and the consumption-neutral direction by
  0.08°. The s_- = 0 comparison isolates exactly this payoff channel: its
  physical paths coincide with the matched deterministic displacement to
  2e-19, and the consumption gap is the payoff times ρ at every horizon.

## 4. Impact neutrality versus path neutrality

Every named "neutral" direction is neutral **on impact only**, at inherited
capital and tax. What happens afterwards depends on which state
combination the capital-tax loop responds to.

- **Claim/rental-neutral (151.8°)** is a structural invariant line at this
  calibration: the capital-growth loading and the tax-speed feedback on
  (z, x) are both proportional to the rental-rate gradient (alignment
  errors 0 and 9e-16), so with equal OU rates (κ_z = κ_x) capital and the
  tax rate never move (|k|, |τ| < 6e-19 over 40 years), the rental base,
  the tax receipts and the claim-loading functional stay at zero for the
  whole path, and output, wages and F decay together at rate κ. Two
  alignments produce this: k_I = log(K̄/L) and κ_z = κ_x. It is the sharpest
  form of R15's "domestic loss invisible to the claim".
- **Output-neutral (115.95°)** unravels through capital: the rental rate is
  positive on impact, capital accumulates (peak 5.1e-4 at 10 years), output
  turns positive within a quarter and reaches 0.61e-4 at 10 years (12% of
  the size of its cancelling components; the 10% threshold is crossed at
  5.5 years). Wages recover at 8.6 years and F turns positive at 10 years.
- **Worker-income-neutral (77.4°)** loses wage neutrality at 1.25 years as
  capital deepening raises wages (+0.43e-4 at 20 years).
- **Primary-resource-neutral (101.55°)** loses F-neutrality at 7.5 years;
  wages recover at 5.9 years.

Under unequal persistence the mixture itself rotates and every neutral
condition unravels much faster (fixed initial (dz, dα), automation
persistence 0.50 / 0.81 / 0.95): the standardized state angle rotates by
−55° / 0° / +3° at 5 years for the worker-income-neutral direction and by
+15° / 0° / −20° for the claim-neutral direction; the claim-neutral line
loses its rental and claim neutrality at 0.75 / never / 1.75 years, the
output-neutral direction at 0.75 / 5.5 / 1.0 years, the primary-resource-
neutral direction at 0.75 / 7.5 / 5.5 years. With persistence 0.95 every
automation-containing direction eventually raises output, wages and F
(capital keeps accumulating for ~19 years); with 0.50 the domestic
responses die within a few years and the impact sign classification is the
only one that lasts. These are propagation differences, not size
differences: the initial displacement is identical across the three cases.

## 5. Capital and taxes over time

- The tax rate falls on impact for every direction with a positive rental
  base (θ ∈ (331.8°, 151.8°)), overshoots once and reverses sign after 10–15
  years; the capital-tax loop's damped oscillation (roots −0.076 ± 0.078i)
  is common to all directions, so the peak-capital time is 10 years
  essentially everywhere and only the amplitude varies with θ.
- Directions with an impact wage loss that becomes a wage gain: by 10 years
  for θ ∈ (77.4°, 120°], by 20 years for θ up to 140°; the discounted
  cumulative wage response is nevertheless negative beyond 101.5°, so the
  recovery does not amount to a cumulative gain for automation-heavy
  mixtures with falling productivity.
- The late sign changes reported near 40 years for θ ∈ [45°, 77°] are tails
  of order 1e-8 (three to four orders below the impact responses) and
  should not be read as reversals.

## 6. What depends on the local approximation and the maintained branch

- All of the above is first order in the displacement and holds exactly
  for the solved linear system; superposition, sign symmetry and scaling
  were verified to 1e-14. Region boundaries are exact zero angles of linear
  functionals; the atlas cannot say how they move for large shocks.
- No reported row leaves the maintained branch: specialisation margins
  stay above 0.58, no scaffolding cap is touched (minimum scaffolding slack
  0.44). The closest genuine boundary is the transfer floor T ≥ 0: a
  one-year OU automation displacement in the negative-automation directions
  (θ ≈ 265°) takes the transfer level from 0.0202 to 0.0086, so a displacement
  about 1.7 times larger would drive transfers to zero and the linear
  extrapolation would fail there first. One-sd Brownian innovations use only
  4% of the floor.
- The unconstrained negative safe position (N − s* = −0.4465) and the
  persistent decline in N along transfer-heavy paths are properties of the
  unconstrained local solution. Nothing here verifies government borrowing
  capacity; the pending fiscal-capacity object C_G(S) (attainable primary
  surpluses with T = 0, wages excluded) is a separate specification task.
- The claim/rental coincidence and the invariant line are calibration
  artifacts of the aligned normalization; the persistence variants show how
  they break.
- The leading portfolio and its welfare coefficients are ex-ante local
  objects; the atlas reports realized first-order paths and makes no
  shock-specific welfare ranking.
