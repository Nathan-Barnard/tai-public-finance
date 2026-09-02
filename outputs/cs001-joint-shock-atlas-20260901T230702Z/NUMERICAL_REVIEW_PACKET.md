# Numerical review packet: CS001 joint productivity-automation shock atlas

- Run started (UTC): `2026-09-01T23:07:03Z`; finished: `2026-09-01T23:07:22Z`
- Commit `ee48affef15034e74f9e3af3d2533ce51f57f1fe` on branch `cs001/joint-shock-atlas` (dirty at start: False)
- Complete-input fingerprint `e521c69dc3fd1a3e92d1ecf603f7449a7c96becd5c0c636abde72ef0024a32a4`; atlas fingerprint `{'commit': 'ee48affef15034e74f9e3af3d2533ce51f57f1fe', 'complete_input_sha256': 'e521c69dc3fd1a3e92d1ecf603f7449a7c96becd5c0c636abde72ef0024a32a4', 'atlas_settings_sha256': 'cb9da78b9ae4d58d103e2f6c80a3ccd08a68446afd7c172ddeece344c775d96f', 'shock_atlas_source_sha256': 'eaaf6b8174e98275a969b0c687a351a13fc3aa86c495b8e488030e011b51eec4'}`
- Outcome of independent checks: **pass**; failures: `[]`
- Machine runtime (seconds by chunk): `{"atlas_brownian_innovation": 3.509, "atlas_finite_window_ou_displacement": 1.772, "atlas_fixed_share_check": 0.215, "atlas_matched_state_displacement": 1.73, "independent_checks": 0.651, "models": 0.218, "persistence_unravelling": 2.333}`; total `10.43` s

Raw-output locators: every row of `atlas_raw_quarterly.csv` (all 161 quarterly horizons) and `atlas_raw.csv` (13 key horizons) is keyed by `model`, `family`, `regime`, `direction_key` (= `theta_ddd.ddd`), `horizon_years`; one row per path in `path_features.csv`; per-chunk raw parts under `parts/`. Locators below use `model::family::regime::direction_key@horizon`.

## Matrix-equation residuals per solved model (baseline pipeline diagnostics, recomputed from primitives)

| model | acceptance | Riccati (full) | Riccati (real block) | Sylvester | disc. Lyapunov | closed-form vs Schur | Hamiltonian axis distance | capital-tax Hurwitz (margin) | full loop Hurwitz |
|---|---|---|---|---|---|---|---|---|---|
| automation_persistence_0.50 | pass | 1.100e-16 | 8.871e-17 | 5.887e-17 | 1.632e-18 | 7.691e-16 | 8.613e-02 | yes (7.603e-02) | yes |
| automation_persistence_0.81 | pass | 1.159e-16 | 8.871e-17 | 6.322e-17 | 1.608e-17 | 7.691e-16 | 8.613e-02 | yes (7.603e-02) | yes |
| automation_persistence_0.95 | pass | 1.216e-16 | 8.871e-17 | 6.477e-17 | 2.367e-17 | 7.691e-16 | 8.613e-02 | yes (7.603e-02) | yes |
| baseline | pass | 1.159e-16 | 8.871e-17 | 6.322e-17 | 1.608e-17 | 7.691e-16 | 8.613e-02 | yes (7.603e-02) | yes |

Finite-difference primitive checks (gradient / Hessian relative errors) and feedback-construction errors:

- automation_persistence_0.50: output: 4.299e-10 / 1.491e-08; rental_rate: 1.840e-10 / 7.013e-09; safe_rate: 3.347e-10 / 4.912e-10; wage_income: 3.126e-10 / 1.068e-08; A_rc/A_c/F construction 0.000e+00/0.000e+00/0.000e+00; resolvent identity 2.711e-16
- automation_persistence_0.81: output: 4.299e-10 / 1.491e-08; rental_rate: 1.840e-10 / 7.013e-09; safe_rate: 1.080e-10 / 1.096e-10; wage_income: 3.126e-10 / 1.068e-08; A_rc/A_c/F construction 0.000e+00/0.000e+00/0.000e+00; resolvent identity 2.709e-16
- automation_persistence_0.95: output: 4.299e-10 / 1.491e-08; rental_rate: 1.840e-10 / 7.013e-09; safe_rate: 2.648e-11 / 4.754e-12; wage_income: 3.126e-10 / 1.068e-08; A_rc/A_c/F construction 0.000e+00/0.000e+00/0.000e+00; resolvent identity 3.040e-16
- baseline: output: 4.299e-10 / 1.491e-08; rental_rate: 1.840e-10 / 7.013e-09; safe_rate: 1.080e-10 / 1.096e-10; wage_income: 3.126e-10 / 1.068e-08; A_rc/A_c/F construction 0.000e+00/0.000e+00/0.000e+00; resolvent identity 2.709e-16

## Worst-case path checks (with locators)

- Matrix exponential vs direct DOP853 ODE integration: max relative error 1.783e-12 over 438 paths at `baseline::matched_deterministic_state_displacement::no_inherited_payoff::theta_151.805`
- Superposition (automation_persistence_0.50): component split max rel. error 4.141e-16 at `automation_persistence_0.50::fixed_initial_displacement_across_persistence::no_inherited_payoff::theta_151.805`; cos/sin basis max rel. error 1.369e-15 over 36 paths at `automation_persistence_0.50::brownian_innovation_short_window::zero_inherited_position::theta_163.533`
- Superposition (automation_persistence_0.81): component split max rel. error 7.633e-16 at `automation_persistence_0.81::brownian_innovation_short_window::optimal_inherited_position::theta_331.805`; cos/sin basis max rel. error 2.245e-15 over 36 paths at `automation_persistence_0.81::brownian_innovation_short_window::optimal_inherited_position::theta_331.805`
- Superposition (automation_persistence_0.95): component split max rel. error 9.348e-16 at `automation_persistence_0.95::brownian_innovation_short_window::optimal_inherited_position::theta_312.624`; cos/sin basis max rel. error 1.519e-15 over 36 paths at `automation_persistence_0.95::brownian_innovation_short_window::optimal_inherited_position::theta_132.624`
- Superposition (baseline): component split max rel. error 8.974e-16 at `baseline::finite_window_ou_state_displacement::no_inherited_payoff::theta_155.000`; cos/sin basis max rel. error 2.460e-15 over 320 paths at `baseline::brownian_innovation_short_window::optimal_inherited_position::theta_155.000`
- Sign symmetry theta vs theta+pi (automation_persistence_0.50): max rel. error 2.689e-15 over 36 pairs at `automation_persistence_0.50::brownian_innovation_short_window::optimal_inherited_position::theta_163.533`; unpaired: []
- Sign symmetry theta vs theta+pi (automation_persistence_0.81): max rel. error 1.481e-15 over 36 pairs at `automation_persistence_0.81::brownian_innovation_short_window::optimal_inherited_position::theta_151.805`; unpaired: []
- Sign symmetry theta vs theta+pi (automation_persistence_0.95): max rel. error 5.024e-15 over 36 pairs at `automation_persistence_0.95::brownian_innovation_short_window::optimal_inherited_position::theta_132.624`; unpaired: []
- Sign symmetry theta vs theta+pi (baseline): max rel. error 6.815e-15 over 330 pairs at `baseline::finite_window_ou_state_displacement::no_inherited_payoff::theta_150.000`; unpaired: []
- Scaling (halve/double selected displacements): max rel. error 0.000e+00 over 8 paths
- Brownian vs matched-state timing distinction over 160 paths: physical-state max abs difference 2.168e-19; (X gap - claim payoff) max abs 1.063e-17; X-gap constancy over horizons 1.409e-17; zero-position vs matched max abs 0.000e+00
- Capital/tax jump at impact: max abs displacement 0.000e+00 over 438 initial conditions
- Accounting identities (max abs error over every row): F_minus_c_equals_tauB_minus_T 2.602e-18; dJ_equals_wage_component_plus_capital_tax_component 2.220e-16; dT_equals_dc_minus_dW 1.735e-18; dX_equals_dN_plus_dJ 6.939e-18; dX_equals_domestic_contribution_plus_payoff_at_impact 0.000e+00; dc_equals_rho_dX 0.000e+00; primary_cash_flow_equals_tauB_minus_T_minus_Psi 1.952e-18; worst locator `baseline::finite_window_ou_state_displacement::no_inherited_payoff::theta_345.000@2.25:primary_cash_flow_equals_tauB_minus_T_minus_Psi`
- Coordinate conversion (automation_persistence_0.50): theta->(dz,dalpha)->theta round trip 2.842e-14 deg; named-direction vs analytic zero-angle max error 5.684e-14 deg ({"claim_payoff_neutral_automation_negative": 0.0, "claim_payoff_neutral_automation_positive": 0.0, "output_neutral_automation_negative": 0.0, "output_neutral_automation_positive": 0.0, "primary_resource_neutral_automation_negative": 0.0, "primary_resource_neutral_automation_positive": 0.0, "rental_tax_base_neutral_automation_negative": 0.0, "rental_tax_base_neutral_automation_positive": 0.0, "worker_income_neutral_automation_negative": 0.0, "worker_income_neutral_automation_positive": 0.0}); claim-neutral orthogonality to lambda_hat 1.515e-16
- Coordinate conversion (automation_persistence_0.81): theta->(dz,dalpha)->theta round trip 5.684e-14 deg; named-direction vs analytic zero-angle max error 0.000e+00 deg ({"claim_payoff_neutral_automation_negative": 0.0, "claim_payoff_neutral_automation_positive": 0.0, "output_neutral_automation_negative": 0.0, "output_neutral_automation_positive": 0.0, "primary_resource_neutral_automation_negative": 0.0, "primary_resource_neutral_automation_positive": 0.0, "rental_tax_base_neutral_automation_negative": 0.0, "rental_tax_base_neutral_automation_positive": 0.0, "worker_income_neutral_automation_negative": 0.0, "worker_income_neutral_automation_positive": 0.0}); claim-neutral orthogonality to lambda_hat 0.000e+00
- Coordinate conversion (automation_persistence_0.95): theta->(dz,dalpha)->theta round trip 2.842e-14 deg; named-direction vs analytic zero-angle max error 0.000e+00 deg ({"claim_payoff_neutral_automation_negative": 0.0, "claim_payoff_neutral_automation_positive": 0.0, "output_neutral_automation_negative": 0.0, "output_neutral_automation_positive": 0.0, "primary_resource_neutral_automation_negative": 0.0, "primary_resource_neutral_automation_positive": 0.0, "rental_tax_base_neutral_automation_negative": 0.0, "rental_tax_base_neutral_automation_positive": 0.0, "worker_income_neutral_automation_negative": 0.0, "worker_income_neutral_automation_positive": 0.0}); claim-neutral orthogonality to lambda_hat 0.000e+00
- Coordinate conversion (baseline): theta->(dz,dalpha)->theta round trip 5.684e-14 deg; named-direction vs analytic zero-angle max error 0.000e+00 deg ({"claim_payoff_neutral_automation_negative": 0.0, "claim_payoff_neutral_automation_positive": 0.0, "output_neutral_automation_negative": 0.0, "output_neutral_automation_positive": 0.0, "primary_resource_neutral_automation_negative": 0.0, "primary_resource_neutral_automation_positive": 0.0, "rental_tax_base_neutral_automation_negative": 0.0, "rental_tax_base_neutral_automation_positive": 0.0, "worker_income_neutral_automation_negative": 0.0, "worker_income_neutral_automation_positive": 0.0}); claim-neutral orthogonality to lambda_hat 0.000e+00
- Analytic neutral directions: worst impact cancellation index 1.439e-15 (tolerance 1.000e-11) over 130 labelled paths
- dalpha=+0.01 reproduction of the baseline pipeline's constructed_* experiments: max abs difference 5.551e-17 over 644 rows at `baseline::fixed_share_displacement_check::no_inherited_payoff::theta_077.400@0.75:public_net_worth_deviation vs constructed_worker_income_neutral:public_net_worth_deviation`; missing rows []
- Row builder vs baseline irfs._row on shared fields: max abs difference 5.551e-17 over 2093 rows at `baseline::brownian_innovation_short_window::optimal_inherited_position::theta_115.952@0.25:safe_position_level`
- First-order budget identity: max abs residual 3.643e-17 over 70518 rows at `baseline::finite_window_ou_state_displacement::no_inherited_payoff::theta_240.000@0.5`
- Discounted cumulative responses: resolvent probe residual max 1.379e-16 at `baseline::finite_window_ou_state_displacement::no_inherited_payoff::theta_235.000`; resolvent vs matrix-exponential integral to T=1500 years max rel. error 1.785e-13 at `baseline::brownian_innovation_short_window::zero_inherited_position::theta_115.952`

## Closest boundaries, failed and infeasible rows, missing rows

- Rows: 70518; rows failing genuine economic conditions: 0; rows hitting numerical scaffolding: 0; nonfinite rows: 0
- Minimum genuine economic slack (specialisation margins, transfer floor c-W, tau<1): 8.565e-03 at `baseline::finite_window_ou_state_displacement::no_inherited_payoff::theta_265.000`
- Minimum numerical-scaffolding slack (portfolio caps, tax box, tax-speed cap): 4.385e-01 at `baseline::finite_window_ou_state_displacement::no_inherited_payoff::theta_240.000`
- Classification: genuine economic conditions (specialisation branch, transfer floor, positive consumption and worker comprehensive resources, tau<1) are reported separately from numerical-scaffolding slack; neither establishes government borrowing capacity, which this calculation does not verify
- Failed/infeasible rows are retained in `failed_rows.csv` (empty header-only file means none).
- Missing rows: every path carries the full quarterly horizon grid by construction; `sign_symmetry` `unpaired` lists any direction without its opposite (must be empty).

## Named directions, coincidences and invariant-line diagnostics

- automation_persistence_0.50: named angles pure_productivity_positive=0.000000, worker_income_neutral_automation_positive=67.932148, pure_automation_positive=90.000000, primary_resource_neutral_automation_positive=110.333251, output_neutral_automation_positive=131.435327, claim_payoff_neutral_automation_positive=163.533390; coincidences [{"dz_per_dalpha": -3.9794706879985, "named_labels": ["claim_payoff_neutral_automation_positive", "rental_tax_base_neutral_automation_positive"], "theta_deg": 163.53338971460084}, {"dz_per_dalpha": -3.9794706879985, "named_labels": ["claim_payoff_neutral_automation_negative", "rental_tax_base_neutral_automation_negative"], "theta_deg": 343.53338971460084}]; h_I-(eta+1/alpha)=0.000e+00; tax-speed feedback alignment with rental gradient rel. error 5.876e-01; capital-growth alignment 0.000e+00; kappa_z-kappa_x=-4.824e-01
- automation_persistence_0.81: named angles pure_productivity_positive=0.000000, worker_income_neutral_automation_positive=77.399980, pure_automation_positive=90.000000, primary_resource_neutral_automation_positive=101.547781, output_neutral_automation_positive=115.952206, claim_payoff_neutral_automation_positive=151.804922; coincidences [{"dz_per_dalpha": -3.9794706879985, "named_labels": ["claim_payoff_neutral_automation_positive", "rental_tax_base_neutral_automation_positive"], "theta_deg": 151.80492206398452}, {"dz_per_dalpha": -3.9794706879985, "named_labels": ["claim_payoff_neutral_automation_negative", "rental_tax_base_neutral_automation_negative"], "theta_deg": 331.8049220639845}]; h_I-(eta+1/alpha)=0.000e+00; tax-speed feedback alignment with rental gradient rel. error 9.300e-16; capital-growth alignment 0.000e+00; kappa_z-kappa_x=0.000e+00
- automation_persistence_0.95: named angles pure_productivity_positive=0.000000, worker_income_neutral_automation_positive=83.706717, pure_automation_positive=90.000000, primary_resource_neutral_automation_positive=95.756349, output_neutral_automation_positive=103.502526, claim_payoff_neutral_automation_positive=132.624253; coincidences [{"dz_per_dalpha": -3.9794706879985, "named_labels": ["claim_payoff_neutral_automation_positive", "rental_tax_base_neutral_automation_positive"], "theta_deg": 132.62425292319114}, {"dz_per_dalpha": -3.9794706879985, "named_labels": ["claim_payoff_neutral_automation_negative", "rental_tax_base_neutral_automation_negative"], "theta_deg": 312.62425292319114}]; h_I-(eta+1/alpha)=0.000e+00; tax-speed feedback alignment with rental gradient rel. error 7.121e-01; capital-growth alignment 0.000e+00; kappa_z-kappa_x=1.594e-01
- baseline: named angles pure_productivity_positive=0.000000, worker_income_neutral_automation_positive=77.399980, pure_automation_positive=90.000000, primary_resource_neutral_automation_positive=101.547781, output_neutral_automation_positive=115.952206, claim_payoff_neutral_automation_positive=151.804922; coincidences [{"dz_per_dalpha": -3.9794706879985, "named_labels": ["claim_payoff_neutral_automation_positive", "rental_tax_base_neutral_automation_positive"], "theta_deg": 151.80492206398452}, {"dz_per_dalpha": -3.9794706879985, "named_labels": ["claim_payoff_neutral_automation_negative", "rental_tax_base_neutral_automation_negative"], "theta_deg": 331.8049220639845}]; h_I-(eta+1/alpha)=0.000e+00; tax-speed feedback alignment with rental gradient rel. error 9.300e-16; capital-growth alignment 0.000e+00; kappa_z-kappa_x=0.000e+00

## Interpretation boundary for the numerical reviewer

All checks are on the solved first-order linear system: exactness of the matrix exponential, linearity (superposition, sign symmetry, scaling), the Brownian-versus-state timing convention, and accounting identities that hold by construction. Passing them establishes internal consistency of the local LQ computation, not global validity of the LQ approximation, borrowing capacity, or welfare.

## Post-hoc addendum (hand-written after the run, from the tables only)

- Present-value primary cash-flow identity, computed from `path_features.csv`
  for all 438 paths: max |∫e^{-ρt}(τB − T)dt + payoff| = 1.5e-16. Without an
  inherited claim, discounted cumulative transfers equal discounted cumulative
  capital-tax receipts; with the claim, the payoff is fully paid out.
- Planner-resource-wealth split at the anchor: j_W = (11.915, 3.218, 1.326, −2.652),
  j_B = (1.088, 0.325, −0.326, +2.652) on (z, x, k, τ); j_W + j_B = j to 2.2e-16 on
  every row. The capital-tax component of the marginal value of capital is
  negative because dB/dk = αR − δ < 0 at the anchor.
- Structural invariant line: the tax-speed feedback and capital-growth loadings on
  (z, x) are proportional to the rental gradient at the baseline (relative errors
  9e-16 and 0); with equal OU rates the claim/rental-neutral displacement leaves
  capital and tax below 6e-19 for 40 years. Under automation persistence 0.50/0.95
  the alignment errors are 0.59/0.71 and the line is not invariant.
- Wage sign changes reported at 39.9–40.0 years for θ ∈ [45°, 77.4°] (and mirrors)
  are tails of order 1e-8, three to four orders below the impact responses.
- Two labelling defects found and fixed between the first and the definitive run
  (persistence-0.81 rows carried the baseline label and duplicated named rows;
  zero-crossing counters counted noise on the identically-zero invariant path);
  no numerical value changed. The superseded first run directory was removed
  before any run record existed.
- Environment incident: the machine's data volume reached 100% (203 MiB free)
  during testing at 22:45 UTC; the uv package cache was pruned (4.0 GiB freed)
  and the session owning the largest recent files compressed them. The
  definitive run started with 4.0 GiB free; no output write failed.
