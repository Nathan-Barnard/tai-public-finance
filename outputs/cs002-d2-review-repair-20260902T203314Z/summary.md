# RUN-20260902T203314Z-CS002-D2-REPAIR-94332783-01

**Exploratory CS002 D2 prototype. CS002 v0.2 remains draft and unfingerprinted; this is not an approved or completed CS002 result.**

- Outcome: **numerical_failure**
- Config: `lq_farhi_d2_mean_reversion_v1` (fingerprint `af6b8d613e26...`)

## Pure productivity (z(0)-z_bar=+0.01, x(0)=x_bar)

- Outcome: **numerical_failure**
- Independent max scaled ODE residual: `2.531e-09` (<= 1e-7 required)
- Independent max scaled boundary residual: `7.404e-17` (<= 1e-8 required)
- Componentwise manual RHS check: `True`
- Horizon/mesh stability: `False`
- LQ-vs-nonlinear convergence ratios (halving amplitude, target ~4.0): k=[4.004, 4.002, 4.001, 4.0], tau=[3.99, 3.995, 3.997, 3.999]
- Route A vs route B branch agreement: `True`
- J(0): `12.699860` (route disagreement rel. `4.943e-09`)
- X routes max relative deviation: `3.395e-07`
- N routes max relative deviation: `6.023e-11`
- varpi(0): `-4.516348e-01` (route disagreement rel. `1.383e-05`); along-path residual `2.850e-08`
- varpi(0) horizon sensitivity: {40y: -4.5163e-01, 80y: -4.5168e-01, 200y: -4.5168e-01}
- Budget-separation residual: `2.241e-04`
- Wrong-r0=rho substitution detected: `True` (|difference|=`1.281e-01`)
- Minimum reported margins (all time points): {specialisation_margin_automation_composite: 1.0383, specialisation_margin_new_task_composite: 0.5842, tax_margin: 0.4481, tax_speed_margin: 0.4986, transfer_margin: 0.0201, net_rental_tax_base_margin: 0.0403}
- Structural continuation solvency: `not_evaluated` (no viability/no-Ponzi calculation is performed)

## Pure automation (alpha(x(0))-alpha_bar=+0.01, implied x(0)=0.133531, z(0)=z_bar)

- Outcome: **numerical_failure**
- Independent max scaled ODE residual: `9.698e-09` (<= 1e-7 required)
- Independent max scaled boundary residual: `7.400e-17` (<= 1e-8 required)
- Componentwise manual RHS check: `True`
- Horizon/mesh stability: `False`
- LQ-vs-nonlinear convergence ratios (halving amplitude, target ~4.0): k=[3.936, 3.969, 3.985, 3.993], tau=[3.94, 3.971, 3.986, 3.993]
- Route A vs route B branch agreement: `True`
- J(0): `13.042627` (route disagreement rel. `1.214e-08`)
- X routes max relative deviation: `1.216e-06`
- N routes max relative deviation: `6.973e-11`
- varpi(0): `-1.766597e+00` (route disagreement rel. `2.680e-05`); along-path residual `4.242e-08`
- varpi(0) horizon sensitivity: {40y: -1.7666e+00, 80y: -1.7668e+00, 200y: -1.7668e+00}
- Budget-separation residual: `5.402e-04`
- Wrong-r0=rho substitution detected: `True` (|difference|=`4.629e-01`)
- Minimum reported margins (all time points): {specialisation_margin_automation_composite: 0.9940, specialisation_margin_new_task_composite: 0.5797, tax_margin: 0.4424, tax_speed_margin: 0.4941, transfer_margin: 0.0193, net_rental_tax_base_margin: 0.0399}
- Structural continuation solvency: `not_evaluated` (no viability/no-Ponzi calculation is performed)

Interpretation: bounded exploratory D2 prototype under CS002 v0.2's draft-specification exception, extending the reviewed D0-D1 frozen-common-state evidence to deterministic mean-reverting productivity and automation paths. These are deterministic local-shock IRFs, not stochastic expected paths, and do not establish global feasibility, structural continuation solvency, or an approved CS002 result. Do not implement D3, V2, R4, order-epsilon^4, a stochastic PDE, jump models, calibration, or the broad experiment grid from this evidence alone.
