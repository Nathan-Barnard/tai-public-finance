# RUN-20260902T170932Z-CS002-D0D1-5c03b3fd-01

**Exploratory CS002 D0-D1 prototype. CS002 v0.2 remains draft and unfingerprinted; this is not an approved or completed CS002 result.**

- Outcome: **computational_pass**
- Config: `lq_farhi_d0_d1_frozen_v1` (fingerprint `1d2fb8c76f39...`)
- Independent max scaled ODE residual: `9.172e-10` (<= 1e-7 required)
- Independent max scaled boundary residual: `7.401e-17` (<= 1e-8 required)
- Horizon/mesh stability: `True`
- LQ-vs-nonlinear convergence ratios (halving amplitude, target ~4.0): k=[4.009, 4.005, 4.002, 4.001], tau=[4.012, 4.006, 4.003, 4.002]
- Route A vs route B branch agreement: `True`
- J(0) recovery (quadratic tail): flow-integral=`12.579182`, HJB-ODE=`12.579182`, relative disagreement=`1.769e-09`
- Net-worth grid: 3/5 feasible; infeasible members retained with failure reasons (see net_worth_grid.csv), not dropped.

Interpretation: bounded exploratory D0-D1 prototype under CS002 v0.2's draft-specification exception. It supports research review of the equation map, the LQ stable-manifold terminal mapping, residual/branch evidence, and resource profile -- it does not promote CS002 toward review_ready or authorize Block D2.
