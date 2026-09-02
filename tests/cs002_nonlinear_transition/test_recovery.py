from __future__ import annotations

import numpy as np
import pytest

from tai_public_finance.cs002_nonlinear_transition.bvp import economic_bc, economic_rhs, lq_path_initial_guess, solve_two_point_bvp
from tai_public_finance.cs002_nonlinear_transition.model import capital_from_log
from tai_public_finance.cs002_nonlinear_transition.recovery import recover_comprehensive_resources, recover_j
from tai_public_finance.cs002_nonlinear_transition.terminal import anchor_value_tail, lq_quadratic_value_tail


def _solve(cs001_local_system, cs001_solution, k0, dtau0, horizon):
    anchor = cs001_local_system.anchor
    tau0 = anchor.tax_rate_bar + dtau0
    capital0 = capital_from_log(k0, anchor.capital_bar)
    x_mesh = np.linspace(0.0, horizon, 161)
    y_guess = lq_path_initial_guess(x_mesh, k0, dtau0, cs001_local_system, cs001_solution)
    fun = economic_rhs(cs001_local_system)
    bc = economic_bc(capital0, tau0, "lq_stable_manifold", cs001_local_system, cs001_solution)
    result = solve_two_point_bvp(fun, bc, x_mesh, y_guess, tol=1e-11)
    assert result.success
    return result


def test_zero_displacement_j_recovery_equals_anchor_fiscal_wealth(cs001_local_system, cs001_solution):
    anchor = cs001_local_system.anchor
    result = _solve(cs001_local_system, cs001_solution, 0.0, 0.0, horizon=40.0)
    tail = anchor.fiscal_wealth_bar
    recovery = recover_j(result, cs001_local_system, "anchor", tail, horizon=40.0)
    assert recovery.j0_hjb == pytest.approx(anchor.fiscal_wealth_bar, rel=1e-7)
    assert recovery.j0_flow_integral == pytest.approx(anchor.fiscal_wealth_bar, rel=1e-6)
    assert recovery.route_disagreement_relative < 1e-6


def test_both_j_routes_agree_on_a_displaced_path(cs001_local_system, cs001_solution):
    anchor = cs001_local_system.anchor
    horizon = 40.0
    result = _solve(cs001_local_system, cs001_solution, 0.01, 0.01, horizon)
    capital_T = capital_from_log(result.sol(np.array([horizon]))[0, 0], anchor.capital_bar)
    tau_T = result.sol(np.array([horizon]))[1, 0]
    tail = lq_quadratic_value_tail(capital_T, tau_T, cs001_local_system, cs001_solution)

    recovery = recover_j(result, cs001_local_system, "quadratic", tail, horizon)
    assert recovery.route_disagreement_relative < 1e-4, (recovery.j0_flow_integral, recovery.j0_hjb)


def test_comprehensive_resources_are_constant_along_a_frozen_state_path(cs001_local_system, cs001_solution):
    """Ẋ=(r0-rho)X=0 exactly when r0=rho (frozen common states): X(t)=N(t)+J(t)
    must stay at X_0 even though N(t) and J(t) individually vary."""

    anchor = cs001_local_system.anchor
    horizon = 40.0
    result = _solve(cs001_local_system, cs001_solution, 0.01, 0.01, horizon)
    capital_T = capital_from_log(result.sol(np.array([horizon]))[0, 0], anchor.capital_bar)
    tau_T = result.sol(np.array([horizon]))[1, 0]
    tail = lq_quadratic_value_tail(capital_T, tau_T, cs001_local_system, cs001_solution)
    recovery = recover_j(result, cs001_local_system, "quadratic", tail, horizon)

    comprehensive = recover_comprehensive_resources(result, cs001_local_system, recovery, net_worth_0=0.0)
    assert comprehensive.x_constancy_max_rel_deviation < 1e-5, comprehensive.x_path
    # N and J themselves are NOT constant -- otherwise the check above would be vacuous.
    assert np.ptp(comprehensive.n_path) > 1e-6
    assert np.ptp(comprehensive.j_path) > 1e-6


def test_tail_convention_gap_shrinks_as_horizon_extends(cs001_local_system, cs001_solution):
    """The quadratic-tail and anchor-tail recoveries of J_0 must disagree at a
    short horizon (the terminal displacement is still material) and converge
    toward each other as the horizon is extended and the terminal state
    approaches the deterministic anchor."""

    anchor = cs001_local_system.anchor
    gaps = {}
    for horizon in (10.0, 40.0, 80.0):
        result = _solve(cs001_local_system, cs001_solution, 0.01, 0.01, horizon)
        capital_T = capital_from_log(result.sol(np.array([horizon]))[0, 0], anchor.capital_bar)
        tau_T = result.sol(np.array([horizon]))[1, 0]

        quad_tail = lq_quadratic_value_tail(capital_T, tau_T, cs001_local_system, cs001_solution)
        anchor_tail = anchor_value_tail(cs001_local_system)
        quad_recovery = recover_j(result, cs001_local_system, "quadratic", quad_tail, horizon)
        anchor_recovery = recover_j(result, cs001_local_system, "anchor", anchor_tail, horizon)
        gaps[horizon] = abs(quad_recovery.j0_hjb - anchor_recovery.j0_hjb)

    assert gaps[80.0] < gaps[40.0] < gaps[10.0], gaps
