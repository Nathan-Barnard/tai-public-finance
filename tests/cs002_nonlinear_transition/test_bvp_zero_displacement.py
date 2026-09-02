"""CS002 checking-matrix row: zero displacement must return the constant
deterministic anchor (k=0, tau=tau_bar, ell=1, m=0) for BOTH terminal
conventions -- this is the cheapest, highest-value end-to-end test of the
real economic RHS/BC/solver wiring."""

from __future__ import annotations

import numpy as np
import pytest

from tai_public_finance.cs002_nonlinear_transition.bvp import economic_bc, economic_rhs, lq_path_initial_guess, solve_two_point_bvp


@pytest.mark.parametrize("terminal_convention", ["lq_stable_manifold", "crude"])
def test_zero_displacement_returns_the_constant_anchor(cs001_local_system, cs001_solution, terminal_convention):
    anchor = cs001_local_system.anchor
    horizon = 40.0
    x_mesh = np.linspace(0.0, horizon, 81)
    y_guess = lq_path_initial_guess(x_mesh, 0.0, 0.0, cs001_local_system, cs001_solution)

    fun = economic_rhs(cs001_local_system)
    bc = economic_bc(anchor.capital_bar, anchor.tax_rate_bar, terminal_convention, cs001_local_system, cs001_solution)
    result = solve_two_point_bvp(fun, bc, x_mesh, y_guess, tol=1e-11)

    assert result.success, result.message
    check_t = np.linspace(0.0, horizon, 401)
    path = result.sol(check_t)
    np.testing.assert_allclose(path[0, :], 0.0, atol=1e-9)  # k
    np.testing.assert_allclose(path[1, :], anchor.tax_rate_bar, atol=1e-9)  # tau
    np.testing.assert_allclose(path[2, :], 1.0, atol=1e-8)  # ell
    np.testing.assert_allclose(path[3, :], 0.0, atol=1e-8)  # m
