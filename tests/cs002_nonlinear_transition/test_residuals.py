from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import CubicHermiteSpline

from tai_public_finance.cs002_nonlinear_transition.bvp import economic_bc, economic_rhs, lq_path_initial_guess, solve_two_point_bvp
from tai_public_finance.cs002_nonlinear_transition.model import capital_from_log
from tai_public_finance.cs002_nonlinear_transition.residuals import independent_boundary_residual, independent_ode_residual

HORIZON = 40.0
ACCEPTANCE_ODE_TOL = 1e-7
ACCEPTANCE_BOUNDARY_TOL = 1e-8


def _solve(cs001_local_system, cs001_solution, k0, dtau0, terminal_convention="lq_stable_manifold"):
    anchor = cs001_local_system.anchor
    tau0 = anchor.tax_rate_bar + dtau0
    capital0 = capital_from_log(k0, anchor.capital_bar)
    x_mesh = np.linspace(0.0, HORIZON, 161)
    y_guess = lq_path_initial_guess(x_mesh, k0, dtau0, cs001_local_system, cs001_solution)
    fun = economic_rhs(cs001_local_system)
    bc = economic_bc(capital0, tau0, terminal_convention, cs001_local_system, cs001_solution)
    result = solve_two_point_bvp(fun, bc, x_mesh, y_guess, tol=1e-11)
    assert result.success
    return result, capital0, tau0


@pytest.mark.parametrize("k0, dtau0", [(0.0, 0.0), (0.01, 0.01), (-0.01, 0.01), (0.01, -0.01)])
def test_true_solution_passes_both_independent_residual_gates(cs001_local_system, cs001_solution, k0, dtau0):
    result, capital0, tau0 = _solve(cs001_local_system, cs001_solution, k0, dtau0)

    ode_report = independent_ode_residual(result, cs001_local_system)
    assert ode_report.max_scaled_residual <= ACCEPTANCE_ODE_TOL, ode_report.per_state_max_scaled_residual

    boundary_report = independent_boundary_residual(result, capital0, tau0, "lq_stable_manifold", cs001_local_system, cs001_solution)
    assert boundary_report.max_scaled_residual <= ACCEPTANCE_BOUNDARY_TOL


def test_ode_residual_evaluator_detects_a_wrong_path(cs001_local_system, cs001_solution):
    """Negative control: an evaluator that reports near-zero for anything is
    useless. Feed it a deliberately WRONG path (a smooth interpolant between
    the true endpoints that does not solve the ODE) and confirm it is
    flagged with a residual many orders above the acceptance tolerance."""

    true_result, capital0, tau0 = _solve(cs001_local_system, cs001_solution, 0.01, 0.01)
    x_mesh = true_result.x_mesh
    y_true = true_result.y_mesh

    # A cubic Hermite spline through the SAME endpoints and endpoint slopes
    # as the true solution, but through only two knots -- it satisfies the
    # boundary conditions on the nose yet does not solve the interior ODE.
    dydt_true = economic_rhs(cs001_local_system)(x_mesh, y_true)
    wrong_spline = CubicHermiteSpline(
        x=[x_mesh[0], x_mesh[-1]], y=y_true[:, [0, -1]].T, dydx=dydt_true[:, [0, -1]].T, axis=0
    )

    class _FakeResult:
        sol = staticmethod(lambda t: wrong_spline(t).T)
        x_mesh = true_result.x_mesh

    ode_report = independent_ode_residual(_FakeResult(), cs001_local_system)
    assert ode_report.max_scaled_residual > 1e-3, "the residual evaluator failed to flag a path that does not solve the ODE"


def test_boundary_residual_evaluator_detects_a_wrong_terminal_value(cs001_local_system, cs001_solution):
    true_result, capital0, tau0 = _solve(cs001_local_system, cs001_solution, 0.01, 0.01)

    class _FakeResult:
        x_mesh = true_result.x_mesh

        @staticmethod
        def sol(t):
            # independent_boundary_residual calls .sol() once for t=[t0] and once for
            # t=[T] separately, each a length-1 array -- corrupt only the call that is
            # actually evaluating the terminal time, not whichever call happens first.
            y = np.array(true_result.sol(t), copy=True)
            t_arr = np.atleast_1d(t)
            terminal_mask = np.isclose(t_arr, true_result.x_mesh[-1])
            y[2, terminal_mask] += 0.05  # displace ell(T) well away from its correct terminal value
            return y

    report = independent_boundary_residual(_FakeResult(), capital0, tau0, "lq_stable_manifold", cs001_local_system, cs001_solution)
    assert report.max_scaled_residual > 1e-3
