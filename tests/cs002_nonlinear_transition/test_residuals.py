from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import CubicHermiteSpline

from tai_public_finance.cs002_nonlinear_transition.bvp import economic_bc, economic_rhs, lq_path_initial_guess, solve_two_point_bvp
from tai_public_finance.cs002_nonlinear_transition.model import capital_derivatives, capital_from_log, characteristic_rates
from tai_public_finance.cs002_nonlinear_transition.residuals import (
    componentwise_manual_rhs_check,
    independent_boundary_residual,
    independent_ode_residual,
    manual_ell_dot,
    manual_k_dot,
    manual_m_dot,
    manual_tau_dot,
)
from tai_public_finance.primitives import evaluate_smooth_branch, safe_rate

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


# --------------------------------------------------------------------------
# D2 mandatory repair #4: componentwise manual RHS reconstruction, at
# manufactured economic points, algebraically separate from
# characteristic_rates/characteristic_rhs_vectorized.
# --------------------------------------------------------------------------

_MANUFACTURED_POINTS = [
    # (k, tau, ell, m, z, x): a mix of the anchor, small and large
    # displacements, and displaced exogenous (z, x) -- the D2 case.
    (0.0, 0.5, 1.0, 0.0, -1.8, 0.0),
    (0.05, 0.55, 0.9, 0.02, -1.8, 0.0),
    (-0.1, 0.4, 1.2, -0.03, -1.79, 0.05),
    (0.02, 0.48, 0.95, 0.01, -1.75, -0.2),
]


@pytest.mark.parametrize("k, tau, ell, m, z, x", _MANUFACTURED_POINTS)
def test_componentwise_manual_rhs_matches_the_composite_at_manufactured_points(primitives, anchor, k, tau, ell, m, z, x):
    report = componentwise_manual_rhs_check(k, tau, ell, m, z, x, anchor.z_bar, anchor.x_bar, anchor.capital_bar, primitives)
    assert report.all_agree, report.max_abs_difference
    for name, diff in report.max_abs_difference.items():
        assert diff < 1e-9, (name, diff)


def test_manual_reconstructions_never_call_the_composite_characteristic_functions():
    """Architectural check, not a numerical one: the manual_*_dot source
    each call only evaluate_smooth_branch/capital_derivatives/safe_rate."""

    import inspect

    from tai_public_finance.cs002_nonlinear_transition import residuals as residuals_module

    for fn in (manual_k_dot, manual_tau_dot, manual_ell_dot, manual_m_dot):
        source = inspect.getsource(fn)
        assert "characteristic_rates(" not in source
        assert "characteristic_rhs_vectorized(" not in source
    # And componentwise_manual_rhs_check calls characteristic_rates only
    # ONCE, for the composite side of the comparison -- not from within any
    # manual_*_dot call.
    check_source = inspect.getsource(residuals_module.componentwise_manual_rhs_check)
    assert check_source.count("characteristic_rates(") == 1


@pytest.mark.parametrize(
    "broken_component",
    ["k_dot", "tau_dot", "ell_dot", "m_dot"],
)
def test_a_perturbed_manual_component_is_flagged_by_the_check(primitives, anchor, broken_component):
    """Negative control (CS002 D2 handoff: 'Perturb each component so the
    check demonstrably fails'). A check that always reports agreement is
    useless -- corrupt exactly one manual_*_dot formula (a deliberately
    wrong sign on one term) and confirm componentwise_manual_rhs_check's
    all_agree flips to False and the corrupted component's own difference
    is large, while the OTHER three components stay in agreement."""

    k, tau, ell, m, z, x = 0.02, 0.51, 0.98, 0.01, -1.79, 0.03
    z_bar, x_bar, capital_bar = anchor.z_bar, anchor.x_bar, anchor.capital_bar
    capital = capital_from_log(k, capital_bar)
    composite = characteristic_rates(k, tau, ell, m, z_bar, x_bar, capital_bar, primitives, z=z, x=x)
    composite_values = {"k_dot": composite.k_dot, "tau_dot": composite.tau_dot, "ell_dot": composite.ell_dot, "m_dot": composite.m_dot}

    state = evaluate_smooth_branch(z, x, capital, tau, primitives)
    derivs = capital_derivatives(state, primitives)
    r0 = safe_rate(z, x, z_bar, x_bar, primitives)
    kappa_tau = primitives.tax_adjustment_scale
    nu = kappa_tau * m / state.output

    correct = {
        "k_dot": manual_k_dot(z, x, capital, tau, primitives),
        "tau_dot": manual_tau_dot(z, x, capital, tau, m, primitives),
        "ell_dot": manual_ell_dot(z, x, z_bar, x_bar, capital, tau, ell, m, primitives),
        "m_dot": manual_m_dot(z, x, z_bar, x_bar, capital, tau, ell, m, primitives),
    }
    broken = dict(correct)
    if broken_component == "k_dot":
        broken["k_dot"] = -state.capital_growth  # flipped sign
    elif broken_component == "tau_dot":
        broken["tau_dot"] = 2.0 * (kappa_tau * m / state.output)  # wrong coefficient
    elif broken_component == "ell_dot":
        # Dropped the F_K term entirely (a plausible real bug: forgetting a subtraction).
        broken["ell_dot"] = (r0 - state.capital_growth - capital * derivs.capital_growth_K) * ell
    elif broken_component == "m_dot":
        broken["m_dot"] = r0 * m - state.tax_base * (ell - 1.0)  # flipped sign on the second term

    max_abs_difference = {name: abs(broken[name] - composite_values[name]) for name in broken}
    all_agree = all(value <= 1e-9 for value in max_abs_difference.values())

    assert not all_agree
    assert max_abs_difference[broken_component] > 1e-6
    for name in max_abs_difference:
        if name != broken_component:
            assert max_abs_difference[name] < 1e-9, "perturbing one component must not disturb the other three"
