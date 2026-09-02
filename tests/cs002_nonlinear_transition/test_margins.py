from __future__ import annotations

import numpy as np
import pytest

from tai_public_finance.cs002_nonlinear_transition.bvp import economic_bc, economic_rhs, lq_path_initial_guess, solve_two_point_bvp
from tai_public_finance.cs002_nonlinear_transition.margins import evaluate_path_margins
from tai_public_finance.cs002_nonlinear_transition.model import capital_from_log

SCAFFOLDING = {"tax_min": -0.5, "tax_max": 0.95, "tax_speed_abs_max": 0.5}


def _solve(cs001_local_system, cs001_solution, k0, dtau0, horizon=40.0):
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


def test_anchor_margins_match_cs001s_own_reported_anchor_margins(cs001_local_system, cs001_solution):
    anchor = cs001_local_system.anchor
    result = _solve(cs001_local_system, cs001_solution, 0.0, 0.0)
    t_grid = np.linspace(0.0, 40.0, 81)
    margins = evaluate_path_margins(result, cs001_local_system, consumption=anchor.worker_consumption_bar, x_0=anchor.fiscal_wealth_bar, numerical_scaffolding=SCAFFOLDING, t_grid=t_grid)

    assert margins.min_specialisation_margin_automation_composite == pytest.approx(anchor.specialisation_margin_automation_composite, rel=1e-8)
    assert margins.min_specialisation_margin_new_task_composite == pytest.approx(anchor.specialisation_margin_new_task_composite, rel=1e-8)
    assert margins.min_transfer_margin == pytest.approx(anchor.transfer_bar, abs=1e-8)  # c_bar - W_bar = transfer_bar by definition
    assert not margins.boundary_reached
    assert margins.failure_reasons == []


def test_small_displacement_stays_interior_on_every_margin(cs001_local_system, cs001_solution):
    anchor = cs001_local_system.anchor
    result = _solve(cs001_local_system, cs001_solution, 0.01, 0.01)
    t_grid = np.linspace(0.0, 40.0, 161)
    consumption = anchor.worker_consumption_bar  # N_0=0 baseline; close enough to the true recovered c for a slack check
    margins = evaluate_path_margins(result, cs001_local_system, consumption=consumption, x_0=anchor.fiscal_wealth_bar, numerical_scaffolding=SCAFFOLDING, t_grid=t_grid)

    assert margins.min_specialisation_margin_automation_composite > 0.5
    assert margins.min_specialisation_margin_new_task_composite > 0.5
    assert margins.min_tax_margin > 0.3
    assert margins.min_tax_speed_margin > 0.3
    # R^K - delta = 2*rho at the anchor exactly (anchor.py); a small displacement
    # should keep this near that level, not push it toward some large arbitrary bound.
    rho = cs001_local_system.parameters.rho
    assert margins.min_net_rental_tax_base_margin == pytest.approx(2.0 * rho, rel=0.2)
    assert margins.min_net_rental_tax_base_margin > 0.0
    assert margins.structural_continuation_solvency == "not_evaluated"
    assert not margins.boundary_reached
    assert margins.failure_reasons == []


def test_net_rental_tax_base_margin_is_not_labelled_structural_solvency(cs001_local_system):
    """D2 mandatory repair #1 regression: min(R^K-delta) going non-positive
    must be flagged as `net_rental_tax_base_boundary_reached`, never the old
    `structural_solvency_boundary_reached` name, and
    structural_continuation_solvency must always read `not_evaluated` --
    this module performs no separate viability/no-Ponzi calculation."""

    anchor = cs001_local_system.anchor
    # A wildly large capital level drives R^K = alpha*Y/K towards zero
    # (Y ~ K**alpha, alpha<1) well below depreciation_rate, without needing a
    # real BVP solve -- only evaluate_path_margins's own path-evaluation
    # logic is under test here.
    huge_capital_k = np.log(1.0e6)

    class _FakeResult:
        @staticmethod
        def sol(t):
            n = np.atleast_1d(t).size
            path = np.empty((4, n))
            path[0, :] = huge_capital_k
            path[1, :] = anchor.tax_rate_bar
            path[2, :] = 1.0
            path[3, :] = 0.0
            return path

    t_grid = np.linspace(0.0, 1.0, 5)
    margins = evaluate_path_margins(
        _FakeResult(), cs001_local_system, consumption=anchor.worker_consumption_bar, x_0=anchor.fiscal_wealth_bar,
        numerical_scaffolding=SCAFFOLDING, t_grid=t_grid,
    )

    assert margins.min_net_rental_tax_base_margin < 0.0
    assert margins.boundary_reached
    assert "net_rental_tax_base_boundary_reached" in margins.failure_reasons
    assert "structural_solvency_boundary_reached" not in margins.failure_reasons
    assert margins.structural_continuation_solvency == "not_evaluated"


def test_margins_flag_a_deliberately_infeasible_tax_scaffolding(cs001_local_system, cs001_solution):
    """Negative control: an artificially tight tax band that the anchor tax
    rate itself violates must be retained as boundary_reached with an
    explicit reason, not silently passed."""

    anchor = cs001_local_system.anchor
    result = _solve(cs001_local_system, cs001_solution, 0.0, 0.0)
    t_grid = np.linspace(0.0, 40.0, 41)
    tight_scaffolding = {"tax_min": anchor.tax_rate_bar + 0.01, "tax_max": 0.95, "tax_speed_abs_max": 0.5}
    margins = evaluate_path_margins(result, cs001_local_system, consumption=anchor.worker_consumption_bar, x_0=anchor.fiscal_wealth_bar, numerical_scaffolding=tight_scaffolding, t_grid=t_grid)

    assert margins.boundary_reached
    assert "tax_boundary_reached" in margins.failure_reasons
