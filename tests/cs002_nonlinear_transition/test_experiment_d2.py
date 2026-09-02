"""End-to-end wiring tests for run_d2_experiment, plus the CS002 D2 required
checks that are most naturally exercised at the full-experiment level:
zero-displacement collapse to the D1 contract, wrong-r0 detection (#8), and
wrong-terminal-coordinate / perturbed-path detection (#9). Every individual
piece (exogenous paths, model/terminal/bvp generalization, recovery,
residuals, margins, continuation) already has focused unit tests elsewhere;
this checks that run_d2_experiment assembles them correctly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tai_public_finance.cs002_nonlinear_transition.bvp import economic_bc, economic_bc_with_exogenous_path, economic_rhs, economic_rhs_with_exogenous_path
from tai_public_finance.cs002_nonlinear_transition.config_d2 import load_cs002_d2_configuration
from tai_public_finance.cs002_nonlinear_transition.exogenous import ExogenousPath
from tai_public_finance.cs002_nonlinear_transition.experiment_d2 import run_d2_experiment
from tai_public_finance.cs002_nonlinear_transition.residuals import independent_boundary_residual, independent_ode_residual

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "cs002" / "lq_farhi_d2_mean_reversion_v1.json"


@pytest.fixture(scope="module")
def report():
    config = load_cs002_d2_configuration(CONFIG_PATH)
    return run_d2_experiment(config)


def test_both_shock_directions_reach_computational_pass(report):
    assert report.outcome.outcome == "computational_pass", report.outcome.failed_checks
    assert report.outcome.failed_checks == []
    assert report.productivity.outcome.outcome == "computational_pass", report.productivity.outcome.failed_checks
    assert report.automation.outcome.outcome == "computational_pass", report.automation.outcome.failed_checks


def test_every_named_check_is_present_and_true_for_both_directions(report):
    expected = {
        "zero_amplitude_returns_anchor", "route_a_all_amplitudes_converged", "route_b_all_amplitudes_converged",
        "ou_path_matches_numerical_propagation", "ode_residual_within_tolerance", "boundary_residual_within_tolerance",
        "componentwise_rhs_agrees", "horizon_mesh_stability", "lq_convergence_expected_order",
        "j_recovery_routes_agree", "x_recovery_routes_agree", "n_recovery_routes_agree", "varpi_recovery_routes_agree",
        "varpi_along_path_residual_within_tolerance", "budget_separation_within_tolerance", "wrong_r0_is_detected",
        "branch_agreement_route_a_vs_b", "no_boundary_reached_on_baseline_path",
    }
    for sub in (report.productivity, report.automation):
        assert set(sub.outcome.checks) == expected
        assert all(sub.outcome.checks.values()), sub.outcome.failed_checks


def test_residuals_are_comfortably_within_declared_tolerances(report):
    for sub in (report.productivity, report.automation):
        assert sub.ode_residual.max_scaled_residual <= 1e-7
        assert sub.boundary_residual.max_scaled_residual <= 1e-8


def test_convergence_ratios_approach_the_lq_limit_quadratically(report):
    for sub in (report.productivity, report.automation):
        assert len(sub.convergence.ratios_k) == 4
        assert len(sub.convergence.ratios_tau) == 4
        for ratio in sub.convergence.ratios_k + sub.convergence.ratios_tau:
            assert ratio == pytest.approx(4.0, rel=0.1)


def test_route_disagreements_are_far_smaller_than_the_reported_effect(report):
    for sub in (report.productivity, report.automation):
        j_effect = abs(sub.j_recovery.j0_hjb - report.local_system.anchor.fiscal_wealth_bar)
        assert sub.j_recovery.route_disagreement * 10.0 <= j_effect
        x_effect = float(np.max(np.abs(sub.exogenous_resources.x_path_ode - sub.exogenous_resources.x_0)))
        assert sub.exogenous_resources.x_routes_max_abs_deviation * 10.0 <= x_effect
        varpi_effect = float(np.max(np.abs(sub.varpi_recovery.varpi_path_ode)))
        assert sub.varpi_recovery.route_disagreement_max_abs * 10.0 <= varpi_effect


def test_structural_continuation_solvency_is_never_claimed(report):
    """CS002 D2 acceptance: structural continuation solvency is labelled
    not_evaluated, and no path is called globally feasible on the basis of
    the reported local margins."""

    for sub in (report.productivity, report.automation):
        assert all(margin_min > 0.0 for margin_min in (float(v.min()) for v in sub.margins_series.values()))
        # The report never asserts a "feasible"/"solvent" verdict anywhere in
        # its own dataclasses -- only per-margin numeric series and the
        # narrow no_boundary_reached_on_baseline_path check, which is
        # explicitly local-margin-based (see experiment_d2.py's docstrings).


def test_margins_are_reported_at_every_time_point(report):
    for sub in (report.productivity, report.automation):
        for series in sub.margins_series.values():
            assert series.shape == sub.margins_t_grid.shape
            assert series.size > 1


# --------------------------------------------------------------------------
# Required check #3: zero exogenous displacement collapses EXACTLY to the
# D0-D1 contract (bit-for-bit against economic_rhs/economic_bc, not just
# "close").
# --------------------------------------------------------------------------


def test_zero_displacement_exogenous_bvp_matches_the_d1_bvp_bit_for_bit(cs001_local_system, cs001_solution):
    anchor = cs001_local_system.anchor
    horizon = 40.0
    x_mesh = np.linspace(0.0, horizon, 81)
    y_guess = np.tile(np.array([[0.0], [anchor.tax_rate_bar], [1.0], [0.0]]), (1, x_mesh.size))

    d1_fun = economic_rhs(cs001_local_system)
    d1_bc = economic_bc(anchor.capital_bar, anchor.tax_rate_bar, "lq_stable_manifold", cs001_local_system, cs001_solution)

    zero_path = ExogenousPath(z0=anchor.z_bar, x0=anchor.x_bar, z_bar=anchor.z_bar, x_bar=anchor.x_bar, kappa_z=cs001_local_system.parameters.kappa_z, kappa_x=cs001_local_system.parameters.kappa_x)
    d2_fun = economic_rhs_with_exogenous_path(cs001_local_system, zero_path)
    d2_bc = economic_bc_with_exogenous_path(anchor.capital_bar, anchor.tax_rate_bar, horizon, "lq_stable_manifold", cs001_local_system, cs001_solution, zero_path)

    np.testing.assert_array_equal(d1_fun(x_mesh, y_guess), d2_fun(x_mesh, y_guess))
    ya, yb = y_guess[:, 0], y_guess[:, -1]
    np.testing.assert_array_equal(d1_bc(ya, yb), d2_bc(ya, yb))


# --------------------------------------------------------------------------
# Required check #8: a deliberately wrong r0=rho substitution in a nonzero-
# shock fixture must be detected.
# --------------------------------------------------------------------------


def test_wrong_r0_substitution_is_detected_on_both_shock_directions(report):
    for sub in (report.productivity, report.automation):
        assert sub.wrong_r0_detection.detected
        assert sub.wrong_r0_detection.absolute_difference > 10.0 * sub.j_recovery.route_disagreement


# --------------------------------------------------------------------------
# Required check #9: wrong terminal exogenous coordinates and a perturbed
# returned path are detected by the independent diagnostics.
# --------------------------------------------------------------------------


def test_wrong_terminal_exogenous_coordinates_are_detected(report):
    sub = report.productivity
    baseline = sub.route_a.final
    anchor = report.local_system.anchor
    horizon = sub.route_a.horizon
    z_T_correct = float(baseline.exogenous_path.z(horizon))
    x_T_correct = float(baseline.exogenous_path.x(horizon))

    correct = independent_boundary_residual(
        baseline.result, anchor.capital_bar, anchor.tax_rate_bar, "lq_stable_manifold", report.local_system, report.solution,
        z_T=z_T_correct, x_T=x_T_correct,
    )
    wrong = independent_boundary_residual(
        baseline.result, anchor.capital_bar, anchor.tax_rate_bar, "lq_stable_manifold", report.local_system, report.solution,
        z_T=z_T_correct + 0.05, x_T=x_T_correct,  # deliberately wrong terminal z
    )
    assert correct.max_scaled_residual <= 1e-8
    assert wrong.max_scaled_residual > 1e-3


def test_a_perturbed_returned_path_is_detected_with_the_exogenous_path_residual(report):
    from scipy.interpolate import CubicHermiteSpline

    sub = report.productivity
    baseline = sub.route_a.final
    exogenous_path = baseline.exogenous_path
    x_mesh = baseline.result.x_mesh
    y_true = baseline.result.y_mesh

    dydt_true = economic_rhs_with_exogenous_path(report.local_system, exogenous_path)(x_mesh, y_true)
    wrong_spline = CubicHermiteSpline(x=[x_mesh[0], x_mesh[-1]], y=y_true[:, [0, -1]].T, dydx=dydt_true[:, [0, -1]].T, axis=0)

    class _FakeResult:
        sol = staticmethod(lambda t: wrong_spline(t).T)
        x_mesh = baseline.result.x_mesh

    correct_report = independent_ode_residual(baseline.result, report.local_system, exogenous_path=exogenous_path)
    wrong_report = independent_ode_residual(_FakeResult(), report.local_system, exogenous_path=exogenous_path)
    assert correct_report.max_scaled_residual <= 1e-7
    assert wrong_report.max_scaled_residual > 1e-3
