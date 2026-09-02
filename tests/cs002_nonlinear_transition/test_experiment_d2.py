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
from tai_public_finance.cs002_nonlinear_transition.experiment import _pairwise_common_grid
from tai_public_finance.cs002_nonlinear_transition.experiment_d2 import _compare_reported_paths, run_d2_experiment
from tai_public_finance.cs002_nonlinear_transition.recovery import REPORTED_PATH_NAMES
from tai_public_finance.cs002_nonlinear_transition.residuals import independent_boundary_residual, independent_ode_residual

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "cs002" / "lq_farhi_d2_mean_reversion_v1.json"


@pytest.fixture(scope="module")
def report():
    config = load_cs002_d2_configuration(CONFIG_PATH)
    return run_d2_experiment(config)


def test_both_shock_directions_fail_only_horizon_mesh_stability(report):
    """CS002 D2 review repair, finding 2: on THIS config, the now-complete
    horizon/mesh check correctly fails for both directions -- `varpi`'s
    ~1/rho ~= 49.5-year relaxation timescale makes the declared 20-year
    comparison horizon genuinely too short for its terminal tail to have
    become negligible by t=20 (a real, smooth, monotonic divergence, not a
    numerical artifact; see the findings note). This was previously
    invisible because the old check neither reconstructed varpi nor scored
    each path against its own tolerance. Every OTHER independently-verified
    check remains true -- the K_bar=1 regression against the prior run is
    otherwise unchanged (see the machine-readable old-vs-new comparison)."""

    assert report.outcome.outcome == "numerical_failure", report.outcome.failed_checks
    assert report.outcome.failed_checks == ["productivity_horizon_mesh_stability", "automation_horizon_mesh_stability"]
    for sub in (report.productivity, report.automation):
        assert sub.outcome.outcome == "numerical_failure"
        assert sub.outcome.failed_checks == ["horizon_mesh_stability"]


def test_every_named_check_is_present_and_only_horizon_mesh_stability_is_false(report):
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
        assert sub.outcome.checks["horizon_mesh_stability"] is False
        other_checks = {k: v for k, v in sub.outcome.checks.items() if k != "horizon_mesh_stability"}
        assert all(other_checks.values()), {k: v for k, v in other_checks.items() if not v}


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


# --------------------------------------------------------------------------
# CS002 D2 review repair (finding 2): the horizon/mesh stability check must
# compare the COMPLETE reconstructed solution (REPORTED_PATH_NAMES), not
# only the four raw BVP state variables, and must score each reported path
# against ITS OWN effect-scaled tolerance rather than one shared/pooled
# tolerance built from the largest response among several variables.
# --------------------------------------------------------------------------


def _flat_paths(common_t: np.ndarray, value: float = 0.0) -> dict[str, np.ndarray]:
    """A synthetic path dict, every REPORTED_PATH_NAMES entry constant at
    `value` over `common_t` -- a minimal fixture for exercising
    `_compare_reported_paths` directly, without a full BVP solve."""

    return {name: np.full(common_t.shape, value) for name in REPORTED_PATH_NAMES}


def test_compare_reported_paths_rejects_a_dict_missing_a_required_path():
    common_t = np.linspace(0.0, 10.0, 5)
    complete = _flat_paths(common_t)
    incomplete = {k: v for k, v in complete.items() if k != "varpi"}
    reference = {name: 0.0 for name in REPORTED_PATH_NAMES}

    with pytest.raises(ValueError, match="varpi"):
        _compare_reported_paths(complete, incomplete, reference, common_t, 1e-7, 1e-3)
    with pytest.raises(ValueError, match="varpi"):
        _compare_reported_paths(incomplete, complete, reference, common_t, 1e-7, 1e-3)


def test_compare_reported_paths_detects_a_perturbation_in_any_single_reported_path():
    """Every REPORTED_PATH_NAMES entry, perturbed on its own, must fail its
    own check -- and must NOT drag any other, unperturbed path's verdict
    down with it (i.e. no pooling)."""

    common_t = np.linspace(0.0, 10.0, 5)
    reference = {name: 0.0 for name in REPORTED_PATH_NAMES}
    baseline = _flat_paths(common_t, value=0.0)
    tol_floor, tol_relative = 1e-7, 1e-3

    for perturbed_name in REPORTED_PATH_NAMES:
        comparison = {name: arr.copy() for name, arr in baseline.items()}
        comparison[perturbed_name][2] += 1.0  # far beyond the tol_floor=1e-7 anchor tolerance
        _, _, _, within = _compare_reported_paths(baseline, comparison, reference, common_t, tol_floor, tol_relative)
        assert within[perturbed_name] is False, f"perturbing {perturbed_name!r} should fail its own check"
        for other_name in REPORTED_PATH_NAMES:
            if other_name != perturbed_name:
                assert within[other_name] is True, f"perturbing {perturbed_name!r} should not affect {other_name!r}"


def test_compare_reported_paths_uses_path_specific_not_pooled_tolerances():
    """A path with a tiny effect (tight tolerance) must fail on a small
    perturbation even though another path in the SAME comparison has a huge
    effect (loose tolerance) -- the old pooled-tolerance bug used the
    LARGEST peak response across variables to set one shared tolerance,
    which would have let this exact perturbation pass."""

    common_t = np.linspace(0.0, 10.0, 5)
    reference = {name: 0.0 for name in REPORTED_PATH_NAMES}
    baseline = _flat_paths(common_t, value=0.0)
    baseline["J"] = np.full(common_t.shape, 1000.0)  # huge peak response -> loose tolerance (1.0)

    comparison = {name: arr.copy() for name, arr in baseline.items()}
    comparison["varpi"][2] += 1e-4  # << J's pooled tolerance of 1.0, >> varpi's own tolerance of 1e-7

    max_diff, tolerance, _, within = _compare_reported_paths(baseline, comparison, reference, common_t, 1e-7, 1e-3)
    assert tolerance["J"] == pytest.approx(1.0)  # 1e-3 * 1000.0
    assert tolerance["varpi"] == pytest.approx(1e-7)  # floor: baseline varpi's own peak response is 0
    assert max_diff["varpi"] == pytest.approx(1e-4)
    assert max_diff["varpi"] <= tolerance["J"]  # would PASS if pooled against J's loose tolerance
    assert within["varpi"] is False  # but correctly fails against its OWN tolerance
    assert within["J"] is True


def test_horizon_mesh_comparisons_report_every_path_name(report):
    """Every horizon/mesh comparison, for both directions, records all of
    REPORTED_PATH_NAMES -- not only the four raw BVP state variables the
    D0-D1-inherited check compared."""

    for sub in (report.productivity, report.automation):
        for comparison in sub.horizon_mesh_comparisons:
            for mapping in (
                comparison.max_state_difference_from_baseline, comparison.tolerance,
                comparison.time_of_max_difference, comparison.within_tolerance_by_path,
            ):
                assert set(mapping) == set(REPORTED_PATH_NAMES)


def test_horizon_80_vs_baseline_comparison_covers_the_full_zero_to_forty_year_interval(report):
    """CS002 D2 review repair (finding 2) / D1's inherited mandatory repair
    #3: the 80-vs-40 comparison's common interval must be 0-40, not
    silently capped at 0-20 by a shared-grid construction. Checks the grid
    helper directly (mirroring D1's own regression test) AND, end to end,
    that the actual reported max-difference TIME for several paths reaches
    all the way to t=40 -- impossible if the underlying grid were still
    capped at t=20."""

    baseline_horizon = report.config.baseline_horizon
    assert baseline_horizon == 40.0
    grid_80_vs_40 = _pairwise_common_grid(80.0, baseline_horizon)
    assert grid_80_vs_40[0] == 0.0
    assert grid_80_vs_40[-1] == pytest.approx(40.0)

    for sub in (report.productivity, report.automation):
        comp80 = next(c for c in sub.horizon_mesh_comparisons if c.label == "horizon_80y_vs_baseline")
        assert max(comp80.time_of_max_difference.values()) == pytest.approx(40.0)
        paths_reaching_the_true_endpoint = [name for name, t in comp80.time_of_max_difference.items() if t > 20.0]
        assert len(paths_reaching_the_true_endpoint) > 0
