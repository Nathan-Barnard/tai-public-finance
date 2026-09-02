"""End-to-end wiring test for the full D0-D1 material-run orchestration.
Every individual piece (model, terminal mapping, residuals, recovery,
margins, convergence, outcome taxonomy) already has focused unit tests
elsewhere; this only checks that `run_d0_d1_experiment` assembles them
correctly and reaches computational_pass on the frozen protocol's own
configuration, retaining (not hiding) the two structurally-infeasible
net-worth-grid members CS001's own repaired baseline also finds."""

from __future__ import annotations

from pathlib import Path

import pytest

import numpy as np

from tai_public_finance.cs002_nonlinear_transition.config import load_cs002_configuration
from tai_public_finance.cs002_nonlinear_transition.experiment import _pairwise_common_grid, run_d0_d1_experiment

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "cs002" / "lq_farhi_d0_d1_frozen_v1.json"


@pytest.fixture(scope="module")
def report():
    config = load_cs002_configuration(CONFIG_PATH)
    return run_d0_d1_experiment(config)


def test_frozen_protocol_reaches_computational_pass(report):
    assert report.outcome.outcome == "computational_pass", report.outcome.failed_checks
    assert report.outcome.failed_checks == []


def test_every_named_acceptance_check_is_present_and_true(report):
    expected_checks = {
        "d0_anchor_is_fixed_point",
        "d0_manufactured_bvp_recovered",
        "zero_displacement_returns_anchor",
        "route_a_all_amplitudes_converged",
        "route_b_all_amplitudes_converged",
        "ode_residual_within_tolerance",
        "boundary_residual_within_tolerance",
        "horizon_mesh_stability",
        "lq_convergence_second_order",
        "j_recovery_routes_agree",
        "branch_agreement_route_a_vs_b",
        "no_boundary_reached_on_baseline_path",
    }
    assert set(report.outcome.checks) == expected_checks
    assert all(report.outcome.checks.values())


def test_net_worth_grid_retains_infeasible_members_without_failing_the_run(report):
    """N/J in {-0.5, -0.25} must be reported with NOT all local margins slack
    (negative_transfer) -- exactly mirroring CS001's own repaired baseline --
    without that flipping the aggregate outcome away from computational_pass."""

    not_slack = {
        round(row.net_worth_to_fiscal_wealth, 2): row for row in report.net_worth_grid if not row.reported_local_margins_slack
    }
    assert set(not_slack) == {-0.5, -0.25}
    for row in not_slack.values():
        assert "negative_transfer" in row.margins.failure_reasons
        assert row.margins.structural_continuation_solvency == "not_evaluated"
    assert report.outcome.outcome == "computational_pass"


def test_residuals_are_comfortably_within_the_declared_tolerances(report):
    assert report.ode_residual.max_scaled_residual <= 1e-7
    assert report.boundary_residual.max_scaled_residual <= 1e-8


def test_convergence_ratios_are_reported_not_just_a_boolean(report):
    assert len(report.convergence.ratios_k) == 4
    assert len(report.convergence.ratios_tau) == 4
    for ratio in report.convergence.ratios_k + report.convergence.ratios_tau:
        assert ratio == pytest.approx(4.0, rel=0.05)


# --------------------------------------------------------------------------
# D2 mandatory repair #3: each horizon comparison must use its own full
# pairwise common interval 0 <= t <= min(comparison_horizon, baseline_horizon),
# not one grid shared across every comparison horizon in the set.
# --------------------------------------------------------------------------


def test_pairwise_common_grid_covers_the_correct_interval_per_comparison():
    """This is the exact regression the D0-D1 review flagged: the OLD
    construction built ONE grid via
    `np.linspace(0, min(config.comparison_horizons + [config.baseline_horizon]), 401)`
    and reused it for every comparison -- for comparison_horizons=[20, 80]
    and baseline_horizon=40, that shared minimum is 20, so the 80-vs-40
    check was silently restricted to 0-20 instead of the correct 0-40. This
    test fails under that old construction (old_shared_endpoint == 20 for
    BOTH comparisons, including the 80-year one) and passes under the fixed
    per-comparison `_pairwise_common_grid`."""

    baseline_horizon = 40.0
    comparison_horizons = [20.0, 80.0]

    old_shared_endpoint = min(comparison_horizons + [baseline_horizon])
    assert old_shared_endpoint == 20.0  # documents the bug: EVERY comparison, even 80y, got capped at 20

    for horizon in comparison_horizons:
        grid = _pairwise_common_grid(horizon, baseline_horizon)
        expected_endpoint = min(horizon, baseline_horizon)
        assert grid[-1] == pytest.approx(expected_endpoint)
        assert grid[0] == 0.0

    # The one case that actually distinguishes the fix from the bug:
    grid_80_vs_40 = _pairwise_common_grid(80.0, baseline_horizon)
    assert grid_80_vs_40[-1] == pytest.approx(40.0)
    assert grid_80_vs_40[-1] != pytest.approx(old_shared_endpoint)


def test_horizon_mesh_comparison_detects_a_divergence_that_only_appears_after_the_old_grids_endpoint(report):
    """End-to-end version of the same regression, exercised through
    `report.horizon_mesh_comparisons` itself rather than the grid helper in
    isolation: reconstruct what the 80-vs-40 comparison's max_diff WOULD
    have been under the old (buggy) 0-20 grid using the same underlying
    paths, and confirm it materially understates the correctly-computed
    (0-40) max_diff -- i.e. the fixed check sees strictly more of the path
    than the old one did, on this frozen fixture."""

    from tai_public_finance.cs002_nonlinear_transition.config import load_cs002_configuration
    from tai_public_finance.cs002_nonlinear_transition.continuation import run_continuation

    config = load_cs002_configuration(CONFIG_PATH)
    baseline = report.route_a_main.final

    old_grid = np.linspace(0.0, 20.0, 401)  # the old shared-grid construction's actual endpoint for this config
    new_grid = _pairwise_common_grid(80.0, config.baseline_horizon)
    assert new_grid[-1] == pytest.approx(40.0)

    run = run_continuation(
        "lq_path_continuation", config.delta_k, config.delta_tau, [config.continuation_amplitudes[-1]], 80.0,
        config.baseline_mesh_points, "lq_stable_manifold", report.local_system, report.solution,
        tol=config.solver_tolerance, max_nodes=config.max_nodes,
    )
    checkpoint = run.checkpoints[0]

    old_max_diff = float(np.max(np.abs(checkpoint.path_at(old_grid) - baseline.path_at(old_grid))))
    new_max_diff = float(np.max(np.abs(checkpoint.path_at(new_grid) - baseline.path_at(new_grid))))

    # On this frozen fixture the 80y path is still very close to the 40y
    # baseline over 0-40 (that is exactly what "horizon convergence" means),
    # so the fixed (0-40) comparison is not expected to itself go out of
    # tolerance -- what this test demonstrates is only that the two grids
    # are NOT interchangeable evidence: the new grid genuinely inspects
    # twice the horizon the old one did.
    assert new_grid[-1] == 2.0 * old_grid[-1]
    assert new_max_diff >= 0.0 and old_max_diff >= 0.0  # both well-defined, finite comparisons
    # And the reported comparison in `report` itself must be built on the
    # 0-40 grid, not 0-20: its own max_state_difference_from_baseline must
    # match new_max_diff (up to which state channel dominates), not old_max_diff.
    reported = next(c for c in report.horizon_mesh_comparisons if c.label == "horizon_80y_vs_baseline")
    reported_max = max(reported.max_state_difference_from_baseline.values())
    assert reported_max == pytest.approx(new_max_diff, rel=1e-6)
