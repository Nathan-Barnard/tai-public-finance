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

from tai_public_finance.cs002_nonlinear_transition.config import load_cs002_configuration
from tai_public_finance.cs002_nonlinear_transition.experiment import run_d0_d1_experiment

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
    """N/J in {-0.5, -0.25} must be reported infeasible (negative_transfer) --
    exactly mirroring CS001's own repaired baseline -- without that flipping
    the aggregate outcome away from computational_pass."""

    infeasible = {round(row.net_worth_to_fiscal_wealth, 2): row for row in report.net_worth_grid if not row.feasible}
    assert set(infeasible) == {-0.5, -0.25}
    for row in infeasible.values():
        assert "negative_transfer" in row.margins.failure_reasons
    assert report.outcome.outcome == "computational_pass"


def test_residuals_are_comfortably_within_the_declared_tolerances(report):
    assert report.ode_residual.max_scaled_residual <= 1e-7
    assert report.boundary_residual.max_scaled_residual <= 1e-8


def test_convergence_ratios_are_reported_not_just_a_boolean(report):
    assert len(report.convergence.ratios_k) == 4
    assert len(report.convergence.ratios_tau) == 4
    for ratio in report.convergence.ratios_k + report.convergence.ratios_tau:
        assert ratio == pytest.approx(4.0, rel=0.05)
