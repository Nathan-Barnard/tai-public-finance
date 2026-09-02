from __future__ import annotations

import pytest

from tai_public_finance.cs002_nonlinear_transition.outcome import AggregateOutcome, determine_outcome


def test_all_checks_pass_gives_computational_pass():
    checks = {"a": True, "b": True}
    result = determine_outcome(checks)
    assert result.outcome == "computational_pass"
    assert result.failed_checks == []


def test_derivation_blocked_takes_priority_over_everything():
    checks = {"ode_residual": False, "boundary_margin": False}
    result = determine_outcome(
        checks,
        derivation_blocked=True,
        derivation_blocked_reason="stable-tail mapping unresolved",
        numerical_failure_check_names=("ode_residual",),
        boundary_check_names=("boundary_margin",),
    )
    assert result.outcome == "derivation_blocked"
    assert "stable-tail mapping unresolved" in result.conclusion


def test_numerical_failure_beats_branch_sensitivity_and_boundary():
    checks = {"ode_residual": False, "branch_agreement": False, "transfer_margin": False}
    result = determine_outcome(
        checks,
        numerical_failure_check_names=("ode_residual",),
        branch_sensitivity_check_names=("branch_agreement",),
        boundary_check_names=("transfer_margin",),
    )
    assert result.outcome == "numerical_failure"


def test_branch_sensitivity_beats_boundary_reaching():
    checks = {"branch_agreement": False, "transfer_margin": False}
    result = determine_outcome(
        checks, numerical_failure_check_names=("ode_residual",), branch_sensitivity_check_names=("branch_agreement",), boundary_check_names=("transfer_margin",)
    )
    assert result.outcome == "branch_sensitive"


def test_boundary_reaching_alone():
    checks = {"transfer_margin": False, "ode_residual": True}
    result = determine_outcome(
        checks, numerical_failure_check_names=("ode_residual",), branch_sensitivity_check_names=("branch_agreement",), boundary_check_names=("transfer_margin",)
    )
    assert result.outcome == "boundary_reaching"


def test_uncategorised_failure_is_conservatively_a_numerical_failure():
    checks = {"some_new_check": False}
    result = determine_outcome(checks)
    assert result.outcome == "numerical_failure"


def test_invalid_outcome_string_is_rejected():
    with pytest.raises(ValueError):
        AggregateOutcome(outcome="not_a_real_outcome", checks={})
