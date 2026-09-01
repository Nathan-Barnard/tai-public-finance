from __future__ import annotations

from tai_public_finance.cs001_lq_anchor.sweep import chi_star, sweep_one_parameter


def test_chi_star_matches_the_closed_form_at_the_baseline_anchor(baseline):
    anchor = baseline["local_system"].anchor
    rho = baseline["local_system"].parameters.rho
    expected = anchor.gamma * (anchor.gamma + rho) ** 2 / (32.0 * rho**2)
    assert chi_star(anchor.gamma, rho) == expected


def test_tax_adjustment_scale_sweep_crosses_the_oscillation_threshold(baseline):
    config = baseline["config"]
    scaffolding = config.experiment["numerical_scaffolding"]
    reporting = config.experiment["reporting"]
    tolerances = config.experiment["acceptance_tolerances"]

    rows = sweep_one_parameter(
        base_parameters=config.parameters,
        parameter_path=("parameters", "tax_adjustment_scale"),
        values=[0.001, 0.5, 2.0],
        scaffolding=scaffolding,
        reporting=reporting,
        risk_scale_epsilon=float(config.experiment["risk_scale_epsilon"]),
        acceptance_tolerances=tolerances,
    )
    assert len(rows) == 3
    for row in rows:
        assert row["outcome"] == "pass", row["failed_checks"]

    by_value = {row["value"]: row for row in rows}
    # Small kappa_tau (expensive adjustment) -> below chi_star -> real roots.
    assert by_value[0.001]["oscillatory"] is False
    # The baseline calibration -> above chi_star -> a complex-conjugate pair.
    assert by_value[0.5]["oscillatory"] is True
    # chi is proportional to tax_adjustment_scale at fixed K_bar/Y_bar, and
    # chi_star itself does not depend on tax_adjustment_scale at all.
    assert by_value[0.001]["chi"] < by_value[0.001]["chi_star"]
    assert by_value[0.5]["chi"] > by_value[0.5]["chi_star"]
    assert by_value[0.001]["chi_star"] == by_value[2.0]["chi_star"]


def test_sweep_retains_a_failing_point_instead_of_crashing(baseline):
    config = baseline["config"]
    scaffolding = config.experiment["numerical_scaffolding"]
    reporting = config.experiment["reporting"]
    tolerances = config.experiment["acceptance_tolerances"]

    # automation_persistence_annual must lie in (0, 1); 1.5 is invalid and
    # must come back as a retained failed row, not raise out of the sweep.
    rows = sweep_one_parameter(
        base_parameters=config.parameters,
        parameter_path=("parameters", "automation_persistence_annual"),
        values=[0.81, 1.5],
        scaffolding=scaffolding,
        reporting=reporting,
        risk_scale_epsilon=float(config.experiment["risk_scale_epsilon"]),
        acceptance_tolerances=tolerances,
    )
    assert len(rows) == 2
    assert rows[0]["outcome"] == "pass"
    assert rows[1]["outcome"] == "error"
    assert rows[1]["failed_checks"]
