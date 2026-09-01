from __future__ import annotations

import numpy as np
import pytest

from tai_public_finance.cs001_lq_anchor.irfs import build_experiments


def test_matrix_exponential_and_direct_ode_propagation_agree(baseline):
    assert baseline["irfs"]["max_matrix_exponential_vs_ode_relative_error"] < 1e-8


def test_discounted_cumulative_response_solves_the_resolvent_identity(baseline):
    for key, entry in baseline["irfs"]["discounted_cumulative"].items():
        assert entry["resolvent_identity_residual"] < 1e-8, key


def test_first_order_budget_identity_holds_on_every_reported_row(baseline):
    assert baseline["irfs"]["max_first_order_budget_residual"] < 1e-8


def test_brownian_innovation_and_ou_state_displacement_are_distinct_normalizations(baseline):
    experiments = {item.name: item for item in baseline["irfs"]["experiments"]}
    brownian = experiments["brownian_automation_1sd_short_window"]
    ou = experiments["ou_automation_conditional_sd_state_displacement"]
    assert brownian.inherited_portfolio_pays
    assert brownian.brownian_increment is not None
    assert not ou.inherited_portfolio_pays
    assert ou.brownian_increment is None
    assert brownian.initial_y[1] != pytest.approx(ou.initial_y[1])


def test_capital_and_inherited_tax_cannot_jump_on_impact(baseline):
    for experiment in baseline["irfs"]["experiments"]:
        if experiment.family in ("primitive_brownian_innovation", "finite_window_ou_state_displacement"):
            assert experiment.initial_y[2] == 0.0, experiment.name
            assert experiment.initial_y[3] == 0.0, experiment.name


def test_inherited_portfolio_pays_only_in_the_full_access_regime(baseline):
    checks = baseline["irfs"]["cross_checks"]
    for shock in ("brownian_productivity_1sd_short_window", "brownian_automation_1sd_short_window"):
        full = checks[f"{shock}::full_access"]
        restricted = checks[f"{shock}::no_external_claim"]
        assert restricted["initial_public_net_worth_payoff"] == 0.0
        assert full["initial_public_net_worth_payoff"] != 0.0


def test_post_shock_position_regime_does_not_exist_for_non_stochastic_experiments(baseline):
    rows = baseline["irfs"]["rows"]
    regimes_by_experiment: dict[str, set[str]] = {}
    for row in rows:
        regimes_by_experiment.setdefault(row["experiment"], set()).add(row["regime"])
    for experiment in baseline["irfs"]["experiments"]:
        if not experiment.inherited_portfolio_pays:
            assert regimes_by_experiment[experiment.name] == {"full_access"}


def test_reported_paths_remain_on_the_maintained_branch_and_inside_scaffolding(baseline):
    boundary = baseline["irfs"]["boundary_summary"]
    for field in boundary:
        assert boundary[field] > 0.0, field


def test_consumption_equals_rho_times_comprehensive_resources_on_every_row(baseline):
    # c = rho * X at leading order holds identically across every experiment
    # and regime; this is exactly the invariant that a change of level/
    # deviation convention or shock normalization in the rendering layer
    # would silently break for some rows but not others.
    p = baseline["local_system"].parameters
    for row in baseline["irfs"]["rows"]:
        expected = p.rho * row["comprehensive_resources_deviation"]
        assert row["worker_consumption_deviation"] == pytest.approx(expected, abs=1e-10), row["experiment"]


def test_construction_of_experiments_is_deterministic(baseline):
    a = build_experiments(baseline["local_system"], baseline["solution"], baseline["config"].experiment["reporting"])
    b = build_experiments(baseline["local_system"], baseline["solution"], baseline["config"].experiment["reporting"])
    assert [item.name for item in a] == [item.name for item in b]
    for left, right in zip(a, b, strict=True):
        np.testing.assert_array_equal(left.initial_y, right.initial_y)
