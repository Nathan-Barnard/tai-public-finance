from __future__ import annotations

import pytest

from tai_public_finance.cs001_lq_anchor.portfolio import (
    leading_portfolio_and_welfare,
    net_worth_grid,
    portfolio_sign_change_net_worth_ratio,
    transfer_boundary_net_worth_ratio,
)


def test_transfer_boundary_matches_the_closed_form_and_the_task_estimate(baseline):
    local_system = baseline["local_system"]
    anchor = local_system.anchor
    rho = local_system.parameters.rho

    ratio = transfer_boundary_net_worth_ratio(local_system)
    net_worth = ratio * anchor.fiscal_wealth_bar
    # rho*(N+J) = W_bar at the boundary, by construction.
    assert rho * (net_worth + anchor.fiscal_wealth_bar) == pytest.approx(anchor.wage_income_bar, abs=1e-10)
    assert ratio == pytest.approx(-0.0796, abs=2e-4)


def test_portfolio_sign_change_matches_the_closed_form_and_the_task_estimate(baseline):
    local_system = baseline["local_system"]
    solution = baseline["solution"]
    anchor = local_system.anchor

    ratio = portfolio_sign_change_net_worth_ratio(local_system, solution)
    net_worth = ratio * anchor.fiscal_wealth_bar
    result = leading_portfolio_and_welfare(
        local_system, solution, risky_short_limit=1e9, safe_debt_limit=1e9, risk_scale_epsilon=1.0, public_net_worth=net_worth
    )
    assert result.leading_unconstrained_position == pytest.approx(0.0, abs=1e-9)
    assert ratio == pytest.approx(-0.0355, abs=2e-4)


def test_sign_flips_across_the_derived_sign_change_point(baseline):
    local_system = baseline["local_system"]
    solution = baseline["solution"]
    ratio = portfolio_sign_change_net_worth_ratio(local_system, solution)
    grid = net_worth_grid(
        local_system, solution, risky_short_limit=1e9, safe_debt_limit=1e9, risk_scale_epsilon=1.0,
        net_worth_to_fiscal_wealth_ratios=[ratio - 0.01, ratio + 0.01],
    )
    assert grid[0]["leading_unconstrained_position"] < 0.0
    assert grid[1]["leading_unconstrained_position"] > 0.0


def test_grid_flags_a_point_below_the_transfer_boundary_as_infeasible_but_retained(baseline):
    local_system = baseline["local_system"]
    solution = baseline["solution"]
    boundary = transfer_boundary_net_worth_ratio(local_system)
    grid = net_worth_grid(
        local_system, solution, risky_short_limit=20.0, safe_debt_limit=20.0, risk_scale_epsilon=1.0,
        net_worth_to_fiscal_wealth_ratios=[boundary - 0.05],
    )
    row = grid[0]
    assert row["transfer_feasible"] is False
    assert row["feasible"] is False
    assert "negative_transfer" in row["failure_reasons"]
    # Retained, not dropped: the row still carries a real, computed position.
    assert row["leading_unconstrained_position"] is not None


def test_every_feasible_labelled_row_actually_satisfies_every_condition(baseline):
    local_system = baseline["local_system"]
    solution = baseline["solution"]
    boundary = transfer_boundary_net_worth_ratio(local_system)
    sign_change = portfolio_sign_change_net_worth_ratio(local_system, solution)
    ratios = [boundary - 0.05, boundary, sign_change, 0.0, 0.25, 0.5]
    grid = net_worth_grid(
        local_system, solution, risky_short_limit=20.0, safe_debt_limit=20.0, risk_scale_epsilon=1.0,
        net_worth_to_fiscal_wealth_ratios=ratios,
    )
    for row in grid:
        if row["feasible"]:
            assert row["comprehensive_resources_positive"] is True
            assert row["transfer_feasible"] is True
            assert row["portfolio_bound_feasible"] is True
            assert row["failure_reasons"] == []
            assert row["worker_consumption"] - row["wage_income"] == pytest.approx(row["transfer"], abs=1e-9)
