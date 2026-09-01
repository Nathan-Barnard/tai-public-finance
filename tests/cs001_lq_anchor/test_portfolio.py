from __future__ import annotations

import pytest


def test_leading_position_decomposes_merton_and_hedge(baseline):
    portfolio = baseline["portfolio"]
    # s_0_bar = X_bar - (marketed hedge amount); the myopic log-Merton demand
    # is X_bar itself, so the hedge subtracts the fiscal-covarying piece.
    assert portfolio.leading_unconstrained_position == pytest.approx(
        portfolio.comprehensive_resources - portfolio.marketed_fiscal_wealth_amount, abs=1e-12
    )
    assert portfolio.merton_position == pytest.approx(portfolio.comprehensive_resources, abs=1e-12)


def test_portfolio_curvature_is_strictly_negative(baseline):
    assert baseline["portfolio"].portfolio_curvature < 0.0


def test_zero_and_merton_comparators_are_both_feasible_at_baseline(baseline):
    portfolio = baseline["portfolio"]
    assert portfolio.zero_position_feasible
    assert portfolio.merton_comparator_feasible


def test_access_and_hedge_welfare_gains_are_nonnegative(baseline):
    # Both are squared terms over a positive denominator (rho, beta_hat,
    # X_bar^2 > 0), so this holds regardless of the sign of the covariance.
    portfolio = baseline["portfolio"]
    assert portfolio.access_source_gain_q >= 0.0
    assert portfolio.hedge_source_gain_q >= 0.0


def test_net_worth_grid_changes_the_leading_position(baseline):
    from tai_public_finance.cs001_lq_anchor.portfolio import net_worth_grid

    config = baseline["config"]
    scaffolding = config.experiment["numerical_scaffolding"]
    grid = net_worth_grid(
        baseline["local_system"],
        baseline["solution"],
        risky_short_limit=float(scaffolding["risky_short_limit"]),
        safe_debt_limit=float(scaffolding["safe_debt_limit"]),
        risk_scale_epsilon=float(config.experiment["risk_scale_epsilon"]),
        net_worth_to_fiscal_wealth_ratios=[-0.5, 0.0, 0.5],
    )
    positions = [row["leading_unconstrained_position"] for row in grid]
    assert len(set(positions)) == len(positions), "Changing N_bar should change the leading position."
