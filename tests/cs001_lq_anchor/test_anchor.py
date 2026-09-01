from __future__ import annotations

import pytest

from tai_public_finance.primitives.production import evaluate_smooth_branch


def test_capital_tax_split_the_net_rental_return_equally(baseline):
    p = baseline["local_system"].parameters
    anchor = baseline["anchor"]
    assert anchor.rental_rate_bar - p.depreciation_rate == pytest.approx(2.0 * p.rho, abs=1e-13)
    assert anchor.tax_rate_bar == pytest.approx(0.5, abs=1e-14)


def test_capital_law_is_stationary_at_the_anchor(baseline):
    anchor = baseline["anchor"]
    p = baseline["local_system"].parameters
    exact = evaluate_smooth_branch(anchor.z_bar, anchor.x_bar, anchor.capital_bar, anchor.tax_rate_bar, p)
    assert exact.capital_growth == pytest.approx(0.0, abs=1e-13)


def test_fiscal_resources_equal_worker_consumption_at_n_bar_zero(baseline):
    anchor = baseline["anchor"]
    assert anchor.public_net_worth_bar == 0.0
    assert anchor.fiscal_resources_bar == pytest.approx(anchor.worker_consumption_bar, abs=1e-12)


def test_anchor_is_strictly_inside_the_maintained_production_branch(baseline):
    anchor = baseline["anchor"]
    assert anchor.specialisation_margin_automation_composite > 0.0
    assert anchor.specialisation_margin_new_task_composite > 0.0


def test_comprehensive_resources_equal_fiscal_wealth_when_n_bar_zero(baseline):
    anchor = baseline["anchor"]
    assert anchor.comprehensive_resources_bar == pytest.approx(anchor.fiscal_wealth_bar, abs=1e-13)


def test_capital_share_anchor_matches_farhi_calibration_target(baseline):
    # alpha_bar = A(x_mean) is the logistic midpoint by construction, which the
    # primitive table's alpha_lower/alpha_upper were chosen to place at 0.34.
    anchor = baseline["anchor"]
    assert anchor.alpha_bar == pytest.approx(0.34, abs=1e-12)
