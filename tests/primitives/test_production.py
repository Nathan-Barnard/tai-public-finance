from __future__ import annotations

from pathlib import Path

import pytest

from tai_public_finance.primitives import load_primitive_parameters
from tai_public_finance.primitives.production import (
    automation_share,
    automation_share_first_derivative,
    evaluate_smooth_branch,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIMITIVE_PATH = REPO_ROOT / "configs" / "primitives" / "lq_farhi_annual_v1.json"


@pytest.fixture(scope="module")
def parameters():
    return load_primitive_parameters(PRIMITIVE_PATH)


def test_automation_share_at_x_mean_matches_the_logistic_midpoint(parameters):
    # alpha_lower=0.19, alpha_upper=0.49 -> midpoint 0.34, matching Farhi's
    # capital-share target.
    assert automation_share(parameters.x_mean, parameters) == pytest.approx(0.34, abs=1e-12)


def test_automation_share_first_derivative_via_finite_difference(parameters):
    step = 1e-6
    analytic = automation_share_first_derivative(parameters.x_mean, parameters)
    numeric = (
        automation_share(parameters.x_mean + step, parameters) - automation_share(parameters.x_mean - step, parameters)
    ) / (2.0 * step)
    assert analytic == pytest.approx(numeric, rel=1e-6)


def test_output_is_positive_and_factor_shares_exhaust_output(parameters):
    state = evaluate_smooth_branch(z=-1.0, x=0.2, capital=1.0, tax_rate=0.5, p=parameters)
    assert state.output > 0.0
    assert state.rental_rate * state.capital + state.wage_income == pytest.approx(state.output, rel=1e-12)


def test_specialisation_margins_are_positive_near_the_calibrated_anchor(parameters):
    state = evaluate_smooth_branch(z=-1.8, x=0.0, capital=1.0, tax_rate=0.5, p=parameters)
    assert state.on_maintained_branch
