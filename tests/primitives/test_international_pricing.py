from __future__ import annotations

from pathlib import Path

import pytest

from tai_public_finance.primitives import load_primitive_parameters
from tai_public_finance.primitives.international_pricing import (
    automation_state_consumption_loading,
    leading_price_of_risk,
    safe_rate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIMITIVE_PATH = REPO_ROOT / "configs" / "primitives" / "lq_farhi_annual_v1.json"


@pytest.fixture(scope="module")
def parameters():
    return load_primitive_parameters(PRIMITIVE_PATH)


def test_safe_rate_equals_rho_at_the_deterministic_mean(parameters):
    z_bar, x_bar = -1.8066165916, 0.0
    assert safe_rate(z_bar, x_bar, z_bar, x_bar, parameters) == pytest.approx(parameters.rho, abs=1e-12)


def test_leading_price_of_risk_uses_the_automation_state_loading(parameters):
    ell_x = automation_state_consumption_loading(0.0, parameters)
    result = leading_price_of_risk(parameters.sigma_z_hat, parameters.sigma_x_hat, ell_x)
    assert result.lambda_hat[0] == pytest.approx(parameters.sigma_z_hat)
    assert result.lambda_hat[1] == pytest.approx(ell_x * parameters.sigma_x_hat)
    assert result.beta_hat == pytest.approx(result.lambda_hat[0] ** 2 + result.lambda_hat[1] ** 2)
    assert result.beta_hat > 0.0
