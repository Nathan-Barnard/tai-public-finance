"""International pricing primitives: safe rate, consumption loading, price of risk.

These implement the specialized international closure (sigma_I = lambda_I,
beta_I = ||lambda_I||^2) documented in EV10 as a price-taking small-open-economy
convention: the domestic government is infinitesimal relative to a residual
foreign counterparty, so its trades do not move sigma_I or lambda_I.
"""

from __future__ import annotations

from dataclasses import dataclass

from .parameters import PrimitiveParameters
from .production import automation_share, automation_share_first_derivative, automation_share_second_derivative, logit


def international_consumption_loading(alpha: float, p: PrimitiveParameters) -> float:
    """h_I(alpha) = 1/alpha + k_I - logit(alpha) + capital_advantage."""

    return 1.0 / alpha + p.international_log_capital_labour_ratio - logit(alpha) + p.capital_advantage


def international_consumption_loading_derivative(alpha: float) -> float:
    """d h_I / d alpha = -1/alpha^2 - 1/(alpha(1-alpha)); independent of k_I, capital_advantage."""

    return -1.0 / alpha**2 - 1.0 / (alpha * (1.0 - alpha))


def automation_state_consumption_loading(x: float, p: PrimitiveParameters) -> float:
    """ell_x(x) = A'(x) h_I(A(x)): the local exposure of international consumption to x."""

    alpha = automation_share(x, p)
    a = automation_share_first_derivative(x, p)
    return a * international_consumption_loading(alpha, p)


def automation_state_consumption_loading_second_derivative(x: float, p: PrimitiveParameters) -> float:
    """ell_xx(x) = A''(x) h_I(A(x)) + A'(x)^2 h_{I,alpha}(A(x))."""

    alpha = automation_share(x, p)
    a = automation_share_first_derivative(x, p)
    q = automation_share_second_derivative(x, p)
    h_i = international_consumption_loading(alpha, p)
    h_i_alpha = international_consumption_loading_derivative(alpha)
    return q * h_i + a * a * h_i_alpha


def safe_rate(z: float, x: float, z_bar: float, x_bar: float, p: PrimitiveParameters) -> float:
    """Exact zero-risk international safe rate r_0(z, x); r_0(z_bar, x_bar) = rho."""

    ell_x = automation_state_consumption_loading(x, p)
    return p.rho + p.kappa_z * (z_bar - z) + p.kappa_x * (x_bar - x) * ell_x


@dataclass(frozen=True)
class LeadingPriceOfRisk:
    lambda_hat: tuple[float, float]
    beta_hat: float


def leading_price_of_risk(sigma_z_hat: float, sigma_x_hat: float, ell_x_bar: float) -> LeadingPriceOfRisk:
    """lambda_hat = (sigma_z_hat, ell_x_bar * sigma_x_hat), beta_hat = ||lambda_hat||^2."""

    lambda_hat = (sigma_z_hat, ell_x_bar * sigma_x_hat)
    beta_hat = lambda_hat[0] ** 2 + lambda_hat[1] ** 2
    return LeadingPriceOfRisk(lambda_hat=lambda_hat, beta_hat=beta_hat)
