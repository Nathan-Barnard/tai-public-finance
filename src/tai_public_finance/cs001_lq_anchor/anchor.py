"""The illustrative deterministic interior steady state.

At the deterministic mean, the frictionless common-discount interior
benchmark gives R_bar^K - delta = 2*rho and tau_bar = 1/2 (splitting the net
rental return equally between the domestic owner and the government). z_bar
is then the unique value consistent with the K_bar = 1 normalization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..primitives import PrimitiveParameters
from ..primitives.international_pricing import (
    automation_state_consumption_loading,
    automation_state_consumption_loading_second_derivative,
    international_consumption_loading,
)
from ..primitives.production import (
    automation_share,
    automation_share_first_derivative,
    automation_share_second_derivative,
    logit,
    omega,
)


@dataclass(frozen=True)
class SteadyState:
    z_bar: float
    x_bar: float
    capital_bar: float
    public_net_worth_bar: float
    tax_rate_bar: float

    alpha_bar: float
    alpha_x_bar: float  # A'(x_bar)
    alpha_xx_bar: float  # A''(x_bar)
    rental_rate_bar: float
    output_bar: float
    wage_income_bar: float
    tax_base_bar: float
    fiscal_resources_bar: float

    fiscal_wealth_normalized_bar: float  # J_bar / K_bar
    fiscal_wealth_bar: float  # J_bar
    comprehensive_resources_bar: float  # X_bar = N_bar + J_bar
    worker_consumption_bar: float  # c_bar = rho * X_bar
    transfer_bar: float  # T_bar = c_bar - W_bar

    eta_output_alpha: float  # log(K_bar/L) - logit(alpha_bar) + capital_advantage
    h_international_bar: float  # h_I(alpha_bar)
    ell_x_bar: float  # A'(x_bar) h_I(alpha_bar)
    ell_xx_bar: float
    gamma: float  # (1 - alpha_bar)/2 * rental_rate_bar; local capital mean-reversion rate
    chi: float  # kappa_tau * capital_bar / output_bar

    specialisation_margin_automation_composite: float
    specialisation_margin_new_task_composite: float


def compute_steady_state(p: PrimitiveParameters) -> SteadyState:
    rho = p.rho
    labour = p.labour
    capital_bar = p.capital_anchor
    net_worth_bar = p.public_net_worth_anchor

    alpha_bar = automation_share(p.x_mean, p)
    a = automation_share_first_derivative(p.x_mean, p)
    q = automation_share_second_derivative(p.x_mean, p)

    rental_rate_bar = p.depreciation_rate + 2.0 * rho
    log_capital_labour = math.log(capital_bar / labour)
    z_bar = (
        math.log(rental_rate_bar)
        - math.log(alpha_bar)
        - p.capital_advantage * alpha_bar
        - math.log(omega(alpha_bar))
        + (1.0 - alpha_bar) * log_capital_labour
    )
    output_bar = rental_rate_bar * capital_bar / alpha_bar
    wage_income_bar = (1.0 - alpha_bar) * output_bar
    tax_rate_bar = 0.5
    tax_base_bar = (rental_rate_bar - p.depreciation_rate) * capital_bar
    fiscal_resources_bar = tax_rate_bar * tax_base_bar + wage_income_bar

    fiscal_wealth_normalized_bar = 1.0 + wage_income_bar / (rho * capital_bar)
    fiscal_wealth_bar = capital_bar * fiscal_wealth_normalized_bar
    comprehensive_resources_bar = net_worth_bar + fiscal_wealth_bar
    worker_consumption_bar = rho * comprehensive_resources_bar
    transfer_bar = worker_consumption_bar - wage_income_bar

    eta_output_alpha = log_capital_labour - logit(alpha_bar) + p.capital_advantage
    h_international_bar = international_consumption_loading(alpha_bar, p)
    ell_x_bar = automation_state_consumption_loading(p.x_mean, p)
    ell_xx_bar = automation_state_consumption_loading_second_derivative(p.x_mean, p)
    gamma = (1.0 - alpha_bar) / 2.0 * rental_rate_bar
    chi = p.tax_adjustment_scale * capital_bar / output_bar

    margin_automation_composite = log_capital_labour - logit(alpha_bar) + p.capital_advantage
    margin_new_task_composite = p.new_task_labour_advantage - log_capital_labour + logit(alpha_bar)

    return SteadyState(
        z_bar=z_bar,
        x_bar=p.x_mean,
        capital_bar=capital_bar,
        public_net_worth_bar=net_worth_bar,
        tax_rate_bar=tax_rate_bar,
        alpha_bar=alpha_bar,
        alpha_x_bar=a,
        alpha_xx_bar=q,
        rental_rate_bar=rental_rate_bar,
        output_bar=output_bar,
        wage_income_bar=wage_income_bar,
        tax_base_bar=tax_base_bar,
        fiscal_resources_bar=fiscal_resources_bar,
        fiscal_wealth_normalized_bar=fiscal_wealth_normalized_bar,
        fiscal_wealth_bar=fiscal_wealth_bar,
        comprehensive_resources_bar=comprehensive_resources_bar,
        worker_consumption_bar=worker_consumption_bar,
        transfer_bar=transfer_bar,
        eta_output_alpha=eta_output_alpha,
        h_international_bar=h_international_bar,
        ell_x_bar=ell_x_bar,
        ell_xx_bar=ell_xx_bar,
        gamma=gamma,
        chi=chi,
        specialisation_margin_automation_composite=margin_automation_composite,
        specialisation_margin_new_task_composite=margin_new_task_composite,
    )
