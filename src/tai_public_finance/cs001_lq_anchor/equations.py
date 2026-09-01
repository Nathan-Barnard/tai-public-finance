"""The primitive-to-matrix map: the only place A, B, D, Q, and j are built.

Coordinates are y = (z - z_bar, x - x_bar, k, t) with k = log(K/K_bar) and
t = tau - 1/2 (COORDINATES fixes this order everywhere downstream). Every
entry below is derived directly from the primitive log-gradients and
log-Hessians of Y, R^K, and W recorded in the local LQ system and
computation plan; nothing here is a hand-entered reduced-form coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..primitives import PrimitiveParameters, leading_price_of_risk
from ..primitives.international_pricing import LeadingPriceOfRisk
from .anchor import SteadyState

COORDINATES = ("z", "x", "k", "tau_deviation")


def _log_hessian(kind: str, alpha: float, a: float, q: float, eta: float) -> np.ndarray:
    """The only nonzero entries (besides symmetry) of the log-Hessian of Y, R^K, or W."""

    hessian = np.zeros((4, 4))
    hessian[1, 2] = hessian[2, 1] = a
    if kind == "output":
        hessian[1, 1] = q * eta - a * a / (alpha * (1.0 - alpha))
    elif kind == "rental":
        hessian[1, 1] = q * (eta + 1.0 / alpha) - a * a / (alpha**2 * (1.0 - alpha))
    elif kind == "wage":
        hessian[1, 1] = q * (eta - 1.0 / (1.0 - alpha)) - a * a / (alpha * (1.0 - alpha) ** 2)
    else:
        raise ValueError(f"Unknown log-Hessian kind: {kind}")
    return hessian


@dataclass(frozen=True)
class LocalSystem:
    parameters: PrimitiveParameters
    anchor: SteadyState

    output_y: np.ndarray
    output_yy: np.ndarray
    rental_y: np.ndarray
    rental_yy: np.ndarray
    wage_y: np.ndarray
    wage_yy: np.ndarray
    tax_base_normalized_y: np.ndarray  # d(B/K_bar)
    tax_base_normalized_yy: np.ndarray
    fiscal_resources_normalized_y: np.ndarray  # d(F/K_bar)
    fiscal_resources_normalized_yy: np.ndarray
    capital_growth_yy: np.ndarray  # g_yy
    safe_rate_y: np.ndarray  # d_r
    safe_rate_yy: np.ndarray  # r_{0,yy}

    A: np.ndarray
    B: np.ndarray
    D: np.ndarray
    Q: np.ndarray
    Q_rr_closed_form: np.ndarray
    Sigma_e_hat: np.ndarray

    linear_fiscal_wealth: np.ndarray  # j
    price_of_risk: LeadingPriceOfRisk


def build_local_system(p: PrimitiveParameters, anchor: SteadyState) -> LocalSystem:
    alpha = anchor.alpha_bar
    a = anchor.alpha_x_bar
    q = anchor.alpha_xx_bar
    eta = anchor.eta_output_alpha
    rho = p.rho
    capital_bar = anchor.capital_bar
    rental_bar = anchor.rental_rate_bar
    output_bar = anchor.output_bar
    wage_bar = anchor.wage_income_bar

    u_output = np.array([1.0, a * eta, alpha, 0.0])
    u_rental = np.array([1.0, a * (eta + 1.0 / alpha), alpha - 1.0, 0.0])
    u_wage = np.array([1.0, a * (eta - 1.0 / (1.0 - alpha)), alpha, 0.0])
    m_output = _log_hessian("output", alpha, a, q, eta)
    m_rental = _log_hessian("rental", alpha, a, q, eta)
    m_wage = _log_hessian("wage", alpha, a, q, eta)

    output_y = output_bar * u_output
    rental_y = rental_bar * u_rental
    wage_y = wage_bar * u_wage
    output_yy = output_bar * (m_output + np.outer(u_output, u_output))
    rental_yy = rental_bar * (m_rental + np.outer(u_rental, u_rental))
    wage_yy = wage_bar * (m_wage + np.outer(u_wage, u_wage))

    e_k = np.array([0.0, 0.0, 1.0, 0.0])
    e_t = np.array([0.0, 0.0, 0.0, 1.0])

    # Normalized net-rental tax base b~ = B/K_bar = (R^K - delta) * (K/K_bar); its
    # gradient/Hessian in y follow from R^K(y) alone since e_k contributes the
    # log-capital direction of B/K_bar = (R^K-delta)*exp(k) linearised at k=0.
    tax_base_normalized_y = rental_y + (rental_bar - p.depreciation_rate) * e_k
    tax_base_normalized_yy = (
        rental_yy
        + np.outer(e_k, rental_y)
        + np.outer(rental_y, e_k)
        + (rental_bar - p.depreciation_rate) * np.outer(e_k, e_k)
    )

    # f = F/K_bar = tau * b~ + W/K_bar, tau = 1/2 + t; expand the tau(y)*b~(y)
    # product around the anchor (tau=1/2, b~=R_bar-delta).
    fiscal_resources_normalized_y = (
        0.5 * tax_base_normalized_y + wage_y / capital_bar + (rental_bar - p.depreciation_rate) * e_t
    )
    fiscal_resources_normalized_yy = (
        0.5 * tax_base_normalized_yy
        + wage_yy / capital_bar
        + np.outer(e_t, tax_base_normalized_y)
        + np.outer(tax_base_normalized_y, e_t)
    )

    # g = (1-tau)(R^K-delta) - rho is capital's percentage drift; its Hessian is
    # the only nonlinear contribution to Q from the state laws (j_k=1, all other
    # drifts are linear in y).
    capital_growth_yy = 0.5 * rental_yy - np.outer(e_t, rental_y) - np.outer(rental_y, e_t)

    rental_x = rental_y[1]
    A = np.array(
        [
            [-p.kappa_z, 0.0, 0.0, 0.0],
            [0.0, -p.kappa_x, 0.0, 0.0],
            [rental_bar / 2.0, rental_x / 2.0, -anchor.gamma, -2.0 * rho],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    B = np.array([0.0, 0.0, 0.0, 1.0])
    D = A[2:, :2].copy()

    safe_rate_y = np.array([-p.kappa_z, -p.kappa_x * anchor.ell_x_bar, 0.0, 0.0])
    safe_rate_yy = np.zeros((4, 4))
    safe_rate_yy[1, 1] = -2.0 * p.kappa_x * anchor.ell_xx_bar

    linear_fiscal_wealth = np.linalg.solve(
        rho * np.eye(4) - A.T,
        fiscal_resources_normalized_y - anchor.fiscal_wealth_normalized_bar * safe_rate_y,
    )

    Q = (
        fiscal_resources_normalized_yy
        + capital_growth_yy
        - (np.outer(safe_rate_y, linear_fiscal_wealth) + np.outer(linear_fiscal_wealth, safe_rate_y))
        - anchor.fiscal_wealth_normalized_bar * safe_rate_yy
    )
    Q_rr_closed_form = np.array([[rho, 2.0 * rho], [2.0 * rho, 0.0]])

    Sigma_e_hat = np.array(
        [
            [p.sigma_z_hat, 0.0],
            [0.0, p.sigma_x_hat],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    price_of_risk = leading_price_of_risk(p.sigma_z_hat, p.sigma_x_hat, anchor.ell_x_bar)

    return LocalSystem(
        parameters=p,
        anchor=anchor,
        output_y=output_y,
        output_yy=output_yy,
        rental_y=rental_y,
        rental_yy=rental_yy,
        wage_y=wage_y,
        wage_yy=wage_yy,
        tax_base_normalized_y=tax_base_normalized_y,
        tax_base_normalized_yy=tax_base_normalized_yy,
        fiscal_resources_normalized_y=fiscal_resources_normalized_y,
        fiscal_resources_normalized_yy=fiscal_resources_normalized_yy,
        capital_growth_yy=capital_growth_yy,
        safe_rate_y=safe_rate_y,
        safe_rate_yy=safe_rate_yy,
        A=A,
        B=B,
        D=D,
        Q=Q,
        Q_rr_closed_form=Q_rr_closed_form,
        Sigma_e_hat=Sigma_e_hat,
        linear_fiscal_wealth=linear_fiscal_wealth,
        price_of_risk=price_of_risk,
    )
