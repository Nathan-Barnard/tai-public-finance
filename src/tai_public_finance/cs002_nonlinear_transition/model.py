"""Exact nonlinear model-function and derivative evaluation.

CS002 D0-D1, frozen common states (z, x) = (z_bar, x_bar). Every economic
quantity is read from the one canonical smooth-branch primitive
(`tai_public_finance.primitives.production.evaluate_smooth_branch`); this
module adds only the K-derivatives the CS002 handoff specifies in closed
form and the characteristic-system right-hand side built from them. There is
exactly one place (`evaluate_smooth_branch`) where Y, R^K, W, B, F, g
themselves are defined -- this module never re-derives their levels.

Closed-form K-derivatives at fixed (z, x, tau), re-derived from
Y = exp(z + capital_advantage*alpha) * Omega(alpha) * K**alpha * L**(1-alpha)
(alpha depends only on x, so it is a constant here) and cross-checked against
finite differences of evaluate_smooth_branch in test_model.py:

    Y_K      = alpha * Y / K
    (R^K)_K  = (alpha - 1) * R^K / K
    B_K      = alpha**2 * Y / K - delta
    F_K      = tau * B_K + (1 - alpha) * Y_K
    g_K      = (1 - tau) * (R^K)_K

All derivatives in the costate equation hold the current control nu fixed
(the envelope theorem does the rest); nu itself is just a number computed
from the current state and plugged in, not a quantity these derivatives
differentiate through.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..primitives import PrimitiveParameters, evaluate_smooth_branch, safe_rate
from ..primitives.production import SmoothBranchState

STATE_NAMES = ("k", "tau", "ell", "m")


@dataclass(frozen=True)
class CapitalDerivatives:
    """First derivatives with respect to REAL capital K, at fixed (z, x, tau)."""

    output_K: float
    rental_rate_K: float
    tax_base_K: float
    fiscal_resources_K: float
    capital_growth_K: float


def capital_from_log(k: float, capital_bar: float) -> float:
    return capital_bar * math.exp(k)


def log_from_capital(capital: float, capital_bar: float) -> float:
    if not capital > 0.0:
        raise ValueError(f"log-capital coordinate requires K > 0; got K={capital}.")
    return math.log(capital / capital_bar)


def capital_derivatives(state: SmoothBranchState, p: PrimitiveParameters) -> CapitalDerivatives:
    alpha = state.alpha
    capital = state.capital
    output_K = alpha * state.output / capital
    rental_rate_K = (alpha - 1.0) * state.rental_rate / capital
    tax_base_K = alpha * alpha * state.output / capital - p.depreciation_rate
    fiscal_resources_K = state.tax_rate * tax_base_K + (1.0 - alpha) * output_K
    capital_growth_K = (1.0 - state.tax_rate) * rental_rate_K
    return CapitalDerivatives(
        output_K=output_K,
        rental_rate_K=rental_rate_K,
        tax_base_K=tax_base_K,
        fiscal_resources_K=fiscal_resources_K,
        capital_growth_K=capital_growth_K,
    )


@dataclass(frozen=True)
class CharacteristicRates:
    """One evaluation of the characteristic system at a single point."""

    k_dot: float
    tau_dot: float
    ell_dot: float
    m_dot: float
    nu: float
    r0: float
    state: SmoothBranchState
    derivatives: CapitalDerivatives


def _require_finite(label: str, value: float) -> None:
    if not math.isfinite(value):
        raise FloatingPointError(f"Non-finite {label}={value!r}; the path has left the smooth interior branch.")


def characteristic_rates(
    k: float,
    tau: float,
    ell: float,
    m: float,
    z_bar: float,
    x_bar: float,
    capital_bar: float,
    p: PrimitiveParameters,
) -> CharacteristicRates:
    """Evaluate (k_dot, tau_dot, ell_dot, m_dot) at one point, frozen (z, x)=(z_bar, x_bar).

    kappa_tau is the primitive tax_adjustment_scale directly (the coefficient
    of the quadratic tax-speed adjustment cost Y*nu**2/(2*kappa_tau)); it is
    NOT the LQ-normalised `chi = kappa_tau * K_bar / Y_bar` CS001 uses
    internally for its y-normalised local system.
    """

    capital = capital_from_log(k, capital_bar)
    _require_finite("K", capital)
    state = evaluate_smooth_branch(z_bar, x_bar, capital, tau, p)
    _require_finite("Y", state.output)
    derivs = capital_derivatives(state, p)
    r0 = safe_rate(z_bar, x_bar, z_bar, x_bar, p)
    kappa_tau = p.tax_adjustment_scale
    nu = kappa_tau * m / state.output

    k_dot = state.capital_growth
    tau_dot = nu
    ell_dot = (r0 - state.capital_growth - capital * derivs.capital_growth_K) * ell - (
        derivs.fiscal_resources_K - (nu * nu) * derivs.output_K / (2.0 * kappa_tau)
    )
    m_dot = r0 * m + state.tax_base * (ell - 1.0)

    for label, value in (("k_dot", k_dot), ("tau_dot", tau_dot), ("ell_dot", ell_dot), ("m_dot", m_dot)):
        _require_finite(label, value)

    return CharacteristicRates(
        k_dot=k_dot, tau_dot=tau_dot, ell_dot=ell_dot, m_dot=m_dot, nu=nu, r0=r0, state=state, derivatives=derivs
    )


def characteristic_rhs_vectorized(
    y: np.ndarray, z_bar: float, x_bar: float, capital_bar: float, p: PrimitiveParameters
) -> np.ndarray:
    """scipy.integrate.solve_bvp's `fun(x, y)` signature: y has shape (4, m).

    Loops in pure Python over mesh columns, calling the exact scalar
    primitive evaluation once per column, rather than a numpy reimplementation
    of evaluate_smooth_branch -- this is the one place that function's levels
    are used, and it must stay exactly that function, not a parallel
    vectorised copy that could silently drift from it. Mesh sizes here (tens
    to low thousands of points) make the Python-level loop negligible next to
    the D0/D1 time budgets.
    """

    m = y.shape[1]
    out = np.empty((4, m))
    for col in range(m):
        k, tau, ell, ell_costate_m = y[:, col]
        rates = characteristic_rates(k, tau, ell, ell_costate_m, z_bar, x_bar, capital_bar, p)
        out[0, col] = rates.k_dot
        out[1, col] = rates.tau_dot
        out[2, col] = rates.ell_dot
        out[3, col] = rates.m_dot
    return out
