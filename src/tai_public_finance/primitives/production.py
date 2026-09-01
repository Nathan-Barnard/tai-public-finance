"""Exact smooth full-specialisation production-branch technology.

Y = exp(z + capital_advantage * alpha) * Omega(alpha) * K**alpha * L**(1-alpha),
alpha = A(x) = alpha_lower + (alpha_upper - alpha_lower) * logistic(x).

Every function here is the *exact* nonlinear primitive, not an LQ
approximation. They are the independent route CS001 requires for checking
the local quadratic system's derivatives (see diagnostics.finite_difference
checks): a matrix entry built from a Taylor expansion of one of these
functions must agree with a finite difference of the function itself.

Valid only on the smooth interior full-specialisation branch; callers must
check that both specialisation_margin_* fields below are positive before
trusting an evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .parameters import PrimitiveParameters


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def automation_share(x: float, p: PrimitiveParameters) -> float:
    return p.alpha_lower + (p.alpha_upper - p.alpha_lower) * logistic(x)


def automation_share_first_derivative(x: float, p: PrimitiveParameters) -> float:
    ell = logistic(x)
    return (p.alpha_upper - p.alpha_lower) * ell * (1.0 - ell)


def automation_share_second_derivative(x: float, p: PrimitiveParameters) -> float:
    ell = logistic(x)
    return (p.alpha_upper - p.alpha_lower) * ell * (1.0 - ell) * (1.0 - 2.0 * ell)


def logit(alpha: float) -> float:
    return math.log(alpha / (1.0 - alpha))


def omega(alpha: float) -> float:
    return math.exp(-alpha * math.log(alpha) - (1.0 - alpha) * math.log(1.0 - alpha))


@dataclass(frozen=True)
class SmoothBranchState:
    z: float
    x: float
    capital: float
    tax_rate: float
    alpha: float
    output: float
    rental_rate: float
    wage_income: float
    tax_base: float
    tax_revenue: float
    fiscal_resources: float
    capital_growth: float
    specialisation_margin_automation_composite: float
    specialisation_margin_new_task_composite: float
    output_automation_semielasticity: float
    wage_automation_semielasticity: float

    @property
    def on_maintained_branch(self) -> bool:
        return (
            self.specialisation_margin_automation_composite > 0.0
            and self.specialisation_margin_new_task_composite > 0.0
        )


def evaluate_smooth_branch(z: float, x: float, capital: float, tax_rate: float, p: PrimitiveParameters) -> SmoothBranchState:
    """Exact nonlinear evaluation at an arbitrary (z, x, K, tau); no LQ approximation."""

    alpha = automation_share(x, p)
    output = math.exp(z + p.capital_advantage * alpha) * omega(alpha) * capital**alpha * p.labour ** (1.0 - alpha)
    rental_rate = alpha * output / capital
    wage_income = (1.0 - alpha) * output
    tax_base = (rental_rate - p.depreciation_rate) * capital
    fiscal_resources = tax_rate * tax_base + wage_income
    capital_growth = (1.0 - tax_rate) * (rental_rate - p.depreciation_rate) - p.rho
    log_capital_labour = math.log(capital / p.labour)
    margin_automation_composite = log_capital_labour - logit(alpha) + p.capital_advantage
    margin_new_task_composite = p.new_task_labour_advantage - log_capital_labour + logit(alpha)
    return SmoothBranchState(
        z=z,
        x=x,
        capital=capital,
        tax_rate=tax_rate,
        alpha=alpha,
        output=output,
        rental_rate=rental_rate,
        wage_income=wage_income,
        tax_base=tax_base,
        tax_revenue=tax_rate * tax_base,
        fiscal_resources=fiscal_resources,
        capital_growth=capital_growth,
        specialisation_margin_automation_composite=margin_automation_composite,
        specialisation_margin_new_task_composite=margin_new_task_composite,
        output_automation_semielasticity=margin_automation_composite,
        wage_automation_semielasticity=margin_automation_composite - 1.0 / (1.0 - alpha),
    )
