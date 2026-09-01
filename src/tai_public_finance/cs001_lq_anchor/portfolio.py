"""Leading small-risk portfolio and order-two welfare objects.

Everything here is algebraic at the deterministic anchor: it needs only the
linear fiscal-wealth coefficient j and the quadratic solution H, not any
higher derivative of the nonlinear deterministic fiscal-wealth function.
Exact precautionary consumption/tax-speed corrections are out of scope
(Stage 2B in the local LQ system and computation plan) and are not computed
here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .anchor import SteadyState
from .equations import LocalSystem
from .solver import LqSolution


@dataclass(frozen=True)
class LeadingPortfolio:
    public_net_worth: float
    fiscal_wealth: float
    comprehensive_resources: float
    risk_scale_epsilon: float

    lambda_hat: np.ndarray
    beta_hat: float
    zeta_j: np.ndarray  # fiscal-wealth shock loading at this (N, J)
    zeta_j_perp: np.ndarray  # component orthogonal to the traded payoff
    marketed_fiscal_wealth_amount: float  # zeta_j . lambda_hat / beta_hat

    return_demand_component: float  # ordinary log-Merton demand = X
    fiscal_hedge_component: float  # -zeta_j.lambda_hat/beta_hat; NEGATIVE here (a short/reduced
    # position against the claim) because fiscal wealth already covaries positively with its
    # payoff -- this is a reduction of exposure, not "holding more of the claim".
    leading_unconstrained_position: float  # s_0_bar = return_demand_component + fiscal_hedge_component
    leading_constrained_position: float
    portfolio_curvature: float  # -beta_hat / (rho X^2), strictly negative
    portfolio_gradient_y: np.ndarray  # d s_0_bar(y) / dy at the anchor
    portfolio_bounds: tuple[float, float]
    portfolio_lower_slack: float
    portfolio_upper_slack: float
    zero_position_feasible: bool
    merton_position: float  # the myopic log-Merton comparator s = X
    merton_comparator_feasible: bool

    access_source_gain_q: float  # Q_access: leading-order value of optimal s vs s=0, same state
    hedge_source_gain_q: float  # Q_hedge: leading-order LOSS AVOIDED by holding s_0_bar instead of
    # being forced into the myopic Merton position s=X -- NOT an additional gain stacked on top of
    # access_source_gain_q; the two compare the optimum against two different baselines and do not add.
    access_value_leading: float  # epsilon^2 Q_access / rho
    hedge_value_leading: float
    access_consumption_equivalent_leading: float  # epsilon^2 Q_access
    hedge_consumption_equivalent_leading: float

    varpi_ee: np.ndarray  # opportunity-value Hessian restricted to (z, x)
    safe_rate_order_two_anchor: float  # r_2_bar
    g2_anchor: float
    v2_anchor: float  # V^2(S_bar) = G2 / rho


def leading_portfolio_and_welfare(
    local_system: LocalSystem,
    solution: LqSolution,
    risky_short_limit: float,
    safe_debt_limit: float,
    risk_scale_epsilon: float,
    public_net_worth: float | None = None,
) -> LeadingPortfolio:
    p = local_system.parameters
    anchor: SteadyState = local_system.anchor
    rho = p.rho
    capital_bar = anchor.capital_bar

    net_worth = anchor.public_net_worth_bar if public_net_worth is None else float(public_net_worth)
    fiscal_wealth = anchor.fiscal_wealth_bar
    comprehensive = net_worth + fiscal_wealth
    if comprehensive <= 0.0:
        raise ValueError("The leading portfolio requires positive comprehensive resources X = N + J.")

    lambda_hat = np.array(local_system.price_of_risk.lambda_hat)
    beta_hat = local_system.price_of_risk.beta_hat
    if beta_hat <= 0.0:
        raise ValueError("The leading portfolio requires a nondegenerate traded payoff (beta_hat > 0).")

    j = local_system.linear_fiscal_wealth
    zeta_j = capital_bar * np.array([p.sigma_z_hat * j[0], p.sigma_x_hat * j[1]])
    marketed_amount = float(zeta_j @ lambda_hat / beta_hat)
    return_demand_component = comprehensive
    fiscal_hedge_component = -marketed_amount
    leading_position = return_demand_component + fiscal_hedge_component
    zeta_j_perp = zeta_j - marketed_amount * lambda_hat
    portfolio_curvature = -beta_hat / (rho * comprehensive**2)

    # d(zeta_J)/dy and d(lambda_hat)/dy at the anchor, using the local quadratic
    # gradient of fiscal wealth (j + H y) and the exact x-dependence of ell_x.
    H = solution.H
    zeta_gradient = capital_bar * np.vstack([p.sigma_z_hat * H[0, :], p.sigma_x_hat * H[1, :]])
    lambda_gradient = np.zeros((2, 4))
    lambda_gradient[1, 1] = anchor.ell_xx_bar * p.sigma_x_hat
    covariance = float(zeta_j @ lambda_hat)
    hedge_ratio_gradient = np.zeros(4)
    for index in range(4):
        zeta_i = zeta_gradient[:, index]
        lambda_i = lambda_gradient[:, index]
        hedge_ratio_gradient[index] = (
            (float(zeta_i @ lambda_hat) + float(zeta_j @ lambda_i)) / beta_hat
            - 2.0 * covariance * float(lambda_hat @ lambda_i) / beta_hat**2
        )
    portfolio_gradient_y = -hedge_ratio_gradient

    lower_bound = -risky_short_limit
    upper_bound = net_worth + safe_debt_limit
    constrained_position = min(max(leading_position, lower_bound), upper_bound)
    merton_position = comprehensive

    access_source_gain = (beta_hat * comprehensive - covariance) ** 2 / (2.0 * rho * beta_hat * comprehensive**2)
    hedge_source_gain = covariance**2 / (2.0 * rho * beta_hat * comprehensive**2)

    varpi_ee = np.zeros((2, 2))
    varpi_ee[1, 1] = -2.0 * p.kappa_x * anchor.ell_xx_bar / (rho * (rho + 2.0 * p.kappa_x))
    r2 = p.sigma_x_hat**2 * anchor.ell_xx_bar / 2.0 - beta_hat / 2.0

    sigma_ee = local_system.Sigma_e_hat[:2, :]
    sigma_cov = sigma_ee @ sigma_ee.T
    g2 = (
        r2 * net_worth / (rho * comprehensive)
        + float(np.trace(sigma_cov @ (capital_bar * solution.H_ee))) / (2.0 * rho * comprehensive)
        + 0.5 * float(np.trace(sigma_cov @ varpi_ee))
        + beta_hat / (2.0 * rho)
        - covariance / (rho * comprehensive)
        - float(zeta_j_perp @ zeta_j_perp) / (2.0 * rho * comprehensive**2)
    )

    return LeadingPortfolio(
        public_net_worth=net_worth,
        fiscal_wealth=fiscal_wealth,
        comprehensive_resources=comprehensive,
        risk_scale_epsilon=risk_scale_epsilon,
        lambda_hat=lambda_hat,
        beta_hat=beta_hat,
        zeta_j=zeta_j,
        zeta_j_perp=zeta_j_perp,
        marketed_fiscal_wealth_amount=marketed_amount,
        return_demand_component=return_demand_component,
        fiscal_hedge_component=fiscal_hedge_component,
        leading_unconstrained_position=leading_position,
        leading_constrained_position=constrained_position,
        portfolio_curvature=portfolio_curvature,
        portfolio_gradient_y=portfolio_gradient_y,
        portfolio_bounds=(lower_bound, upper_bound),
        portfolio_lower_slack=leading_position - lower_bound,
        portfolio_upper_slack=upper_bound - leading_position,
        zero_position_feasible=lower_bound <= 0.0 <= upper_bound,
        merton_position=merton_position,
        merton_comparator_feasible=lower_bound <= merton_position <= upper_bound,
        access_source_gain_q=access_source_gain,
        hedge_source_gain_q=hedge_source_gain,
        access_value_leading=risk_scale_epsilon**2 * access_source_gain / rho,
        hedge_value_leading=risk_scale_epsilon**2 * hedge_source_gain / rho,
        access_consumption_equivalent_leading=risk_scale_epsilon**2 * access_source_gain,
        hedge_consumption_equivalent_leading=risk_scale_epsilon**2 * hedge_source_gain,
        varpi_ee=varpi_ee,
        safe_rate_order_two_anchor=r2,
        g2_anchor=g2,
        v2_anchor=g2 / rho,
    )


def transfer_boundary_net_worth_ratio(local_system: LocalSystem) -> float:
    """The N/J ratio at which the transfer floor c=rho*(N+J) >= W_bar binds
    exactly: rho*(N+J) = W_bar => N = W_bar/rho - J => N/J = W_bar/(rho*J) - 1.
    Independent of the portfolio solve -- a pure function of the anchor."""

    anchor = local_system.anchor
    rho = local_system.parameters.rho
    return anchor.wage_income_bar / (rho * anchor.fiscal_wealth_bar) - 1.0


def portfolio_sign_change_net_worth_ratio(local_system: LocalSystem, solution: LqSolution) -> float:
    """The N/J ratio at which the leading position s_0_bar = X - zeta_J.lambda/beta
    is exactly zero: N+J = zeta_J.lambda/beta => N/J = (zeta_J.lambda/beta)/J - 1.
    The marketed amount zeta_J.lambda/beta does not depend on N, so this needs
    only one portfolio evaluation (any N) to read it off."""

    anchor = local_system.anchor
    reference = leading_portfolio_and_welfare(local_system, solution, risky_short_limit=1e9, safe_debt_limit=1e9, risk_scale_epsilon=1.0)
    return reference.marketed_fiscal_wealth_amount / anchor.fiscal_wealth_bar - 1.0


def net_worth_grid(
    local_system: LocalSystem,
    solution: LqSolution,
    risky_short_limit: float,
    safe_debt_limit: float,
    risk_scale_epsilon: float,
    net_worth_to_fiscal_wealth_ratios: list[float],
) -> list[dict]:
    """Repeat the leading-portfolio and welfare calculation over an N/J grid.

    N_bar=0 selects one member of the deterministic fiscal-wealth family; not
    every member is economically admissible. Every row is retained -- a row
    that fails the maintained non-negative-transfer condition, or any other
    feasibility check, is reported with feasible=False and explicit
    failure_reasons rather than silently dropped or allowed to crash the grid.
    """

    anchor = local_system.anchor
    fiscal_wealth = anchor.fiscal_wealth_bar
    wage_income = anchor.wage_income_bar
    rho = local_system.parameters.rho
    rows = []
    for ratio in net_worth_to_fiscal_wealth_ratios:
        net_worth = float(ratio) * fiscal_wealth
        row: dict = {"public_net_worth_to_fiscal_wealth": float(ratio), "public_net_worth": net_worth}
        failure_reasons: list[str] = []

        comprehensive_resources = net_worth + fiscal_wealth
        row["comprehensive_resources"] = comprehensive_resources
        row["comprehensive_resources_positive"] = comprehensive_resources > 0.0
        if not row["comprehensive_resources_positive"]:
            failure_reasons.append("comprehensive_resources_not_positive")

        try:
            result = leading_portfolio_and_welfare(
                local_system, solution, risky_short_limit, safe_debt_limit, risk_scale_epsilon, public_net_worth=net_worth
            )
        except ValueError as error:
            row["portfolio_error"] = str(error)
            failure_reasons.append("portfolio_computation_failed")
            row["worker_consumption"] = rho * comprehensive_resources if comprehensive_resources > 0.0 else None
            row["wage_income"] = wage_income
            row["transfer"] = (row["worker_consumption"] - wage_income) if row["worker_consumption"] is not None else None
            row["transfer_feasible"] = False
            row["portfolio_bound_feasible"] = False
            row["feasible"] = False
            row["failure_reasons"] = failure_reasons
            rows.append(row)
            continue

        worker_consumption = rho * comprehensive_resources
        transfer = worker_consumption - wage_income
        transfer_feasible = transfer >= 0.0
        if not transfer_feasible:
            failure_reasons.append("negative_transfer")
        portfolio_bound_feasible = result.portfolio_lower_slack >= 0.0 and result.portfolio_upper_slack >= 0.0
        if not portfolio_bound_feasible:
            failure_reasons.append("portfolio_bound_infeasible")

        row.update(result.__dict__)
        row["worker_consumption"] = worker_consumption
        row["wage_income"] = wage_income
        row["transfer"] = transfer
        row["transfer_feasible"] = transfer_feasible
        row["portfolio_bound_feasible"] = portfolio_bound_feasible
        row["feasible"] = row["comprehensive_resources_positive"] and transfer_feasible and portfolio_bound_feasible
        row["failure_reasons"] = failure_reasons
        # N - s_0_bar: the residual signed safe position at the unconstrained optimum
        # (s_0_bar>N means the risky position is partly debt financed).
        row["residual_safe_position"] = net_worth - result.leading_unconstrained_position
        # Same number as hedge_consumption_equivalent_leading, under the name this
        # exercise's brief uses: the loss avoided relative to the forced Merton s=X.
        row["merton_misallocation_consumption_equivalent"] = result.hedge_consumption_equivalent_leading
        rows.append(row)
    return rows
