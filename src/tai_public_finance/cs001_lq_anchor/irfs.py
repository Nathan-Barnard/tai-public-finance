"""Impulse-response experiments, kept in explicitly distinct families.

A Brownian innovation is realized on the portfolio held immediately before
it (s_-); it is fundamentally different from an inherited-state
displacement or a finite-window OU-conditional-sd displacement, both of
which hold public net worth fixed and carry no traded-claim payoff. Mixing
these into one undifferentiated "shock" is exactly the confusion the local
LQ system and computation plan and CS001 warn against, so every experiment
below is tagged with its family and each row records which family produced
it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eig, expm

from ..primitives.production import evaluate_smooth_branch
from .diagnostics import resolvent_probe_residual
from .equations import LocalSystem
from .portfolio import LeadingPortfolio
from .solver import LqSolution


@dataclass(frozen=True)
class Experiment:
    name: str
    family: str
    description: str
    initial_y: np.ndarray
    brownian_increment: np.ndarray | None = None
    inherited_portfolio_pays: bool = False


def _normalised(vector: np.ndarray, scale: float) -> np.ndarray:
    norm = float(np.max(np.abs(vector)))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero modal vector.")
    return scale * vector / norm


def build_experiments(local_system: LocalSystem, solution: LqSolution, reporting: dict) -> list[Experiment]:
    p = local_system.parameters
    anchor = local_system.anchor
    brownian_delta = float(reporting["short_brownian_window_years"])
    state_window = float(reporting["finite_ou_window_years"])
    delta_alpha = float(reporting["constructed_automation_share_displacement"])
    state_scale = float(reporting["inherited_state_displacement"])
    sigma_z, sigma_x = p.sigma_z_hat, p.sigma_x_hat
    a = anchor.alpha_x_bar
    eta = anchor.eta_output_alpha
    h_i = anchor.h_international_bar
    alpha = anchor.alpha_bar
    tau = anchor.tax_rate_bar
    chi_tau = 1.0 - alpha + tau * alpha

    experiments = [
        Experiment(
            "brownian_productivity_1sd_short_window",
            "primitive_brownian_innovation",
            "One standardized productivity Brownian innovation over the declared short window; the inherited portfolio pays.",
            np.array([sigma_z * np.sqrt(brownian_delta), 0.0, 0.0, 0.0]),
            np.array([np.sqrt(brownian_delta), 0.0]),
            True,
        ),
        Experiment(
            "brownian_automation_1sd_short_window",
            "primitive_brownian_innovation",
            "One standardized automation Brownian innovation over the declared short window; the inherited portfolio pays.",
            np.array([0.0, sigma_x * np.sqrt(brownian_delta), 0.0, 0.0]),
            np.array([0.0, np.sqrt(brownian_delta)]),
            True,
        ),
        Experiment(
            "ou_productivity_conditional_sd_state_displacement",
            "finite_window_ou_state_displacement",
            "Productivity state displaced by its finite-window conditional OU standard deviation; public net worth is held fixed.",
            np.array(
                [sigma_z * np.sqrt((1.0 - np.exp(-2.0 * p.kappa_z * state_window)) / (2.0 * p.kappa_z)), 0.0, 0.0, 0.0]
            ),
        ),
        Experiment(
            "ou_automation_conditional_sd_state_displacement",
            "finite_window_ou_state_displacement",
            "Automation state displaced by its finite-window conditional OU standard deviation; public net worth is held fixed.",
            np.array(
                [0.0, sigma_x * np.sqrt((1.0 - np.exp(-2.0 * p.kappa_x * state_window)) / (2.0 * p.kappa_x)), 0.0, 0.0]
            ),
        ),
    ]

    # Under the current normalization k_I = log(K_bar/L) = 0, h_I(alpha_bar) =
    # eta + 1/alpha_bar, so the rental/tax-base-neutral and claim-neutral
    # directions coincide (local LQ system and computation plan); report once.
    directions = {
        "combined_claim_rental_base_neutral": -h_i,
        "worker_income_neutral": -(eta - 1.0 / (1.0 - alpha)),
        "output_neutral": -eta,
        "primary_resource_neutral": -(tau - 1.0 + chi_tau * eta) / chi_tau,
    }
    for name, dz_per_dalpha in directions.items():
        experiments.append(
            Experiment(
                f"constructed_{name}",
                "economically_constructed_state_displacement",
                f"A {delta_alpha:.6g} automation-share displacement with dz/dalpha={dz_per_dalpha:.12g}; public net worth is held fixed.",
                np.array([dz_per_dalpha * delta_alpha, delta_alpha / a, 0.0, 0.0]),
            )
        )

    experiments.extend(
        [
            Experiment(
                "inherited_capital_1pct",
                "inherited_state_displacement",
                "A one-percent log-capital displacement; not a stochastic innovation.",
                np.array([0.0, 0.0, state_scale, 0.0]),
            ),
            Experiment(
                "inherited_tax_1pp",
                "inherited_state_displacement",
                "A one-percentage-point inherited tax displacement; not a stochastic innovation.",
                np.array([0.0, 0.0, 0.0, state_scale]),
            ),
        ]
    )

    values, vectors = eig(solution.A_rc)
    if np.max(np.abs(values.imag)) > 1e-10:
        index = int(np.argmax(values.imag))
        vector = vectors[:, index]
        real_basis = _normalised(vector.real, state_scale)
        imag_basis = _normalised(vector.imag, state_scale)
        experiments.extend(
            [
                Experiment(
                    "capital_tax_mode_real_basis",
                    "closed_loop_invariant_basis",
                    "Real basis vector of the damped capital-tax invariant plane; a propagation mode, not primitive uncertainty.",
                    np.array([0.0, 0.0, real_basis[0], real_basis[1]]),
                ),
                Experiment(
                    "capital_tax_mode_quadrature_basis",
                    "closed_loop_invariant_basis",
                    "Quadrature basis vector of the damped capital-tax invariant plane; a propagation mode, not primitive uncertainty.",
                    np.array([0.0, 0.0, imag_basis[0], imag_basis[1]]),
                ),
            ]
        )
    else:
        for index in range(2):
            vector = _normalised(vectors[:, index].real, state_scale)
            experiments.append(
                Experiment(
                    f"capital_tax_eigenmode_{index + 1}",
                    "closed_loop_eigenbasis",
                    "Real capital-tax closed-loop eigenmode; a propagation mode, not primitive uncertainty.",
                    np.array([0.0, 0.0, vector[0], vector[1]]),
                )
            )
    return experiments


def _row(
    local_system: LocalSystem,
    solution: LqSolution,
    portfolio: LeadingPortfolio,
    scaffolding: dict,
    experiment: Experiment,
    regime: str,
    horizon: float,
    y: np.ndarray,
    x_deviation: float,
) -> dict:
    p = local_system.parameters
    anchor = local_system.anchor
    j = local_system.linear_fiscal_wealth
    capital_bar = anchor.capital_bar

    nu = anchor.chi * float(solution.H[3, :] @ y)
    output_dev = float(local_system.output_y @ y)
    rental_dev = float(local_system.rental_y @ y)
    wage_dev = float(local_system.wage_y @ y)
    tax_base_dev = capital_bar * float(local_system.tax_base_normalized_y @ y)
    tax_revenue_dev = anchor.tax_rate_bar * tax_base_dev + anchor.tax_base_bar * float(y[3])
    fiscal_dev = capital_bar * float(local_system.fiscal_resources_normalized_y @ y)
    n_deviation = float(x_deviation - capital_bar * (j @ y))
    consumption_dev = p.rho * float(x_deviation)
    transfer_dev = consumption_dev - wage_dev

    if regime == "full_access":
        risky_dev = float(x_deviation + portfolio.portfolio_gradient_y @ y)
        risky_level = portfolio.leading_unconstrained_position + risky_dev
    else:
        risky_dev = 0.0
        risky_level = 0.0
    net_worth_level = anchor.public_net_worth_bar + n_deviation
    safe_level = net_worth_level - risky_level

    z = anchor.z_bar + y[0]
    x = anchor.x_bar + y[1]
    capital = capital_bar * float(np.exp(y[2]))
    tax_rate = anchor.tax_rate_bar + y[3]
    exact = evaluate_smooth_branch(z, x, capital, tax_rate, p)

    x_level = anchor.comprehensive_resources_bar + x_deviation
    consumption_level = anchor.worker_consumption_bar + consumption_dev
    transfer_level = consumption_level - exact.wage_income

    lower_slack = risky_level + float(scaffolding["risky_short_limit"])
    upper_slack = net_worth_level + float(scaffolding["safe_debt_limit"]) - risky_level

    first_order_budget_left = float(
        anchor.comprehensive_resources_bar * (local_system.safe_rate_y @ y) - capital_bar * (j @ (solution.A_c @ y))
    )
    first_order_budget_right = float(
        p.rho * n_deviation + anchor.public_net_worth_bar * (local_system.safe_rate_y @ y) + fiscal_dev - consumption_dev
    )

    return {
        "experiment": experiment.name,
        "experiment_family": experiment.family,
        "regime": regime,
        "horizon_years": float(horizon),
        "z_deviation": float(y[0]),
        "x_deviation": float(y[1]),
        "alpha_deviation_linear": float(anchor.alpha_x_bar * y[1]),
        "log_capital_deviation": float(y[2]),
        "capital_deviation_linear": float(capital_bar * y[2]),
        "tax_rate_deviation": float(y[3]),
        "tax_speed": nu,
        "output_deviation_linear": output_dev,
        "rental_rate_deviation_linear": rental_dev,
        "wage_income_deviation_linear": wage_dev,
        "tax_base_deviation_linear": tax_base_dev,
        "tax_revenue_deviation_linear": tax_revenue_dev,
        "fiscal_resources_deviation_linear": fiscal_dev,
        "public_net_worth_deviation": n_deviation,
        "comprehensive_resources_deviation": float(x_deviation),
        "worker_consumption_deviation": consumption_dev,
        "transfer_deviation_linear": transfer_dev,
        "risky_position_deviation": risky_dev,
        "safe_position_level": safe_level,
        "tax_adjustment_cost_quadratic_diagnostic": exact.output * nu * nu / (2.0 * p.tax_adjustment_scale),
        "specialisation_margin_automation_composite": exact.specialisation_margin_automation_composite,
        "specialisation_margin_new_task_composite": exact.specialisation_margin_new_task_composite,
        "output_automation_semielasticity": exact.output_automation_semielasticity,
        "worker_consumption_level": consumption_level,
        "transfer_level": transfer_level,
        "comprehensive_resources_level": x_level,
        "risky_position_level": risky_level,
        "portfolio_lower_slack": lower_slack,
        "portfolio_upper_slack": upper_slack,
        "tax_lower_slack": tax_rate - float(scaffolding["tax_min"]),
        "tax_upper_slack": float(scaffolding["tax_max"]) - tax_rate,
        "tax_structural_ceiling_slack": 1.0 - tax_rate,
        "tax_speed_slack": float(scaffolding["tax_speed_abs_max"]) - abs(nu),
        "first_order_budget_residual": first_order_budget_left - first_order_budget_right,
    }


def run_irfs(
    local_system: LocalSystem,
    solution: LqSolution,
    portfolio: LeadingPortfolio,
    reporting: dict,
    scaffolding: dict,
) -> dict:
    grid = reporting["horizon_grid_years"]
    horizons = np.arange(float(grid["start"]), float(grid["stop"]) + float(grid["step"]) / 2.0, float(grid["step"]))
    if np.any(np.diff(horizons) < 0.0) or horizons[0] != 0.0:
        raise ValueError("IRF horizons must be sorted and start at zero.")

    experiments = build_experiments(local_system, solution, reporting)
    F = solution.F
    j = local_system.linear_fiscal_wealth
    capital_bar = local_system.anchor.capital_bar
    lambda_hat = portfolio.lambda_hat

    rows: list[dict] = []
    cross_checks: dict[str, dict] = {}
    discounted_cumulative: dict[str, dict] = {}

    for experiment in experiments:
        regimes = ["full_access", "no_external_claim"] if experiment.inherited_portfolio_pays else ["full_access"]
        for regime in regimes:
            inherited_position = portfolio.leading_unconstrained_position if regime == "full_access" else 0.0
            initial_net_worth = 0.0
            if experiment.inherited_portfolio_pays:
                initial_net_worth = inherited_position * float(lambda_hat @ experiment.brownian_increment)
            initial_x = initial_net_worth + capital_bar * float(j @ experiment.initial_y)
            w0 = np.concatenate([experiment.initial_y, [initial_x]])

            path = np.vstack([expm(F * horizon) @ w0 for horizon in horizons])
            ode = solve_ivp(
                lambda _t, value: F @ value,
                (float(horizons[0]), float(horizons[-1])),
                w0,
                t_eval=horizons,
                rtol=1e-12,
                atol=1e-14,
                method="DOP853",
            )
            if not ode.success:
                raise RuntimeError(f"Direct ODE cross-check failed for {experiment.name}: {ode.message}")
            ode_path = ode.y.T
            difference = float(np.max(np.abs(path - ode_path)))
            scale = 1.0 + float(np.max(np.abs(path)))
            key = f"{experiment.name}::{regime}"
            cross_checks[key] = {
                "matrix_exponential_vs_ode_relative_error": difference / scale,
                "initial_public_net_worth_payoff": initial_net_worth,
                "initial_comprehensive_resources": initial_x,
            }
            cumulative_vector = solution.resolvent @ w0
            rho = local_system.parameters.rho
            discounted_cumulative[key] = {
                "discounted_state_and_comprehensive_response": cumulative_vector,
                "resolvent_identity_residual": resolvent_probe_residual(rho, F, cumulative_vector, w0),
            }
            for horizon, values in zip(horizons, path, strict=True):
                rows.append(
                    _row(local_system, solution, portfolio, scaffolding, experiment, regime, float(horizon), values[:4], float(values[4]))
                )

    boundary_fields = [
        "specialisation_margin_automation_composite",
        "specialisation_margin_new_task_composite",
        "output_automation_semielasticity",
        "worker_consumption_level",
        "transfer_level",
        "comprehensive_resources_level",
        "portfolio_lower_slack",
        "portfolio_upper_slack",
        "tax_lower_slack",
        "tax_upper_slack",
        "tax_structural_ceiling_slack",
        "tax_speed_slack",
    ]
    boundary_summary = {field: float(min(row[field] for row in rows)) for field in boundary_fields}

    return {
        "experiments": experiments,
        "rows": rows,
        "cross_checks": cross_checks,
        "discounted_cumulative": discounted_cumulative,
        "boundary_summary": boundary_summary,
        "max_matrix_exponential_vs_ode_relative_error": max(
            entry["matrix_exponential_vs_ode_relative_error"] for entry in cross_checks.values()
        ),
        "max_first_order_budget_residual": max(abs(row["first_order_budget_residual"]) for row in rows),
    }
