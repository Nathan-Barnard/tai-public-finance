"""L2_local_batch one-dimensional parameter sweeps.

CS001 explicitly anticipates this lane ("Performance and escalation:
L2_local_batch: broad parameter map with checkpointed tabular output") and
its own acceptance standard ("Parameter sweeps return intervals/regions and
failure codes; failed calibrations are not silently dropped"). Each sweep
point reruns the full Stage 1 + Stage 2A pipeline from a primitive table
with exactly one field overridden; nothing here changes the maintained
economic setup, and a point that fails to solve (non-hyperbolic Hamiltonian,
degenerate traded payoff, non-positive comprehensive resources, ...) is
retained as a failed row, never silently dropped or allowed to crash the
sweep.
"""

from __future__ import annotations

import copy
from typing import Any

from ..primitives import PrimitiveParameters
from .anchor import compute_steady_state
from .diagnostics import acceptance as compute_acceptance
from .diagnostics import run_diagnostics
from .equations import build_local_system
from .irfs import run_irfs
from .portfolio import leading_portfolio_and_welfare
from .solver import solve_lq_system


def chi_star(gamma: float, rho: float) -> float:
    """The closed-form monotone/oscillatory transition for the capital-tax
    closed loop (local LQ system and computation plan): chi < chi_star gives
    real, distinct roots; chi > chi_star gives a complex-conjugate pair."""

    return gamma * (gamma + rho) ** 2 / (32.0 * rho**2)


def _parameters_with_override(base_raw: dict, path: tuple[str, ...], value: float) -> PrimitiveParameters:
    raw = copy.deepcopy(base_raw)
    node = raw
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return PrimitiveParameters.from_dict(raw)


def sweep_one_parameter(
    base_parameters: PrimitiveParameters,
    parameter_path: tuple[str, ...],
    values: list[float],
    scaffolding: dict,
    reporting: dict,
    risk_scale_epsilon: float,
    acceptance_tolerances: dict,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in values:
        row: dict[str, Any] = {"parameter_path": ".".join(parameter_path), "value": float(value)}
        try:
            parameters = _parameters_with_override(base_parameters.raw, parameter_path, float(value))
            anchor = compute_steady_state(parameters)
            local_system = build_local_system(parameters, anchor)
            solution = solve_lq_system(local_system)
            portfolio = leading_portfolio_and_welfare(
                local_system,
                solution,
                risky_short_limit=float(scaffolding["risky_short_limit"]),
                safe_debt_limit=float(scaffolding["safe_debt_limit"]),
                risk_scale_epsilon=risk_scale_epsilon,
            )
            irfs = run_irfs(local_system, solution, portfolio, reporting, scaffolding)
            diagnostics = run_diagnostics(local_system, solution, portfolio)
            accept = compute_acceptance(
                local_system,
                solution,
                diagnostics,
                portfolio,
                irfs["boundary_summary"],
                irfs["max_matrix_exponential_vs_ode_relative_error"],
                irfs["max_first_order_budget_residual"],
                acceptance_tolerances,
            )
        except Exception as error:  # noqa: BLE001 -- a sweep must retain, not crash on, a failed point
            row.update(
                {
                    "outcome": "error",
                    "failed_checks": [f"{type(error).__name__}: {error}"],
                    "chi": None,
                    "real_closed_loop_root_real": None,
                    "real_closed_loop_root_imag_abs": None,
                    "oscillatory": None,
                    "leading_unconstrained_position": None,
                    "return_demand_component": None,
                    "fiscal_hedge_component": None,
                    "hedge_consumption_equivalent_leading": None,
                    "min_boundary_slack": None,
                }
            )
            rows.append(row)
            continue

        real_roots = diagnostics.closed_loop["real_closed_loop_eigenvalues"]
        dominant = real_roots[0]
        row.update(
            {
                "outcome": accept.outcome,
                "failed_checks": accept.failed_checks,
                "chi": local_system.anchor.chi,
                "chi_star": chi_star(local_system.anchor.gamma, local_system.parameters.rho),
                "real_closed_loop_root_real": float(dominant.real),
                "real_closed_loop_root_imag_abs": float(abs(dominant.imag)),
                "oscillatory": bool(abs(dominant.imag) > 1e-9),
                "leading_unconstrained_position": portfolio.leading_unconstrained_position,
                "return_demand_component": portfolio.return_demand_component,
                "fiscal_hedge_component": portfolio.fiscal_hedge_component,
                "hedge_consumption_equivalent_leading": portfolio.hedge_consumption_equivalent_leading,
                "min_boundary_slack": min(irfs["boundary_summary"].values()),
            }
        )
        rows.append(row)
    return rows
