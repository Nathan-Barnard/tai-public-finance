"""Economic boundary margins along a solved path: specialization, tax,
tax-speed, transfer, comprehensive-resource, and net-rental tax-base
margins (CS002 D0-D1 handoff, acceptance #9). Every margin is a SIGNED
distance to its boundary -- positive and slack is interior, non-positive
means the path has reached (or crossed) that boundary and the case must be
retained with an explicit reason, never dropped or smoothed over.

D2 mandatory repair #1 (D0-D1 review finding): this module previously named
min(R^K(t) - delta) "structural solvency". It is NOT structural/continuation
solvency, the no-Ponzi condition, or the viability frontier -- it only
measures whether the net rental flow supporting B=(R^K-delta)*K is positive
(R08 dossier section 3's "load-bearing local conditions"). It is renamed
here to `min_net_rental_tax_base_margin`, and `structural_continuation_
solvency` is reported literally as "not_evaluated" (no separately specified
viability/no-Ponzi calculation is performed by this module or anywhere else
in this package) rather than implied by this proxy. This margin remains a
genuinely different object from the comprehensive-resource margin (is
X=N+J positive), which CS001's own net_worth_grid already checks under the
name comprehensive_resources_positive. Renaming is a reporting/labelling
correction, not an equation change -- the computed value of min(R^K-delta)
along the path is bit-for-bit identical to the D0-D1 code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..cs001_lq_anchor.equations import LocalSystem
from ..primitives import evaluate_smooth_branch
from .bvp import BvpSolveResult
from .exogenous import ExogenousEvaluator, current_state
from .model import capital_from_log

StructuralContinuationSolvencyLabel = Literal["not_evaluated"]
NOT_EVALUATED: StructuralContinuationSolvencyLabel = "not_evaluated"


@dataclass(frozen=True)
class PathMargins:
    min_specialisation_margin_automation_composite: float
    min_specialisation_margin_new_task_composite: float
    min_tax_margin: float
    min_tax_speed_margin: float
    min_transfer_margin: float
    comprehensive_resource_margin: float
    min_net_rental_tax_base_margin: float
    structural_continuation_solvency: StructuralContinuationSolvencyLabel
    boundary_reached: bool
    failure_reasons: list[str] = field(default_factory=list)


def evaluate_path_margins(
    result: BvpSolveResult,
    local_system: LocalSystem,
    consumption: float,
    x_0: float,
    numerical_scaffolding: dict,
    t_grid: np.ndarray,
) -> PathMargins:
    anchor = local_system.anchor
    p = local_system.parameters
    tax_min = float(numerical_scaffolding["tax_min"])
    tax_max = float(numerical_scaffolding["tax_max"])
    tax_speed_abs_max = float(numerical_scaffolding["tax_speed_abs_max"])

    path = result.sol(t_grid)
    spec_auto = np.empty(t_grid.size)
    spec_new_task = np.empty(t_grid.size)
    tax_margin = np.empty(t_grid.size)
    tax_speed_margin = np.empty(t_grid.size)
    transfer_margin = np.empty(t_grid.size)
    net_rental_tax_base_margin = np.empty(t_grid.size)

    for i in range(t_grid.size):
        k, tau, _ell, m = path[:, i]
        capital = capital_from_log(k, anchor.capital_bar)
        state = evaluate_smooth_branch(anchor.z_bar, anchor.x_bar, capital, tau, p)
        nu = p.tax_adjustment_scale * m / state.output

        spec_auto[i] = state.specialisation_margin_automation_composite
        spec_new_task[i] = state.specialisation_margin_new_task_composite
        tax_margin[i] = min(tau - tax_min, tax_max - tau)
        tax_speed_margin[i] = tax_speed_abs_max - abs(nu)
        transfer_margin[i] = consumption - state.wage_income
        net_rental_tax_base_margin[i] = state.rental_rate - p.depreciation_rate

    min_spec_auto = float(spec_auto.min())
    min_spec_new_task = float(spec_new_task.min())
    min_tax = float(tax_margin.min())
    min_tax_speed = float(tax_speed_margin.min())
    min_transfer = float(transfer_margin.min())
    min_net_rental_tax_base = float(net_rental_tax_base_margin.min())

    failure_reasons: list[str] = []
    if min_spec_auto <= 0.0:
        failure_reasons.append("specialisation_automation_boundary_reached")
    if min_spec_new_task <= 0.0:
        failure_reasons.append("specialisation_new_task_boundary_reached")
    if min_tax <= 0.0:
        failure_reasons.append("tax_boundary_reached")
    if min_tax_speed <= 0.0:
        failure_reasons.append("tax_speed_boundary_reached")
    if min_transfer <= 0.0:
        failure_reasons.append("negative_transfer")
    if x_0 <= 0.0:
        failure_reasons.append("comprehensive_resources_not_positive")
    if min_net_rental_tax_base <= 0.0:
        failure_reasons.append("net_rental_tax_base_boundary_reached")

    return PathMargins(
        min_specialisation_margin_automation_composite=min_spec_auto,
        min_specialisation_margin_new_task_composite=min_spec_new_task,
        min_tax_margin=min_tax,
        min_tax_speed_margin=min_tax_speed,
        min_transfer_margin=min_transfer,
        comprehensive_resource_margin=x_0,
        min_net_rental_tax_base_margin=min_net_rental_tax_base,
        structural_continuation_solvency=NOT_EVALUATED,
        boundary_reached=bool(failure_reasons),
        failure_reasons=failure_reasons,
    )


def margins_time_series(
    result: BvpSolveResult,
    local_system: LocalSystem,
    consumption_path: np.ndarray,
    numerical_scaffolding: dict,
    t_grid: np.ndarray,
    exogenous_path: ExogenousEvaluator | None = None,
) -> dict[str, np.ndarray]:
    """Same per-point margin formulas as `evaluate_path_margins`, but
    returning the FULL time series rather than reducing to a minimum --
    CS002 D2 acceptance: 'specialization, tax, tax-speed, transfer, ...
    margins are reported at every time point.' `consumption_path` is
    time-varying in D2 (c(t)=rho*X(t)), unlike D0-D1's single scalar; pass
    an array matching `t_grid`. Threading `exogenous_path` evaluates every
    margin at the ACTUAL current (z(t), x(t)), not the frozen anchor --
    omitting it reproduces the D0-D1 frozen-state evaluation exactly."""

    anchor = local_system.anchor
    p = local_system.parameters
    tax_min = float(numerical_scaffolding["tax_min"])
    tax_max = float(numerical_scaffolding["tax_max"])
    tax_speed_abs_max = float(numerical_scaffolding["tax_speed_abs_max"])

    path = result.sol(t_grid)
    n = t_grid.size
    z_at_t, x_at_t = current_state(t_grid, anchor.z_bar, anchor.x_bar, exogenous_path)
    spec_auto = np.empty(n)
    spec_new_task = np.empty(n)
    tax_margin = np.empty(n)
    tax_speed_margin = np.empty(n)
    transfer_margin = np.empty(n)
    net_rental_tax_base_margin = np.empty(n)

    for i in range(n):
        k, tau, _ell, m = path[:, i]
        capital = capital_from_log(k, anchor.capital_bar)
        state = evaluate_smooth_branch(float(z_at_t[i]), float(x_at_t[i]), capital, tau, p)
        nu = p.tax_adjustment_scale * m / state.output

        spec_auto[i] = state.specialisation_margin_automation_composite
        spec_new_task[i] = state.specialisation_margin_new_task_composite
        tax_margin[i] = min(tau - tax_min, tax_max - tau)
        tax_speed_margin[i] = tax_speed_abs_max - abs(nu)
        transfer_margin[i] = consumption_path[i] - state.wage_income
        net_rental_tax_base_margin[i] = state.rental_rate - p.depreciation_rate

    return {
        "specialisation_margin_automation_composite": spec_auto,
        "specialisation_margin_new_task_composite": spec_new_task,
        "tax_margin": tax_margin,
        "tax_speed_margin": tax_speed_margin,
        "transfer_margin": transfer_margin,
        "net_rental_tax_base_margin": net_rental_tax_base_margin,
    }
