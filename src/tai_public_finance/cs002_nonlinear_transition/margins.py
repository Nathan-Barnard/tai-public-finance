"""Economic boundary margins along a solved path: specialization, tax,
tax-speed, transfer, comprehensive-resource, and structural-solvency (CS002
D0-D1 handoff, acceptance #9). Every margin is a SIGNED distance to its
boundary -- positive and slack is interior, non-positive means the path has
reached (or crossed) that boundary and the case must be retained with an
explicit reason, never dropped or smoothed over.

structural-solvency margin is operationalised here as min(R^K(t) - delta):
the net rental return above depreciation is what B=(R^K-delta)*K and hence
the whole capital-tax fiscal channel is built on (R08 dossier section 3's
"load-bearing local conditions"); it is a genuinely different object from
the comprehensive-resource margin (is X=N+J positive), which CS001's own
net_worth_grid already checks under the name comprehensive_resources_positive.
This is a reporting/labelling choice, not an equation choice -- flagged as
such in the CS002 D0-D1 return report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..cs001_lq_anchor.equations import LocalSystem
from ..primitives import evaluate_smooth_branch
from .bvp import BvpSolveResult
from .model import capital_from_log


@dataclass(frozen=True)
class PathMargins:
    min_specialisation_margin_automation_composite: float
    min_specialisation_margin_new_task_composite: float
    min_tax_margin: float
    min_tax_speed_margin: float
    min_transfer_margin: float
    comprehensive_resource_margin: float
    min_structural_solvency_margin: float
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
    solvency_margin = np.empty(t_grid.size)

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
        solvency_margin[i] = state.rental_rate - p.depreciation_rate

    min_spec_auto = float(spec_auto.min())
    min_spec_new_task = float(spec_new_task.min())
    min_tax = float(tax_margin.min())
    min_tax_speed = float(tax_speed_margin.min())
    min_transfer = float(transfer_margin.min())
    min_solvency = float(solvency_margin.min())

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
    if min_solvency <= 0.0:
        failure_reasons.append("structural_solvency_boundary_reached")

    return PathMargins(
        min_specialisation_margin_automation_composite=min_spec_auto,
        min_specialisation_margin_new_task_composite=min_spec_new_task,
        min_tax_margin=min_tax,
        min_tax_speed_margin=min_tax_speed,
        min_transfer_margin=min_transfer,
        comprehensive_resource_margin=x_0,
        min_structural_solvency_margin=min_solvency,
        boundary_reached=bool(failure_reasons),
        failure_reasons=failure_reasons,
    )
