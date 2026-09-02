"""Orchestrates one full CS002 D0-D1 material run from a configuration:
continuation (both routes, both terminal conventions), horizon and mesh
comparisons, the LQ-convergence amplitude sweep, independent residuals, J/X/
N recovery over the net-worth grid, path margins, the two diagnostic-basis
cases, and the aggregate outcome. cli.py only loads a config, calls
`run_d0_d1_experiment`, and writes the result out (reporting.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

from ..cs001_lq_anchor import build_local_system, compute_steady_state, solve_lq_system
from ..cs001_lq_anchor.equations import LocalSystem
from ..cs001_lq_anchor.solver import LqSolution
from .bvp import economic_bc, economic_rhs, solve_two_point_bvp
from .config import Cs002Configuration
from .continuation import Checkpoint, ContinuationRun, run_continuation
from .margins import PathMargins, evaluate_path_margins
from .model import capital_from_log, characteristic_rates
from .outcome import AggregateOutcome, determine_outcome
from .recovery import ComprehensiveResourcesPath, JRecovery, recover_comprehensive_resources, recover_j
from .residuals import BoundaryResidualReport, OdeResidualReport, independent_boundary_residual, independent_ode_residual
from .terminal import anchor_value_tail, lq_linear_kt_path, lq_quadratic_value_tail


@dataclass(frozen=True)
class HorizonMeshComparison:
    label: str
    horizon: float
    n_mesh_points: int
    max_state_difference_from_baseline: dict[str, float]
    tolerance: float
    within_tolerance: bool


@dataclass(frozen=True)
class ConvergenceOrderReport:
    amplitudes: list[float]
    errors_k: list[float]
    errors_tau: list[float]
    ratios_k: list[float]
    ratios_tau: list[float]
    expected_ratio: float
    within_tolerance: bool


@dataclass(frozen=True)
class NetWorthGridRow:
    net_worth_to_fiscal_wealth: float
    net_worth_0: float
    comprehensive: ComprehensiveResourcesPath
    margins: PathMargins
    feasible: bool


@dataclass(frozen=True)
class D0Checks:
    """Cheap, self-contained re-verification of the Block D0 checking-matrix
    rows, recorded directly in the material run's own evidence rather than
    only in the separate pytest suite."""

    anchor_is_fixed_point: bool
    anchor_fixed_point_residual: float
    manufactured_bvp_recovered: bool
    manufactured_bvp_max_error: float


@dataclass(frozen=True)
class ExperimentReport:
    config: Cs002Configuration
    local_system: LocalSystem
    solution: LqSolution
    d0_checks: D0Checks
    route_a_main: ContinuationRun
    route_b_main: ContinuationRun
    route_a_crude_tail: ContinuationRun
    diagnostic_basis_runs: dict[str, ContinuationRun]
    horizon_mesh_comparisons: list[HorizonMeshComparison]
    convergence: ConvergenceOrderReport
    ode_residual: OdeResidualReport
    boundary_residual: BoundaryResidualReport
    j_recovery_quadratic: JRecovery
    j_recovery_anchor: JRecovery
    net_worth_grid: list[NetWorthGridRow]
    outcome: AggregateOutcome


def _manufactured_bvp_check() -> tuple[bool, float]:
    m = np.array([[-0.5, 0.1, 0.0, 0.0], [0.05, -0.3, 0.02, 0.0], [0.0, 0.03, 0.4, 0.01], [0.0, 0.0, 0.02, 0.6]])
    horizon = 3.0
    ya_target = np.array([1.0, -0.5])
    yb_target = np.array([0.2, 0.3])
    e_full = expm(m * horizon)
    c23 = np.linalg.solve(e_full[2:4, 2:4], yb_target - e_full[2:4, 0:2] @ ya_target)
    c = np.array([ya_target[0], ya_target[1], c23[0], c23[1]])

    def fun(t, y):
        del t
        return m @ y

    def bc(ya, yb):
        return np.array([ya[0] - ya_target[0], ya[1] - ya_target[1], yb[2] - yb_target[0], yb[3] - yb_target[1]])

    x_mesh = np.linspace(0.0, horizon, 11)
    y_guess = np.zeros((4, x_mesh.size))
    y_guess[0, :] = ya_target[0]
    y_guess[1, :] = ya_target[1]
    result = solve_two_point_bvp(fun, bc, x_mesh, y_guess, tol=1e-11)
    if not result.success:
        return False, float("nan")
    check_t = np.linspace(0.0, horizon, 25)
    exact = np.stack([expm(m * ti) @ c for ti in check_t], axis=1)
    numeric = result.sol(check_t)
    max_error = float(np.max(np.abs(numeric - exact)))
    return max_error < 1e-6, max_error


def _run_d0_checks(local_system: LocalSystem) -> D0Checks:
    anchor = local_system.anchor
    rates = characteristic_rates(0.0, anchor.tax_rate_bar, 1.0, 0.0, anchor.z_bar, anchor.x_bar, anchor.capital_bar, local_system.parameters)
    residual = float(max(abs(rates.k_dot), abs(rates.tau_dot), abs(rates.ell_dot), abs(rates.m_dot)))
    manufactured_ok, manufactured_error = _manufactured_bvp_check()
    return D0Checks(
        anchor_is_fixed_point=residual < 1e-9,
        anchor_fixed_point_residual=residual,
        manufactured_bvp_recovered=manufactured_ok,
        manufactured_bvp_max_error=manufactured_error,
    )


def _peak_abs_response(checkpoint: Checkpoint, t: np.ndarray) -> dict[str, float]:
    path = checkpoint.path_at(t)
    names = ("k", "tau_deviation", "ell_deviation", "m")
    anchor_center = np.array([0.0, 0.5, 1.0, 0.0])
    return {name: float(np.max(np.abs(path[i, :] - anchor_center[i]))) for i, name in enumerate(names)}


def _horizon_mesh_comparisons(config: Cs002Configuration, local_system: LocalSystem, solution: LqSolution, baseline: Checkpoint) -> list[HorizonMeshComparison]:
    tol_floor = config.acceptance_tolerances["horizon_mesh_effect_floor"]
    tol_relative = config.acceptance_tolerances["horizon_mesh_effect_relative"]
    peak_response = _peak_abs_response(baseline, np.linspace(0.0, config.baseline_horizon, 401))
    peak_overall = max(peak_response.values())
    tolerance = max(tol_floor, tol_relative * peak_overall)

    comparisons: list[HorizonMeshComparison] = []
    shared_t = np.linspace(0.0, min(config.comparison_horizons + [config.baseline_horizon]), 401)
    baseline_path = baseline.path_at(shared_t)
    names = ("k", "tau", "ell", "m")

    for horizon in config.comparison_horizons:
        run = run_continuation(
            "lq_path_continuation", config.delta_k, config.delta_tau, [config.continuation_amplitudes[-1]], horizon,
            config.baseline_mesh_points, "lq_stable_manifold", local_system, solution, tol=config.solver_tolerance, max_nodes=config.max_nodes,
        )
        checkpoint = run.checkpoints[0]
        path = checkpoint.path_at(shared_t)
        max_diff = {names[i]: float(np.max(np.abs(path[i, :] - baseline_path[i, :]))) for i in range(4)}
        comparisons.append(
            HorizonMeshComparison(
                label=f"horizon_{horizon:g}y_vs_baseline", horizon=horizon, n_mesh_points=config.baseline_mesh_points,
                max_state_difference_from_baseline=max_diff, tolerance=tolerance, within_tolerance=max(max_diff.values()) <= tolerance,
            )
        )

    refined_run = run_continuation(
        "lq_path_continuation", config.delta_k, config.delta_tau, [config.continuation_amplitudes[-1]], config.baseline_horizon,
        config.refined_mesh_points, "lq_stable_manifold", local_system, solution, tol=config.solver_tolerance, max_nodes=config.max_nodes,
    )
    refined_checkpoint = refined_run.checkpoints[0]
    refined_t = np.linspace(0.0, config.baseline_horizon, 401)
    refined_path = refined_checkpoint.path_at(refined_t)
    base_path_full = baseline.path_at(refined_t)
    max_diff = {names[i]: float(np.max(np.abs(refined_path[i, :] - base_path_full[i, :]))) for i in range(4)}
    comparisons.append(
        HorizonMeshComparison(
            label=f"mesh_{config.refined_mesh_points}_vs_{config.baseline_mesh_points}", horizon=config.baseline_horizon,
            n_mesh_points=config.refined_mesh_points, max_state_difference_from_baseline=max_diff, tolerance=tolerance,
            within_tolerance=max(max_diff.values()) <= tolerance,
        )
    )
    return comparisons


def _convergence_order(config: Cs002Configuration, local_system: LocalSystem, solution: LqSolution) -> ConvergenceOrderReport:
    fun = economic_rhs(local_system)
    x_mesh = np.linspace(0.0, config.baseline_horizon, config.baseline_mesh_points)
    check_t = np.linspace(0.0, config.baseline_horizon, 401)
    errors_k: list[float] = []
    errors_tau: list[float] = []
    for amplitude in config.convergence_amplitude_sequence:
        dk = amplitude * config.delta_k
        dtau = amplitude * config.delta_tau
        run = run_continuation(
            "lq_path_continuation", config.delta_k, config.delta_tau, [amplitude], config.baseline_horizon, config.baseline_mesh_points,
            "lq_stable_manifold", local_system, solution, tol=config.solver_tolerance, max_nodes=config.max_nodes,
        )
        checkpoint = run.checkpoints[0]
        nonlinear = checkpoint.path_at(check_t)
        lq = lq_linear_kt_path(check_t, dk, dtau, solution)
        errors_k.append(float(np.max(np.abs(nonlinear[0, :] - lq[0, :]))))
        errors_tau.append(float(np.max(np.abs((nonlinear[1, :] - 0.5) - lq[1, :]))))

    ratios_k = [errors_k[i] / errors_k[i + 1] for i in range(len(errors_k) - 1)]
    ratios_tau = [errors_tau[i] / errors_tau[i + 1] for i in range(len(errors_tau) - 1)]
    expected_order = config.acceptance_tolerances["convergence_order_expected"]
    rel_tol = config.acceptance_tolerances["convergence_order_relative_tolerance"]
    expected_ratio = 2.0**expected_order
    within = all(abs(r - expected_ratio) <= rel_tol * expected_ratio for r in ratios_k + ratios_tau)
    return ConvergenceOrderReport(
        amplitudes=list(config.convergence_amplitude_sequence), errors_k=errors_k, errors_tau=errors_tau,
        ratios_k=ratios_k, ratios_tau=ratios_tau, expected_ratio=expected_ratio, within_tolerance=within,
    )


def run_d0_d1_experiment(config: Cs002Configuration) -> ExperimentReport:
    primitives = config.cs001.parameters
    anchor = compute_steady_state(primitives)
    local_system = build_local_system(primitives, anchor)
    solution = solve_lq_system(local_system)

    d0_checks = _run_d0_checks(local_system)

    amplitudes = config.continuation_amplitudes
    route_a_main = run_continuation(
        "lq_path_continuation", config.delta_k, config.delta_tau, amplitudes, config.baseline_horizon, config.baseline_mesh_points,
        "lq_stable_manifold", local_system, solution, tol=config.solver_tolerance, max_nodes=config.max_nodes,
    )
    route_b_main = run_continuation(
        "crude_direct", config.delta_k, config.delta_tau, amplitudes, config.baseline_horizon, config.baseline_mesh_points,
        "lq_stable_manifold", local_system, solution, tol=config.solver_tolerance, max_nodes=config.max_nodes,
    )
    route_a_crude_tail = run_continuation(
        "lq_path_continuation", config.delta_k, config.delta_tau, amplitudes, config.baseline_horizon, config.baseline_mesh_points,
        "crude", local_system, solution, tol=config.solver_tolerance, max_nodes=config.max_nodes,
    )

    diagnostic_basis_runs: dict[str, ContinuationRun] = {}
    for label, (dk, dtau) in zip(("delta_k_only", "delta_tau_only"), config.diagnostic_basis_displacements):
        diagnostic_basis_runs[label] = run_continuation(
            "lq_path_continuation", dk, dtau, [1.0], config.baseline_horizon, config.baseline_mesh_points,
            "lq_stable_manifold", local_system, solution, tol=config.solver_tolerance, max_nodes=config.max_nodes,
        )

    baseline = route_a_main.final
    horizon_mesh_comparisons = _horizon_mesh_comparisons(config, local_system, solution, baseline)
    convergence = _convergence_order(config, local_system, solution)

    ode_residual = independent_ode_residual(baseline.result, local_system)
    boundary_residual = independent_boundary_residual(baseline.result, baseline.capital_0, baseline.tau_0, "lq_stable_manifold", local_system, solution)

    capital_T = capital_from_log(baseline.result.sol(np.array([config.baseline_horizon]))[0, 0], anchor.capital_bar)
    tau_T = baseline.result.sol(np.array([config.baseline_horizon]))[1, 0]
    quad_tail = lq_quadratic_value_tail(capital_T, tau_T, local_system, solution)
    anchor_tail = anchor_value_tail(local_system)
    j_recovery_quadratic = recover_j(baseline.result, local_system, "quadratic", quad_tail, config.baseline_horizon)
    j_recovery_anchor = recover_j(baseline.result, local_system, "anchor", anchor_tail, config.baseline_horizon)

    margin_t = np.linspace(0.0, config.baseline_horizon, 161)
    net_worth_grid: list[NetWorthGridRow] = []
    for ratio in config.net_worth_grid_ratios:
        net_worth_0 = ratio * anchor.fiscal_wealth_bar
        comprehensive = recover_comprehensive_resources(baseline.result, local_system, j_recovery_quadratic, net_worth_0)
        margins = evaluate_path_margins(baseline.result, local_system, comprehensive.consumption, comprehensive.x_0, config.numerical_scaffolding, margin_t)
        feasible = comprehensive.x_0 > 0.0 and not margins.boundary_reached
        net_worth_grid.append(NetWorthGridRow(net_worth_to_fiscal_wealth=ratio, net_worth_0=net_worth_0, comprehensive=comprehensive, margins=margins, feasible=feasible))

    # Path-only margins (independent of N_0: specialization, tax, tax-speed,
    # structural-solvency) gate the aggregate outcome. Transfer and
    # comprehensive-resource margins are inherently N_0-specific -- the
    # net-worth grid above is a deliberate sweep expected to include
    # infeasible low-N_0 members (exactly as CS001's own repaired baseline
    # does; see outputs/cs001-lq-anchor-baseline-repair-01/FINDINGS.md
    # section 4), and that does not by itself make the underlying transition
    # numerically or economically invalid.
    baseline_path_margins = evaluate_path_margins(baseline.result, local_system, consumption=0.0, x_0=1.0, numerical_scaffolding=config.numerical_scaffolding, t_grid=margin_t)
    path_only_interior = (
        baseline_path_margins.min_specialisation_margin_automation_composite > 0.0
        and baseline_path_margins.min_specialisation_margin_new_task_composite > 0.0
        and baseline_path_margins.min_tax_margin > 0.0
        and baseline_path_margins.min_tax_speed_margin > 0.0
        and baseline_path_margins.min_structural_solvency_margin > 0.0
    )

    checks = {
        "d0_anchor_is_fixed_point": d0_checks.anchor_is_fixed_point,
        "d0_manufactured_bvp_recovered": d0_checks.manufactured_bvp_recovered,
        "zero_displacement_returns_anchor": route_a_main.checkpoints[0].accepted,
        "route_a_all_amplitudes_converged": route_a_main.all_accepted,
        "route_b_all_amplitudes_converged": route_b_main.all_accepted,
        "ode_residual_within_tolerance": ode_residual.max_scaled_residual <= config.acceptance_tolerances["ode_residual_max"],
        "boundary_residual_within_tolerance": boundary_residual.max_scaled_residual <= config.acceptance_tolerances["boundary_residual_max"],
        "horizon_mesh_stability": all(c.within_tolerance for c in horizon_mesh_comparisons),
        "lq_convergence_second_order": convergence.within_tolerance,
        "j_recovery_routes_agree": j_recovery_quadratic.route_disagreement_relative < 1e-3,
        "branch_agreement_route_a_vs_b": _routes_agree(route_a_main, route_b_main, config),
        "no_boundary_reached_on_baseline_path": path_only_interior,
    }

    outcome = determine_outcome(
        checks,
        numerical_failure_check_names=(
            "d0_anchor_is_fixed_point", "d0_manufactured_bvp_recovered", "zero_displacement_returns_anchor",
            "route_a_all_amplitudes_converged", "route_b_all_amplitudes_converged", "ode_residual_within_tolerance",
            "boundary_residual_within_tolerance", "horizon_mesh_stability", "lq_convergence_second_order", "j_recovery_routes_agree",
        ),
        branch_sensitivity_check_names=("branch_agreement_route_a_vs_b",),
        boundary_check_names=("no_boundary_reached_on_baseline_path",),
    )

    return ExperimentReport(
        config=config, local_system=local_system, solution=solution, d0_checks=d0_checks,
        route_a_main=route_a_main, route_b_main=route_b_main, route_a_crude_tail=route_a_crude_tail,
        diagnostic_basis_runs=diagnostic_basis_runs, horizon_mesh_comparisons=horizon_mesh_comparisons,
        convergence=convergence, ode_residual=ode_residual, boundary_residual=boundary_residual,
        j_recovery_quadratic=j_recovery_quadratic, j_recovery_anchor=j_recovery_anchor,
        net_worth_grid=net_worth_grid, outcome=outcome,
    )


def _routes_agree(route_a: ContinuationRun, route_b: ContinuationRun, config: Cs002Configuration) -> bool:
    atol = config.acceptance_tolerances["branch_agreement_atol"]
    rtol = config.acceptance_tolerances["branch_agreement_rtol"]
    check_t = np.linspace(0.0, config.baseline_horizon, 81)
    for ckpt_a, ckpt_b in zip(route_a.checkpoints, route_b.checkpoints):
        if not (ckpt_a.accepted and ckpt_b.accepted):
            return False
        path_a = ckpt_a.path_at(check_t)
        path_b = ckpt_b.path_at(check_t)
        if not np.allclose(path_a, path_b, atol=atol, rtol=rtol):
            return False
    return True
