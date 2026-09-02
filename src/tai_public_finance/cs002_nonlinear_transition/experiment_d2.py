"""Orchestrates one full CS002 D2 material run from a configuration: two
shock directions (pure productivity, pure automation), each with continuation
(both routes), OU-path verification, horizon/mesh comparisons on their own
pairwise common interval, LQ-limit convergence, independent residuals
(off-mesh, componentwise, boundary), two-route J/X/N/varpi recovery, budget
separation, margins reported at every time point, and the aggregate outcome.
cli_d2.py only loads a config, calls `run_d2_experiment`, and writes the
result out (reporting_d2.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..cs001_lq_anchor import build_local_system, compute_steady_state, solve_lq_system
from ..cs001_lq_anchor.equations import LocalSystem
from ..cs001_lq_anchor.solver import LqSolution
from ..primitives.production import automation_share
from .config_d2 import Cs002D2Configuration
from .continuation_d2 import ExogenousCheckpoint, ExogenousContinuationRun, ShockDirection, run_exogenous_shock_continuation
from .exogenous import invert_automation_share, propagate_exogenous_numerically
from .experiment import _pairwise_common_grid  # reused pure helper; D2 mandatory repair #3 applies identically here
from .margins import margins_time_series
from .model import capital_from_log
from .outcome import AggregateOutcome, determine_outcome
from .recovery import (
    REPORTED_PATH_NAMES,
    ExogenousResourcesPath,
    JRecovery,
    OpportunityValueRecovery,
    anchor_reference_values,
    reconstruct_reported_paths,
    recover_exogenous_resources,
    recover_j,
    recover_varpi,
)
from .residuals import (
    BoundaryResidualReport,
    BudgetSeparationResidualReport,
    ComponentwiseRhsReport,
    OdeResidualReport,
    VarpiAlongPathResidualReport,
    budget_separation_residual,
    componentwise_manual_rhs_check,
    independent_boundary_residual,
    independent_ode_residual,
    independent_varpi_along_path_residual,
)
from .terminal import lq_full_state_path, lq_quadratic_value_tail

_OUTCOME_PRIORITY = {"derivation_blocked": 4, "numerical_failure": 3, "branch_sensitive": 2, "boundary_reaching": 1, "computational_pass": 0}


@dataclass(frozen=True)
class OuPathCheck:
    max_abs_difference_z: float
    max_abs_difference_x: float
    within_tolerance: bool


@dataclass(frozen=True)
class HorizonMeshComparisonD2:
    """CS002 D2 review repair (finding 2): every field below is keyed by
    REPORTED_PATH_NAMES -- the complete reconstructed solution (z, x, r0,
    K, log-capital deviation, tau, nu, ell, m, output, fiscal resources, J,
    X, N, c, varpi, and the LQ comparator k/tau paths), not only the four
    raw BVP state variables. `tolerance` is evaluated SEPARATELY for each
    path from that path's OWN peak absolute response on the baseline run
    (never pooled across variables via a shared max()), and `within_tolerance`
    is true only when every single path passes its own tolerance."""

    label: str
    horizon: float
    n_mesh_points: int
    max_state_difference_from_baseline: dict[str, float]
    tolerance: dict[str, float]
    time_of_max_difference: dict[str, float]
    within_tolerance_by_path: dict[str, bool]
    within_tolerance: bool


@dataclass(frozen=True)
class ConvergenceOrderReportD2:
    amplitudes: list[float]
    errors_k: list[float]
    errors_tau: list[float]
    ratios_k: list[float]
    ratios_tau: list[float]
    expected_ratio: float
    within_tolerance: bool


@dataclass(frozen=True)
class WrongR0DetectionCheck:
    """CS002 D2 required check #8: a deliberately wrong r0=rho substitution
    for this nonzero shock must be detected -- i.e. materially disagree with
    the correct state-dependent-r0 recovery, well beyond the two CORRECT
    routes' own disagreement with each other."""

    j0_correct: float
    j0_wrong_r0_rho: float
    absolute_difference: float
    detected: bool


@dataclass(frozen=True)
class ShockDirectionReport:
    direction: ShockDirection
    z0_target: float
    x0_target: float
    alpha_target: float | None
    route_a: ExogenousContinuationRun
    route_b: ExogenousContinuationRun
    ou_path_check: OuPathCheck
    horizon_mesh_comparisons: list[HorizonMeshComparisonD2]
    convergence: ConvergenceOrderReportD2
    ode_residual: OdeResidualReport
    boundary_residual: BoundaryResidualReport
    componentwise_rhs_reports: list[ComponentwiseRhsReport]
    j_recovery: JRecovery
    exogenous_resources: ExogenousResourcesPath
    budget_separation: BudgetSeparationResidualReport
    varpi_recovery: OpportunityValueRecovery
    varpi_along_path_residual: VarpiAlongPathResidualReport
    varpi_horizon_sensitivity: dict[float, float]
    margins_t_grid: np.ndarray
    margins_series: dict[str, np.ndarray]
    wrong_r0_detection: WrongR0DetectionCheck
    outcome: AggregateOutcome


@dataclass(frozen=True)
class ExperimentReportD2:
    config: Cs002D2Configuration
    local_system: LocalSystem
    solution: LqSolution
    productivity: ShockDirectionReport
    automation: ShockDirectionReport
    outcome: AggregateOutcome


def _compare_reported_paths(
    baseline_paths: dict[str, np.ndarray],
    comparison_paths: dict[str, np.ndarray],
    reference_values: dict[str, float],
    common_t: np.ndarray,
    tol_floor: float,
    tol_relative: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, bool]]:
    """Score every REPORTED_PATH_NAMES path SEPARATELY: `tolerance[name] =
    max(tol_floor, tol_relative * peak_absolute_response_of_that_path)`,
    where the peak response is the baseline path's own max absolute
    deviation from ITS OWN anchor/steady-state reference over `common_t` --
    CS002 D2 review repair (finding 2): never use the largest response among
    several variables to set one shared tolerance for all of them. Returns
    (max_diff, tolerance, time_of_max_difference, within_tolerance_by_path),
    each a dict keyed by REPORTED_PATH_NAMES."""

    missing = [name for name in REPORTED_PATH_NAMES if name not in baseline_paths or name not in comparison_paths]
    if missing:
        raise ValueError(f"Horizon/mesh comparison is missing reconstructed path(s): {missing}")

    max_diff: dict[str, float] = {}
    tolerance: dict[str, float] = {}
    time_of_max: dict[str, float] = {}
    within_tolerance_by_path: dict[str, bool] = {}
    for name in REPORTED_PATH_NAMES:
        diff = np.abs(comparison_paths[name] - baseline_paths[name])
        idx = int(np.argmax(diff))
        max_diff[name] = float(diff[idx])
        time_of_max[name] = float(common_t[idx])
        peak_response = float(np.max(np.abs(baseline_paths[name] - reference_values[name])))
        tol = max(tol_floor, tol_relative * peak_response)
        tolerance[name] = tol
        within_tolerance_by_path[name] = max_diff[name] <= tol
    return max_diff, tolerance, time_of_max, within_tolerance_by_path


def _horizon_mesh_comparisons(
    config: Cs002D2Configuration,
    direction: ShockDirection,
    z0_target: float,
    x0_target: float,
    local_system: LocalSystem,
    solution: LqSolution,
    baseline: ExogenousCheckpoint,
    baseline_j_recovery: JRecovery,
    baseline_exogenous_resources: ExogenousResourcesPath,
    baseline_varpi_recovery: OpportunityValueRecovery,
) -> list[HorizonMeshComparisonD2]:
    """CS002 D2 review repair (finding 2): each comparison reconstructs the
    COMPLETE reported solution for both runs (not only the raw BVP state)
    on their full pairwise common interval, and scores every path against
    its own effect-scaled tolerance. The comparison run's own J/X/N/c/varpi
    are recovered FRESH here, at ITS OWN horizon-appropriate terminal tail
    (same convention as the primary run -- quadratic LQ tail for J, tail_
    value=0 for varpi -- evaluated at this run's own terminal state/horizon,
    never reusing the baseline's tail value); the baseline side reuses the
    ALREADY-recovered `baseline_j_recovery`/`baseline_exogenous_resources`/
    `baseline_varpi_recovery` (the same objects the final report uses), only
    interpolated onto each comparison's own common grid -- never a separate
    shadow recomputation that could silently drift from the reported baseline."""

    tol_floor = config.acceptance_tolerances["horizon_mesh_effect_floor"]
    tol_relative = config.acceptance_tolerances["horizon_mesh_effect_relative"]
    reference_values = anchor_reference_values(local_system)
    anchor = local_system.anchor

    def _compare_one(label: str, horizon: float, n_mesh_points: int, common_t: np.ndarray, checkpoint: ExogenousCheckpoint) -> HorizonMeshComparisonD2:
        capital_T = capital_from_log(float(checkpoint.result.sol(np.array([horizon]))[0, 0]), anchor.capital_bar)
        tau_T = float(checkpoint.result.sol(np.array([horizon]))[1, 0])
        z_T = float(checkpoint.exogenous_path.z(horizon))
        x_T = float(checkpoint.exogenous_path.x(horizon))
        quad_tail = lq_quadratic_value_tail(capital_T, tau_T, local_system, solution, z=z_T, x=x_T)
        comparison_j_recovery = recover_j(checkpoint.result, local_system, "quadratic", quad_tail, horizon, exogenous_path=checkpoint.exogenous_path)
        comparison_exogenous_resources = recover_exogenous_resources(
            checkpoint.result, local_system, comparison_j_recovery, config.initial_public_net_worth, checkpoint.exogenous_path
        )
        comparison_varpi_recovery = recover_varpi(local_system, checkpoint.exogenous_path, horizon, tail_value=0.0)

        comparison_paths = reconstruct_reported_paths(
            checkpoint.result, checkpoint.exogenous_path, local_system, solution,
            comparison_j_recovery, comparison_exogenous_resources, comparison_varpi_recovery, common_t,
        )
        baseline_paths = reconstruct_reported_paths(
            baseline.result, baseline.exogenous_path, local_system, solution,
            baseline_j_recovery, baseline_exogenous_resources, baseline_varpi_recovery, common_t,
        )
        max_diff, tolerance, time_of_max, within_by_path = _compare_reported_paths(
            baseline_paths, comparison_paths, reference_values, common_t, tol_floor, tol_relative
        )
        return HorizonMeshComparisonD2(
            label=label, horizon=horizon, n_mesh_points=n_mesh_points,
            max_state_difference_from_baseline=max_diff, tolerance=tolerance, time_of_max_difference=time_of_max,
            within_tolerance_by_path=within_by_path, within_tolerance=all(within_by_path.values()),
        )

    comparisons: list[HorizonMeshComparisonD2] = []
    for horizon in config.comparison_horizons:
        common_t = _pairwise_common_grid(horizon, config.baseline_horizon)
        run = run_exogenous_shock_continuation(
            "warm_start", direction, z0_target, x0_target, [config.continuation_amplitudes[-1]], horizon,
            config.baseline_mesh_points, "lq_stable_manifold", local_system, solution, tol=config.solver_tolerance, max_nodes=config.max_nodes,
        )
        checkpoint = run.checkpoints[0]
        comparisons.append(_compare_one(f"horizon_{horizon:g}y_vs_baseline", horizon, config.baseline_mesh_points, common_t, checkpoint))

    refined_run = run_exogenous_shock_continuation(
        "warm_start", direction, z0_target, x0_target, [config.continuation_amplitudes[-1]], config.baseline_horizon,
        config.refined_mesh_points, "lq_stable_manifold", local_system, solution, tol=config.solver_tolerance, max_nodes=config.max_nodes,
    )
    refined_checkpoint = refined_run.checkpoints[0]
    refined_t = np.linspace(0.0, config.baseline_horizon, 401)
    comparisons.append(
        _compare_one(
            f"mesh_{config.refined_mesh_points}_vs_{config.baseline_mesh_points}", config.baseline_horizon,
            config.refined_mesh_points, refined_t, refined_checkpoint,
        )
    )
    return comparisons


def _convergence_order(
    config: Cs002D2Configuration,
    direction: ShockDirection,
    z_bar: float,
    x_bar: float,
    z0_target: float,
    x0_target: float,
    local_system: LocalSystem,
    solution: LqSolution,
) -> ConvergenceOrderReportD2:
    check_t = np.linspace(0.0, config.baseline_horizon, 401)
    errors_k: list[float] = []
    errors_tau: list[float] = []
    for amplitude in config.convergence_amplitude_sequence:
        run = run_exogenous_shock_continuation(
            "warm_start", direction, z0_target, x0_target, [amplitude], config.baseline_horizon, config.baseline_mesh_points,
            "lq_stable_manifold", local_system, solution, tol=config.solver_tolerance, max_nodes=config.max_nodes,
        )
        checkpoint = run.checkpoints[0]
        nonlinear = checkpoint.path_at(check_t)
        y0 = amplitude * np.array([z0_target - z_bar, x0_target - x_bar, 0.0, 0.0])
        lq_path = lq_full_state_path(check_t, y0, solution)
        errors_k.append(float(np.max(np.abs(nonlinear[0, :] - lq_path[2, :]))))
        errors_tau.append(float(np.max(np.abs((nonlinear[1, :] - local_system.anchor.tax_rate_bar) - lq_path[3, :]))))

    ratios_k = [errors_k[i] / errors_k[i + 1] for i in range(len(errors_k) - 1)]
    ratios_tau = [errors_tau[i] / errors_tau[i + 1] for i in range(len(errors_tau) - 1)]
    expected_order = config.acceptance_tolerances["convergence_order_expected"]
    rel_tol = config.acceptance_tolerances["convergence_order_relative_tolerance"]
    expected_ratio = 2.0**expected_order
    within = all(abs(r - expected_ratio) <= rel_tol * expected_ratio for r in ratios_k + ratios_tau)
    return ConvergenceOrderReportD2(
        amplitudes=list(config.convergence_amplitude_sequence), errors_k=errors_k, errors_tau=errors_tau,
        ratios_k=ratios_k, ratios_tau=ratios_tau, expected_ratio=expected_ratio, within_tolerance=within,
    )


def _routes_agree(route_a: ExogenousContinuationRun, route_b: ExogenousContinuationRun, config: Cs002D2Configuration) -> bool:
    atol = config.acceptance_tolerances["branch_agreement_atol"]
    rtol = config.acceptance_tolerances["branch_agreement_rtol"]
    check_t = np.linspace(0.0, config.baseline_horizon, 81)
    for ckpt_a, ckpt_b in zip(route_a.checkpoints, route_b.checkpoints):
        if not (ckpt_a.accepted and ckpt_b.accepted):
            return False
        if not np.allclose(ckpt_a.path_at(check_t), ckpt_b.path_at(check_t), atol=atol, rtol=rtol):
            return False
    return True


def _run_shock_direction(
    direction: ShockDirection,
    z0_target: float,
    x0_target: float,
    alpha_target: float | None,
    config: Cs002D2Configuration,
    local_system: LocalSystem,
    solution: LqSolution,
) -> ShockDirectionReport:
    anchor = local_system.anchor
    amplitudes = config.continuation_amplitudes
    route_a = run_exogenous_shock_continuation(
        "warm_start", direction, z0_target, x0_target, amplitudes, config.baseline_horizon, config.baseline_mesh_points,
        "lq_stable_manifold", local_system, solution, tol=config.solver_tolerance, max_nodes=config.max_nodes,
    )
    route_b = run_exogenous_shock_continuation(
        "crude_direct", direction, z0_target, x0_target, amplitudes, config.baseline_horizon, config.baseline_mesh_points,
        "lq_stable_manifold", local_system, solution, tol=config.solver_tolerance, max_nodes=config.max_nodes,
    )

    baseline = route_a.final  # full-amplitude (1.0) checkpoint: the "one checked transition"
    exogenous_path = baseline.exogenous_path
    horizon = config.baseline_horizon

    # Required check #2: analytic OU paths agree with numerical propagation.
    check_t = np.linspace(0.0, horizon, 401)
    z_analytic, x_analytic = exogenous_path(check_t)
    z_numeric, x_numeric = propagate_exogenous_numerically(
        check_t, exogenous_path.z0, exogenous_path.x0, exogenous_path.z_bar, exogenous_path.x_bar, exogenous_path.kappa_z, exogenous_path.kappa_x
    )
    max_diff_z = float(np.max(np.abs(z_analytic - z_numeric)))
    max_diff_x = float(np.max(np.abs(x_analytic - x_numeric)))
    ou_path_check = OuPathCheck(max_abs_difference_z=max_diff_z, max_abs_difference_x=max_diff_x, within_tolerance=max(max_diff_z, max_diff_x) < 1e-6)

    convergence = _convergence_order(config, direction, anchor.z_bar, anchor.x_bar, z0_target, x0_target, local_system, solution)

    ode_residual = independent_ode_residual(baseline.result, local_system, exogenous_path=exogenous_path)
    z_T = float(exogenous_path.z(horizon))
    x_T = float(exogenous_path.x(horizon))
    boundary_residual = independent_boundary_residual(
        baseline.result, anchor.capital_bar, anchor.tax_rate_bar, "lq_stable_manifold", local_system, solution, z_T=z_T, x_T=x_T
    )

    # Required check #10: componentwise manual RHS identities, at several
    # manufactured points sampled along this shock's own solved path.
    sample_t = np.linspace(0.0, horizon, 5)
    path_samples = baseline.path_at(sample_t)
    z_samples, x_samples = exogenous_path(sample_t)
    componentwise_rhs_reports = [
        componentwise_manual_rhs_check(
            path_samples[0, i], path_samples[1, i], path_samples[2, i], path_samples[3, i], float(z_samples[i]), float(x_samples[i]),
            anchor.z_bar, anchor.x_bar, anchor.capital_bar, local_system.parameters,
            tolerance=config.acceptance_tolerances["componentwise_rhs_tolerance"],
        )
        for i in range(sample_t.size)
    ]

    capital_T = capital_from_log(float(baseline.result.sol(np.array([horizon]))[0, 0]), anchor.capital_bar)
    tau_T = float(baseline.result.sol(np.array([horizon]))[1, 0])
    quad_tail = lq_quadratic_value_tail(capital_T, tau_T, local_system, solution, z=z_T, x=x_T)
    j_recovery = recover_j(baseline.result, local_system, "quadratic", quad_tail, horizon, exogenous_path=exogenous_path)

    exogenous_resources = recover_exogenous_resources(baseline.result, local_system, j_recovery, config.initial_public_net_worth, exogenous_path)
    budget_separation = budget_separation_residual(exogenous_resources)

    varpi_recovery = recover_varpi(local_system, exogenous_path, horizon, tail_value=0.0)

    # Required check #12/CS002 D2 review repair (finding 2): reconstruct and
    # compare the COMPLETE reported solution (not just the raw BVP state) on
    # each comparison's own full pairwise common interval, with a tolerance
    # scored separately per path. Runs after j_recovery/exogenous_resources/
    # varpi_recovery so the baseline side of every comparison reuses these
    # SAME already-recovered objects (never a separate shadow recomputation).
    horizon_mesh_comparisons = _horizon_mesh_comparisons(
        config, direction, z0_target, x0_target, local_system, solution, baseline, j_recovery, exogenous_resources, varpi_recovery
    )

    varpi_along_path_residual = independent_varpi_along_path_residual(varpi_recovery, local_system, exogenous_path)
    varpi_horizon_sensitivity = {h: recover_varpi(local_system, exogenous_path, h, tail_value=0.0).varpi_0 for h in config.varpi_tail_horizon_sequence}

    margins_t = np.linspace(0.0, horizon, 161)
    consumption_on_margins_grid = np.interp(margins_t, exogenous_resources.t_grid, exogenous_resources.consumption_path)
    margins_series = margins_time_series(baseline.result, local_system, consumption_on_margins_grid, config.numerical_scaffolding, margins_t, exogenous_path=exogenous_path)

    # Required check #8: a deliberately wrong r0=rho substitution, on this
    # SAME nonzero-shock path, must be detected against the two CORRECT
    # routes' own (small) disagreement.
    j_wrong = recover_j(baseline.result, local_system, "quadratic", quad_tail, horizon, exogenous_path=None)
    absolute_difference = abs(j_wrong.j0_hjb - j_recovery.j0_hjb)
    wrong_r0_detection = WrongR0DetectionCheck(
        j0_correct=j_recovery.j0_hjb, j0_wrong_r0_rho=j_wrong.j0_hjb, absolute_difference=absolute_difference,
        detected=absolute_difference > max(10.0 * j_recovery.route_disagreement, 1e-9),
    )

    effect_floor = 1e-10
    factor = config.acceptance_tolerances["route_disagreement_effect_factor"]
    j_effect = max(abs(j_recovery.j0_hjb - anchor.fiscal_wealth_bar), effect_floor)
    x_effect = max(float(np.max(np.abs(exogenous_resources.x_path_ode - exogenous_resources.x_0))), effect_floor)
    varpi_effect = max(float(np.max(np.abs(varpi_recovery.varpi_path_ode))), effect_floor)

    margins_min = {name: float(np.min(series)) for name, series in margins_series.items()}
    no_boundary_reached = all(value > 0.0 for value in margins_min.values()) and exogenous_resources.x_0 > 0.0

    checks = {
        "zero_amplitude_returns_anchor": route_a.checkpoints[0].accepted,
        "route_a_all_amplitudes_converged": route_a.all_accepted,
        "route_b_all_amplitudes_converged": route_b.all_accepted,
        "ou_path_matches_numerical_propagation": ou_path_check.within_tolerance,
        "ode_residual_within_tolerance": ode_residual.max_scaled_residual <= config.acceptance_tolerances["ode_residual_max"],
        "boundary_residual_within_tolerance": boundary_residual.max_scaled_residual <= config.acceptance_tolerances["boundary_residual_max"],
        "componentwise_rhs_agrees": all(r.all_agree for r in componentwise_rhs_reports),
        "horizon_mesh_stability": all(c.within_tolerance for c in horizon_mesh_comparisons),
        "lq_convergence_expected_order": convergence.within_tolerance,
        "j_recovery_routes_agree": j_recovery.route_disagreement * factor <= j_effect,
        "x_recovery_routes_agree": exogenous_resources.x_routes_max_abs_deviation * factor <= x_effect,
        "n_recovery_routes_agree": exogenous_resources.n_routes_max_abs_deviation * factor <= max(exogenous_resources.x_0, effect_floor),
        "varpi_recovery_routes_agree": varpi_recovery.route_disagreement_max_abs * factor <= varpi_effect,
        "varpi_along_path_residual_within_tolerance": varpi_along_path_residual.max_scaled_residual <= config.acceptance_tolerances["ode_residual_max"],
        "budget_separation_within_tolerance": budget_separation.max_scaled_residual <= config.acceptance_tolerances["budget_separation_residual_max"],
        "wrong_r0_is_detected": wrong_r0_detection.detected,
        "branch_agreement_route_a_vs_b": _routes_agree(route_a, route_b, config),
        "no_boundary_reached_on_baseline_path": no_boundary_reached,
    }

    outcome = determine_outcome(
        checks,
        numerical_failure_check_names=(
            "zero_amplitude_returns_anchor", "route_a_all_amplitudes_converged", "route_b_all_amplitudes_converged",
            "ou_path_matches_numerical_propagation", "ode_residual_within_tolerance", "boundary_residual_within_tolerance",
            "componentwise_rhs_agrees", "horizon_mesh_stability", "lq_convergence_expected_order",
            "j_recovery_routes_agree", "x_recovery_routes_agree", "n_recovery_routes_agree", "varpi_recovery_routes_agree",
            "varpi_along_path_residual_within_tolerance", "budget_separation_within_tolerance", "wrong_r0_is_detected",
        ),
        branch_sensitivity_check_names=("branch_agreement_route_a_vs_b",),
        boundary_check_names=("no_boundary_reached_on_baseline_path",),
    )

    return ShockDirectionReport(
        direction=direction, z0_target=z0_target, x0_target=x0_target, alpha_target=alpha_target,
        route_a=route_a, route_b=route_b, ou_path_check=ou_path_check, horizon_mesh_comparisons=horizon_mesh_comparisons,
        convergence=convergence, ode_residual=ode_residual, boundary_residual=boundary_residual,
        componentwise_rhs_reports=componentwise_rhs_reports, j_recovery=j_recovery, exogenous_resources=exogenous_resources,
        budget_separation=budget_separation, varpi_recovery=varpi_recovery, varpi_along_path_residual=varpi_along_path_residual,
        varpi_horizon_sensitivity=varpi_horizon_sensitivity, margins_t_grid=margins_t, margins_series=margins_series,
        wrong_r0_detection=wrong_r0_detection, outcome=outcome,
    )


def _combine_outcomes(productivity: AggregateOutcome, automation: AggregateOutcome) -> AggregateOutcome:
    worse = productivity if _OUTCOME_PRIORITY[productivity.outcome] >= _OUTCOME_PRIORITY[automation.outcome] else automation
    combined_checks = {f"productivity_{k}": v for k, v in productivity.checks.items()}
    combined_checks.update({f"automation_{k}": v for k, v in automation.checks.items()})
    failed = [name for name, ok in combined_checks.items() if not ok]
    conclusion = (
        "Both shock directions reached computational_pass. Exploratory prototype evidence only -- not an approved CS002 result."
        if worse.outcome == "computational_pass"
        else f"outcome={worse.outcome}; failed checks: {failed}."
    )
    return AggregateOutcome(outcome=worse.outcome, checks=combined_checks, failed_checks=failed, conclusion=conclusion)


def run_d2_experiment(config: Cs002D2Configuration) -> ExperimentReportD2:
    primitives = config.cs001.parameters
    anchor = compute_steady_state(primitives)
    local_system = build_local_system(primitives, anchor)
    solution = solve_lq_system(local_system)

    z0_target_productivity = anchor.z_bar + config.delta_z_productivity
    alpha_bar = automation_share(anchor.x_bar, primitives)
    alpha_target = alpha_bar + config.delta_alpha_automation
    x0_target_automation = invert_automation_share(alpha_target, primitives)

    productivity = _run_shock_direction("productivity", z0_target_productivity, anchor.x_bar, None, config, local_system, solution)
    automation = _run_shock_direction("automation", anchor.z_bar, x0_target_automation, alpha_target, config, local_system, solution)

    outcome = _combine_outcomes(productivity.outcome, automation.outcome)

    return ExperimentReportD2(config=config, local_system=local_system, solution=solution, productivity=productivity, automation=automation, outcome=outcome)
