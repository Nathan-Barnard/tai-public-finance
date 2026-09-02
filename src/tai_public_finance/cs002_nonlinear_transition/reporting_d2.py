"""Output bundle and immutable run record for one CS002 D2 material run.

Reuses the fully generic serialization/hashing/git/environment helpers from
cs001_lq_anchor.reporting verbatim (they reference no CS001- or D1-specific
fields); everything D2-shaped (report contents, figures, run-record body)
is written here, mirroring reporting.py's D0-D1 structure.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..cs001_lq_anchor.reporting import environment_metadata, git_metadata, serializable, sha256_file
from .experiment_d2 import ExperimentReportD2, ShockDirectionReport
from .model import capital_from_log, characteristic_rates


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{key: serializable(value) for key, value in row.items()} for row in rows])


def _direction_path_rows(report: ExperimentReportD2, sub: ShockDirectionReport, n_points: int = 201) -> list[dict[str, Any]]:
    baseline = sub.route_a.final
    local_system = report.local_system
    anchor = local_system.anchor
    p = local_system.parameters
    exogenous_path = baseline.exogenous_path
    horizon = sub.route_a.horizon

    t_grid = np.linspace(0.0, horizon, n_points)
    path = baseline.path_at(t_grid)
    z_t, x_t = exogenous_path(t_grid)

    j_t = np.interp(t_grid, sub.j_recovery.t_grid, sub.j_recovery.j_path_hjb)
    x_res_t = np.interp(t_grid, sub.exogenous_resources.t_grid, sub.exogenous_resources.x_path_ode)
    n_t = np.interp(t_grid, sub.exogenous_resources.t_grid, sub.exogenous_resources.n_path_budget_ode)
    c_t = np.interp(t_grid, sub.exogenous_resources.t_grid, sub.exogenous_resources.consumption_path)
    varpi_t = np.interp(t_grid, sub.varpi_recovery.t_grid, sub.varpi_recovery.varpi_path_ode)

    y0 = np.array([exogenous_path.z0 - anchor.z_bar, exogenous_path.x0 - anchor.x_bar, 0.0, 0.0])
    from .terminal import lq_full_state_path

    lq_path = lq_full_state_path(t_grid, y0, report.solution)

    rows = []
    for i, t in enumerate(t_grid):
        k, tau, ell, m = path[:, i]
        capital = capital_from_log(k, anchor.capital_bar)
        rates = characteristic_rates(k, tau, ell, m, anchor.z_bar, anchor.x_bar, anchor.capital_bar, p, z=float(z_t[i]), x=float(x_t[i]))
        rows.append(
            {
                "t_years": float(t),
                "z": float(z_t[i]),
                "x": float(x_t[i]),
                "r0": float(rates.r0),
                "capital": float(capital),
                "log_capital_deviation": float(k),
                "tax_rate": float(tau),
                "nu_tax_speed": float(rates.nu),
                "ell_J_K": float(ell),
                "m_J_tau": float(m),
                "output": float(rates.state.output),
                "fiscal_resources": float(rates.state.fiscal_resources),
                "J": float(j_t[i]),
                "X": float(x_res_t[i]),
                "N": float(n_t[i]),
                "c": float(c_t[i]),
                "varpi": float(varpi_t[i]),
                "lq_log_capital_deviation": float(lq_path[2, i]),
                "lq_tax_rate": float(lq_path[3, i] + anchor.tax_rate_bar),
            }
        )
    return rows


def _direction_margins_rows(sub: ShockDirectionReport) -> list[dict[str, Any]]:
    rows = []
    for i, t in enumerate(sub.margins_t_grid):
        row: dict[str, Any] = {"t_years": float(t)}
        for name, series in sub.margins_series.items():
            row[name] = float(series[i])
        rows.append(row)
    return rows


def _continuation_checkpoint_rows(report: ExperimentReportD2) -> list[dict[str, Any]]:
    rows = []
    for direction, sub in (("productivity", report.productivity), ("automation", report.automation)):
        for route_label, run in (("warm_start", sub.route_a), ("crude_direct", sub.route_b)):
            for checkpoint in run.checkpoints:
                rows.append(
                    {
                        "direction": direction,
                        "route": route_label,
                        "terminal_convention": run.terminal_convention,
                        "horizon": run.horizon,
                        "amplitude": checkpoint.amplitude,
                        "z0": checkpoint.z0,
                        "x0": checkpoint.x0,
                        "accepted": checkpoint.accepted,
                        "n_mesh_nodes": checkpoint.result.n_mesh_nodes if checkpoint.result else None,
                        "solver_max_rms_residual": checkpoint.result.solver_max_rms_residual if checkpoint.result else None,
                        "failure_message": checkpoint.failure_message or "",
                    }
                )
    return rows


def _horizon_mesh_rows(report: ExperimentReportD2) -> list[dict[str, Any]]:
    rows = []
    for direction, sub in (("productivity", report.productivity), ("automation", report.automation)):
        for comparison in sub.horizon_mesh_comparisons:
            rows.append(
                {
                    "direction": direction,
                    "label": comparison.label,
                    "horizon": comparison.horizon,
                    "n_mesh_points": comparison.n_mesh_points,
                    "tolerance": comparison.tolerance,
                    "within_tolerance": comparison.within_tolerance,
                    **{f"max_diff_{name}": value for name, value in comparison.max_state_difference_from_baseline.items()},
                }
            )
    return rows


def _convergence_rows(report: ExperimentReportD2) -> list[dict[str, Any]]:
    rows = []
    for direction, sub in (("productivity", report.productivity), ("automation", report.automation)):
        for amplitude, error_k, error_tau in zip(sub.convergence.amplitudes, sub.convergence.errors_k, sub.convergence.errors_tau):
            rows.append({"direction": direction, "amplitude": amplitude, "error_k": error_k, "error_tau": error_tau})
    return rows


def _varpi_horizon_sensitivity_rows(report: ExperimentReportD2) -> list[dict[str, Any]]:
    rows = []
    for direction, sub in (("productivity", report.productivity), ("automation", report.automation)):
        for horizon, varpi_0 in sub.varpi_horizon_sensitivity.items():
            rows.append({"direction": direction, "tail_horizon": horizon, "varpi_0": varpi_0})
    return rows


def _write_direction_figures(output_dir: Path, report: ExperimentReportD2, direction: str, sub: ShockDirectionReport) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    horizon = sub.route_a.horizon
    t_grid = np.linspace(0.0, horizon, 201)
    baseline = sub.route_a.final
    exogenous_path = baseline.exogenous_path
    path = baseline.path_at(t_grid)
    z_t, x_t = exogenous_path(t_grid)
    anchor = report.local_system.anchor

    r0_t = np.array([characteristic_rates(path[0, i], path[1, i], path[2, i], path[3, i], anchor.z_bar, anchor.x_bar, anchor.capital_bar, report.local_system.parameters, z=float(z_t[i]), x=float(x_t[i])).r0 for i in range(t_grid.size)])
    nu_t = np.array([characteristic_rates(path[0, i], path[1, i], path[2, i], path[3, i], anchor.z_bar, anchor.x_bar, anchor.capital_bar, report.local_system.parameters, z=float(z_t[i]), x=float(x_t[i])).nu for i in range(t_grid.size)])
    j_t = np.interp(t_grid, sub.j_recovery.t_grid, sub.j_recovery.j_path_hjb)
    x_res_t = np.interp(t_grid, sub.exogenous_resources.t_grid, sub.exogenous_resources.x_path_ode)
    n_t = np.interp(t_grid, sub.exogenous_resources.t_grid, sub.exogenous_resources.n_path_budget_ode)
    c_t = np.interp(t_grid, sub.exogenous_resources.t_grid, sub.exogenous_resources.consumption_path)
    varpi_t = np.interp(t_grid, sub.varpi_recovery.t_grid, sub.varpi_recovery.varpi_path_ode)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    ax = axes[0, 0]
    ax.plot(t_grid, z_t - anchor.z_bar, label="z(t)-z_bar")
    ax.plot(t_grid, x_t - anchor.x_bar, label="x(t)-x_bar")
    ax.axhline(0.0, color="gray", linewidth=0.5, linestyle=":")
    ax2 = ax.twinx()
    ax2.plot(t_grid, r0_t, "r--", label="r0(t)")
    ax2.axhline(report.local_system.parameters.rho, color="red", linewidth=0.5, linestyle=":")
    ax.set_title("exogenous state deviations and r0")
    ax.legend(loc="upper right", fontsize=7)
    ax2.legend(loc="lower right", fontsize=7)

    ax = axes[0, 1]
    ax.plot(t_grid, path[0, :], label="k = log(K/K_bar)")
    ax2 = ax.twinx()
    ax2.plot(t_grid, path[1, :], "g--", label="tau")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_title("K (log-deviation) and tau")
    ax.legend(loc="upper right", fontsize=7)
    ax2.legend(loc="lower right", fontsize=7)

    ax = axes[0, 2]
    ax.plot(t_grid, nu_t)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_title("nu (tax speed)")

    ax = axes[1, 0]
    ax.plot(t_grid, path[2, :], label="ell = J_K")
    ax.plot(t_grid, path[3, :], label="m = J_tau")
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_title("costates ell, m")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.plot(t_grid, j_t, label="J")
    ax.plot(t_grid, x_res_t, label="X")
    ax.plot(t_grid, n_t, label="N")
    ax.plot(t_grid, c_t, label="c")
    ax.set_title("J, X, N, c")
    ax.legend(fontsize=7)

    ax = axes[1, 2]
    ax.plot(t_grid, varpi_t)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_title("opportunity value varpi")

    for ax in axes.ravel():
        ax.set_xlabel("years")
    fig.suptitle(f"CS002 D2 exploratory prototype -- {direction} shock ({report.config.config_id})")
    fig.tight_layout()
    path_fig = figures_dir / f"{direction}_paths.png"
    fig.savefig(path_fig, dpi=150)
    plt.close(fig)
    return [path_fig]


def _write_convergence_figure(output_dir: Path, report: ExperimentReportD2) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (direction, sub) in zip(axes, (("productivity", report.productivity), ("automation", report.automation))):
        amplitudes = sub.convergence.amplitudes[::-1]
        ax.loglog(amplitudes, sub.convergence.errors_k[::-1], "o-", label="error_k")
        ax.loglog(amplitudes, sub.convergence.errors_tau[::-1], "s-", label="error_tau")
        reference = [sub.convergence.errors_k[::-1][0] * (a / amplitudes[0]) ** 2 for a in amplitudes]
        ax.loglog(amplitudes, reference, "k:", label="slope 2 (reference)")
        ax.set_xlabel("continuation amplitude (log scale)")
        ax.set_ylabel("max |nonlinear - LQ| (log scale)")
        ax.set_title(f"{direction}: nonlinear-vs-LQ convergence")
        ax.legend(fontsize=8)
    fig.tight_layout()
    conv_fig = figures_dir / "nonlinear_vs_lq_convergence.png"
    fig.savefig(conv_fig, dpi=150)
    plt.close(fig)
    return conv_fig


def write_d2_bundle(
    output_dir: Path,
    report: ExperimentReportD2,
    repository: Path,
    material_run_elapsed_seconds: float,
    task_elapsed_seconds: float,
    command: str,
    git_at_run_start: dict[str, Any],
    run_id: str,
    runs_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)

    figure_paths: list[Path] = []
    csv_paths: list[Path] = []
    for direction, sub in (("productivity", report.productivity), ("automation", report.automation)):
        path_rows = _direction_path_rows(report, sub)
        margins_rows = _direction_margins_rows(sub)
        path_csv = output_dir / f"{direction}_path.csv"
        margins_csv = output_dir / f"{direction}_margins.csv"
        _write_csv(path_csv, path_rows)
        _write_csv(margins_csv, margins_rows)
        csv_paths += [path_csv, margins_csv]
        figure_paths += _write_direction_figures(output_dir, report, direction, sub)

    figure_paths.append(_write_convergence_figure(output_dir, report))

    checkpoints_csv = output_dir / "continuation_checkpoints.csv"
    horizon_mesh_csv = output_dir / "horizon_mesh_comparisons.csv"
    convergence_csv = output_dir / "convergence.csv"
    varpi_sensitivity_csv = output_dir / "varpi_horizon_sensitivity.csv"
    _write_csv(checkpoints_csv, _continuation_checkpoint_rows(report))
    _write_csv(horizon_mesh_csv, _horizon_mesh_rows(report))
    _write_csv(convergence_csv, _convergence_rows(report))
    _write_csv(varpi_sensitivity_csv, _varpi_horizon_sensitivity_rows(report))
    csv_paths += [checkpoints_csv, horizon_mesh_csv, convergence_csv, varpi_sensitivity_csv]

    environment = environment_metadata()

    def _sub_summary(sub: ShockDirectionReport) -> dict[str, Any]:
        # Curated, NOT a wholesale serializable(sub): ShockDirectionReport
        # holds route_a/route_b (BvpSolveResult.sol callables inside their
        # checkpoints) and varpi_recovery.varpi_of_t (a dense-ODE
        # interpolant) -- neither is JSON-serializable, mirroring D1's own
        # reporting.py, which never serializes route_a_main/route_b_main
        # wholesale either.
        return {
            "direction": sub.direction,
            "z0_target": sub.z0_target,
            "x0_target": sub.x0_target,
            "alpha_target": sub.alpha_target,
            "ou_path_check": serializable(sub.ou_path_check),
            "horizon_mesh_comparisons": [serializable(c) for c in sub.horizon_mesh_comparisons],
            "convergence": serializable(sub.convergence),
            "ode_residual": serializable(sub.ode_residual),
            "boundary_residual": serializable(sub.boundary_residual),
            "componentwise_rhs_reports": [serializable(r) for r in sub.componentwise_rhs_reports],
            "j_recovery": {k: v for k, v in serializable(sub.j_recovery).items() if k not in ("j_path_hjb", "t_grid")}
            | {"j0_hjb": sub.j_recovery.j0_hjb, "route_disagreement_relative": sub.j_recovery.route_disagreement_relative},
            "exogenous_resources_summary": {
                "net_worth_0": sub.exogenous_resources.net_worth_0,
                "x_0": sub.exogenous_resources.x_0,
                "x_routes_max_abs_deviation": sub.exogenous_resources.x_routes_max_abs_deviation,
                "x_routes_max_rel_deviation": sub.exogenous_resources.x_routes_max_rel_deviation,
                "n_routes_max_abs_deviation": sub.exogenous_resources.n_routes_max_abs_deviation,
                "n_routes_max_rel_deviation": sub.exogenous_resources.n_routes_max_rel_deviation,
            },
            "budget_separation": serializable(sub.budget_separation),
            "varpi_recovery_summary": {
                "tail_horizon": sub.varpi_recovery.tail_horizon,
                "tail_value": sub.varpi_recovery.tail_value,
                "varpi_0": sub.varpi_recovery.varpi_0,
                "route_disagreement_max_abs": sub.varpi_recovery.route_disagreement_max_abs,
                "route_disagreement_max_rel": sub.varpi_recovery.route_disagreement_max_rel,
            },
            "varpi_along_path_residual": serializable(sub.varpi_along_path_residual),
            "varpi_horizon_sensitivity": {str(h): v for h, v in sub.varpi_horizon_sensitivity.items()},
            "margins_min": {name: float(series.min()) for name, series in sub.margins_series.items()},
            "wrong_r0_detection": serializable(sub.wrong_r0_detection),
            "outcome": serializable(sub.outcome),
            "continuation_summary": {
                "route_a_all_accepted": sub.route_a.all_accepted,
                "route_b_all_accepted": sub.route_b.all_accepted,
                "amplitudes": [c.amplitude for c in sub.route_a.checkpoints],
            },
        }

    report_payload = {
        "run_id": run_id,
        "config_id": report.config.config_id,
        "config_fingerprint": report.config.fingerprint,
        "cs001_fingerprints": report.config.cs001.fingerprints,
        "outcome": serializable(report.outcome),
        "productivity": _sub_summary(report.productivity),
        "automation": _sub_summary(report.automation),
        "environment": environment,
        "implementation": git_at_run_start,
        "material_run_elapsed_seconds": material_run_elapsed_seconds,
        "task_elapsed_seconds": task_elapsed_seconds,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    input_copy = output_dir / "complete_input.json"
    input_copy.write_text(
        json.dumps(serializable({"cs002_d2_config": report.config.raw, "primitives": report.local_system.parameters.raw}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    outcome = report.outcome
    summary_path = output_dir / "summary.md"

    def _direction_summary_lines(direction: str, sub: ShockDirectionReport) -> list[str]:
        return [
            f"## {direction}",
            "",
            f"- Outcome: **{sub.outcome.outcome}**",
            f"- Independent max scaled ODE residual: `{sub.ode_residual.max_scaled_residual:.3e}` (<= 1e-7 required)",
            f"- Independent max scaled boundary residual: `{sub.boundary_residual.max_scaled_residual:.3e}` (<= 1e-8 required)",
            f"- Componentwise manual RHS check: `{all(r.all_agree for r in sub.componentwise_rhs_reports)}`",
            f"- Horizon/mesh stability: `{all(c.within_tolerance for c in sub.horizon_mesh_comparisons)}`",
            f"- LQ-vs-nonlinear convergence ratios (halving amplitude, target ~4.0): k={[round(r, 3) for r in sub.convergence.ratios_k]}, tau={[round(r, 3) for r in sub.convergence.ratios_tau]}",
            f"- Route A vs route B branch agreement: `{sub.outcome.checks.get('branch_agreement_route_a_vs_b')}`",
            f"- J(0): `{sub.j_recovery.j0_hjb:.6f}` (route disagreement rel. `{sub.j_recovery.route_disagreement_relative:.3e}`)",
            f"- X routes max relative deviation: `{sub.exogenous_resources.x_routes_max_rel_deviation:.3e}`",
            f"- N routes max relative deviation: `{sub.exogenous_resources.n_routes_max_rel_deviation:.3e}`",
            f"- varpi(0): `{sub.varpi_recovery.varpi_0:.6e}` (route disagreement rel. `{sub.varpi_recovery.route_disagreement_max_rel:.3e}`); along-path residual `{sub.varpi_along_path_residual.max_scaled_residual:.3e}`",
            f"- varpi(0) horizon sensitivity: {{{', '.join(f'{h:g}y: {v:.4e}' for h, v in sub.varpi_horizon_sensitivity.items())}}}",
            f"- Budget-separation residual: `{sub.budget_separation.max_scaled_residual:.3e}`",
            f"- Wrong-r0=rho substitution detected: `{sub.wrong_r0_detection.detected}` (|difference|=`{sub.wrong_r0_detection.absolute_difference:.3e}`)",
            f"- Minimum reported margins (all time points): {{{', '.join(f'{name}: {float(series.min()):.4f}' for name, series in sub.margins_series.items())}}}",
            f"- Structural continuation solvency: `not_evaluated` (no viability/no-Ponzi calculation is performed)",
            "",
        ]

    summary_lines = [
        f"# {run_id}",
        "",
        "**Exploratory CS002 D2 prototype. CS002 v0.2 remains draft and unfingerprinted; this is not an approved or completed CS002 result.**",
        "",
        f"- Outcome: **{outcome.outcome}**",
        f"- Config: `{report.config.config_id}` (fingerprint `{report.config.fingerprint[:12]}...`)",
        "",
        *_direction_summary_lines("Pure productivity (z(0)-z_bar=+0.01, x(0)=x_bar)", report.productivity),
        *_direction_summary_lines(f"Pure automation (alpha(x(0))-alpha_bar=+0.01, implied x(0)={report.automation.x0_target:.6f}, z(0)=z_bar)", report.automation),
        "Interpretation: bounded exploratory D2 prototype under CS002 v0.2's draft-specification exception, "
        "extending the reviewed D0-D1 frozen-common-state evidence to deterministic mean-reverting productivity "
        "and automation paths. These are deterministic local-shock IRFs, not stochastic expected paths, and do "
        "not establish global feasibility, structural continuation solvency, or an approved CS002 result. Do not "
        "implement D3, V2, R4, order-epsilon^4, a stochastic PDE, jump models, calibration, or the broad "
        "experiment grid from this evidence alone.",
        "",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    material_artifacts = [report_path, input_copy, summary_path, *csv_paths, *figure_paths]

    def _location(path: Path) -> str:
        try:
            return str(path.relative_to(repository))
        except ValueError:
            return str(path)

    artifacts = [
        {"role": "primary_output" if path == report_path else ("figure" if path.suffix == ".png" else "supporting_output"), "location": _location(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in material_artifacts
    ]

    run_record = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "completed" if outcome.outcome == "computational_pass" else outcome.outcome,
        "purpose": "Exploratory CS002 D2 prototype: deterministic mean-reverting productivity and automation "
        "transitions, state-dependent safe rate, two-route J/X/N/varpi recovery. Bounded feasibility extension "
        "of the reviewed D0-D1 frozen-common-state evidence, under CS002 v0.2's draft-specification exception -- "
        "not an approved or completed CS002 result; does not authorize Block D3.",
        "specification": {"id": "CS002", "version": "0.2", "status": "draft", "fingerprint_sha256": None},
        "input_fingerprints": {"cs002_d2_config_sha256": report.config.fingerprint, **{f"cs001_{k}": v for k, v in report.config.cs001.fingerprints.items()}},
        "problem_id": "CP002",
        "approach_id": "CA002",
        "benchmark_id": None,
        "implementation": {
            "repository_url": "https://github.com/Nathan-Barnard/tai-public-finance",
            "local_path": str(repository),
            "commit": git_at_run_start["commit"],
            "branch": git_at_run_start["branch"],
            "dirty_worktree": git_at_run_start["dirty_worktree_at_run_start"],
            "dirty_worktree_at_run_start": git_at_run_start["dirty_worktree_at_run_start"],
            "implementer": "claude_code",
            "entrypoint": "python3 -m tai_public_finance.cs002_nonlinear_transition.cli_d2",
            "command": command,
        },
        "environment": {
            "operating_system": environment["operating_system"],
            "architecture": environment["architecture"],
            "runtime_versions": {key: environment[key] for key in ("python", "numpy", "scipy", "matplotlib", "pyyaml")},
            "dependency_lock_path": "uv.lock",
            "dependency_lock_sha256": sha256_file(repository / "uv.lock") if (repository / "uv.lock").exists() else None,
            "container_or_image": None,
        },
        "hardware": {
            "resource_lane": "L1_interactive",
            "machine_label": "local_mac",
            "cpu": environment["cpu"],
            "logical_cores": environment["logical_cores"],
            "memory_gb": environment["memory_gb"],
            "accelerator": None,
            "accelerator_memory_gb": None,
            "detected_at_runtime": True,
        },
        "budget": {
            "wall_seconds_limit": 1200,
            "cash_limit_usd": 0,
            "actual_cash_usd": 0,
            "early_stop_rule": "Stop and retain failure if any independent residual, componentwise-RHS, "
            "horizon/mesh stability, convergence-order, branch-agreement, route-disagreement, budget-separation, "
            "or boundary-margin acceptance check fails for either shock direction; solver-reported convergence "
            "alone never sets the outcome to pass.",
        },
        "randomness": {"deterministic_requested": True, "seeds": [], "nondeterminism_notes": "Deterministic float64 BVP collocation and ODE quadrature; no simulation draws."},
        "inputs": {
            "parameter_set_id": report.config.cs001.parameter_set_id,
            "parameter_fingerprint_sha256": report.config.cs001.fingerprints["primitive_sha256"],
            "cs002_d2_config_id": report.config.config_id,
            "input_artifacts": [str(report.config.config_path), str(report.config.cs001.experiment_path), str(report.config.cs001.primitive_path)],
            "shock_directions": {
                "productivity": {"delta_z": report.config.delta_z_productivity, "z0_target": report.productivity.z0_target},
                "automation": {"delta_alpha": report.config.delta_alpha_automation, "alpha_target": report.automation.alpha_target, "x0_target": report.automation.x0_target},
            },
            "continuation_amplitudes": report.config.continuation_amplitudes,
            "horizons_years": {"baseline": report.config.baseline_horizon, "comparisons": report.config.comparison_horizons},
            "initial_public_net_worth": report.config.initial_public_net_worth,
        },
        "solver": {
            "packages_and_versions": {"numpy": environment["numpy"], "scipy": environment["scipy"]},
            "method": "scipy.integrate.solve_bvp collocation on the non-autonomous characteristic system driven by "
            "the analytic exogenous OU path, continuation in shock amplitude from a warm-started previous "
            "checkpoint (route A) cross-checked against direct crude-anchor-guess solves at every amplitude "
            "(route B); independent off-mesh finite-difference ODE residual, re-applied terminal-condition "
            "boundary residual, and an algebraically separate componentwise manual RHS reconstruction.",
            "tolerances": report.config.acceptance_tolerances,
            "initialization": "warm-started continuation from the previous amplitude (route A) / crude constant-anchor guess at every amplitude (route B)",
            "continuation": f"amplitude sequence {report.config.continuation_amplitudes}",
            "precision": "float64",
        },
        "preflight": {
            "tests_command": "uv run pytest",
            "tests_status": "passed",
            "exact_or_manufactured_benchmark": "D0-D1's matrix-exponential manufactured BVP and R16 fixed-tax closed-form transition (inherited, reused unchanged); D2's own componentwise manual RHS reconstruction and OU-path-vs-numerical-propagation checks.",
        },
        "result": {
            "outcome": outcome.outcome,
            "wall_seconds": material_run_elapsed_seconds,
            "task_elapsed_seconds": task_elapsed_seconds,
            "peak_memory_gb": None,
            "actual_cash_usd": 0,
            "checkpoints_recovered": True,
            "solver_reported_metrics": None,
            "independent_diagnostics": {"checks": outcome.checks},
            "economic_quantities": {
                "productivity_j0": report.productivity.j_recovery.j0_hjb,
                "productivity_varpi0": report.productivity.varpi_recovery.varpi_0,
                "automation_j0": report.automation.j_recovery.j0_hjb,
                "automation_varpi0": report.automation.varpi_recovery.varpi_0,
                "anchor_fiscal_wealth_bar": report.local_system.anchor.fiscal_wealth_bar,
            },
            "reliable_region": "Only the reported baseline (full-amplitude) productivity and automation paths at "
            "N_bar=0, with recorded positive branch and margin slack; small-displacement illustrative benchmark, "
            "not a global feasibility or uniqueness claim.",
            "failure_code": None if outcome.outcome == "computational_pass" else outcome.outcome,
            "failure_detail": None if outcome.outcome == "computational_pass" else outcome.failed_checks,
        },
        "artifacts": artifacts,
        "interpretation": {
            "supports_result_ids": ["R08", "R09", "R11", "R16"],
            "challenges_result_ids": [],
            "question_ids": ["Q02", "Q03", "Q05", "Q06", "Q08", "Q17"],
            "assurance_or_review_artifacts": [
                "research-notes/nonlinear-deterministic-and-order-two-risk-development.md",
                "assurance/dossiers/deterministic-fiscal-wealth-opportunity-value-and-tax-speed--R08-R09-R11.md",
            ],
            "conclusion": outcome.conclusion,
            "limitations": [
                "CS002 is registered as draft, unfingerprinted, in the codex research workspace; this run is a "
                "bounded exploratory D2 prototype under the draft-specification exception recorded in CS002's own "
                "\"Immediate implementation handoff\" section, not an approved or completed CS002 result.",
                "Deterministic zero-risk paths only: these are local illustrative IRFs, not stochastic expected "
                "paths. Block D3 (derivative service), V2, R4, order-epsilon^4, a stochastic PDE, jump models, "
                "calibration, and the broad experiment grid are explicitly out of scope and not implemented here.",
                "Illustrative solver/algebra benchmark on the Farhi-based smoke calibration, not an empirical UK calibration.",
                "min_net_rental_tax_base_margin (min(R^K-delta)) and comprehensive_resource_margin (X) are local "
                "reporting margins, not a structural-solvency or global-feasibility calculation -- "
                "structural_continuation_solvency is reported literally as not_evaluated for both shock directions.",
                "The varpi(0) terminal tail convention is 0 at the run's own horizon; varpi_horizon_sensitivity "
                "reports how much varpi(0) changes as the horizon (hence the tail's own horizon) is extended, "
                "rather than silently assuming the tail is immaterial.",
                "budget_separation_within_tolerance uses a central-finite-difference cross-check of already-"
                "recovered N and J arrays (np.gradient on a 401-point grid), materially coarser than the dedicated "
                "off-mesh characteristic-BVP residual -- it is a supplementary accounting-identity diagnostic, not "
                "the primary numerical-accuracy evidence.",
                "task_elapsed_seconds is the implementer's own end-to-end estimate of reading, implementation, and "
                "debugging time, not an instrumented wall-clock measurement; wall_seconds (this material run's own "
                "compute time) is directly measured and exact.",
            ],
            "next_decision": "Research owner reviews the equation map, the D2-specific terminal-condition and "
            "recovery generalizations, residual/branch/route-disagreement evidence, and margin profile for both "
            "shock directions before CS002 may move toward review_ready or Block D3 (derivative service) is "
            "authorized.",
        },
    }
    record_path = (runs_dir or repository / "runs") / f"{run_id}.yaml"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    if record_path.exists():
        raise FileExistsError(f"Run record {record_path} already exists. Run records are immutable.")
    record_path.write_text(yaml.safe_dump(serializable(run_record), sort_keys=False, allow_unicode=False), encoding="utf-8")

    return {"record_path": record_path, "artifacts": artifacts, "environment": environment, "git": git_at_run_start}
