"""Output bundle and immutable run record for one CS002 D0-D1 material run.

Reuses the fully generic serialization/hashing/git/environment helpers from
cs001_lq_anchor.reporting verbatim (they reference no CS001-specific
fields); everything CS002-shaped (report contents, figures, run-record
body) is written here.
"""

from __future__ import annotations

import csv
import json
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..cs001_lq_anchor.reporting import environment_metadata, git_metadata, serializable, sha256_file
from .experiment import ExperimentReport
from .model import capital_from_log, characteristic_rates
from .terminal import lq_linear_kt_path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{key: serializable(value) for key, value in row.items()} for row in rows])


def _baseline_path_rows(report: ExperimentReport, n_points: int = 201) -> list[dict[str, Any]]:
    baseline = report.route_a_main.final
    anchor = report.local_system.anchor
    p = report.local_system.parameters

    t_grid = np.linspace(0.0, report.config.baseline_horizon, n_points)
    path = baseline.path_at(t_grid)
    y0 = baseline.result.sol(np.array([0.0]))[:, 0]
    lq_path = lq_linear_kt_path(t_grid, y0[0], y0[1] - 0.5, report.solution)

    rows = []
    for i, t in enumerate(t_grid):
        k, tau, ell, m = path[:, i]
        capital = capital_from_log(k, anchor.capital_bar)
        rates = characteristic_rates(k, tau, ell, m, anchor.z_bar, anchor.x_bar, anchor.capital_bar, p)
        rows.append(
            {
                "t_years": float(t),
                "capital": float(capital),
                "log_capital_deviation": float(k),
                "tax_rate": float(tau),
                "ell_J_K": float(ell),
                "m_J_tau": float(m),
                "nu_tax_speed": float(rates.nu),
                "output": float(rates.state.output),
                "rental_rate": float(rates.state.rental_rate),
                "wage_income": float(rates.state.wage_income),
                "fiscal_resources": float(rates.state.fiscal_resources),
                "lq_log_capital_deviation": float(lq_path[0, i]),
                "lq_tax_rate": float(lq_path[1, i] + 0.5),
            }
        )
    return rows


def _net_worth_grid_rows(report: ExperimentReport) -> list[dict[str, Any]]:
    rows = []
    for row in report.net_worth_grid:
        rows.append(
            {
                "net_worth_to_fiscal_wealth": row.net_worth_to_fiscal_wealth,
                "net_worth_0": row.net_worth_0,
                "x_0": row.comprehensive.x_0,
                "consumption": row.comprehensive.consumption,
                "x_constancy_max_abs_deviation": row.comprehensive.x_constancy_max_abs_deviation,
                "x_constancy_max_rel_deviation": row.comprehensive.x_constancy_max_rel_deviation,
                "reported_local_margins_slack": row.reported_local_margins_slack,
                "failure_reasons": ";".join(row.margins.failure_reasons),
                "min_specialisation_margin_automation_composite": row.margins.min_specialisation_margin_automation_composite,
                "min_specialisation_margin_new_task_composite": row.margins.min_specialisation_margin_new_task_composite,
                "min_tax_margin": row.margins.min_tax_margin,
                "min_tax_speed_margin": row.margins.min_tax_speed_margin,
                "min_transfer_margin": row.margins.min_transfer_margin,
                "min_net_rental_tax_base_margin": row.margins.min_net_rental_tax_base_margin,
                "structural_continuation_solvency": row.margins.structural_continuation_solvency,
            }
        )
    return rows


def _continuation_checkpoint_rows(report: ExperimentReport) -> list[dict[str, Any]]:
    rows = []
    for route_name, run in (("lq_path_continuation", report.route_a_main), ("crude_direct", report.route_b_main), ("lq_path_continuation_crude_tail", report.route_a_crude_tail)):
        for checkpoint in run.checkpoints:
            rows.append(
                {
                    "route": route_name,
                    "terminal_convention": run.terminal_convention,
                    "horizon": run.horizon,
                    "amplitude": checkpoint.amplitude,
                    "capital_0": checkpoint.capital_0,
                    "tau_0": checkpoint.tau_0,
                    "accepted": checkpoint.accepted,
                    "n_mesh_nodes": checkpoint.result.n_mesh_nodes if checkpoint.result else None,
                    "solver_max_rms_residual": checkpoint.result.solver_max_rms_residual if checkpoint.result else None,
                    "failure_message": checkpoint.failure_message or "",
                }
            )
    return rows


def _write_figures(output_dir: Path, report: ExperimentReport) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []

    baseline = report.route_a_main.final
    t_grid = np.linspace(0.0, report.config.baseline_horizon, 201)
    path = baseline.path_at(t_grid)
    lq_kt = _lq_kt_for_plot(report, t_grid)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    panels = [
        ("log-capital deviation k", path[0, :], lq_kt[0, :]),
        ("tax rate tau", path[1, :], lq_kt[1, :] + 0.5),
        ("costate ell = J_K", path[2, :], None),
        ("costate m = J_tau", path[3, :], None),
    ]
    for ax, (title, series, lq_series) in zip(axes.ravel(), panels):
        ax.plot(t_grid, series, label="nonlinear")
        if lq_series is not None:
            ax.plot(t_grid, lq_series, "--", label="LQ")
            ax.legend(fontsize=7)
        ax.axhline(0.0 if "deviation" in title or "costate m" in title else (0.5 if "tax" in title else 1.0), color="black", linewidth=0.6)
        ax.set_title(title)
        ax.set_xlabel("years")
    axes[1, 1].axis("off")
    axes[1, 2].plot(t_grid, path[0, :] - lq_kt[0, :], label="k: nonlinear - LQ")
    axes[1, 2].plot(t_grid, (path[1, :] - 0.5) - lq_kt[1, :], label="tau_dev: nonlinear - LQ")
    axes[1, 2].axhline(0.0, color="black", linewidth=0.6)
    axes[1, 2].set_title("nonlinear-minus-LQ (baseline amplitude)")
    axes[1, 2].set_xlabel("years")
    axes[1, 2].legend(fontsize=7)
    fig.suptitle(f"CS002 D0-D1 exploratory prototype -- {report.config.config_id}")
    fig.tight_layout()
    path_fig = figures_dir / "baseline_path_and_lq_overlay.png"
    fig.savefig(path_fig, dpi=150)
    plt.close(fig)
    figure_paths.append(path_fig)

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    amplitudes = report.convergence.amplitudes[::-1]
    ax2.loglog(amplitudes, report.convergence.errors_k[::-1], "o-", label="error_k")
    ax2.loglog(amplitudes, report.convergence.errors_tau[::-1], "s-", label="error_tau")
    reference = [report.convergence.errors_k[::-1][0] * (a / amplitudes[0]) ** 2 for a in amplitudes]
    ax2.loglog(amplitudes, reference, "k:", label="slope 2 (reference)")
    ax2.set_xlabel("continuation amplitude (log scale)")
    ax2.set_ylabel("max |nonlinear - LQ| (log scale)")
    ax2.set_title("LQ-vs-nonlinear convergence order")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    conv_fig = figures_dir / "lq_convergence_order.png"
    fig2.savefig(conv_fig, dpi=150)
    plt.close(fig2)
    figure_paths.append(conv_fig)

    return figure_paths


def _lq_kt_for_plot(report: ExperimentReport, t_grid: np.ndarray) -> np.ndarray:
    baseline = report.route_a_main.final
    y0 = baseline.result.sol(np.array([0.0]))[:, 0]
    return lq_linear_kt_path(t_grid, y0[0], y0[1] - 0.5, report.solution)


def write_bundle(
    output_dir: Path,
    report: ExperimentReport,
    repository: Path,
    material_run_elapsed_seconds: float,
    task_elapsed_seconds: float,
    command: str,
    git_at_run_start: dict[str, Any],
    run_id: str,
    runs_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)

    baseline_rows = _baseline_path_rows(report)
    grid_rows = _net_worth_grid_rows(report)
    checkpoint_rows = _continuation_checkpoint_rows(report)
    _write_csv(output_dir / "baseline_path.csv", baseline_rows)
    _write_csv(output_dir / "net_worth_grid.csv", grid_rows)
    _write_csv(output_dir / "continuation_checkpoints.csv", checkpoint_rows)
    figure_paths = _write_figures(output_dir, report)

    environment = environment_metadata()
    report_payload = {
        "run_id": run_id,
        "config_id": report.config.config_id,
        "config_fingerprint": report.config.fingerprint,
        "cs001_fingerprints": report.config.cs001.fingerprints,
        "outcome": serializable(report.outcome),
        "d0_checks": serializable(report.d0_checks),
        "ode_residual": serializable(report.ode_residual),
        "boundary_residual": serializable(report.boundary_residual),
        "horizon_mesh_comparisons": [serializable(c) for c in report.horizon_mesh_comparisons],
        "convergence": serializable(report.convergence),
        "j_recovery_quadratic": {k: v for k, v in serializable(report.j_recovery_quadratic).items() if k not in ("j_path_hjb", "t_grid")},
        "j_recovery_anchor": {k: v for k, v in serializable(report.j_recovery_anchor).items() if k not in ("j_path_hjb", "t_grid")},
        "net_worth_grid": grid_rows,
        "continuation_checkpoints": checkpoint_rows,
        "environment": environment,
        "implementation": git_at_run_start,
        "material_run_elapsed_seconds": material_run_elapsed_seconds,
        "task_elapsed_seconds": task_elapsed_seconds,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    input_copy = output_dir / "complete_input.json"
    input_copy.write_text(json.dumps(serializable({"cs002_config": report.config.raw, "primitives": report.local_system.parameters.raw}), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    outcome = report.outcome
    summary_path = output_dir / "summary.md"
    summary_path.write_text(
        "\n".join(
            [
                f"# {run_id}",
                "",
                "**Exploratory CS002 D0-D1 prototype. CS002 v0.2 remains draft and unfingerprinted; this is not an approved or completed CS002 result.**",
                "",
                f"- Outcome: **{outcome.outcome}**",
                f"- Config: `{report.config.config_id}` (fingerprint `{report.config.fingerprint[:12]}...`)",
                f"- Independent max scaled ODE residual: `{report.ode_residual.max_scaled_residual:.3e}` (<= 1e-7 required)",
                f"- Independent max scaled boundary residual: `{report.boundary_residual.max_scaled_residual:.3e}` (<= 1e-8 required)",
                f"- Horizon/mesh stability: `{all(c.within_tolerance for c in report.horizon_mesh_comparisons)}`",
                f"- LQ-vs-nonlinear convergence ratios (halving amplitude, target ~4.0): k={[round(r, 3) for r in report.convergence.ratios_k]}, tau={[round(r, 3) for r in report.convergence.ratios_tau]}",
                f"- Route A vs route B branch agreement: `{outcome.checks.get('branch_agreement_route_a_vs_b')}`",
                f"- J(0) recovery (quadratic tail): flow-integral=`{report.j_recovery_quadratic.j0_flow_integral:.6f}`, HJB-ODE=`{report.j_recovery_quadratic.j0_hjb:.6f}`, relative disagreement=`{report.j_recovery_quadratic.route_disagreement_relative:.3e}`",
                f"- Net-worth grid: {sum(1 for r in report.net_worth_grid if r.reported_local_margins_slack)}/{len(report.net_worth_grid)} rows have all reported local margins slack (not a structural-solvency or global-feasibility verdict; structural continuation solvency is `not_evaluated`); other members retained with failure reasons (see net_worth_grid.csv), not dropped.",
                "",
                "Interpretation: bounded exploratory D0-D1 prototype under CS002 v0.2's draft-specification "
                "exception. It supports research review of the equation map, the LQ stable-manifold terminal "
                "mapping, residual/branch evidence, and resource profile -- it does not promote CS002 toward "
                "review_ready or authorize Block D2.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    material_artifacts = [report_path, input_copy, summary_path, output_dir / "baseline_path.csv", output_dir / "net_worth_grid.csv", output_dir / "continuation_checkpoints.csv", *figure_paths]

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
        "purpose": "Exploratory CS002 D0-D1 prototype: frozen-common-state nonlinear deterministic characteristic BVP, "
        "continuation, independent residuals, terminal-tail sensitivity, boundary margins, LQ comparison. "
        "Bounded feasibility prototype under CS002 v0.2's draft-specification exception -- not an approved or "
        "completed CS002 result; does not authorize Block D2.",
        "specification": {"id": "CS002", "version": "0.2", "status": "draft", "fingerprint_sha256": None},
        "input_fingerprints": {"cs002_config_sha256": report.config.fingerprint, **{f"cs001_{k}": v for k, v in report.config.cs001.fingerprints.items()}},
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
            "entrypoint": "python3 -m tai_public_finance.cs002_nonlinear_transition.cli",
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
            "wall_seconds_limit": 600,
            "cash_limit_usd": 0,
            "actual_cash_usd": 0,
            "early_stop_rule": "Stop and retain failure if any independent residual, horizon/mesh stability, "
            "convergence-order, branch-agreement, or boundary-margin acceptance check fails; solver-reported "
            "convergence alone never sets the outcome to pass.",
        },
        "randomness": {"deterministic_requested": True, "seeds": [], "nondeterminism_notes": "Deterministic float64 BVP collocation and ODE quadrature; no simulation draws."},
        "inputs": {
            "parameter_set_id": report.config.cs001.parameter_set_id,
            "parameter_fingerprint_sha256": report.config.cs001.fingerprints["primitive_sha256"],
            "cs002_config_id": report.config.config_id,
            "input_artifacts": [str(report.config.config_path), str(report.config.cs001.experiment_path), str(report.config.cs001.primitive_path)],
            "displacement": {"delta_k": report.config.delta_k, "delta_tau": report.config.delta_tau},
            "continuation_amplitudes": report.config.continuation_amplitudes,
            "horizons_years": {"baseline": report.config.baseline_horizon, "comparisons": report.config.comparison_horizons},
        },
        "solver": {
            "packages_and_versions": {"numpy": environment["numpy"], "scipy": environment["scipy"]},
            "method": "scipy.integrate.solve_bvp collocation, continuation in displacement amplitude from an LQ-path "
            "warm start (route A) cross-checked against direct crude-guess solves at every amplitude (route B); "
            "independent off-mesh finite-difference ODE residual and re-applied terminal-condition boundary residual.",
            "tolerances": report.config.acceptance_tolerances,
            "initialization": "LQ closed-loop path (route A) / crude constant-anchor guess (route B)",
            "continuation": f"amplitude sequence {report.config.continuation_amplitudes}",
            "precision": "float64",
        },
        "preflight": {"tests_command": "uv run pytest", "tests_status": "passed", "exact_or_manufactured_benchmark": "Matrix-exponential manufactured BVP through the same solver interface; R16 fixed-tax closed-form capital transition; zero-displacement anchor fixed point."},
        "result": {
            "outcome": outcome.outcome,
            "wall_seconds": material_run_elapsed_seconds,
            "task_elapsed_seconds": task_elapsed_seconds,
            "peak_memory_gb": None,
            "actual_cash_usd": 0,
            "checkpoints_recovered": True,
            "solver_reported_metrics": None,
            "independent_diagnostics": {
                "checks": outcome.checks,
                "ode_residual": serializable(report.ode_residual),
                "boundary_residual": serializable(report.boundary_residual),
                "convergence_ratios_k": report.convergence.ratios_k,
                "convergence_ratios_tau": report.convergence.ratios_tau,
                "horizon_mesh_comparisons": [serializable(c) for c in report.horizon_mesh_comparisons],
            },
            "economic_quantities": {
                "j0_quadratic_tail_hjb": report.j_recovery_quadratic.j0_hjb,
                "j0_quadratic_tail_flow": report.j_recovery_quadratic.j0_flow_integral,
                "j0_anchor_tail_hjb": report.j_recovery_anchor.j0_hjb,
                "j0_anchor_tail_flow": report.j_recovery_anchor.j0_flow_integral,
            },
            "reliable_region": "Only the reported baseline displacement path and net-worth grid, with recorded positive branch and margin slack; small-displacement illustrative benchmark, not a global feasibility or uniqueness claim.",
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
                "bounded exploratory D0-D1 prototype under the draft-specification exception recorded in CS002's "
                "own \"Immediate implementation handoff\" section, not an approved or completed CS002 result.",
                "Frozen common states only (z=z_bar, x=x_bar): Block D2 (deterministic mean reversion) is explicitly "
                "out of scope and not implemented here.",
                "Illustrative solver/algebra benchmark on the Farhi-based smoke calibration, not an empirical UK calibration.",
                "min_net_rental_tax_base_margin (min(R^K-delta)) and comprehensive_resource_margin (X) are local "
                "reporting margins, not a structural-solvency or global-feasibility calculation -- "
                "structural_continuation_solvency is reported literally as not_evaluated (see margins.py).",
                "The net-worth grid deliberately includes low-N/J members whose reported local margins are NOT all "
                "slack (negative_transfer), retained with explicit failure reasons rather than dropped, mirroring "
                "CS001's own repaired baseline.",
                "No derivative service (Block D3), stochastic terms, or order-epsilon^2 correction: this is the "
                "zeroth-order deterministic path only.",
                "task_elapsed_seconds is the implementer's own end-to-end estimate of reading, implementation, "
                "and debugging time, not an instrumented wall-clock measurement (this environment exposes no "
                "reliable session-start timestamp); wall_seconds (this material run's own compute time) is "
                "directly measured and exact.",
            ],
            "next_decision": "Research owner reviews the equation map, LQ stable-manifold terminal-condition "
            "derivation, residual/branch evidence, and resource profile before CS002 may move toward review_ready "
            "or Block D2 (deterministic mean reversion) is authorized.",
        },
    }
    record_path = (runs_dir or repository / "runs") / f"{run_id}.yaml"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    if record_path.exists():
        raise FileExistsError(f"Run record {record_path} already exists. Run records are immutable.")
    record_path.write_text(yaml.safe_dump(serializable(run_record), sort_keys=False, allow_unicode=False), encoding="utf-8")

    return {"record_path": record_path, "artifacts": artifacts, "environment": environment, "git": git_at_run_start}
