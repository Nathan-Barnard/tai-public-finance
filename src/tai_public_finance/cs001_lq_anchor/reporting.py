"""Assembling the material-run bundle: JSON report, CSVs, figures, run record."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import yaml

from .equations import COORDINATES


def serializable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: serializable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            if value.ndim == 1:
                return [{"real": float(item.real), "imag": float(item.imag)} for item in value]
            return [[{"real": float(item.real), "imag": float(item.imag)} for item in row] for row in value]
        return value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git_metadata(repository: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(["git", *args], cwd=repository, check=True, capture_output=True, text=True)
        return completed.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        branch = run("branch", "--show-current")
        dirty = bool(run("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, branch, dirty = None, None, None
    return {"commit": commit, "branch": branch, "dirty_worktree_at_run_start": dirty}


def environment_metadata() -> dict[str, Any]:
    import matplotlib

    memory_gb = None
    try:
        memory_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError, AttributeError):
        pass
    return {
        "operating_system": platform.platform(),
        "architecture": platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "pyyaml": yaml.__version__,
        "cpu": platform.processor() or None,
        "logical_cores": os.cpu_count(),
        "memory_gb": memory_gb,
        "precision": "float64",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{key: serializable(value) for key, value in row.items()} for row in rows])


def _write_matrix(path: Path, matrix: np.ndarray, row_names: list[str], column_names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["coordinate", *column_names])
        for name, row in zip(row_names, matrix, strict=True):
            writer.writerow([name, *[float(value) for value in row]])


def _write_figures(output_dir: Path, irf_rows: list[dict[str, Any]]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    def rows_by_family(family: str, regime: str = "full_access") -> dict[str, list[dict[str, Any]]]:
        # Grouping by experiment_family (already on every row) rather than a
        # hand-maintained list of experiment-name strings means a rename in
        # irfs.py's build_experiments can't silently drop a series from these
        # figures without also being reflected here.
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in irf_rows:
            if row["experiment_family"] == family and row["regime"] == regime:
                grouped.setdefault(row["experiment"], []).append(row)
        return grouped

    def series_label(experiment: str) -> str:
        return experiment.replace("constructed_", "").replace("brownian_", "")

    primitive_rows = rows_by_family("primitive_brownian_innovation")
    constructed_rows = rows_by_family("economically_constructed_state_displacement")
    figure_paths: list[Path] = []

    state_variables = [
        ("worker_consumption_deviation", "Worker consumption"),
        ("tax_rate_deviation", "Tax rate"),
        ("log_capital_deviation", "Log capital"),
        ("public_net_worth_deviation", "Public net worth"),
    ]
    for rows_by_experiment, filename in (
        (primitive_rows, "stage1_primitive_brownian_irfs.png"),
        (constructed_rows, "stage1_constructed_displacement_irfs.png"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        for ax, (variable, title) in zip(axes.ravel(), state_variables, strict=True):
            for experiment, rows in rows_by_experiment.items():
                ax.plot(
                    [row["horizon_years"] for row in rows],
                    [row[variable] for row in rows],
                    label=series_label(experiment),
                )
            ax.axhline(0.0, color="black", linewidth=0.7)
            ax.set_title(title)
            ax.set_xlabel("Years")
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        path = figures_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figure_paths.append(path)

    fiscal_variables = [
        ("output_deviation_linear", "Output"),
        ("wage_income_deviation_linear", "Wage income"),
        ("tax_base_deviation_linear", "Net-rental tax base"),
        ("fiscal_resources_deviation_linear", "Fiscal resources F"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (variable, title) in zip(axes.ravel(), fiscal_variables, strict=True):
        for experiment, rows in primitive_rows.items():
            ax.plot(
                [row["horizon_years"] for row in rows],
                [row[variable] for row in rows],
                label=series_label(experiment),
            )
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set_title(title)
        ax.set_xlabel("Years")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fiscal_path = figures_dir / "stage1_fiscal_aggregate_irfs.png"
    fig.savefig(fiscal_path, dpi=160)
    plt.close(fig)
    figure_paths.append(fiscal_path)

    access_rows = rows_by_family("primitive_brownian_innovation", regime="no_external_claim")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for experiment in primitive_rows:
        for regime, rows_by_regime, style in (
            ("full_access", primitive_rows, "-"),
            ("no_external_claim", access_rows, "--"),
        ):
            rows = rows_by_regime.get(experiment, [])
            ax.plot(
                [row["horizon_years"] for row in rows],
                [row["worker_consumption_deviation"] for row in rows],
                linestyle=style,
                label=f"{series_label(experiment)}: {regime}",
            )
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xlabel("Years")
    ax.set_ylabel("Worker-consumption deviation")
    ax.legend(fontsize=7)
    fig.tight_layout()
    access_path = figures_dir / "stage2a_consumption_with_and_without_claim.png"
    fig.savefig(access_path, dpi=160)
    plt.close(fig)
    figure_paths.append(access_path)

    return figure_paths


def write_bundle(
    output_dir: Path,
    report: dict[str, Any],
    repository: Path,
    elapsed_seconds: float,
    command: str,
    git_at_run_start: dict[str, Any],
    runs_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    matrices_dir = output_dir / "matrices"
    matrices_dir.mkdir()

    local_system = report["local_system"]
    solution = report["solution"]
    coordinates = list(COORDINATES)
    _write_matrix(matrices_dir / "A.csv", local_system.A, coordinates, coordinates)
    _write_matrix(matrices_dir / "Q.csv", local_system.Q, coordinates, coordinates)
    _write_matrix(matrices_dir / "H.csv", solution.H, coordinates, coordinates)
    _write_matrix(matrices_dir / "A_closed_loop.csv", solution.A_c, coordinates, coordinates)
    _write_matrix(matrices_dir / "stationary_covariance_y.csv", solution.stationary_covariance_y, coordinates, coordinates)
    _write_csv(output_dir / "irfs.csv", report["irfs"]["rows"])
    _write_csv(output_dir / "portfolio_net_worth_grid.csv", report["portfolio_net_worth_grid"])
    figure_paths = _write_figures(output_dir, report["irfs"]["rows"])

    environment = environment_metadata()
    report_to_write = serializable(report | {"environment": environment, "implementation": git_at_run_start, "elapsed_seconds": elapsed_seconds})
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report_to_write, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    input_copy = output_dir / "complete_input.json"
    input_copy.write_text(
        json.dumps(serializable({"experiment": report["experiment_config"], "primitives": local_system.parameters.raw}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    portfolio = report["portfolio_anchor"]
    diagnostics = report["diagnostics"]
    acceptance = report["acceptance"]
    boundary = report["irfs"]["boundary_summary"]
    summary_path = output_dir / "summary.md"
    real_roots = diagnostics.closed_loop["real_closed_loop_eigenvalues"]
    summary_path.write_text(
        "\n".join(
            [
                f"# {report['run_id']}",
                "",
                f"- Outcome: **{acceptance.outcome}**",
                f"- Parameter set: `{report['parameter_set_id']}` (`{report['calibration_role']}`)",
                f"- Stabilizing capital-tax roots: `{[complex(v) for v in real_roots]}`",
                f"- Full closed-loop stability margin: `{diagnostics.closed_loop['full_closed_loop_stability_margin']:.12g}`",
                f"- Riccati scaled residual: `{diagnostics.riccati_full_scaled_residual:.3e}`",
                f"- Sylvester scaled residual: `{diagnostics.sylvester_scaled_residual:.3e}`",
                f"- Discounted Lyapunov scaled residual: `{diagnostics.discounted_lyapunov_scaled_residual:.3e}`",
                f"- Leading risky position: `{portfolio.leading_unconstrained_position:.12g}`",
                f"- Portfolio curvature: `{portfolio.portfolio_curvature:.12g}`",
                f"- Leading access consumption equivalent at epsilon={portfolio.risk_scale_epsilon}: `{portfolio.access_consumption_equivalent_leading:.12g}`",
                f"- Leading hedge consumption equivalent at epsilon={portfolio.risk_scale_epsilon}: `{portfolio.hedge_consumption_equivalent_leading:.12g}`",
                f"- Minimum transfer slack across reported local paths: `{boundary['transfer_level']:.12g}`",
                f"- Minimum specialization margins: `{boundary['specialisation_margin_automation_composite']:.12g}`, `{boundary['specialisation_margin_new_task_composite']:.12g}`",
                "",
                "Interpretation: the calculation is local and leading-small-risk. It does not establish exact "
                "precautionary policy functions, a global constrained solution, stochastic public-wealth "
                "stationarity, or an empirical policy magnitude.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    material_artifacts = [
        report_path,
        input_copy,
        summary_path,
        output_dir / "irfs.csv",
        output_dir / "portfolio_net_worth_grid.csv",
        *sorted(matrices_dir.glob("*.csv")),
        *figure_paths,
    ]
    def _location(path: Path) -> str:
        try:
            return str(path.relative_to(repository))
        except ValueError:
            return str(path)

    artifacts = [
        {
            "role": "primary_output" if path == report_path else ("figure" if path.suffix == ".png" else "supporting_output"),
            "location": _location(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in material_artifacts
    ]

    run_record = {
        "schema_version": "1.0",
        "run_id": report["run_id"],
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "completed" if acceptance.outcome == "pass" else "failed",
        "purpose": "Exploratory first CS001 tranche: Stage 1 deterministic 4x4 LQ system and Stage 2A leading "
        "small-risk baseline calculation. Not a completed or approved CS001 -- see specification.status.",
        # The specification document's own fingerprint (if/when CS001 is approved and fingerprinted in the
        # codex registry) is NOT the same object as this run's input fingerprints below -- do not conflate
        # a hash of what we fed in with a hash of the (currently unfingerprinted, draft) contract itself.
        "specification": {"id": "CS001", "version": "0.1", "status": "draft", "fingerprint_sha256": None},
        "input_fingerprints": {
            "primitive_sha256": report["fingerprints"]["primitive_sha256"],
            "experiment_sha256": report["fingerprints"]["experiment_sha256"],
            "complete_input_sha256": report["fingerprints"]["complete_input_sha256"],
        },
        "problem_id": "CP001",
        "approach_id": "CA001",
        "benchmark_id": None,
        "implementation": {
            "repository_url": "https://github.com/Nathan-Barnard/tai-public-finance",
            "local_path": str(repository),
            "commit": git_at_run_start["commit"],
            "branch": git_at_run_start["branch"],
            "dirty_worktree": git_at_run_start["dirty_worktree_at_run_start"],
            "dirty_worktree_at_run_start": git_at_run_start["dirty_worktree_at_run_start"],
            "implementer": "claude_code",
            "entrypoint": "python3 -m tai_public_finance.cs001_lq_anchor.cli",
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
            "early_stop_rule": "Stop and retain failure if any residual, concavity, feasibility, or path-validity acceptance check fails.",
        },
        "randomness": {"deterministic_requested": True, "seeds": [], "nondeterminism_notes": "Dense deterministic float64 linear algebra; no simulation draws."},
        "inputs": {
            "parameter_set_id": report["parameter_set_id"],
            "parameter_fingerprint_sha256": report["fingerprints"]["primitive_sha256"],
            "experiment_ids": [item.name for item in report["irfs"]["experiments"]],
            "input_artifacts": [report["config_paths"]["experiment"], report["config_paths"]["primitives"]],
            "domain": {"local_anchor": serializable(local_system.anchor), "numerical_scaffolding": report["experiment_config"]["numerical_scaffolding"]},
            "maintained_economic_closure": report["experiment_config"]["maintained_economic_closure"],
            "counterfactual_metadata": report["experiment_config"]["counterfactual_metadata"],
            "discretization": report["experiment_config"]["reporting"]["horizon_grid_years"],
        },
        "solver": {
            "packages_and_versions": {"numpy": environment["numpy"], "scipy": environment["scipy"]},
            "method": "ordered complex Schur stable invariant subspace; closed-form Hamiltonian-root cross-check; scipy solve_sylvester for the Sylvester/Lyapunov blocks; matrix exponential with direct DOP853 ODE cross-check",
            "tolerances": report["experiment_config"]["acceptance_tolerances"],
            "initialization": "deterministic interior anchor",
            "continuation": None,
            "precision": "float64",
        },
        "preflight": {
            "tests_command": "uv run pytest",
            "tests_status": report["preflight_tests_status"],
            "exact_or_manufactured_benchmark": "Q_rr cancellation, closed-form Riccati roots, manufactured non-hyperbolic refusal, repeated-root ordered-Schur regression, and matrix-exponential/ODE agreement",
        },
        "result": {
            "outcome": acceptance.outcome,
            "wall_seconds": elapsed_seconds,
            "peak_memory_gb": None,
            "actual_cash_usd": 0,
            "checkpoints_recovered": False,
            "solver_reported_metrics": None,
            "independent_diagnostics": serializable(diagnostics),
            "economic_quantities": {"anchor": serializable(local_system.anchor), "portfolio_anchor": serializable(portfolio), "boundary_summary": boundary},
            "reliable_region": "Only the reported local paths with recorded positive branch and feasibility slack.",
            "failure_code": None if acceptance.outcome == "pass" else "acceptance_check_failed",
            "failure_detail": None if acceptance.outcome == "pass" else acceptance.failed_checks,
        },
        "artifacts": artifacts,
        "interpretation": {
            "supports_result_ids": ["R10", "R11", "R13", "R14", "R15", "R17", "R18", "R19", "R20", "R22", "R23", "R24"],
            "challenges_result_ids": [],
            "question_ids": ["Q01", "Q03", "Q04", "Q05", "Q06", "Q07", "Q17"],
            "assurance_or_review_artifacts": [
                "research-notes/local-lq-system-computation-and-proof-plan.md",
                "economics-verification/reviews/conceptual-and-economic-audit-of-the-current-full-ramsey-planner--EV10.md",
            ],
            "conclusion": acceptance.conclusion,
            "limitations": report["limitations"],
            "next_decision": "Use the identical primitive fingerprint for the matched PDE-LQ bridge once CS004 is approved; reserve exact precautionary policies for a Stage 2B specification.",
        },
    }
    record_path = (runs_dir or repository / "runs") / f"{report['run_id']}.yaml"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    if record_path.exists():
        raise FileExistsError(
            f"Run record {record_path} already exists. Run records are immutable — reuse of a run_id is not permitted."
        )
    record_path.write_text(yaml.safe_dump(serializable(run_record), sort_keys=False, allow_unicode=False), encoding="utf-8")
    return {"record_path": record_path, "artifacts": artifacts, "environment": environment, "git": git_at_run_start}
