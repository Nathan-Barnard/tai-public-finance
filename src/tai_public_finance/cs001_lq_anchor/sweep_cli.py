"""Command-line entry point for a CS001 L2_local_batch one-parameter sweep."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import load_cs001_configuration
from .reporting import environment_metadata, git_metadata, sha256_file
from .sweep import sweep_one_parameter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--parameter-path", required=True, help="Dotted path into the primitive table, e.g. parameters.tax_adjustment_scale")
    parser.add_argument("--values", required=True, help="Comma-separated float values")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[3]
    git_at_run_start = git_metadata(repository)
    started = time.perf_counter()

    config = load_cs001_configuration(args.config)
    experiment = config.experiment
    scaffolding = experiment["numerical_scaffolding"]
    parameter_path = tuple(args.parameter_path.split("."))
    values = [float(v) for v in args.values.split(",")]

    rows = sweep_one_parameter(
        base_parameters=config.parameters,
        parameter_path=parameter_path,
        values=values,
        scaffolding=scaffolding,
        reporting=experiment["reporting"],
        risk_scale_epsilon=float(experiment["risk_scale_epsilon"]),
        acceptance_tolerances=experiment["acceptance_tolerances"],
    )
    elapsed = time.perf_counter() - started

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "sweep.csv"
    fieldnames = list(rows[0].keys())
    assert all(row.keys() == rows[0].keys() for row in rows), "every sweep row must carry the same fields"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in row.items()})

    run_id = args.run_id or f"RUN-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-CS001-SWEEP-{(git_at_run_start['commit'] or 'nogit')[:8]}-01"
    n_pass = sum(1 for r in rows if r["outcome"] == "pass")
    n_fail = len(rows) - n_pass
    environment = environment_metadata()
    record = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "completed",
        "purpose": f"CS001 L2_local_batch parameter sweep over {args.parameter_path} ({len(values)} points); "
        "exploratory first CS001 tranche, not a completed/approved CS001 result.",
        "specification": {"id": "CS001", "version": "0.1", "status": "draft", "fingerprint_sha256": None},
        "problem_id": "CP001",
        "implementation": {
            "repository_url": "https://github.com/Nathan-Barnard/tai-public-finance",
            "local_path": str(repository),
            "commit": git_at_run_start["commit"],
            "branch": git_at_run_start["branch"],
            "dirty_worktree_at_run_start": git_at_run_start["dirty_worktree_at_run_start"],
            "implementer": "claude_code",
            "entrypoint": "python3 -m tai_public_finance.cs001_lq_anchor.sweep_cli",
            "command": f"uv run python -m tai_public_finance.cs001_lq_anchor.sweep_cli --config {args.config} "
            f"--parameter-path {args.parameter_path} --values {args.values} --output-dir {args.output_dir} --run-id {run_id}",
        },
        "environment": {
            "operating_system": environment["operating_system"],
            "runtime_versions": {key: environment[key] for key in ("python", "numpy", "scipy")},
        },
        "hardware": {"resource_lane": "L2_local_batch", "machine_label": "local_mac", "accelerator": None},
        "inputs": {
            "parameter_set_id": config.parameter_set_id,
            "base_input_fingerprints": config.fingerprints,
            "swept_parameter_path": args.parameter_path,
            "swept_values": values,
        },
        "result": {
            "outcome": "pass" if n_fail == 0 else "mixed",
            "wall_seconds": elapsed,
            "n_points": len(rows),
            "n_pass": n_pass,
            "n_fail_or_error": n_fail,
            "failed_values": [r["value"] for r in rows if r["outcome"] != "pass"],
        },
        "artifacts": [
            {"role": "primary_output", "location": str(csv_path.relative_to(repository)) if repository in csv_path.parents else str(csv_path), "sha256": sha256_file(csv_path), "bytes": csv_path.stat().st_size}
        ],
        "interpretation": {
            "conclusion": f"{n_pass}/{len(rows)} sweep points passed every CS001 acceptance check.",
            "limitations": [
                "One-dimensional sweep only; other primitives held at the illustrative Farhi-based baseline.",
                "Exploratory first CS001 tranche; CS001 itself remains draft/unfingerprinted.",
            ],
        },
    }
    record_path = repository / "runs" / f"{run_id}.yaml"
    record_path.write_text(yaml.safe_dump(record, sort_keys=False, allow_unicode=False), encoding="utf-8")

    print(json.dumps({"run_id": run_id, "n_pass": n_pass, "n_fail_or_error": n_fail, "csv": str(csv_path), "record": str(record_path)}, indent=2))
    return 0 if n_fail == 0 else 0  # a sweep with some infeasible/failed points is still a successful sweep


if __name__ == "__main__":
    raise SystemExit(main())
