"""CLI for a focused N/J public-net-worth sensitivity grid (reuses portfolio.net_worth_grid)."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import load_cs001_configuration
from .equations import build_local_system
from .anchor import compute_steady_state
from .portfolio import net_worth_grid, portfolio_sign_change_net_worth_ratio, transfer_boundary_net_worth_ratio
from .reporting import environment_metadata, git_metadata, sha256_file, serializable
from .solver import solve_lq_system


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ratios", required=True, help="Comma-separated N/J ratios")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[3]
    git_at_run_start = git_metadata(repository)
    started = time.perf_counter()

    config = load_cs001_configuration(args.config)
    experiment = config.experiment
    scaffolding = experiment["numerical_scaffolding"]
    anchor = compute_steady_state(config.parameters)
    local_system = build_local_system(config.parameters, anchor)
    solution = solve_lq_system(local_system)

    transfer_boundary = transfer_boundary_net_worth_ratio(local_system)
    sign_change = portfolio_sign_change_net_worth_ratio(local_system, solution)

    ratios = [float(v) for v in args.ratios.split(",")]
    rows = net_worth_grid(
        local_system,
        solution,
        risky_short_limit=float(scaffolding["risky_short_limit"]),
        safe_debt_limit=float(scaffolding["safe_debt_limit"]),
        risk_scale_epsilon=float(experiment["risk_scale_epsilon"]),
        net_worth_to_fiscal_wealth_ratios=ratios,
    )
    elapsed = time.perf_counter() - started

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "net_worth_grid.csv"
    serial_rows = [serializable(row) for row in rows]
    fieldnames = list(serial_rows[0].keys())
    assert all(r.keys() == serial_rows[0].keys() for r in serial_rows), "every row must carry the same fields"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in serial_rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()})

    run_id = args.run_id or f"RUN-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-CS001-NJGRID-{(git_at_run_start['commit'] or 'nogit')[:8]}-01"
    n_feasible = sum(1 for r in rows if r["feasible"])
    environment = environment_metadata()
    record = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "completed",
        "purpose": f"Focused public-net-worth (N/J) sensitivity grid, {len(ratios)} points, resolving the transfer "
        "boundary and the risky-position sign-change point. Exploratory first CS001 tranche, not a completed/"
        "approved CS001 result.",
        "specification": {"id": "CS001", "version": "0.1", "status": "draft", "fingerprint_sha256": None},
        "problem_id": "CP001",
        "supersedes": None,
        "based_on_run": "RUN-20260901T194938Z-CS001-d876a61e-01",
        "implementation": {
            "repository_url": "https://github.com/Nathan-Barnard/tai-public-finance",
            "local_path": str(repository),
            "commit": git_at_run_start["commit"],
            "branch": git_at_run_start["branch"],
            "dirty_worktree_at_run_start": git_at_run_start["dirty_worktree_at_run_start"],
            "implementer": "claude_code",
            "entrypoint": "python3 -m tai_public_finance.cs001_lq_anchor.net_worth_grid_cli",
            "command": f"uv run python -m tai_public_finance.cs001_lq_anchor.net_worth_grid_cli --config {args.config} "
            f"--ratios {args.ratios} --output-dir {args.output_dir} --run-id {run_id}",
        },
        "environment": {
            "operating_system": environment["operating_system"],
            "runtime_versions": {key: environment[key] for key in ("python", "numpy", "scipy")},
        },
        "hardware": {"resource_lane": "L1_interactive", "machine_label": "local_mac", "accelerator": None},
        "inputs": {
            "parameter_set_id": config.parameter_set_id,
            "input_fingerprints": config.fingerprints,
            "swept_ratios": ratios,
            "derived_thresholds": {
                "transfer_boundary_net_worth_ratio": transfer_boundary,
                "portfolio_sign_change_net_worth_ratio": sign_change,
            },
        },
        "result": {
            "outcome": "mixed",
            "wall_seconds": elapsed,
            "n_points": len(rows),
            "n_feasible": n_feasible,
            "n_infeasible": len(rows) - n_feasible,
            "infeasible_ratios_and_reasons": [
                {"ratio": r["public_net_worth_to_fiscal_wealth"], "reasons": r["failure_reasons"]} for r in rows if not r["feasible"]
            ],
        },
        "artifacts": [{"role": "primary_output", "location": str(csv_path.relative_to(repository)), "sha256": sha256_file(csv_path), "bytes": csv_path.stat().st_size}],
        "interpretation": {
            "conclusion": f"{n_feasible}/{len(rows)} grid points are feasible (positive-X, non-negative transfer, "
            "portfolio-bound). Different rows are different members of the deterministic fiscal-wealth family, not "
            "a causal comparison across changing public net worth.",
            "limitations": [
                "Exploratory first CS001 tranche; CS001 itself remains draft/unfingerprinted.",
                "Deterministic-family comparative statics only; not a claim about the dynamics of N itself.",
            ],
        },
    }
    record_path = repository / "runs" / f"{run_id}.yaml"
    record_path.write_text(yaml.safe_dump(record, sort_keys=False, allow_unicode=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "run_id": run_id,
                "n_points": len(rows),
                "n_feasible": n_feasible,
                "transfer_boundary_net_worth_ratio": transfer_boundary,
                "portfolio_sign_change_net_worth_ratio": sign_change,
                "csv": str(csv_path),
                "record": str(record_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
