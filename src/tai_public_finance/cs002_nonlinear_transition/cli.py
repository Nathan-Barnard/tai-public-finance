"""Command-line entry point for the CS002 D0-D1 exploratory material run."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import load_cs002_configuration
from .experiment import run_d0_d1_experiment
from .reporting import git_metadata, write_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--runs-dir", help="Where to write the immutable run-record YAML (default: <repository>/runs).")
    parser.add_argument(
        "--task-elapsed-seconds",
        type=float,
        default=None,
        help="Total task wall-clock elapsed (reading, setup, tests), distinct from this material run's own "
        "compute time; if omitted, recorded equal to the material run time.",
    )
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[3]
    git_at_run_start = git_metadata(repository)
    cache = repository / ".cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    config = load_cs002_configuration(args.config)
    report = run_d0_d1_experiment(config)
    elapsed = time.perf_counter() - started

    run_id = args.run_id or f"RUN-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-CS002-D0D1-{(git_at_run_start['commit'] or 'nogit')[:8]}-01"
    command = f"uv run python -m tai_public_finance.cs002_nonlinear_transition.cli --config {args.config} --output-dir {args.output_dir} --run-id {run_id}"
    runs_dir = Path(args.runs_dir).resolve() if args.runs_dir else None
    task_elapsed = args.task_elapsed_seconds if args.task_elapsed_seconds is not None else elapsed

    output = write_bundle(
        Path(args.output_dir).resolve(), report, repository, elapsed, task_elapsed, command, git_at_run_start, run_id, runs_dir
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "outcome": report.outcome.outcome,
                "record": str(output["record_path"]),
                "failed_checks": report.outcome.failed_checks,
            },
            indent=2,
        )
    )
    return 0 if report.outcome.outcome == "computational_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
