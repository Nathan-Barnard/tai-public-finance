"""Command-line entry point for the CS001 Stage 1 / Stage 2A baseline calculation."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .anchor import compute_steady_state
from .config import load_cs001_configuration
from .diagnostics import acceptance as compute_acceptance
from .diagnostics import run_diagnostics
from .equations import build_local_system
from .irfs import run_irfs
from .portfolio import leading_portfolio_and_welfare, net_worth_grid
from .reporting import git_metadata, write_bundle
from .solver import solve_lq_system


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--runs-dir", help="Where to write the immutable run-record YAML (default: <repository>/runs).")
    parser.add_argument("--preflight-tests-status", default="not_recorded")
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[3]
    git_at_run_start = git_metadata(repository)
    cache = repository / ".cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    config = load_cs001_configuration(args.config)
    experiment = config.experiment
    scaffolding = experiment["numerical_scaffolding"]

    anchor = compute_steady_state(config.parameters)
    local_system = build_local_system(config.parameters, anchor)
    solution = solve_lq_system(local_system)
    portfolio = leading_portfolio_and_welfare(
        local_system,
        solution,
        risky_short_limit=float(scaffolding["risky_short_limit"]),
        safe_debt_limit=float(scaffolding["safe_debt_limit"]),
        risk_scale_epsilon=float(experiment["risk_scale_epsilon"]),
    )
    grid = net_worth_grid(
        local_system,
        solution,
        risky_short_limit=float(scaffolding["risky_short_limit"]),
        safe_debt_limit=float(scaffolding["safe_debt_limit"]),
        risk_scale_epsilon=float(experiment["risk_scale_epsilon"]),
        net_worth_to_fiscal_wealth_ratios=experiment["reporting"]["public_net_worth_to_fiscal_wealth_grid"],
    )
    irfs = run_irfs(local_system, solution, portfolio, experiment["reporting"], scaffolding)
    diagnostics = run_diagnostics(local_system, solution, portfolio)

    run_id = args.run_id or f"RUN-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-CS001-{(git_at_run_start['commit'] or 'nogit')[:8]}-01"
    accept = compute_acceptance(
        local_system,
        solution,
        diagnostics,
        portfolio,
        irfs["boundary_summary"],
        irfs["max_matrix_exponential_vs_ode_relative_error"],
        irfs["max_first_order_budget_residual"],
        experiment["acceptance_tolerances"],
    )

    report = {
        "run_id": run_id,
        "parameter_set_id": config.parameter_set_id,
        "calibration_role": config.calibration_role,
        "fingerprints": config.fingerprints,
        "config_paths": {"experiment": str(config.experiment_path), "primitives": str(config.primitive_path)},
        "experiment_config": experiment,
        "local_system": local_system,
        "solution": solution,
        "portfolio_anchor": portfolio,
        "portfolio_net_worth_grid": grid,
        "irfs": irfs,
        "diagnostics": diagnostics,
        "acceptance": accept,
        "preflight_tests_status": args.preflight_tests_status,
        "limitations": [
            "Local T3/LQ evidence only; no global constrained or off-specialisation optimality claim.",
            "The Farhi-based vector is illustrative rather than an empirical UK calibration.",
            "epsilon=1 reports the leading small-risk coefficient at the primitive volatility scale; it is not an exact finite-risk welfare result.",
            "No exact precautionary consumption or tax-speed correction is reported; those require a Stage 2B specification with higher deterministic fiscal-wealth derivatives, or an explicit truncated stochastic-LQ closure.",
            "The stable four-state block has a stationary covariance, but public net worth and worker consumption retain a neutral direction and need not be stationary.",
            "Finite-horizon positive-X and boundary-slack checks are local continuation plausibility, not a global stochastic solvency or transversality proof.",
            "CS001 is registered as draft, unfingerprinted, in the codex research workspace; this is an exploratory "
            "first CS001 tranche run under Nathan's direct commission, not a completed or approved CS001 result. "
            "This run's own input fingerprints (primitive/experiment/complete-input) are recorded separately from, "
            "and must never be read as, a specification fingerprint -- none exists yet.",
            "This run supersedes RUN-20260901T184527Z-CS001-89f6a939-01, which is preserved unchanged as historical "
            "evidence; that run's portfolio-decomposition sign labelling, welfare-comparison language, claim/no-claim "
            "gap description, N/J grid feasibility reporting, and specification-fingerprint field were corrected here.",
        ],
    }
    command = f"uv run python -m tai_public_finance.cs001_lq_anchor.cli --config {args.config} --output-dir {args.output_dir} --run-id {run_id}"
    runs_dir = Path(args.runs_dir).resolve() if args.runs_dir else None
    output = write_bundle(Path(args.output_dir).resolve(), report, repository, time.perf_counter() - started, command, git_at_run_start, runs_dir)
    print(
        json.dumps(
            {"run_id": run_id, "outcome": accept.outcome, "record": str(output["record_path"]), "failed_checks": accept.failed_checks},
            indent=2,
        )
    )
    return 0 if accept.outcome == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
