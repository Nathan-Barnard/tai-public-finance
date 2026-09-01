from __future__ import annotations

from pathlib import Path

import pytest

from tai_public_finance.cs001_lq_anchor import (
    build_local_system,
    compute_steady_state,
    leading_portfolio_and_welfare,
    load_cs001_configuration,
    solve_lq_system,
)
from tai_public_finance.cs001_lq_anchor.diagnostics import run_diagnostics
from tai_public_finance.cs001_lq_anchor.irfs import run_irfs

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "cs001" / "lq_farhi_smoke.json"


@pytest.fixture(scope="session")
def baseline():
    config = load_cs001_configuration(CONFIG_PATH)
    anchor = compute_steady_state(config.parameters)
    local_system = build_local_system(config.parameters, anchor)
    solution = solve_lq_system(local_system)
    scaffolding = config.experiment["numerical_scaffolding"]
    portfolio = leading_portfolio_and_welfare(
        local_system,
        solution,
        risky_short_limit=float(scaffolding["risky_short_limit"]),
        safe_debt_limit=float(scaffolding["safe_debt_limit"]),
        risk_scale_epsilon=float(config.experiment["risk_scale_epsilon"]),
    )
    irfs = run_irfs(local_system, solution, portfolio, config.experiment["reporting"], scaffolding)
    diagnostics = run_diagnostics(local_system, solution, portfolio)
    return {
        "config": config,
        "anchor": anchor,
        "local_system": local_system,
        "solution": solution,
        "portfolio": portfolio,
        "irfs": irfs,
        "diagnostics": diagnostics,
    }
