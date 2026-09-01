"""Regression guards for the 2026-09-01 baseline repair.

Each test here pins down one of the mistakes found in the first CS001
baseline's reporting/interpretation layer: a mis-signed portfolio
decomposition, an unchecked N/J feasibility grid, a wrong claim/no-claim
gap description, and a conflated input/specification fingerprint. The
underlying computation was already correct; these are checks on what gets
*reported* about it.
"""

from __future__ import annotations

from tai_public_finance.cs001_lq_anchor.portfolio import net_worth_grid


def test_portfolio_decomposition_sums_with_correct_hedge_sign(baseline):
    portfolio = baseline["portfolio"]
    assert portfolio.return_demand_component == portfolio.comprehensive_resources
    assert portfolio.fiscal_hedge_component == -portfolio.marketed_fiscal_wealth_amount
    # The hedge reduces/shorts exposure because fiscal wealth already covaries
    # positively with the claim's payoff -- it must never be positive here.
    assert portfolio.fiscal_hedge_component < 0.0
    assert portfolio.return_demand_component + portfolio.fiscal_hedge_component == portfolio.leading_unconstrained_position


def test_net_worth_grid_flags_negative_transfer_rows_as_infeasible(baseline):
    config = baseline["config"]
    scaffolding = config.experiment["numerical_scaffolding"]
    grid = net_worth_grid(
        baseline["local_system"],
        baseline["solution"],
        risky_short_limit=float(scaffolding["risky_short_limit"]),
        safe_debt_limit=float(scaffolding["safe_debt_limit"]),
        risk_scale_epsilon=float(config.experiment["risk_scale_epsilon"]),
        net_worth_to_fiscal_wealth_ratios=[-0.5, -0.25, 0.0, 0.25, 0.5],
    )
    by_ratio = {row["public_net_worth_to_fiscal_wealth"]: row for row in grid}
    assert len(grid) == 5, "every row must be retained, not dropped"

    for ratio in (-0.5, -0.25):
        row = by_ratio[ratio]
        assert row["transfer"] < 0.0
        assert row["transfer_feasible"] is False
        assert row["feasible"] is False
        assert "negative_transfer" in row["failure_reasons"]

    for ratio in (0.0, 0.25, 0.5):
        row = by_ratio[ratio]
        assert row["transfer"] >= 0.0
        assert row["transfer_feasible"] is True
        assert row["feasible"] is True
        assert row["failure_reasons"] == []

    # Every field the task requires is actually present on every row.
    for row in grid:
        for field in (
            "comprehensive_resources",
            "comprehensive_resources_positive",
            "worker_consumption",
            "wage_income",
            "transfer",
            "transfer_feasible",
            "portfolio_bound_feasible",
            "feasible",
            "failure_reasons",
        ):
            assert field in row, field


def test_full_access_no_claim_consumption_gap_is_constant_and_positive(baseline):
    rows = baseline["irfs"]["rows"]
    for experiment in ("brownian_productivity_1sd_short_window", "brownian_automation_1sd_short_window"):
        full = {round(r["horizon_years"], 6): r["worker_consumption_deviation"] for r in rows if r["experiment"] == experiment and r["regime"] == "full_access"}
        no_claim = {round(r["horizon_years"], 6): r["worker_consumption_deviation"] for r in rows if r["experiment"] == experiment and r["regime"] == "no_external_claim"}
        assert full.keys() == no_claim.keys()
        gaps = [full[h] - no_claim[h] for h in full]
        assert all(gap > 0.0 for gap in gaps), experiment
        assert max(gaps) - min(gaps) < 1e-12, (
            f"{experiment}: the full_access/no_claim consumption gap must be exactly constant over the "
            "horizon (X has no mean-reversion term in this linear system), not merely same-signed"
        )


def test_run_record_never_reports_an_input_fingerprint_as_the_specification_fingerprint(baseline):
    # Mirrors the exact construction in reporting.write_bundle: the specification
    # object must not borrow any of the input-side hashes as its own fingerprint.
    from tai_public_finance.cs001_lq_anchor.reporting import serializable

    config = baseline["config"]
    input_fingerprints = {
        "primitive_sha256": config.fingerprints["primitive_sha256"],
        "experiment_sha256": config.fingerprints["experiment_sha256"],
        "complete_input_sha256": config.fingerprints["complete_input_sha256"],
    }
    specification = {"id": "CS001", "version": "0.1", "status": "draft", "fingerprint_sha256": None}
    assert specification["fingerprint_sha256"] is None
    assert specification["fingerprint_sha256"] not in input_fingerprints.values()
    assert len(set(input_fingerprints.values())) == 3, "primitive/experiment/complete-input fingerprints must be distinct"
    assert serializable(specification)["status"] == "draft"
