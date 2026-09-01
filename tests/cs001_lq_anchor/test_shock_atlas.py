"""Tests for the joint productivity-automation shock atlas.

The atlas reuses the maintained CS001 solve and adds directions, families and
checks on top; these tests pin the coordinate conversion, the analytic neutral
directions, linearity (superposition, sign symmetry, scaling), the
Brownian-versus-state timing convention, the separate-accounts identities
(including F - c = tau B - T), the reproduction of the baseline pipeline's
dalpha=+0.01 experiments, atomic chunked writes with resumption, and the
structural invariant line at the baseline calibration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tai_public_finance.cs001_lq_anchor import shock_atlas as sa
from tai_public_finance.cs001_lq_anchor.config import load_cs001_configuration

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "cs001" / "lq_farhi_smoke.json"


@pytest.fixture(scope="module")
def atlas():
    config = load_cs001_configuration(CONFIG_PATH)
    settings = dict(sa.DEFAULT_SETTINGS)
    horizons = sa.make_horizons(settings)
    model = sa.solve_atlas_model("baseline", config.parameters, config.experiment, horizons)
    directions = sa.build_directions(config.parameters, model.anchor, 5.0, named_only=True)
    reporting = config.experiment["reporting"]
    scaffolding = config.experiment["numerical_scaffolding"]
    brownian = sa.compute_path_sets(model, sa.brownian_initial_conditions(model, directions, reporting), horizons, scaffolding, settings)
    matched = sa.compute_path_sets(model, sa.matched_state_initial_conditions(model, directions, reporting), horizons, scaffolding, settings)
    ou = sa.compute_path_sets(model, sa.ou_window_initial_conditions(model, directions, reporting), horizons, scaffolding, settings)
    fixed = sa.compute_path_sets(model, sa.fixed_share_initial_conditions(model, directions, reporting), horizons, scaffolding, settings)
    return {"config": config, "settings": settings, "horizons": horizons, "model": model, "directions": directions, "brownian": brownian, "matched": matched, "ou": ou, "fixed": fixed}


def _named(path_sets, label):
    return next(ps for ps in path_sets if label in ps.ic.direction.named_labels)


# --- coordinates and named directions -------------------------------------------------------


def test_baseline_pipeline_reproduces_and_capital_tax_block_is_hurwitz(atlas):
    model = atlas["model"]
    assert model.acceptance.outcome == "pass", model.acceptance.failed_checks
    assert model.diagnostics.closed_loop["real_closed_loop_hurwitz"]
    assert model.portfolio.leading_unconstrained_position == pytest.approx(0.4464893494362929, abs=2e-11)


def test_named_direction_angles_and_conversion(atlas):
    model = atlas["model"]
    p = model.parameters
    slopes = sa.named_direction_slopes(model.anchor)
    assert sa.theta_from_dz_dalpha_slope(slopes[sa.NAMED_PURE_AUTOMATION], p, model.anchor) == pytest.approx(90.0)
    assert sa.theta_from_dz_dalpha_slope(slopes[sa.NAMED_PURE_PRODUCTIVITY], p, model.anchor) == 0.0
    for name, slope in slopes.items():
        if slope is None:
            continue
        theta = sa.theta_from_dz_dalpha_slope(slope, p, model.anchor)
        assert 0.0 < theta < 180.0  # positive dalpha
        back = sa.dz_dalpha_slope_from_theta(theta, p, model.anchor)
        assert back == pytest.approx(slope, rel=1e-12, abs=1e-14), name
    # claim-neutral direction is orthogonal to lambda_hat in standardized innovation space
    claim = next(d for d in atlas["directions"] if f"{sa.NAMED_CLAIM_NEUTRAL}_positive" in d.named_labels)
    lam = model.portfolio.lambda_hat
    assert abs(float(lam @ claim.unit)) / np.linalg.norm(lam) < 1e-12


def test_claim_and_rental_base_neutral_directions_coincide_under_aligned_normalization(atlas):
    model = atlas["model"]
    coincidences = sa.named_direction_coincidences(atlas["directions"])
    labels = {tuple(sorted(c["named_labels"])) for c in coincidences}
    assert (f"{sa.NAMED_CLAIM_NEUTRAL}_positive", f"{sa.NAMED_RENTAL_BASE_NEUTRAL}_positive") in labels
    assert sa.invariant_line_checks(model)["h_I_minus_eta_plus_one_over_alpha"] == 0.0


def test_every_direction_has_its_opposite(atlas):
    thetas = {round(d.theta_deg, 6) for d in atlas["directions"]}
    for theta in thetas:
        assert round((theta + 180.0) % 360.0, 6) in thetas


def test_full_grid_contains_five_degree_circle_and_off_grid_named_directions(atlas):
    model = atlas["model"]
    directions = sa.build_directions(model.parameters, model.anchor, 5.0)
    grid = [d for d in directions if d.on_grid]
    assert len(grid) == 72
    assert any(not d.on_grid for d in directions)
    assert len({d.key for d in directions}) == len(directions)


# --- neutrality on impact ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,field",
    [
        (sa.NAMED_OUTPUT_NEUTRAL, "output_deviation_linear"),
        (sa.NAMED_WORKER_INCOME_NEUTRAL, "wage_income_deviation_linear"),
        (sa.NAMED_PRIMARY_RESOURCE_NEUTRAL, "fiscal_resources_deviation_linear"),
        (sa.NAMED_RENTAL_BASE_NEUTRAL, "rental_rate_deviation_linear"),
        (sa.NAMED_CLAIM_NEUTRAL, "claim_loading_state_functional"),
    ],
)
def test_analytic_neutral_directions_zero_their_impact_object(atlas, label, field):
    for family in ("brownian", "matched", "ou"):
        ps = _named(atlas[family], f"{label}_positive")
        assert ps.features[f"cancellation_index_impact_{field}"] < 1e-11, (family, label)
        assert ps.features[f"impact_neutral_{field}"]


def test_claim_neutral_brownian_innovation_has_zero_inherited_claim_payoff(atlas):
    ps = _named(atlas["brownian"], f"{sa.NAMED_CLAIM_NEUTRAL}_positive")
    assert ps.ic.regime == sa.REGIME_OPTIMAL
    assert abs(ps.ic.claim_payoff) < 1e-18


def test_impact_neutrality_does_not_imply_path_neutrality_for_output_neutral_direction(atlas):
    ps = _named(atlas["matched"], f"{sa.NAMED_OUTPUT_NEUTRAL}_positive")
    assert ps.features["neutrality_unravel_time_years_output_deviation_linear"] is not None
    assert ps.features["neutrality_unravel_time_years_output_deviation_linear"] > 0.0


def test_claim_rental_neutral_line_is_invariant_for_capital_and_tax_at_baseline(atlas):
    """Structural coincidence at the aligned baseline (k_I = log(K/L), kappa_z = kappa_x):
    the tax-speed feedback and capital-growth loadings on (z, x) are proportional to the
    rental gradient, so the claim/rental-neutral displacement never moves capital or tax."""

    model = atlas["model"]
    inv = sa.invariant_line_checks(model)
    assert inv["tax_speed_feedback_alignment_relative_error"] < 1e-10
    assert inv["capital_growth_alignment_relative_error"] < 1e-12
    ps = _named(atlas["matched"], f"{sa.NAMED_CLAIM_NEUTRAL}_positive")
    assert np.max(np.abs(ps.path[:, 2])) < 1e-15
    assert np.max(np.abs(ps.path[:, 3])) < 1e-15


# --- linearity ---------------------------------------------------------------------------------


def test_superposition_component_split_and_cos_sin_basis(atlas):
    for family in ("brownian", "matched", "ou"):
        result = sa.superposition_checks(atlas["model"], atlas[family])
        assert result["component_split"]["error"] < 1e-10, family
        assert result["cos_sin_basis"]["count"] > 0
        assert result["cos_sin_basis"]["error"] < 1e-10, family


def test_sign_symmetry_theta_and_theta_plus_pi(atlas):
    for family in ("brownian", "matched", "ou", "fixed"):
        result = sa.sign_symmetry_checks(atlas[family])
        assert result["unpaired"] == []
        assert result["pairs"] > 0
        assert result["error"] < 1e-10, family


def test_halving_and_doubling_scale_every_first_order_response(atlas):
    ps = _named(atlas["brownian"], f"{sa.NAMED_WORKER_INCOME_NEUTRAL}_positive")
    assert sa.scaling_checks(atlas["model"], ps.ic, atlas["horizons"])["error"] < 1e-12


# --- timing distinction and accounting -----------------------------------------------------------


def test_brownian_and_matched_state_families_share_physical_paths_and_differ_by_the_payoff(atlas):
    result = sa.timing_distinction_checks(atlas["brownian"], atlas["matched"])
    assert result["compared_paths"] == len(atlas["brownian"])
    assert result["physical_state_max_abs_difference"] < 1e-15  # roundoff only: F has a zero last column
    assert result["optimal_x_gap_minus_payoff_max_abs"] < 1e-15
    assert result["optimal_x_gap_constancy_max_abs"] < 1e-15
    assert result["zero_position_vs_matched_max_abs"] == 0.0


def test_capital_and_inherited_tax_do_not_jump_in_any_family(atlas):
    ics = [ps.ic for family in ("brownian", "matched", "ou", "fixed") for ps in atlas[family]]
    assert sa.no_jump_checks(ics)["max_abs_capital_or_tax_impact_displacement"] == 0.0


def test_accounting_identities_including_separate_accounts(atlas):
    rows = [row for family in ("brownian", "matched", "ou", "fixed") for ps in atlas[family] for row in ps.rows]
    result = sa.accounting_identity_checks(rows, atlas["model"].rho)
    for key, value in result.items():
        if key == "locator":
            continue
        assert value < 1e-14, key
    # The planner object F contains wages; F - c = tau B - T is the government's primary cash flow.
    row = rows[0]
    assert row["government_primary_cash_flow_deviation_linear"] == pytest.approx(row["tax_revenue_deviation_linear"] - row["transfer_deviation_linear"], abs=1e-15)
    assert row["tax_adjustment_cost_deviation_first_order"] == 0.0


def test_rows_carry_separate_accounts_and_split_economic_from_scaffolding_flags(atlas):
    row = atlas["brownian"][0].rows[0]
    for field in (
        "wage_income_deviation_linear",
        "tax_revenue_deviation_linear",
        "transfer_deviation_linear",
        "tax_adjustment_cost_deviation_first_order",
        "tax_adjustment_cost_quadratic_diagnostic",
        "government_primary_cash_flow_deviation_linear",
        "public_net_worth_deviation",
        "risky_position_deviation",
        "safe_position_deviation",
        "transfer_slack",
        "economic_conditions_ok",
        "numerical_scaffolding_ok",
        "failure_reasons",
    ):
        assert field in row, field
    assert "feasible" not in row


def test_planner_resource_wealth_splits_into_wage_and_capital_tax_components(atlas):
    model = atlas["model"]
    G = model.outcome_matrix
    idx = sa.LINEAR_INDEX
    total = G[idx["planner_resource_wealth_deviation"]]
    wage = G[idx["planner_resource_wealth_wage_component_deviation"]]
    tax = G[idx["planner_resource_wealth_capital_tax_component_deviation"]]
    np.testing.assert_allclose(wage + tax, total, atol=1e-12)
    # k-coefficients: a unit of capital is worth one unit of J at the margin (j_k = 1). Its
    # capitalized-wage part exceeds one while its capitalized-capital-tax part is negative:
    # capital deepening raises future wages but shrinks the net-rental tax base (dB/dk = alpha R - delta < 0).
    assert wage[2] > 1.0 and tax[2] < 0.0
    assert wage[2] + tax[2] == pytest.approx(1.0, abs=1e-10)
    # The inherited tax rate redistributes J between the two components with zero net first-order effect.
    assert wage[3] + tax[3] == pytest.approx(0.0, abs=1e-10) and wage[3] < 0.0 < tax[3]
    rows = [row for ps in atlas["brownian"] for row in ps.rows]
    assert sa.accounting_identity_checks(rows, model.rho)["dJ_equals_wage_component_plus_capital_tax_component"] < 1e-14


def test_exactly_invariant_path_reports_no_noise_sign_reversals(atlas):
    ps = _named(atlas["matched"], f"{sa.NAMED_CLAIM_NEUTRAL}_positive")
    assert ps.features["tax_sign_reversal_count"] == 0
    assert abs(ps.features["tax_rate_max"]) < 1e-15 and abs(ps.features["tax_rate_min"]) < 1e-15


def test_anchor_decomposition_matches_documented_values(atlas):
    d = sa.anchor_decomposition(atlas["model"])
    assert d["wage_income_W"] == pytest.approx(0.2337282, abs=1e-6)
    assert d["capital_tax_receipts_tauB"] == pytest.approx(0.0202027, abs=1e-6)
    assert d["worker_wage_endowment_value_W_over_rho"] == pytest.approx(11.56915, abs=1e-4)
    assert d["capital_tax_resource_value_tauB_over_rho"] == pytest.approx(1.0, abs=1e-9)
    assert d["planner_resource_wealth_J"] == pytest.approx(12.56915, abs=1e-4)
    assert abs(d["decomposition_residual"]) < 1e-12
    assert "unverified" in d["portfolio_classification"]


# --- reproduction of the baseline pipeline -----------------------------------------------------


def test_fixed_share_family_reproduces_baseline_constructed_experiments(atlas):
    result = sa.fixed_share_reproduction_check(atlas["model"], atlas["fixed"], atlas["horizons"])
    assert result["missing"] == []
    assert result["compared_rows"] > 0
    assert result["max_abs_difference"] < 1e-12, result


def test_row_builder_agrees_with_baseline_irfs_row(atlas):
    result = sa.row_builder_cross_check(atlas["model"], atlas["brownian"] + atlas["matched"], atlas["config"].experiment["numerical_scaffolding"])
    assert result["compared_rows"] > 0
    assert result["max_abs_difference"] < 1e-12, result


def test_matrix_exponential_paths_agree_with_direct_ode_integration(atlas):
    for family in ("brownian", "matched", "ou"):
        assert max(ps.ode_relative_error for ps in atlas[family]) < 1e-8


def test_discounted_cumulative_resolvent_and_expm_integral_agree(atlas):
    model = atlas["model"]
    ps = _named(atlas["brownian"], f"{sa.NAMED_PURE_AUTOMATION}_positive")
    cumulative, residual = sa.discounted_cumulative(model, ps.ic.w0)
    assert residual < 1e-10
    integral = sa.discounted_cumulative_expm_integral(model, ps.ic.w0, 1500.0)
    assert np.max(np.abs(cumulative - integral)) / np.max(np.abs(cumulative)) < 1e-9


# --- persistence variant ----------------------------------------------------------------------


def test_persistence_variant_unravels_the_claim_neutral_invariant_line(atlas):
    config = atlas["config"]
    variant = sa.persistence_variant_parameters(config.parameters, 0.50)
    model = sa.solve_atlas_model("automation_persistence_0.50", variant, config.experiment, atlas["horizons"], {"automation_persistence_annual": 0.5})
    assert model.acceptance.outcome == "pass", model.acceptance.failed_checks
    base = atlas["model"]
    named = [d for d in atlas["directions"] if d.named_labels]
    fixed = sa.matched_state_initial_conditions(model, named, config.experiment["reporting"], magnitudes=sa.brownian_magnitudes(base, config.experiment["reporting"]), family=sa.FAMILY_FIXED_ACROSS_PERSISTENCE)
    sets = sa.compute_path_sets(model, fixed, atlas["horizons"], config.experiment["numerical_scaffolding"], atlas["settings"], ode_check=lambda ic: False)
    ps = _named(sets, f"{sa.NAMED_CLAIM_NEUTRAL}_positive")
    assert np.max(np.abs(ps.path[:, 2])) > 1e-7  # capital now moves: the neutral mixture rotates
    assert ps.features["state_rotation_deg_5y"] != pytest.approx(0.0, abs=1e-6)
    assert ps.ic.dz == pytest.approx(_named(atlas["matched"], f"{sa.NAMED_CLAIM_NEUTRAL}_positive").ic.dz)


# --- runner: atomic writes, state file, resumption --------------------------------------------


def test_runner_smoke_mode_writes_tables_state_and_events(tmp_path):
    config = load_cs001_configuration(CONFIG_PATH)
    runner = sa.AtlasRunner(config, tmp_path / "run", {}, "testcommit", resume=False, mode="smoke")
    result = runner.run()
    assert result["checks"]["outcome"] == "pass", result["checks"]["failures"]
    for name in ("state.json", "events.log", "models.json", "numerical_diagnostics.json", "atlas_raw.csv", "atlas_raw_quarterly.csv.gz", "path_features.csv", "named_directions.csv", "zero_impact_thresholds.csv", "impact_sign_regions.csv", "failed_rows.csv", "runtime.json"):
        assert (tmp_path / "run" / name).exists(), name
    state = json.loads((tmp_path / "run" / "state.json").read_text())
    assert state["completed_chunks"] == [sa.CHUNK_MODELS, sa.CHUNK_BROWNIAN, sa.CHUNK_MATCHED, sa.CHUNK_OU, sa.CHUNK_CHECKS, sa.CHUNK_TABLES]
    assert not list((tmp_path / "run").rglob("*.tmp"))
    events = [json.loads(line) for line in (tmp_path / "run" / "events.log").read_text().splitlines()]
    assert events[0]["event"] == "runner_initialized"
    assert events[-1]["event"] == "run_completed"
    rows = sa.read_csv_typed(tmp_path / "run" / "atlas_raw.csv", sa.ROW_STRING_FIELDS, sa.ROW_BOOL_FIELDS)
    assert {row["horizon_years"] for row in rows} == set(sa.KEY_HORIZONS)
    assert all(row["economic_conditions_ok"] and row["numerical_scaffolding_ok"] for row in rows)


class _InterruptedRunner(sa.AtlasRunner):
    attempts = 0

    def _chunk_sets(self, chunk):
        if chunk == sa.CHUNK_OU and _InterruptedRunner.attempts == 0:
            _InterruptedRunner.attempts += 1
            raise RuntimeError("simulated interruption before the OU chunk was written")
        return super()._chunk_sets(chunk)


def test_runner_resumes_only_unfinished_chunks_and_reproduces_a_clean_run(tmp_path):
    config = load_cs001_configuration(CONFIG_PATH)
    clean = sa.AtlasRunner(config, tmp_path / "clean", {}, "testcommit", resume=False, mode="smoke")
    clean.run()
    _InterruptedRunner.attempts = 0
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _InterruptedRunner(config, tmp_path / "resumed", {}, "testcommit", resume=False, mode="smoke").run()
    state = json.loads((tmp_path / "resumed" / "state.json").read_text())
    assert state["completed_chunks"] == [sa.CHUNK_MODELS, sa.CHUNK_BROWNIAN, sa.CHUNK_MATCHED]
    resumed = _InterruptedRunner(config, tmp_path / "resumed", {}, "testcommit", resume=True, mode="smoke")
    resumed.run()
    events = [json.loads(line) for line in (tmp_path / "resumed" / "events.log").read_text().splitlines()]
    skipped = [e["chunk"] for e in events if e["event"] == "chunk_skipped_already_complete"]
    assert skipped == [sa.CHUNK_MODELS, sa.CHUNK_BROWNIAN, sa.CHUNK_MATCHED]
    for name in ("atlas_raw.csv", "atlas_raw_quarterly.csv.gz", "path_features.csv", "named_directions.csv", "zero_impact_thresholds.csv"):
        assert (tmp_path / "resumed" / name).read_bytes() == (tmp_path / "clean" / name).read_bytes(), name


def test_runner_refuses_to_resume_under_a_different_fingerprint(tmp_path):
    config = load_cs001_configuration(CONFIG_PATH)
    runner = sa.AtlasRunner(config, tmp_path / "run", {}, "commitA", resume=False, mode="smoke")
    runner.run()
    with pytest.raises(RuntimeError, match="different commit/configuration fingerprint"):
        sa.AtlasRunner(config, tmp_path / "run", {}, "commitB", resume=True, mode="smoke")
    with pytest.raises(RuntimeError, match="resume=True"):
        sa.AtlasRunner(config, tmp_path / "run", {}, "commitA", resume=False, mode="smoke")
