"""Command-line entry point for the CS001 joint productivity-automation shock atlas.

    uv run python -m tai_public_finance.cs001_lq_anchor.shock_atlas_cli \
        --config configs/cs001/lq_farhi_smoke.json \
        --output-dir outputs/cs001-joint-shock-atlas-<UTC timestamp> [--mode full|smoke] [--resume]

The run writes a state file after every completed chunk and an append-only
event log; ``--resume`` continues only unfinished chunks under the same commit
and configuration fingerprint. ``--finalize`` (after the hand-written review
packets exist) hashes every artifact and writes the immutable run record.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import load_cs001_configuration
from .reporting import environment_metadata, git_metadata
from .shock_atlas import (
    DEFAULT_SETTINGS,
    FAMILY_BROWNIAN,
    FAMILY_FIXED_ACROSS_PERSISTENCE,
    FAMILY_MATCHED_STATE,
    REGIME_NONE,
    REGIME_OPTIMAL,
    REGIME_ZERO,
    ROW_BOOL_FIELDS,
    ROW_STRING_FIELDS,
    AtlasCheckFailure,
    AtlasRunner,
    atomic_write_json,
    atomic_write_text,
    read_csv_typed,
    serial,
    sha256_file,
)

FEATURE_STRING_FIELDS = ("model", "family", "experiment_id", "regime", "direction_key", "named_labels", "impact_sign_pattern")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Figures (generated last, from the tidy tables only; every figure has a data CSV)
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        atomic_write_text(path, "")
        return
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def write_figures(output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures_dir = output_dir / "figures"
    data_dir = output_dir / "figure_data"
    figures_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    features = read_csv_typed(output_dir / "path_features.csv", FEATURE_STRING_FIELDS)
    paths: list[Path] = []

    def subset(family: str, regime: str, model: str = "baseline") -> list[dict[str, Any]]:
        return sorted((f for f in features if f["family"] == family and f["regime"] == regime and f["model"] == model), key=lambda f: f["theta_deg"])

    named_angles = sorted({(f["theta_deg"], f["named_labels"]) for f in features if f["named_labels"] and f["model"] == "baseline"})

    # 1. Impact incidence by angle (Brownian, optimal inherited position)
    optimal = subset(FAMILY_BROWNIAN, REGIME_OPTIMAL)
    zero = subset(FAMILY_BROWNIAN, REGIME_ZERO)
    series = [
        ("impact_output_deviation_linear", "Output Y"),
        ("impact_wage_income_deviation_linear", "Wage income W"),
        ("impact_tax_revenue_deviation_linear", "Capital-tax receipts tau B"),
        ("impact_fiscal_resources_deviation_linear", "Planner resources F = tau B + W"),
        ("claim_payoff_impact", "Inherited claim payoff (s_- = s*)"),
        ("impact_worker_consumption_deviation", "Worker consumption c"),
        ("impact_transfer_deviation_linear", "Transfer T"),
        ("impact_government_primary_cash_flow_deviation_linear", "Government primary cash flow tau B - T"),
    ]
    rows = [{"theta_deg": f["theta_deg"], "named_labels": f["named_labels"], **{k: f[k] for k, _ in series}} for f in optimal]
    _write_csv(data_dir / "impact_incidence_by_angle.csv", rows)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    theta = [f["theta_deg"] for f in optimal]
    for key, label in series[:4]:
        axes[0].plot(theta, [f[key] for f in optimal], label=label)
    for key, label in series[4:]:
        axes[1].plot(theta, [f[key] for f in optimal], label=label)
    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=0.7)
        for angle, labels in named_angles:
            ax.axvline(angle, color="grey", linewidth=0.5, linestyle=":")
        ax.legend(fontsize=7)
    axes[0].set_title("Impact responses to one standardized joint Brownian innovation, by direction (s_- = s*)")
    axes[1].set_xlabel("theta (degrees): 0 = +productivity, 90 = +automation, 180 = -productivity, 270 = -automation")
    fig.tight_layout()
    path = figures_dir / "impact_incidence_by_angle.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # 2. Worker comprehensive resources and transfers with and without the inherited claim
    rows = []
    zero_by_theta = {f["theta_deg"]: f for f in zero}
    for f in optimal:
        z = zero_by_theta.get(f["theta_deg"])
        rows.append(
            {
                "theta_deg": f["theta_deg"],
                "named_labels": f["named_labels"],
                "worker_comprehensive_resources_impact_optimal": f["worker_comprehensive_resources_impact"],
                "worker_comprehensive_resources_impact_zero_position": z["worker_comprehensive_resources_impact"] if z else None,
                "claim_payoff_impact": f["claim_payoff_impact"],
                "domestic_planner_resource_wealth_contribution_impact": f["domestic_planner_resource_wealth_contribution_impact"],
                "transfer_impact_optimal": f["impact_transfer_deviation_linear"],
                "transfer_impact_zero_position": z["impact_transfer_deviation_linear"] if z else None,
                "wage_impact": f["impact_wage_income_deviation_linear"],
            }
        )
    _write_csv(data_dir / "resources_and_transfers_by_angle.csv", rows)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(theta, [r["worker_comprehensive_resources_impact_optimal"] for r in rows], label="dX, s_- = s* (payoff + domestic J change)")
    axes[0].plot(theta, [r["worker_comprehensive_resources_impact_zero_position"] for r in rows], label="dX, s_- = 0 (domestic J change only)")
    axes[0].plot(theta, [r["claim_payoff_impact"] for r in rows], label="inherited claim payoff s* lambda.dW")
    axes[1].plot(theta, [r["transfer_impact_optimal"] for r in rows], label="dT, s_- = s*")
    axes[1].plot(theta, [r["transfer_impact_zero_position"] for r in rows], label="dT, s_- = 0")
    axes[1].plot(theta, [r["wage_impact"] for r in rows], label="dW")
    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=0.7)
        for angle, labels in named_angles:
            ax.axvline(angle, color="grey", linewidth=0.5, linestyle=":")
        ax.legend(fontsize=7)
    axes[0].set_title("Worker comprehensive resources X = N + J and transfers, with and without the inherited claim payoff")
    axes[1].set_xlabel("theta (degrees)")
    fig.tight_layout()
    path = figures_dir / "resources_and_transfers_by_angle.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # 3. Path features by angle (matched-state family: pure propagation, no payoff)
    matched = subset(FAMILY_MATCHED_STATE, REGIME_NONE)
    rows = [
        {
            "theta_deg": f["theta_deg"],
            "named_labels": f["named_labels"],
            "wage_sign_change_time_years": f["wage_sign_change_time_years"],
            "peak_abs_capital_time_years": f["peak_abs_capital_time_years"],
            "peak_abs_capital_value": f["peak_abs_capital_value"],
            "tax_sign_reversal_count": f["tax_sign_reversal_count"],
            "tax_rate_max": f["tax_rate_max"],
            "tax_rate_min": f["tax_rate_min"],
            "discounted_cumulative_wage": f["discounted_cumulative_wage_income_deviation_linear"],
            "discounted_cumulative_consumption": f["discounted_cumulative_worker_consumption_deviation"],
        }
        for f in matched
    ]
    _write_csv(data_dir / "path_features_by_angle.csv", rows)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    th = [r["theta_deg"] for r in rows]
    axes[0].plot(th, [r["wage_sign_change_time_years"] if r["wage_sign_change_time_years"] is not None else np.nan for r in rows], ".-", label="wage sign-change time (years)")
    axes[0].plot(th, [r["peak_abs_capital_time_years"] for r in rows], ".-", label="peak |capital| time (years)")
    axes[1].plot(th, [r["peak_abs_capital_value"] for r in rows], ".-", label="peak capital response (signed)")
    axes[1].plot(th, [r["tax_rate_max"] for r in rows], ".-", label="tax-rate max")
    axes[1].plot(th, [r["tax_rate_min"] for r in rows], ".-", label="tax-rate min")
    axes[2].plot(th, [r["discounted_cumulative_wage"] for r in rows], ".-", label="discounted cumulative wage")
    axes[2].plot(th, [r["discounted_cumulative_consumption"] for r in rows], ".-", label="discounted cumulative consumption")
    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=0.7)
        for angle, labels in named_angles:
            ax.axvline(angle, color="grey", linewidth=0.5, linestyle=":")
        ax.legend(fontsize=7)
    axes[0].set_title("Propagation features by direction (matched deterministic state displacement)")
    axes[2].set_xlabel("theta (degrees)")
    fig.tight_layout()
    path = figures_dir / "path_features_by_angle.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # 4. Persistence unravelling for the named directions (fixed initial displacement)
    persistence_paths = output_dir / "persistence_named_paths.csv"
    if persistence_paths.exists() and persistence_paths.stat().st_size > 0:
        prow = read_csv_typed(persistence_paths, ROW_STRING_FIELDS, ROW_BOOL_FIELDS)
        base_rows = read_csv_typed(output_dir / "atlas_raw.csv", ROW_STRING_FIELDS, ROW_BOOL_FIELDS)
        fixed = [r for r in prow if r["family"] == FAMILY_FIXED_ACROSS_PERSISTENCE] + [r for r in base_rows if r["family"] == FAMILY_MATCHED_STATE and r["named_labels"] and r["model"] == "baseline"]
        labels = sorted({r["named_labels"] for r in fixed if "positive" in r["named_labels"]})
        variables = [("output_deviation_linear", "Output"), ("wage_income_deviation_linear", "Wage income"), ("fiscal_resources_deviation_linear", "Planner resources F"), ("claim_loading_state_functional", "Claim-loading state functional dz + ell_x dx")]
        fig, axes = plt.subplots(len(labels), len(variables), figsize=(4 * len(variables), 2.6 * len(labels)), sharex=True)
        data_rows = []
        for i, label in enumerate(labels):
            for jdx, (variable, title) in enumerate(variables):
                ax = axes[i, jdx] if len(labels) > 1 else axes[jdx]
                for model in sorted({r["model"] for r in fixed}):
                    pts = sorted((r for r in fixed if r["named_labels"] == label and r["model"] == model), key=lambda r: r["horizon_years"])
                    if not pts:
                        continue
                    ax.plot([r["horizon_years"] for r in pts], [r[variable] for r in pts], ".-", label=model, markersize=3)
                    data_rows.extend({"named_labels": label, "model": model, "variable": variable, "horizon_years": r["horizon_years"], "value": r[variable]} for r in pts)
                ax.axhline(0.0, color="black", linewidth=0.6)
                if i == 0:
                    ax.set_title(title, fontsize=9)
                if jdx == 0:
                    ax.set_ylabel(label.replace("_automation_positive", "").replace("_positive", ""), fontsize=7)
                if i == 0 and jdx == 0:
                    ax.legend(fontsize=6)
        fig.suptitle("Fixed initial (dz, dalpha) displacement propagated under different automation persistence", fontsize=10)
        fig.tight_layout()
        path = figures_dir / "persistence_unravelling_named_directions.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
        _write_csv(data_dir / "persistence_unravelling_named_directions.csv", data_rows)

    # 5. Polar sign map of impact incidence (Brownian, optimal inherited position)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="polar")
    bands = [("impact_output_deviation_linear", "Y > 0", 1.0), ("impact_wage_income_deviation_linear", "W > 0", 0.85), ("impact_fiscal_resources_deviation_linear", "F > 0", 0.7), ("impact_tax_revenue_deviation_linear", "tau B > 0", 0.55), ("claim_payoff_impact", "payoff > 0", 0.4), ("impact_transfer_deviation_linear", "T > 0", 0.25)]
    data_rows = []
    for key, label, radius in bands:
        angles = np.radians([f["theta_deg"] for f in optimal])
        positive = np.array([f[key] > 0 for f in optimal])
        ax.scatter(angles[positive], np.full(int(positive.sum()), radius), s=14, label=label)
        data_rows.extend({"object": key, "theta_deg": f["theta_deg"], "positive": bool(f[key] > 0)} for f in optimal)
    for angle, labels in named_angles:
        ax.plot([np.radians(angle), np.radians(angle)], [0.0, 1.05], color="grey", linewidth=0.5, linestyle=":")
    ax.set_yticks([])
    ax.set_title("Where each impact object is positive (dots), by innovation direction", fontsize=9)
    ax.legend(loc="lower left", bbox_to_anchor=(1.0, 0.0), fontsize=7)
    fig.tight_layout()
    path = figures_dir / "polar_impact_sign_map.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    _write_csv(data_dir / "polar_impact_sign_map.csv", data_rows)
    return paths


# ---------------------------------------------------------------------------
# Numerical review packet (auto-generated from the diagnostics; no log reading needed)
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3e}"
    return str(value)


def write_numerical_review_packet(output_dir: Path, manifest: dict[str, Any]) -> Path:
    diagnostics = json.loads((output_dir / "numerical_diagnostics.json").read_text(encoding="utf-8")) if (output_dir / "numerical_diagnostics.json").exists() else {}
    models = json.loads((output_dir / "models.json").read_text(encoding="utf-8")) if (output_dir / "models.json").exists() else {}
    runtime = json.loads((output_dir / "runtime.json").read_text(encoding="utf-8")) if (output_dir / "runtime.json").exists() else {}
    lines = [
        "# Numerical review packet: CS001 joint productivity-automation shock atlas",
        "",
        f"- Run started (UTC): `{manifest.get('started_utc')}`; finished: `{manifest.get('finished_utc')}`",
        f"- Commit `{manifest.get('git', {}).get('commit')}` on branch `{manifest.get('git', {}).get('branch')}` (dirty at start: {manifest.get('git', {}).get('dirty_worktree_at_run_start')})",
        f"- Complete-input fingerprint `{manifest.get('fingerprints', {}).get('complete_input_sha256')}`; atlas fingerprint `{manifest.get('atlas_fingerprint')}`",
        f"- Outcome of independent checks: **{diagnostics.get('outcome', 'not run')}**; failures: `{diagnostics.get('failures', [])}`",
        f"- Machine runtime (seconds by chunk): `{json.dumps({k: round(v, 3) for k, v in runtime.get('seconds_by_chunk', {}).items()})}`; total `{runtime.get('total_seconds', 0.0):.2f}` s",
        "",
        "Raw-output locators: every row of `atlas_raw_quarterly.csv` (all 161 quarterly horizons) and `atlas_raw.csv` (13 key horizons) is keyed by `model`, `family`, `regime`, `direction_key` (= `theta_ddd.ddd`), `horizon_years`; one row per path in `path_features.csv`; per-chunk raw parts under `parts/`. Locators below use `model::family::regime::direction_key@horizon`.",
        "",
        "## Matrix-equation residuals per solved model (baseline pipeline diagnostics, recomputed from primitives)",
        "",
        "| model | acceptance | Riccati (full) | Riccati (real block) | Sylvester | disc. Lyapunov | closed-form vs Schur | Hamiltonian axis distance | capital-tax Hurwitz (margin) | full loop Hurwitz |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, m in diagnostics.get("models", {}).items():
        lines.append(
            f"| {label} | {m['acceptance_outcome']} | {_fmt(m['riccati_full_scaled_residual'])} | {_fmt(m['riccati_real_block_scaled_residual'])} | {_fmt(m['sylvester_scaled_residual'])} | {_fmt(m['discounted_lyapunov_scaled_residual'])} | {_fmt(m['closed_form_vs_invariant_subspace_relative_error'])} | {_fmt(m['hamiltonian_imaginary_axis_distance'])} | {_fmt(m['real_closed_loop_hurwitz'])} ({_fmt(m['real_closed_loop_stability_margin'])}) | {_fmt(m['full_closed_loop_hurwitz'])} |"
        )
    lines += ["", "Finite-difference primitive checks (gradient / Hessian relative errors) and feedback-construction errors:", ""]
    for label, m in diagnostics.get("models", {}).items():
        fd = m["finite_difference_checks"]
        fc = m["feedback_construction_errors"]
        lines.append(f"- {label}: " + "; ".join(f"{k}: {_fmt(v['gradient_relative_error'])} / {_fmt(v['hessian_relative_error'])}" for k, v in fd.items()) + f"; A_rc/A_c/F construction {_fmt(fc['A_rc_relative_error'])}/{_fmt(fc['A_c_relative_error'])}/{_fmt(fc['F_relative_error'])}; resolvent identity {_fmt(m['resolvent_identity_residual'])}")
    d = diagnostics
    if d:
        lines += [
            "",
            "## Worst-case path checks (with locators)",
            "",
            f"- Matrix exponential vs direct DOP853 ODE integration: max relative error {_fmt(d['matrix_exponential_vs_ode']['max_relative_error'])} over {d['matrix_exponential_vs_ode']['paths_checked']} paths at `{d['matrix_exponential_vs_ode']['locator']}`",
        ]
        for label, sp in d["superposition"].items():
            lines.append(f"- Superposition ({label}): component split max rel. error {_fmt(sp['component_split']['error'])} at `{sp['component_split']['locator']}`; cos/sin basis max rel. error {_fmt(sp['cos_sin_basis']['error'])} over {sp['cos_sin_basis']['count']} paths at `{sp['cos_sin_basis']['locator']}`")
        for label, ss in d["sign_symmetry"].items():
            lines.append(f"- Sign symmetry theta vs theta+pi ({label}): max rel. error {_fmt(ss['error'])} over {ss['pairs']} pairs at `{ss['locator']}`; unpaired: {ss['unpaired']}")
        lines.append(f"- Scaling (halve/double selected displacements): max rel. error {_fmt(max((s['error'] for s in d['scaling']), default=0.0))} over {len(d['scaling'])} paths")
        td = d["timing_distinction"]
        lines.append(f"- Brownian vs matched-state timing distinction over {td['compared_paths']} paths: physical-state max abs difference {_fmt(td['physical_state_max_abs_difference'])}; (X gap - claim payoff) max abs {_fmt(td['optimal_x_gap_minus_payoff_max_abs'])}; X-gap constancy over horizons {_fmt(td['optimal_x_gap_constancy_max_abs'])}; zero-position vs matched max abs {_fmt(td['zero_position_vs_matched_max_abs'])}")
        lines.append(f"- Capital/tax jump at impact: max abs displacement {_fmt(d['no_jump']['max_abs_capital_or_tax_impact_displacement'])} over {d['no_jump']['count']} initial conditions")
        acc = d["accounting_identities"]
        lines.append("- Accounting identities (max abs error over every row): " + "; ".join(f"{k} {_fmt(v)}" for k, v in acc.items() if k != "locator") + f"; worst locator `{acc['locator']}`")
        for label, cc in d["coordinate_conversion"].items():
            lines.append(f"- Coordinate conversion ({label}): theta->(dz,dalpha)->theta round trip {_fmt(cc['round_trip_max_abs_degrees'])} deg; named-direction vs analytic zero-angle max error {_fmt(cc['named_zero_angle_max_error_degrees'])} deg ({json.dumps({k: round(v, 12) for k, v in cc['named_zero_angle_errors_degrees'].items()})}); claim-neutral orthogonality to lambda_hat {_fmt(cc['claim_neutral_orthogonal_to_lambda_hat_relative'])}")
        nz = d["neutral_zero"]
        lines.append(f"- Analytic neutral directions: worst impact cancellation index {_fmt(nz['worst_cancellation_index'])} (tolerance {_fmt(nz['tolerance'])}) over {len(nz['per_direction'])} labelled paths")
        fs = d["fixed_share_reproduction"]
        lines.append(f"- dalpha=+0.01 reproduction of the baseline pipeline's constructed_* experiments: max abs difference {_fmt(fs.get('max_abs_difference'))} over {fs.get('compared_rows')} rows at `{fs.get('locator')}`; missing rows {fs.get('missing', [])}" if "skipped" not in fs else f"- dalpha=+0.01 reproduction: {fs['skipped']}")
        rb = d["row_builder_cross_check"]
        lines.append(f"- Row builder vs baseline irfs._row on shared fields: max abs difference {_fmt(rb['max_abs_difference'])} over {rb['compared_rows']} rows at `{rb['locator']}`")
        bi = d["first_order_budget_identity"]
        lines.append(f"- First-order budget identity: max abs residual {_fmt(bi['max_abs_residual'])} over {bi['rows']} rows at `{bi['locator']}`")
        lines.append(f"- Discounted cumulative responses: resolvent probe residual max {_fmt(d['discounted_cumulative_resolvent']['max_probe_residual'])} at `{d['discounted_cumulative_resolvent']['locator']}`; resolvent vs matrix-exponential integral to T={d['discounted_cumulative_expm_integral']['horizon_years']:g} years max rel. error {_fmt(d['discounted_cumulative_expm_integral']['max_relative_error'])} at `{d['discounted_cumulative_expm_integral']['locator']}`")
        fe = d["feasibility"]
        lines += [
            "",
            "## Closest boundaries, failed and infeasible rows, missing rows",
            "",
            f"- Rows: {fe['rows']}; rows failing genuine economic conditions: {fe['rows_failing_economic_conditions']}; rows hitting numerical scaffolding: {fe['rows_failing_numerical_scaffolding']}; nonfinite rows: {d['nonfinite_rows']}",
            f"- Minimum genuine economic slack (specialisation margins, transfer floor c-W, tau<1): {_fmt(fe['min_economic_slack'])} at `{fe['min_economic_slack_locator']}`",
            f"- Minimum numerical-scaffolding slack (portfolio caps, tax box, tax-speed cap): {_fmt(fe['min_numerical_scaffolding_slack'])} at `{fe['min_numerical_scaffolding_slack_locator']}`",
            f"- Classification: {fe['classification']}",
            "- Failed/infeasible rows are retained in `failed_rows.csv` (empty header-only file means none).",
            "- Missing rows: every path carries the full quarterly horizon grid by construction; `sign_symmetry` `unpaired` lists any direction without its opposite (must be empty).",
        ]
    if models:
        lines += ["", "## Named directions, coincidences and invariant-line diagnostics", ""]
        for label, m in models.items():
            inv = m.get("invariant_line", {})
            lines.append(f"- {label}: named angles " + ", ".join(f"{v['labels'][0]}={v['theta_deg']:.6f}" for v in m.get("named_direction_angles_deg", {}).values() if v["labels"] and v["labels"][0].endswith("positive")) + f"; coincidences {json.dumps(m.get('named_direction_coincidences'))}; h_I-(eta+1/alpha)={_fmt(inv.get('h_I_minus_eta_plus_one_over_alpha'))}; tax-speed feedback alignment with rental gradient rel. error {_fmt(inv.get('tax_speed_feedback_alignment_relative_error'))}; capital-growth alignment {_fmt(inv.get('capital_growth_alignment_relative_error'))}; kappa_z-kappa_x={_fmt(inv.get('kappa_z_minus_kappa_x'))}")
    lines += [
        "",
        "## Interpretation boundary for the numerical reviewer",
        "",
        "All checks are on the solved first-order linear system: exactness of the matrix exponential, linearity (superposition, sign symmetry, scaling), the Brownian-versus-state timing convention, and accounting identities that hold by construction. Passing them establishes internal consistency of the local LQ computation, not global validity of the LQ approximation, borrowing capacity, or welfare.",
        "",
    ]
    path = output_dir / "NUMERICAL_REVIEW_PACKET.md"
    atomic_write_text(path, "\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------


def _hash_tree(output_dir: Path, repository: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file() and p.name != ".gitignore" and not p.name.endswith(".tmp")):
        try:
            location = str(path.relative_to(repository))
        except ValueError:
            location = str(path)
        role = "figure" if path.suffix == ".png" else ("review_packet" if path.suffix == ".md" else ("raw_part" if path.parent.name == "parts" else "primary_output" if path.name in ("atlas_raw.csv", "numerical_diagnostics.json") else "supporting_output"))
        artifacts.append({"role": role, "location": location, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return artifacts


def write_run_record(output_dir: Path, repository: Path, runs_dir: Path, run_id: str | None, manifest: dict[str, Any], preflight_tests_status: str) -> Path:
    diagnostics = json.loads((output_dir / "numerical_diagnostics.json").read_text(encoding="utf-8")) if (output_dir / "numerical_diagnostics.json").exists() else {}
    runtime = json.loads((output_dir / "runtime.json").read_text(encoding="utf-8")) if (output_dir / "runtime.json").exists() else {}
    models = json.loads((output_dir / "models.json").read_text(encoding="utf-8")) if (output_dir / "models.json").exists() else {}
    environment = environment_metadata()
    started = manifest.get("started_utc", _now())
    stamp = started.replace("-", "").replace(":", "")
    commit = (manifest.get("git", {}).get("commit") or "nogit")[:8]
    run_id = run_id or f"RUN-{stamp}-CS001-ATLAS-{commit}-01"
    artifacts = _hash_tree(output_dir, repository)
    record = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_utc": _now(),
        "status": "completed" if diagnostics.get("outcome") == "pass" else "failed",
        "purpose": "CS001 joint productivity-automation shock atlas (exploratory local-LQ evidence on the illustrative Farhi-based vector): "
        "one standardized joint Brownian innovation scanned over the full direction circle at five-degree resolution plus every analytically "
        "constructed direction, with matched deterministic state displacements, finite-window OU displacements, and a persistence-unravelling "
        "experiment. Exploratory CS001 tranche; CS001 remains draft/unfingerprinted.",
        "specification": {"id": "CS001", "version": "0.1", "status": "draft", "fingerprint_sha256": None},
        "problem_id": "CP001",
        "approach_id": "CA001",
        "based_on_run": "RUN-20260901T194938Z-CS001-d876a61e-01",
        "implementation": {
            "repository_url": "https://github.com/Nathan-Barnard/tai-public-finance",
            "local_path": str(repository),
            "commit": manifest.get("git", {}).get("commit"),
            "branch": manifest.get("git", {}).get("branch"),
            "dirty_worktree_at_run_start": manifest.get("git", {}).get("dirty_worktree_at_run_start"),
            "implementer": "claude_code",
            "entrypoint": "python3 -m tai_public_finance.cs001_lq_anchor.shock_atlas_cli",
            "command": manifest.get("command"),
            "resume_events": manifest.get("resume_events", []),
        },
        "input_fingerprints": manifest.get("fingerprints"),
        "atlas_fingerprint": manifest.get("atlas_fingerprint"),
        "atlas_settings": manifest.get("settings"),
        "environment": {
            "operating_system": environment["operating_system"],
            "architecture": environment["architecture"],
            "runtime_versions": {key: environment[key] for key in ("python", "numpy", "scipy", "matplotlib", "pyyaml")},
            "dependency_lock_path": "uv.lock",
            "dependency_lock_sha256": sha256_file(repository / "uv.lock") if (repository / "uv.lock").exists() else None,
        },
        "hardware": {"resource_lane": "L2_local_batch", "machine_label": "local_mac", "cpu": environment["cpu"], "logical_cores": environment["logical_cores"], "memory_gb": environment["memory_gb"], "accelerator": None},
        "budget": {"wall_seconds_limit": 7200, "cash_limit_usd": 0, "actual_cash_usd": 0, "early_stop_rule": "Stop on a reproducible failure of baseline reproduction, superposition, sign symmetry, coordinate conversion, budget accounting, or the Brownian/state-displacement timing distinction; retain all rows."},
        "randomness": {"deterministic_requested": True, "seeds": [], "nondeterminism_notes": "Dense deterministic float64 linear algebra; no simulation draws."},
        "inputs": {
            "parameter_set_id": manifest.get("parameter_set_id"),
            "experiment_ids": ["E07", "E08", "E01", "E02", "E03", "E04"],
            "models": {label: {"overrides": m.get("overrides"), "acceptance_outcome": m.get("acceptance_outcome"), "direction_count": m.get("direction_count")} for label, m in models.items()},
            "maintained_setup": manifest.get("maintained_setup"),
        },
        "preflight": {"tests_command": "uv run pytest", "tests_status": preflight_tests_status},
        "result": {
            "outcome": diagnostics.get("outcome"),
            "failures": diagnostics.get("failures"),
            "wall_seconds_computation": runtime.get("total_seconds"),
            "wall_seconds_by_chunk": runtime.get("seconds_by_chunk"),
            "wall_seconds_figures": manifest.get("figures_seconds"),
            "feasibility": diagnostics.get("feasibility"),
            "worst_checks": {k: diagnostics.get(k) for k in ("matrix_exponential_vs_ode", "timing_distinction", "accounting_identities", "first_order_budget_identity", "discounted_cumulative_resolvent", "discounted_cumulative_expm_integral", "fixed_share_reproduction", "row_builder_cross_check", "neutral_zero")},
            "reliable_region": "Only the reported local paths with recorded positive specialisation and transfer-floor slack; numerical-scaffolding slack is not economic feasibility.",
            "portfolio_classification": "unconstrained local desired portfolio; genuine fiscal-capacity feasibility unverified",
        },
        "artifacts": artifacts,
        "interpretation": {
            "supports_result_ids": ["R13", "R14", "R15", "R20", "R23", "R24"],
            "hypothesis_ids": ["HYP001", "HYP002", "HYP003", "HYP004", "HYP005"],
            "question_ids": ["Q04"],
            "conclusion": "See MORNING_REPORT.md and ECONOMIC_REVIEW_PACKET.md in the output directory; NUMERICAL_REVIEW_PACKET.md carries the worst residuals with locators.",
            "limitations": [
                "Local first-order LQ evidence on the illustrative Farhi-based vector; a directional mechanism map, not a probability statement about shocks, a global constrained solution, or an exact welfare calculation.",
                "J is planner-resource wealth (includes future worker wages), X = N + J is worker comprehensive resources; the leading portfolio is unconstrained and genuine fiscal-capacity feasibility is unverified; wide portfolio/debt caps are numerical scaffolding only.",
                "No shock-specific welfare ranking or exact precautionary policy is computed; the leading access coefficient is an ex-ante local object.",
                "CS001 remains draft and unfingerprinted in the codex registry; this is an exploratory tranche under Nathan's direct commission.",
            ],
            "next_decision": "Route the fiscal-capacity object C_G(S) (attainable primary surpluses with T=0, wages excluded) to an economic specification before any borrowing-capacity claim; carry the impact-versus-path neutrality distinction into CS002/CS004 comparisons.",
        },
    }
    runs_dir.mkdir(parents=True, exist_ok=True)
    record_path = runs_dir / f"{run_id}.yaml"
    if record_path.exists():
        raise FileExistsError(f"Run record {record_path} already exists; run records are immutable.")
    record_path.write_text(yaml.safe_dump(serial(record), sort_keys=False, allow_unicode=False), encoding="utf-8")
    atomic_write_json(output_dir / "artifact_hashes.json", {"run_id": run_id, "artifacts": artifacts})
    return record_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--resume", action="store_true", help="Continue unfinished chunks under the same commit and configuration fingerprint.")
    parser.add_argument("--angular-step-degrees", type=float, default=float(DEFAULT_SETTINGS["angular_step_degrees"]))
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--finalize", action="store_true", help="Hash every artifact (including hand-written packets) and write the immutable run record.")
    parser.add_argument("--run-id")
    parser.add_argument("--runs-dir")
    parser.add_argument("--preflight-tests-status", default="not_recorded")
    args = parser.parse_args(argv)

    repository = Path(__file__).resolve().parents[3]
    git = git_metadata(repository)
    cache = repository / ".cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir).resolve()
    manifest_path = output_dir / "manifest.json"
    command = "uv run python -m tai_public_finance.cs001_lq_anchor.shock_atlas_cli " + " ".join(sys.argv[1:] if argv is None else argv)

    if args.finalize:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        runs_dir = Path(args.runs_dir).resolve() if args.runs_dir else repository / "runs"
        record = write_run_record(output_dir, repository, runs_dir, args.run_id, manifest, args.preflight_tests_status)
        manifest["run_record"] = str(record)
        manifest["finalized_utc"] = _now()
        atomic_write_json(manifest_path, manifest)
        print(json.dumps({"run_record": str(record), "artifacts": len(json.loads((output_dir / 'artifact_hashes.json').read_text())['artifacts'])}, indent=2))
        return 0

    config = load_cs001_configuration(args.config)
    settings = {**DEFAULT_SETTINGS, "angular_step_degrees": float(args.angular_step_degrees)}
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if not manifest:
        manifest = {
            "started_utc": _now(),
            "command": command,
            "mode": args.mode,
            "git": git,
            "parameter_set_id": config.parameter_set_id,
            "config_paths": {"experiment": str(config.experiment_path), "primitives": str(config.primitive_path)},
            "fingerprints": config.fingerprints,
            "settings": settings,
            "maintained_setup": {
                "installed_capital_price_q_D": config.parameters.installed_capital_price,
                "public_net_worth_anchor_N_bar": config.parameters.public_net_worth_anchor,
                "welfare": config.experiment["maintained_economic_closure"]["welfare_population"],
                "government_risky_asset": config.experiment["maintained_economic_closure"]["government_risky_asset"],
                "international_pricing_closure": config.experiment["maintained_economic_closure"]["international_pricing_closure"],
                "firm_branch": config.experiment["maintained_economic_closure"]["firm_branch_assumption"],
                "tax_adjustment_scale": config.parameters.tax_adjustment_scale,
                "productivity_persistence_annual": config.parameters.productivity_persistence_annual,
                "automation_persistence_annual_baseline": config.parameters.automation_persistence_annual,
                "future_innovations": config.experiment["counterfactual_metadata"]["future_shock_coupling"],
                "short_brownian_window_years": config.experiment["reporting"]["short_brownian_window_years"],
                "finite_ou_window_years": config.experiment["reporting"]["finite_ou_window_years"],
                "shock_covariance_law": "unchanged: two orthogonal Brownian innovations; a joint direction is a joint realization, not a correlated law",
            },
            "resume_events": [],
            "naming": {
                "J": "planner-resource wealth (worker fiscal-endowment wealth; includes future worker wages)",
                "X": "worker comprehensive resources N + J",
                "s_star": "unconstrained leading small-risk portfolio",
                "portfolio_classification": "unconstrained local desired portfolio; genuine fiscal-capacity feasibility unverified",
            },
        }
        atomic_write_json(manifest_path, manifest)
    elif args.resume:
        manifest.setdefault("resume_events", []).append({"utc": _now(), "command": command, "git": git})
        atomic_write_json(manifest_path, manifest)
    atomic_write_text(output_dir / ".gitignore", "# large or regenerable per-run artifacts: hash-referenced from the run record\natlas_raw_quarterly.csv.gz\nparts/\nfigures/\n*.tmp\n")

    outcome = "fail"
    failure = None
    try:
        runner = AtlasRunner(config, output_dir, settings, git["commit"], resume=args.resume, mode=args.mode)
        manifest["atlas_fingerprint"] = runner.fingerprint
        atomic_write_json(manifest_path, manifest)
        result = runner.run()
        outcome = result["checks"].get("outcome", "fail")
    except AtlasCheckFailure as error:
        failure = str(error)
        print(f"ATLAS CHECK FAILURE: {failure}", file=sys.stderr)

    figures_seconds = None
    if outcome == "pass" and args.mode == "full" and not args.skip_figures:
        started = time.perf_counter()
        write_figures(output_dir)
        figures_seconds = time.perf_counter() - started
    manifest.update({"finished_utc": _now(), "outcome": outcome, "failure": failure, "figures_seconds": figures_seconds})
    atomic_write_json(manifest_path, manifest)
    write_numerical_review_packet(output_dir, manifest)
    print(json.dumps({"outcome": outcome, "failure": failure, "output_dir": str(output_dir)}, indent=2))
    return 0 if outcome == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
