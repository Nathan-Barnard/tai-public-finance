from __future__ import annotations

import json
import sys
from pathlib import Path

from tai_public_finance.cs002_nonlinear_transition.cli_d2 import main

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "cs002" / "lq_farhi_d2_mean_reversion_v1.json"


def test_cli_end_to_end_run_completes_and_writes_a_complete_bundle(tmp_path, monkeypatch, capsys):
    """CS002 D2 review repair, finding 2: on THIS config, the now-complete
    horizon/mesh check correctly fails `horizon_mesh_stability` for both
    directions -- the 20-year comparison horizon is genuinely too short for
    `varpi`'s ~1/rho ~= 49.5-year relaxation timescale, previously invisible
    because the old check neither compared varpi nor scored each path
    against its own tolerance (see the findings note for the full
    explanation). cli_d2.py's own documented contract is exit code 2 for a
    completed run whose outcome isn't computational_pass (0 is reserved for
    an actual pass); this test asserts that CORRECT, EXPECTED-given-this-
    finding behavior, not a crash -- the bundle is still written in full."""

    output_dir = tmp_path / "cs002-d2-cli-test"
    runs_dir = tmp_path / "runs"
    argv = [
        "cs002-d2-cli",
        "--config",
        str(CONFIG_PATH),
        "--output-dir",
        str(output_dir),
        "--run-id",
        "RUN-TEST-CS002-D2-CLI",
        "--runs-dir",
        str(runs_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    exit_code = main()
    assert exit_code == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["outcome"] == "numerical_failure"
    assert payload["failed_checks"] == ["productivity_horizon_mesh_stability", "automation_horizon_mesh_stability"]

    assert (output_dir / "report.json").exists()
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "complete_input.json").exists()
    for direction in ("productivity", "automation"):
        assert (output_dir / f"{direction}_path.csv").exists()
        assert (output_dir / f"{direction}_margins.csv").exists()
    assert (output_dir / "continuation_checkpoints.csv").exists()
    assert (output_dir / "horizon_mesh_comparisons.csv").exists()
    assert (output_dir / "convergence.csv").exists()
    assert (output_dir / "varpi_horizon_sensitivity.csv").exists()
    assert (runs_dir / "RUN-TEST-CS002-D2-CLI.yaml").exists()
    figures = sorted((output_dir / "figures").glob("*.png"))
    assert len(figures) == 3  # productivity_paths, automation_paths, nonlinear_vs_lq_convergence

    report = json.loads((output_dir / "report.json").read_text())
    assert report["outcome"]["outcome"] == "numerical_failure"
    for direction in ("productivity", "automation"):
        checks = report[direction]["outcome"]["checks"]
        assert checks["horizon_mesh_stability"] is False
        assert all(v for k, v in checks.items() if k != "horizon_mesh_stability")
