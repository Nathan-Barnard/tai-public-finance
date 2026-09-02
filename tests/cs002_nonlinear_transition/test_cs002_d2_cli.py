from __future__ import annotations

import json
import sys
from pathlib import Path

from tai_public_finance.cs002_nonlinear_transition.cli_d2 import main

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "cs002" / "lq_farhi_d2_mean_reversion_v1.json"


def test_cli_end_to_end_run_passes_and_writes_a_complete_bundle(tmp_path, monkeypatch, capsys):
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
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["outcome"] == "computational_pass"
    assert payload["failed_checks"] == []

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
    assert report["outcome"]["outcome"] == "computational_pass"
    assert set(report["productivity"]["outcome"]["checks"].values()) == {True}
    assert set(report["automation"]["outcome"]["checks"].values()) == {True}
