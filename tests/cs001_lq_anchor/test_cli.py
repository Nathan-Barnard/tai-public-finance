from __future__ import annotations

import json
import sys
from pathlib import Path

from tai_public_finance.cs001_lq_anchor.cli import main

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "cs001" / "lq_farhi_smoke.json"


def test_cli_end_to_end_run_passes_and_writes_a_complete_bundle(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "cs001-cli-test"
    runs_dir = tmp_path / "runs"
    argv = [
        "cs001-cli",
        "--config",
        str(CONFIG_PATH),
        "--output-dir",
        str(output_dir),
        "--run-id",
        "RUN-TEST-CLI",
        "--runs-dir",
        str(runs_dir),
        "--preflight-tests-status",
        "passed",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    exit_code = main()
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["outcome"] == "pass"
    assert payload["failed_checks"] == []

    assert (output_dir / "report.json").exists()
    assert (output_dir / "irfs.csv").exists()
    assert (output_dir / "portfolio_net_worth_grid.csv").exists()
    assert (runs_dir / "RUN-TEST-CLI.yaml").exists()
    assert (output_dir / "matrices" / "H.csv").exists()
    figures = sorted((output_dir / "figures").glob("*.png"))
    assert len(figures) == 4
