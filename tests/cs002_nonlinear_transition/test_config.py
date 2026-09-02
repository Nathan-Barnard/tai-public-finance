from __future__ import annotations

from pathlib import Path

from tai_public_finance.cs002_nonlinear_transition.config import load_cs002_configuration

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "cs002" / "lq_farhi_d0_d1_frozen_v1.json"


def test_config_loads_and_references_the_cs001_experiment():
    config = load_cs002_configuration(CONFIG_PATH)
    assert config.config_id == "lq_farhi_d0_d1_frozen_v1"
    assert config.cs001.parameter_set_id == "lq_farhi_illustrative_smoke_v1"
    assert config.delta_k == 0.01
    assert config.delta_tau == 0.01
    assert config.continuation_amplitudes == [0.0, 0.25, 0.5, 1.0]
    assert config.baseline_horizon == 40.0
    assert config.comparison_horizons == [20.0, 80.0]
    assert len(config.diagnostic_basis_displacements) == 2
    assert config.net_worth_grid_ratios == [-0.5, -0.25, 0.0, 0.25, 0.5]


def test_config_fingerprint_is_stable_and_deterministic():
    config_a = load_cs002_configuration(CONFIG_PATH)
    config_b = load_cs002_configuration(CONFIG_PATH)
    assert config_a.fingerprint == config_b.fingerprint
    assert len(config_a.fingerprint) == 64  # sha256 hex digest
