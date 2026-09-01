from __future__ import annotations

from pathlib import Path

import pytest

from tai_public_finance.primitives import load_primitive_parameters

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIMITIVE_PATH = REPO_ROOT / "configs" / "primitives" / "lq_farhi_annual_v1.json"


def test_loads_the_farhi_annual_primitive_table():
    p = load_primitive_parameters(PRIMITIVE_PATH)
    assert p.primitive_set_id == "lq_farhi_annual_primitives_v1"
    assert p.installed_capital_price == 1.0


def test_continuous_time_translation_matches_the_documented_farhi_bridge():
    import math

    p = load_primitive_parameters(PRIMITIVE_PATH)
    assert p.rho == pytest.approx(-math.log(0.98), abs=1e-15)
    assert p.kappa_z == pytest.approx(-math.log(0.81), abs=1e-15)
    assert p.kappa_x == pytest.approx(-math.log(0.81), abs=1e-15)
    assert p.sigma_z_hat == pytest.approx(0.04 * math.sqrt(2.0 * p.kappa_z), abs=1e-15)
    assert p.sigma_x_hat == pytest.approx(0.25 * math.sqrt(2.0 * p.kappa_x), abs=1e-15)


def test_fingerprint_changes_when_a_parameter_changes():
    p = load_primitive_parameters(PRIMITIVE_PATH)
    mutated_raw = {**p.raw, "parameters": {**p.raw["parameters"], "depreciation_rate": 0.09}}
    from tai_public_finance.primitives.parameters import PrimitiveParameters

    mutated = PrimitiveParameters.from_dict(mutated_raw)
    assert mutated.fingerprint != p.fingerprint


def test_rejects_q_d_not_equal_to_one():
    p = load_primitive_parameters(PRIMITIVE_PATH)
    from tai_public_finance.primitives.parameters import PrimitiveParameters

    bad_raw = {**p.raw, "parameters": {**p.raw["parameters"], "installed_capital_price": 0.9}}
    with pytest.raises(ValueError, match="q_D=1"):
        PrimitiveParameters.from_dict(bad_raw)


def test_rejects_alpha_bounds_out_of_order():
    p = load_primitive_parameters(PRIMITIVE_PATH)
    from tai_public_finance.primitives.parameters import PrimitiveParameters

    bad_raw = {**p.raw, "parameters": {**p.raw["parameters"], "alpha_lower": 0.5, "alpha_upper": 0.4}}
    with pytest.raises(ValueError):
        PrimitiveParameters.from_dict(bad_raw)
