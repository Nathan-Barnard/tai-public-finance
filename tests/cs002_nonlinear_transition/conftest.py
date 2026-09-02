from __future__ import annotations

from pathlib import Path

import pytest

from tai_public_finance.cs001_lq_anchor import build_local_system, compute_steady_state, load_cs001_configuration, solve_lq_system
from tai_public_finance.primitives import load_primitive_parameters

REPO_ROOT = Path(__file__).resolve().parents[2]
CS001_CONFIG_PATH = REPO_ROOT / "configs" / "cs001" / "lq_farhi_smoke.json"
PRIMITIVE_PATH = REPO_ROOT / "configs" / "primitives" / "lq_farhi_annual_v1.json"


@pytest.fixture(scope="session")
def primitives():
    return load_primitive_parameters(PRIMITIVE_PATH)


@pytest.fixture(scope="session")
def anchor(primitives):
    return compute_steady_state(primitives)


@pytest.fixture(scope="session")
def cs001_local_system(primitives, anchor):
    return build_local_system(primitives, anchor)


@pytest.fixture(scope="session")
def cs001_solution(cs001_local_system):
    return solve_lq_system(cs001_local_system)


@pytest.fixture(scope="session")
def cs001_config():
    return load_cs001_configuration(CS001_CONFIG_PATH)
