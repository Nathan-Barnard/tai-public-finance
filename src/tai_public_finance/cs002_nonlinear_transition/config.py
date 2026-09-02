"""CS002 D0-D1 experiment configuration.

References (never duplicates) the CS001 primitive/experiment configuration
it reuses -- `cs001_experiment_file` points at the same
lq_farhi_illustrative_smoke_v1 experiment CS001 itself loads, so both specs
are provably running against identical primitives with a single fingerprint
chain, and adds only the D0-D1-specific displacement, continuation, horizon,
mesh, and tolerance fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..cs001_lq_anchor.config import Cs001Configuration, load_cs001_configuration
from ..primitives.parameters import sha256_of

_REQUIRED_FIELDS = {
    "cs002_config_id",
    "cs001_experiment_file",
    "initial_displacement",
    "diagnostic_basis_displacements",
    "continuation_amplitudes",
    "horizons_years",
    "mesh",
    "solver_tolerance",
    "max_nodes",
    "numerical_scaffolding",
    "public_net_worth_to_fiscal_wealth_grid",
    "convergence_amplitude_sequence",
    "acceptance_tolerances",
}


@dataclass(frozen=True)
class Cs002Configuration:
    cs001: Cs001Configuration
    raw: dict[str, Any]
    config_path: Path

    @property
    def config_id(self) -> str:
        return str(self.raw["cs002_config_id"])

    @property
    def fingerprint(self) -> str:
        return sha256_of(self.raw)

    @property
    def delta_k(self) -> float:
        return float(self.raw["initial_displacement"]["delta_k"])

    @property
    def delta_tau(self) -> float:
        return float(self.raw["initial_displacement"]["delta_tau"])

    @property
    def diagnostic_basis_displacements(self) -> list[tuple[float, float]]:
        return [(float(d["delta_k"]), float(d["delta_tau"])) for d in self.raw["diagnostic_basis_displacements"]]

    @property
    def continuation_amplitudes(self) -> list[float]:
        return [float(a) for a in self.raw["continuation_amplitudes"]]

    @property
    def baseline_horizon(self) -> float:
        return float(self.raw["horizons_years"]["baseline"])

    @property
    def comparison_horizons(self) -> list[float]:
        return [float(h) for h in self.raw["horizons_years"]["comparisons"]]

    @property
    def baseline_mesh_points(self) -> int:
        return int(self.raw["mesh"]["baseline_points"])

    @property
    def refined_mesh_points(self) -> int:
        return int(self.raw["mesh"]["refined_points"])

    @property
    def solver_tolerance(self) -> float:
        return float(self.raw["solver_tolerance"])

    @property
    def max_nodes(self) -> int:
        return int(self.raw["max_nodes"])

    @property
    def numerical_scaffolding(self) -> dict[str, float]:
        return self.raw["numerical_scaffolding"]

    @property
    def net_worth_grid_ratios(self) -> list[float]:
        return [float(v) for v in self.raw["public_net_worth_to_fiscal_wealth_grid"]]

    @property
    def convergence_amplitude_sequence(self) -> list[float]:
        return [float(a) for a in self.raw["convergence_amplitude_sequence"]]

    @property
    def acceptance_tolerances(self) -> dict[str, float]:
        return self.raw["acceptance_tolerances"]


def load_cs002_configuration(path: str | Path) -> Cs002Configuration:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    missing = _REQUIRED_FIELDS - raw.keys()
    if missing:
        raise ValueError(f"CS002 configuration is missing required fields: {sorted(missing)}")

    cs001_experiment_path = (config_path.parent / raw["cs001_experiment_file"]).resolve()
    cs001_config = load_cs001_configuration(cs001_experiment_path)
    return Cs002Configuration(cs001=cs001_config, raw=raw, config_path=config_path)
