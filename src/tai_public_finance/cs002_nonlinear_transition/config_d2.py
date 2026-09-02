"""CS002 D2 (deterministic mean reversion) experiment configuration.

References (never duplicates) the same CS001 primitive/experiment
configuration config.py's D0-D1 Cs002Configuration reuses, so D1 and D2 both
provably run against identical primitives with a single fingerprint chain.
This is a SEPARATE dataclass from Cs002Configuration (not a subclass or an
extension of it): D2's displacement is in the exogenous state (a shock
direction plus its target size), not D1's (delta_k, delta_tau) fiscal-state
displacement, so the two configurations' shapes genuinely differ.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..cs001_lq_anchor.config import Cs001Configuration, load_cs001_configuration
from ..primitives.parameters import sha256_of

_REQUIRED_FIELDS = {
    "cs002_d2_config_id",
    "cs001_experiment_file",
    "shock_directions",
    "continuation_amplitudes",
    "horizons_years",
    "mesh",
    "solver_tolerance",
    "max_nodes",
    "numerical_scaffolding",
    "initial_public_net_worth",
    "convergence_amplitude_sequence",
    "varpi_tail_horizon_sequence",
    "acceptance_tolerances",
}
_REQUIRED_SHOCK_FIELDS = {"productivity", "automation"}


@dataclass(frozen=True)
class Cs002D2Configuration:
    cs001: Cs001Configuration
    raw: dict[str, Any]
    config_path: Path

    @property
    def config_id(self) -> str:
        return str(self.raw["cs002_d2_config_id"])

    @property
    def fingerprint(self) -> str:
        return sha256_of(self.raw)

    @property
    def delta_z_productivity(self) -> float:
        return float(self.raw["shock_directions"]["productivity"]["delta_z"])

    @property
    def delta_alpha_automation(self) -> float:
        return float(self.raw["shock_directions"]["automation"]["delta_alpha"])

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
    def initial_public_net_worth(self) -> float:
        return float(self.raw["initial_public_net_worth"])

    @property
    def convergence_amplitude_sequence(self) -> list[float]:
        return [float(a) for a in self.raw["convergence_amplitude_sequence"]]

    @property
    def varpi_tail_horizon_sequence(self) -> list[float]:
        return [float(h) for h in self.raw["varpi_tail_horizon_sequence"]]

    @property
    def acceptance_tolerances(self) -> dict[str, float]:
        return self.raw["acceptance_tolerances"]


def load_cs002_d2_configuration(path: str | Path) -> Cs002D2Configuration:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    missing = _REQUIRED_FIELDS - raw.keys()
    if missing:
        raise ValueError(f"CS002 D2 configuration is missing required fields: {sorted(missing)}")
    missing_shocks = _REQUIRED_SHOCK_FIELDS - raw["shock_directions"].keys()
    if missing_shocks:
        raise ValueError(f"CS002 D2 configuration's shock_directions is missing: {sorted(missing_shocks)}")

    cs001_experiment_path = (config_path.parent / raw["cs001_experiment_file"]).resolve()
    cs001_config = load_cs001_configuration(cs001_experiment_path)
    return Cs002D2Configuration(cs001=cs001_config, raw=raw, config_path=config_path)
