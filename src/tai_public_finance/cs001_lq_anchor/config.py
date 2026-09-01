"""CS001 experiment configuration: an experiment file plus its one primitive table."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..primitives import PrimitiveParameters, load_primitive_parameters
from ..primitives.parameters import sha256_of

_REQUIRED_EXPERIMENT_FIELDS = {
    "parameter_set_id",
    "calibration_role",
    "primitive_file",
    "risk_scale_epsilon",
    "maintained_economic_closure",
    "counterfactual_metadata",
    "reporting",
    "numerical_scaffolding",
    "acceptance_tolerances",
}
_REQUIRED_CLOSURE_FIELDS = {
    "welfare_population",
    "domestic_capital_ownership",
    "government_risky_asset",
    "international_pricing_closure",
    "government_spending",
    "adoption_assumption",
    "firm_branch_assumption",
}


@dataclass(frozen=True)
class Cs001Configuration:
    parameters: PrimitiveParameters
    experiment: dict[str, Any]
    experiment_path: Path
    primitive_path: Path

    @property
    def parameter_set_id(self) -> str:
        return str(self.experiment["parameter_set_id"])

    @property
    def calibration_role(self) -> str:
        return str(self.experiment["calibration_role"])

    @property
    def fingerprints(self) -> dict[str, str]:
        return {
            "primitive_sha256": self.parameters.fingerprint,
            "experiment_sha256": sha256_of(self.experiment),
            "complete_input_sha256": sha256_of({"experiment": self.experiment, "primitives": self.parameters.raw}),
        }


def load_cs001_configuration(path: str | Path) -> Cs001Configuration:
    experiment_path = Path(path).resolve()
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    missing = _REQUIRED_EXPERIMENT_FIELDS - experiment.keys()
    if missing:
        raise ValueError(f"CS001 experiment configuration is missing required fields: {sorted(missing)}")
    closure = experiment["maintained_economic_closure"]
    missing_closure = _REQUIRED_CLOSURE_FIELDS - closure.keys()
    if missing_closure:
        raise ValueError(f"maintained_economic_closure is missing required fields: {sorted(missing_closure)}")
    if closure["government_spending"] != 0.0:
        raise ValueError("CS001 is restricted to G=0.")

    primitive_path = (experiment_path.parent / experiment["primitive_file"]).resolve()
    parameters = load_primitive_parameters(primitive_path)
    return Cs001Configuration(
        parameters=parameters,
        experiment=experiment,
        experiment_path=experiment_path,
        primitive_path=primitive_path,
    )
