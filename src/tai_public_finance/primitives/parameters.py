"""The primitive parameter vector: loading, validation, and fingerprinting.

A primitive table fixes the Version 5.1, q_D=1 economic environment in its
as-given annual/reduced-form units. Continuous-time objects (rho, kappa_z,
kappa_x, sigma_z_hat, sigma_x_hat) are derived properties, not stored
fields, so there is exactly one place the annual-to-continuous-time
translation happens.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REQUIRED_PARAMETER_FIELDS = {
    "annual_discount_factor",
    "depreciation_rate",
    "tax_adjustment_scale",
    "alpha_lower",
    "alpha_upper",
    "x_mean",
    "capital_advantage",
    "new_task_labour_advantage",
    "international_log_capital_labour_ratio",
    "productivity_persistence_annual",
    "productivity_stationary_sd",
    "automation_persistence_annual",
    "automation_stationary_sd",
    "installed_capital_price",
}
_REQUIRED_NORMALIZATION_FIELDS = {"labour", "capital_anchor", "public_net_worth_anchor"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_of(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PrimitiveParameters:
    """One versioned, fingerprintable primitive vector."""

    primitive_set_id: str
    model_branch: str

    annual_discount_factor: float
    depreciation_rate: float
    tax_adjustment_scale: float
    alpha_lower: float
    alpha_upper: float
    x_mean: float
    capital_advantage: float
    new_task_labour_advantage: float
    international_log_capital_labour_ratio: float
    productivity_persistence_annual: float
    productivity_stationary_sd: float
    automation_persistence_annual: float
    automation_stationary_sd: float
    installed_capital_price: float

    labour: float
    capital_anchor: float
    public_net_worth_anchor: float

    provenance: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.annual_discount_factor < 1.0:
            raise ValueError("annual_discount_factor must lie in (0, 1).")
        if not 0.0 < self.productivity_persistence_annual < 1.0:
            raise ValueError("productivity_persistence_annual must lie in (0, 1).")
        if not 0.0 < self.automation_persistence_annual < 1.0:
            raise ValueError("automation_persistence_annual must lie in (0, 1).")
        if not 0.0 < self.alpha_lower < self.alpha_upper < 1.0:
            raise ValueError("Require 0 < alpha_lower < alpha_upper < 1.")
        for name in (
            "depreciation_rate",
            "tax_adjustment_scale",
            "productivity_stationary_sd",
            "automation_stationary_sd",
            "labour",
            "capital_anchor",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.installed_capital_price != 1.0:
            raise ValueError("This model branch is restricted to q_D=1 (frictionless installation).")

    @property
    def rho(self) -> float:
        return -math.log(self.annual_discount_factor)

    @property
    def kappa_z(self) -> float:
        return -math.log(self.productivity_persistence_annual)

    @property
    def kappa_x(self) -> float:
        return -math.log(self.automation_persistence_annual)

    @property
    def sigma_z_hat(self) -> float:
        return self.productivity_stationary_sd * math.sqrt(2.0 * self.kappa_z)

    @property
    def sigma_x_hat(self) -> float:
        return self.automation_stationary_sd * math.sqrt(2.0 * self.kappa_x)

    @property
    def fingerprint(self) -> str:
        return sha256_of(self.raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PrimitiveParameters:
        for key in ("primitive_set_id", "model_branch", "parameters", "normalizations", "provenance"):
            if key not in raw:
                raise ValueError(f"Primitive table is missing required field: {key}")
        parameters = raw["parameters"]
        normalizations = raw["normalizations"]
        missing_params = _REQUIRED_PARAMETER_FIELDS - parameters.keys()
        if missing_params:
            raise ValueError(f"Primitive parameters missing: {sorted(missing_params)}")
        missing_norms = _REQUIRED_NORMALIZATION_FIELDS - normalizations.keys()
        if missing_norms:
            raise ValueError(f"Normalizations missing: {sorted(missing_norms)}")
        return cls(
            primitive_set_id=raw["primitive_set_id"],
            model_branch=raw["model_branch"],
            annual_discount_factor=float(parameters["annual_discount_factor"]),
            depreciation_rate=float(parameters["depreciation_rate"]),
            tax_adjustment_scale=float(parameters["tax_adjustment_scale"]),
            alpha_lower=float(parameters["alpha_lower"]),
            alpha_upper=float(parameters["alpha_upper"]),
            x_mean=float(parameters["x_mean"]),
            capital_advantage=float(parameters["capital_advantage"]),
            new_task_labour_advantage=float(parameters["new_task_labour_advantage"]),
            international_log_capital_labour_ratio=float(parameters["international_log_capital_labour_ratio"]),
            productivity_persistence_annual=float(parameters["productivity_persistence_annual"]),
            productivity_stationary_sd=float(parameters["productivity_stationary_sd"]),
            automation_persistence_annual=float(parameters["automation_persistence_annual"]),
            automation_stationary_sd=float(parameters["automation_stationary_sd"]),
            installed_capital_price=float(parameters["installed_capital_price"]),
            labour=float(normalizations["labour"]),
            capital_anchor=float(normalizations["capital_anchor"]),
            public_net_worth_anchor=float(normalizations["public_net_worth_anchor"]),
            provenance={str(k): str(v) for k, v in raw["provenance"].items()},
            raw=raw,
        )


def load_primitive_parameters(path: str | Path) -> PrimitiveParameters:
    text = Path(path).read_text(encoding="utf-8")
    return PrimitiveParameters.from_dict(json.loads(text))
