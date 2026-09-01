"""Canonical primitive parameter vector and technology primitives.

Shared across every computational specification for the Version 5.1,
q_D=1 Brownian Ramsey model. A specification package (e.g. cs001_lq_anchor)
must build its equations from these primitives rather than re-deriving or
hand-entering reduced-form parameters.
"""

from .parameters import PrimitiveParameters, load_primitive_parameters
from .production import (
    SmoothBranchState,
    automation_share,
    automation_share_first_derivative,
    automation_share_second_derivative,
    evaluate_smooth_branch,
    logistic,
    logit,
    omega,
)
from .international_pricing import (
    LeadingPriceOfRisk,
    international_consumption_loading,
    international_consumption_loading_derivative,
    leading_price_of_risk,
    safe_rate,
)

__all__ = [
    "PrimitiveParameters",
    "load_primitive_parameters",
    "SmoothBranchState",
    "automation_share",
    "automation_share_first_derivative",
    "automation_share_second_derivative",
    "evaluate_smooth_branch",
    "logistic",
    "logit",
    "omega",
    "LeadingPriceOfRisk",
    "international_consumption_loading",
    "international_consumption_loading_derivative",
    "leading_price_of_risk",
    "safe_rate",
]
