"""Joint productivity-automation shock atlas for the CS001 local LQ system.

Exploratory local-LQ evidence on the illustrative Farhi-based vector: the
maintained model, policy calibration, and shock covariance law are held
fixed; only the *direction*, *normalization*, and *persistence composition*
of one small disturbance vary. Nothing here is an empirical statement about
the probability of particular shocks, a global constrained solution, or an
exact welfare calculation.

Directions
----------
A standardized joint Brownian innovation is ``u(theta) = (cos theta, sin theta)``
in (productivity, automation) innovation space. Over a reporting window
``Delta`` the inherited-state displacement is

    dz = sigma_z_hat * sqrt(Delta) * cos(theta)
    dx = sigma_x_hat * sqrt(Delta) * sin(theta)
    dalpha = A'(x_bar) * dx

so every angle carries the same standardized innovation magnitude. A
direction written in (dz, dalpha) space with slope ``m = dz/dalpha`` maps to
``theta = atan2(sigma_z_hat, sigma_x_hat * m * A'(x_bar))``. Both theta and
theta + pi are always present (opposite realizations).

Experiment families (kept distinct, never mixed)
-------------------------------------------------
1. ``brownian_innovation_short_window`` (tracker experiment E07): a realized
   Brownian innovation over the short reporting interval. The position held
   *before* the innovation pays: ``s_- = s*`` (optimal inherited position) or
   ``s_- = 0`` (zero inherited position). Both are followed by the same
   full-access Ramsey continuation (one closed-loop F). The second is NOT a
   separately solved no-access economy. Capital and the inherited tax rate
   cannot jump.
2. ``matched_deterministic_state_displacement`` (E08): exactly the same
   (dz, dx) treated as an abstract inherited state; public net worth held
   fixed, no inherited claim payoff, future innovations off. Not stochastic
   news.
3. ``finite_window_ou_state_displacement``: a one-standard-deviation joint
   state displacement using each state's one-year conditional OU standard
   deviation; public net worth fixed, no payoff.
4. ``fixed_share_displacement_check``: the project's existing
   ``dalpha = +0.01`` normalization for automation-containing constructed
   directions, kept separate from the equal-standardized-innovation atlas and
   used only as a reproduction check against the baseline pipeline.

Every reported path carries its own feasibility/specialisation slacks; rows
that fail are retained and flagged, never dropped.

Names and claims (economic correction of 2026-09-01)
-----------------------------------------------------
The government cannot tax worker wages. Worker consumption is ``c = W + T``
with transfers ``T >= 0``; the public budget drift is
``r_I N + s beta_I + tau B - T - Psi``. The planner object
``F = tau B + W`` therefore contains worker wage income, and ``F - c = tau B - T``
is the government's primary cash flow, not a statement that wages are revenue.

* ``J`` (``planner_resource_wealth``) is *planner-resource wealth*, also called
  worker fiscal-endowment wealth: the discounted value of future worker wages
  plus capital-tax resources net of adjustment costs. It includes future
  worker wages and is NOT government borrowing capacity or collateral.
* ``X = N + J`` (``worker_comprehensive_resources``) is *worker comprehensive
  resources*.
* ``s*`` is the *unconstrained leading small-risk portfolio*; a negative safe
  position ``N - s* < 0`` is an unconstrained desired debt-financed risky
  holding, conditional on the genuine transfer and continuation-solvency
  boundaries being slack. Genuine fiscal-capacity feasibility is unverified
  here; the wide portfolio/debt limits in ``numerical_scaffolding`` are
  numerical scaffolding and establish nothing about borrowing capacity.

Every row therefore reports the separate accounts (W, tau B, T, Psi,
tau B - T - Psi, N, s, N - s, transfer-floor slack) and the runner verifies
``F - c = tau B - T`` on every row. Genuine economic conditions
(specialisation branch, transfer floor, positive consumption/resources) are
flagged separately from numerical-scaffolding slack.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from ..primitives import PrimitiveParameters
from ..primitives.parameters import sha256_of
from ..primitives.production import evaluate_smooth_branch
from .anchor import SteadyState, compute_steady_state
from .config import Cs001Configuration
from .diagnostics import AcceptanceReport, DiagnosticsReport, resolvent_probe_residual
from .diagnostics import acceptance as compute_acceptance
from .diagnostics import run_diagnostics
from .equations import LocalSystem, build_local_system
from .irfs import run_irfs
from .portfolio import LeadingPortfolio, leading_portfolio_and_welfare
from .solver import LqSolution, solve_lq_system
from .sweep import _parameters_with_override

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

FAMILY_BROWNIAN = "brownian_innovation_short_window"
FAMILY_MATCHED_STATE = "matched_deterministic_state_displacement"
FAMILY_OU_WINDOW = "finite_window_ou_state_displacement"
FAMILY_FIXED_SHARE = "fixed_share_displacement_check"
FAMILY_FIXED_ACROSS_PERSISTENCE = "fixed_initial_displacement_across_persistence"

EXPERIMENT_IDS = {
    FAMILY_BROWNIAN: "E07",
    FAMILY_MATCHED_STATE: "E08",
    FAMILY_OU_WINDOW: "E08_finite_window_ou_normalization",
    FAMILY_FIXED_SHARE: "E03_E04_fixed_share_check",
    FAMILY_FIXED_ACROSS_PERSISTENCE: "E08_fixed_displacement_persistence_variant",
}

REGIME_OPTIMAL = "optimal_inherited_position"  # s_- = s* (the baseline pipeline's "full_access")
REGIME_ZERO = "zero_inherited_position"  # s_- = 0, same full-access continuation
REGIME_NONE = "no_inherited_payoff"  # deterministic state displacement; N held fixed

NAMED_PURE_PRODUCTIVITY = "pure_productivity"
NAMED_PURE_AUTOMATION = "pure_automation"
NAMED_OUTPUT_NEUTRAL = "output_neutral_automation"
NAMED_WORKER_INCOME_NEUTRAL = "worker_income_neutral_automation"
NAMED_PRIMARY_RESOURCE_NEUTRAL = "primary_resource_neutral_automation"
NAMED_CLAIM_NEUTRAL = "claim_payoff_neutral_automation"
NAMED_RENTAL_BASE_NEUTRAL = "rental_tax_base_neutral_automation"
NAMED_ORDER = (
    NAMED_PURE_PRODUCTIVITY,
    NAMED_PURE_AUTOMATION,
    NAMED_OUTPUT_NEUTRAL,
    NAMED_WORKER_INCOME_NEUTRAL,
    NAMED_PRIMARY_RESOURCE_NEUTRAL,
    NAMED_CLAIM_NEUTRAL,
    NAMED_RENTAL_BASE_NEUTRAL,
)
# What each named direction holds fixed ON IMPACT (at inherited capital and tax); nothing
# about later horizons is implied by the name.
NAMED_IMPACT_NEUTRAL_OBJECT = {
    NAMED_OUTPUT_NEUTRAL: "output_deviation_linear",
    NAMED_WORKER_INCOME_NEUTRAL: "wage_income_deviation_linear",
    NAMED_PRIMARY_RESOURCE_NEUTRAL: "fiscal_resources_deviation_linear",
    NAMED_CLAIM_NEUTRAL: "claim_loading_state_functional",
    NAMED_RENTAL_BASE_NEUTRAL: "rental_rate_deviation_linear",
}

KEY_HORIZONS = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0)
REQUIRED_HORIZONS = (0.0, 0.25, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0)

DEFAULT_SETTINGS: dict[str, Any] = {
    "angular_step_degrees": 5.0,
    "horizon_step_years": 0.25,
    "horizon_end_years": 40.0,
    "key_horizons": list(KEY_HORIZONS),
    "persistence_cases": [0.50, 0.81, 0.95],
    "neutrality_unravel_fraction": 0.10,
    "superposition_relative_tolerance": 1e-9,
    "sign_symmetry_relative_tolerance": 1e-9,
    "scaling_relative_tolerance": 1e-9,
    "timing_distinction_absolute_tolerance": 1e-13,
    "neutral_zero_relative_tolerance": 1e-11,
    "coordinate_conversion_tolerance_degrees": 1e-9,
    "accounting_identity_absolute_tolerance": 1e-14,
    "fixed_share_reproduction_absolute_tolerance": 1e-12,
    "row_builder_cross_check_absolute_tolerance": 1e-12,
    "cumulative_expm_integral_horizon_years": 1500.0,
    "cumulative_expm_integral_relative_tolerance": 1e-9,
    "ode_cross_check": "all",
}

# ---------------------------------------------------------------------------
# Model container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtlasModel:
    label: str
    overrides: dict[str, float]
    parameters: PrimitiveParameters
    anchor: SteadyState
    local_system: LocalSystem
    solution: LqSolution
    portfolio: LeadingPortfolio
    diagnostics: DiagnosticsReport
    acceptance: AcceptanceReport
    baseline_irfs: dict[str, Any]
    outcome_matrix: np.ndarray  # LINEAR_FIELDS x 5, maps w = (y, X) to linear outcomes
    expm_cache: dict[str, np.ndarray]  # keyed by repr(horizon)

    @property
    def rho(self) -> float:
        return self.parameters.rho


def solve_atlas_model(
    label: str,
    parameters: PrimitiveParameters,
    experiment: dict[str, Any],
    horizons: np.ndarray,
    overrides: dict[str, float] | None = None,
) -> AtlasModel:
    """Solve one parameter configuration through the maintained CS001 pipeline
    (anchor -> primitives-to-matrices -> Riccati/Sylvester/Lyapunov -> portfolio),
    rerun the pipeline's own independent diagnostics and acceptance, and cache
    the matrix exponentials used to propagate every direction."""

    scaffolding = experiment["numerical_scaffolding"]
    anchor = compute_steady_state(parameters)
    local_system = build_local_system(parameters, anchor)
    solution = solve_lq_system(local_system)
    portfolio = leading_portfolio_and_welfare(
        local_system,
        solution,
        risky_short_limit=float(scaffolding["risky_short_limit"]),
        safe_debt_limit=float(scaffolding["safe_debt_limit"]),
        risk_scale_epsilon=float(experiment["risk_scale_epsilon"]),
    )
    baseline_irfs = run_irfs(local_system, solution, portfolio, experiment["reporting"], scaffolding)
    diagnostics = run_diagnostics(local_system, solution, portfolio)
    accept = compute_acceptance(
        local_system,
        solution,
        diagnostics,
        portfolio,
        baseline_irfs["boundary_summary"],
        baseline_irfs["max_matrix_exponential_vs_ode_relative_error"],
        baseline_irfs["max_first_order_budget_residual"],
        experiment["acceptance_tolerances"],
    )
    outcome_matrix = build_outcome_matrix(local_system, solution, portfolio)
    cache = {repr(float(h)): expm(solution.F * float(h)) for h in horizons}
    return AtlasModel(
        label=label,
        overrides=dict(overrides or {}),
        parameters=parameters,
        anchor=anchor,
        local_system=local_system,
        solution=solution,
        portfolio=portfolio,
        diagnostics=diagnostics,
        acceptance=accept,
        baseline_irfs=baseline_irfs,
        outcome_matrix=outcome_matrix,
        expm_cache=cache,
    )


def persistence_variant_parameters(base: PrimitiveParameters, automation_persistence_annual: float) -> PrimitiveParameters:
    return _parameters_with_override(base.raw, ("parameters", "automation_persistence_annual"), float(automation_persistence_annual))


# ---------------------------------------------------------------------------
# Linear outcome map
# ---------------------------------------------------------------------------

LINEAR_FIELDS = (
    "z_deviation",
    "x_deviation",
    "alpha_deviation_linear",
    "log_capital_deviation",
    "capital_deviation_linear",
    "tax_rate_deviation",
    "tax_speed",
    "capital_growth_drift_deviation",
    "output_deviation_linear",
    "rental_rate_deviation_linear",
    "wage_income_deviation_linear",
    "tax_base_deviation_linear",
    "tax_revenue_deviation_linear",
    "fiscal_resources_deviation_linear",
    "planner_resource_wealth_deviation",
    "planner_resource_wealth_wage_component_deviation",
    "planner_resource_wealth_capital_tax_component_deviation",
    "public_net_worth_deviation",
    "worker_comprehensive_resources_deviation",
    "worker_consumption_deviation",
    "transfer_deviation_linear",
    "government_primary_cash_flow_deviation_linear",
    "tax_adjustment_cost_deviation_first_order",
    "risky_position_deviation",
    "safe_position_deviation",
    "output_exogenous_state_component",
    "wage_income_exogenous_state_component",
    "fiscal_resources_exogenous_state_component",
    "rental_rate_exogenous_state_component",
    "planner_resource_wealth_exogenous_state_component",
    "claim_loading_state_functional",
)
LINEAR_INDEX = {name: index for index, name in enumerate(LINEAR_FIELDS)}


def build_outcome_matrix(local_system: LocalSystem, solution: LqSolution, portfolio: LeadingPortfolio) -> np.ndarray:
    """Every first-order reported object as one linear functional of the
    augmented state w = (z, x, k, t, X). Built here from the local system's
    primitive gradients, the solved H, and the leading-portfolio gradient; the
    superposition, sign-symmetry, and scaling checks all act on this map."""

    p = local_system.parameters
    anchor = local_system.anchor
    j = local_system.linear_fiscal_wealth
    capital_bar = anchor.capital_bar
    rho = p.rho
    e = np.eye(5)

    def y_part(vector4: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(vector4, dtype=float), [0.0]])

    tax_base = capital_bar * local_system.tax_base_normalized_y
    tax_revenue_y = anchor.tax_rate_bar * tax_base + anchor.tax_base_bar * np.array([0.0, 0.0, 0.0, 1.0])
    # Split planner-resource wealth J = int exp(-int r_0)(W + tau B - Psi) into its capitalized-wage
    # and capitalized-capital-tax components. The linear coefficient is additive, j = j_W + j_B, with
    # (rho I - A^T) j_W = W_y/K_bar - J_W_bar d_r and (rho I - A^T) j_B = (tau B)_y/K_bar - J_B_bar d_r,
    # J_W_bar = W_bar/(rho K_bar), J_B_bar = tau_bar B_bar/(rho K_bar); exact for the first-order object.
    A = local_system.A
    d_r = local_system.safe_rate_y
    j_w_bar = anchor.wage_income_bar / (rho * capital_bar)
    j_b_bar = anchor.tax_rate_bar * anchor.tax_base_bar / (rho * capital_bar)
    j_w = np.linalg.solve(rho * np.eye(4) - A.T, local_system.wage_y / capital_bar - j_w_bar * d_r)
    j_b = np.linalg.solve(rho * np.eye(4) - A.T, tax_revenue_y / capital_bar - j_b_bar * d_r)
    rows = {
        "z_deviation": e[0],
        "x_deviation": e[1],
        "alpha_deviation_linear": anchor.alpha_x_bar * e[1],
        "log_capital_deviation": e[2],
        "capital_deviation_linear": capital_bar * e[2],
        "tax_rate_deviation": e[3],
        "tax_speed": y_part(anchor.chi * solution.H[3, :]),
        "capital_growth_drift_deviation": y_part(local_system.A[2, :]),
        "output_deviation_linear": y_part(local_system.output_y),
        "rental_rate_deviation_linear": y_part(local_system.rental_y),
        "wage_income_deviation_linear": y_part(local_system.wage_y),
        "tax_base_deviation_linear": y_part(tax_base),
        "tax_revenue_deviation_linear": y_part(anchor.tax_rate_bar * tax_base) + anchor.tax_base_bar * e[3],
        "fiscal_resources_deviation_linear": y_part(capital_bar * local_system.fiscal_resources_normalized_y),
        "planner_resource_wealth_deviation": y_part(capital_bar * j),
        "planner_resource_wealth_wage_component_deviation": y_part(capital_bar * j_w),
        "planner_resource_wealth_capital_tax_component_deviation": y_part(capital_bar * j_b),
        "public_net_worth_deviation": e[4] - y_part(capital_bar * j),
        "worker_comprehensive_resources_deviation": e[4],
        "worker_consumption_deviation": rho * e[4],
        "transfer_deviation_linear": rho * e[4] - y_part(local_system.wage_y),
        # Government primary cash flow tau B - T - Psi; Psi has no first-order response
        # because nu_bar = 0 (its quadratic diagnostic is reported separately per row).
        "government_primary_cash_flow_deviation_linear": (y_part(anchor.tax_rate_bar * tax_base) + anchor.tax_base_bar * e[3]) - (rho * e[4] - y_part(local_system.wage_y)),
        "tax_adjustment_cost_deviation_first_order": np.zeros(5),
        "risky_position_deviation": e[4] + y_part(portfolio.portfolio_gradient_y),
        "safe_position_deviation": -y_part(capital_bar * j) - y_part(portfolio.portfolio_gradient_y),
        "output_exogenous_state_component": y_part([local_system.output_y[0], local_system.output_y[1], 0.0, 0.0]),
        "wage_income_exogenous_state_component": y_part([local_system.wage_y[0], local_system.wage_y[1], 0.0, 0.0]),
        "fiscal_resources_exogenous_state_component": y_part(
            [
                capital_bar * local_system.fiscal_resources_normalized_y[0],
                capital_bar * local_system.fiscal_resources_normalized_y[1],
                0.0,
                0.0,
            ]
        ),
        "rental_rate_exogenous_state_component": y_part([local_system.rental_y[0], local_system.rental_y[1], 0.0, 0.0]),
        "planner_resource_wealth_exogenous_state_component": y_part([capital_bar * j[0], capital_bar * j[1], 0.0, 0.0]),
        # dz + ell_x dx: the combination the traded claim prices (lambda_hat . dW in state units)
        "claim_loading_state_functional": y_part([1.0, anchor.ell_x_bar, 0.0, 0.0]),
    }
    return np.vstack([rows[name] for name in LINEAR_FIELDS])


# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Direction:
    key: str
    theta_deg: float
    on_grid: bool
    named_labels: tuple[str, ...]
    dz_per_dalpha: float | None  # slope m in (dz, dalpha) space; None when dalpha = 0

    @property
    def unit(self) -> np.ndarray:
        theta = math.radians(self.theta_deg)
        return np.array([math.cos(theta), math.sin(theta)])

    @property
    def opposite_theta_deg(self) -> float:
        return (self.theta_deg + 180.0) % 360.0


def named_direction_slopes(anchor: SteadyState) -> dict[str, float | None]:
    """dz/dalpha slopes of the analytically constructed directions at fixed inherited
    capital and tax (local LQ system and computation plan, 'Economically constructed
    directions'); None marks the pure productivity direction (dalpha = 0)."""

    eta = anchor.eta_output_alpha
    alpha = anchor.alpha_bar
    tau = anchor.tax_rate_bar
    chi_tau = 1.0 - alpha + tau * alpha
    return {
        NAMED_PURE_PRODUCTIVITY: None,
        NAMED_PURE_AUTOMATION: 0.0,
        NAMED_OUTPUT_NEUTRAL: -eta,
        NAMED_WORKER_INCOME_NEUTRAL: -(eta - 1.0 / (1.0 - alpha)),
        NAMED_PRIMARY_RESOURCE_NEUTRAL: -(tau - 1.0 + chi_tau * eta) / chi_tau,
        NAMED_CLAIM_NEUTRAL: -anchor.h_international_bar,
        NAMED_RENTAL_BASE_NEUTRAL: -(eta + 1.0 / alpha),
    }


def theta_from_dz_dalpha_slope(slope: float | None, p: PrimitiveParameters, anchor: SteadyState) -> float:
    """Convert a (dz, dalpha)-space direction with dalpha > 0 into the standardized
    innovation angle: dz = sigma_z sqrt(D) cos, dx = sigma_x sqrt(D) sin, dalpha = a dx
    => tan(theta) = sigma_z / (sigma_x * m * a). Pure productivity (slope None) is 0 deg."""

    if slope is None:
        return 0.0
    theta = math.degrees(math.atan2(p.sigma_z_hat, p.sigma_x_hat * slope * anchor.alpha_x_bar))
    return theta % 360.0


def dz_dalpha_slope_from_theta(theta_deg: float, p: PrimitiveParameters, anchor: SteadyState) -> float | None:
    """Inverse map used by the coordinate-conversion check."""

    theta = math.radians(theta_deg)
    dx = p.sigma_x_hat * math.sin(theta)
    if abs(dx) < 1e-15:
        return None
    return (p.sigma_z_hat * math.cos(theta)) / (anchor.alpha_x_bar * dx)


def _direction_key(theta_deg: float) -> str:
    return f"theta_{theta_deg:07.3f}"


def build_directions(p: PrimitiveParameters, anchor: SteadyState, angular_step_degrees: float, named_only: bool = False) -> list[Direction]:
    """The full five-degree circle plus every analytically constructed direction at
    its exact angle (merged onto a grid point when it coincides), each with its
    opposite realization theta + pi."""

    entries: dict[float, dict[str, Any]] = {}

    def add(theta: float, on_grid: bool, label: str | None, slope: float | None) -> None:
        theta = theta % 360.0
        for existing in list(entries):
            if abs(existing - theta) < 1e-9 or abs(abs(existing - theta) - 360.0) < 1e-9:
                if label:
                    entries[existing]["labels"].append(label)
                if slope is not None or entries[existing]["slope"] is None:
                    entries[existing]["slope"] = slope if slope is not None else entries[existing]["slope"]
                entries[existing]["on_grid"] = entries[existing]["on_grid"] or on_grid
                return
        entries[theta] = {"labels": [label] if label else [], "slope": slope, "on_grid": on_grid}

    if not named_only:
        n = int(round(360.0 / angular_step_degrees))
        for i in range(n):
            add(i * angular_step_degrees, True, None, None)

    slopes = named_direction_slopes(anchor)
    for name in NAMED_ORDER:
        slope = slopes[name]
        theta = theta_from_dz_dalpha_slope(slope, p, anchor)
        add(theta, False, f"{name}_positive", slope)
        add(theta + 180.0, False, f"{name}_negative", slope)

    directions = []
    for theta in sorted(entries):
        entry = entries[theta]
        slope = entry["slope"]
        if slope is None and not entry["labels"]:
            slope = dz_dalpha_slope_from_theta(theta, p, anchor)
        if entry["labels"] and slope is None:
            slope = None  # pure productivity
        directions.append(
            Direction(
                key=_direction_key(theta),
                theta_deg=float(theta),
                on_grid=bool(entry["on_grid"]),
                named_labels=tuple(entry["labels"]),
                dz_per_dalpha=slope,
            )
        )
    return directions


def named_direction_coincidences(directions: list[Direction]) -> list[dict[str, Any]]:
    """Named labels that share one exact direction (e.g. claim-neutral and rental/tax-base-neutral
    under the aligned normalization k_I = log(K_bar/L))."""

    found = []
    for direction in directions:
        names = [label for label in direction.named_labels]
        if len(names) > 1:
            found.append({"theta_deg": direction.theta_deg, "named_labels": names, "dz_per_dalpha": direction.dz_per_dalpha})
    return found


# ---------------------------------------------------------------------------
# Initial conditions per family
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InitialCondition:
    model: str
    family: str
    regime: str
    direction: Direction
    y0: np.ndarray  # (z, x, k, t) displacement; k = t = 0 always
    brownian_increment: np.ndarray | None  # sqrt(Delta) * u for the Brownian family, else None
    inherited_position: float  # s_- (0 when nothing pays)
    claim_payoff: float  # s_- * lambda_hat . dW (0 unless the Brownian family)
    domestic_planner_resource_wealth_contribution: float  # K_bar * j . y0
    normalization: str
    magnitudes: tuple[float, float]  # (per-unit-cos productivity, per-unit-sin automation) state scales

    @property
    def x0(self) -> float:
        return self.claim_payoff + self.domestic_planner_resource_wealth_contribution

    @property
    def w0(self) -> np.ndarray:
        return np.concatenate([self.y0, [self.x0]])

    @property
    def dz(self) -> float:
        return float(self.y0[0])

    @property
    def dx(self) -> float:
        return float(self.y0[1])

    @property
    def experiment_id(self) -> str:
        return EXPERIMENT_IDS[self.family]

    @property
    def path_key(self) -> str:
        return f"{self.model}::{self.family}::{self.regime}::{self.direction.key}"


def brownian_magnitudes(model: AtlasModel, reporting: dict[str, Any]) -> tuple[float, float]:
    delta = float(reporting["short_brownian_window_years"])
    p = model.parameters
    return (p.sigma_z_hat * math.sqrt(delta), p.sigma_x_hat * math.sqrt(delta))


def ou_window_magnitudes(model: AtlasModel, reporting: dict[str, Any]) -> tuple[float, float]:
    window = float(reporting["finite_ou_window_years"])
    p = model.parameters
    sd_z = p.sigma_z_hat * math.sqrt((1.0 - math.exp(-2.0 * p.kappa_z * window)) / (2.0 * p.kappa_z))
    sd_x = p.sigma_x_hat * math.sqrt((1.0 - math.exp(-2.0 * p.kappa_x * window)) / (2.0 * p.kappa_x))
    return (sd_z, sd_x)


def _make_initial_condition(
    model: AtlasModel,
    family: str,
    regime: str,
    direction: Direction,
    dz: float,
    dx: float,
    brownian_increment: np.ndarray | None,
    normalization: str,
    magnitudes: tuple[float, float],
) -> InitialCondition:
    y0 = np.array([dz, dx, 0.0, 0.0])
    contribution = model.anchor.capital_bar * float(model.local_system.linear_fiscal_wealth @ y0)
    if regime == REGIME_OPTIMAL:
        inherited = model.portfolio.leading_unconstrained_position
    elif regime == REGIME_ZERO:
        inherited = 0.0
    elif regime == REGIME_NONE:
        inherited = 0.0
    else:
        raise ValueError(f"Unknown regime {regime}")
    payoff = 0.0
    if brownian_increment is not None and regime != REGIME_NONE:
        payoff = inherited * float(model.portfolio.lambda_hat @ brownian_increment)
    return InitialCondition(
        model=model.label,
        family=family,
        regime=regime,
        direction=direction,
        y0=y0,
        brownian_increment=None if brownian_increment is None else np.asarray(brownian_increment, dtype=float),
        inherited_position=inherited,
        claim_payoff=payoff,
        domestic_planner_resource_wealth_contribution=contribution,
        normalization=normalization,
        magnitudes=magnitudes,
    )


def brownian_initial_conditions(model: AtlasModel, directions: list[Direction], reporting: dict[str, Any]) -> list[InitialCondition]:
    delta = float(reporting["short_brownian_window_years"])
    mags = brownian_magnitudes(model, reporting)
    text = f"one standardized joint Brownian innovation over Delta={delta:.9g} years; dz=sigma_z sqrt(Delta) cos, dx=sigma_x sqrt(Delta) sin"
    out = []
    for direction in directions:
        u = direction.unit
        increment = math.sqrt(delta) * u
        for regime in (REGIME_OPTIMAL, REGIME_ZERO):
            out.append(_make_initial_condition(model, FAMILY_BROWNIAN, regime, direction, mags[0] * u[0], mags[1] * u[1], increment, text, mags))
    return out


def matched_state_initial_conditions(
    model: AtlasModel, directions: list[Direction], reporting: dict[str, Any], magnitudes: tuple[float, float] | None = None, family: str = FAMILY_MATCHED_STATE
) -> list[InitialCondition]:
    mags = brownian_magnitudes(model, reporting) if magnitudes is None else magnitudes
    text = "abstract inherited-state displacement with the same (dz, dx) as the Brownian family; public net worth fixed; no payoff; future innovations off"
    if magnitudes is not None:
        text = "abstract inherited-state displacement with the BASELINE-model (dz, dx) held fixed across persistence cases; public net worth fixed; no payoff"
    return [_make_initial_condition(model, family, REGIME_NONE, d, mags[0] * d.unit[0], mags[1] * d.unit[1], None, text, mags) for d in directions]


def ou_window_initial_conditions(model: AtlasModel, directions: list[Direction], reporting: dict[str, Any]) -> list[InitialCondition]:
    mags = ou_window_magnitudes(model, reporting)
    window = float(reporting["finite_ou_window_years"])
    text = f"joint state displacement by each state's {window:g}-year conditional OU standard deviation; public net worth fixed; no payoff"
    return [_make_initial_condition(model, FAMILY_OU_WINDOW, REGIME_NONE, d, mags[0] * d.unit[0], mags[1] * d.unit[1], None, text, mags) for d in directions]


def fixed_share_initial_conditions(model: AtlasModel, directions: list[Direction], reporting: dict[str, Any]) -> list[InitialCondition]:
    """The project's existing dalpha=+0.01 (and -0.01) normalization for every
    automation-containing named direction, as a reproduction check only."""

    dalpha_size = float(reporting["constructed_automation_share_displacement"])
    a = model.anchor.alpha_x_bar
    out = []
    for direction in directions:
        if not direction.named_labels or direction.dz_per_dalpha is None:
            continue
        sign = 1.0 if 0.0 < direction.theta_deg < 180.0 else -1.0
        dalpha = sign * dalpha_size
        dx = dalpha / a
        dz = direction.dz_per_dalpha * dalpha
        text = f"fixed automation-share displacement dalpha={dalpha:+.6g} with dz=(dz/dalpha)*dalpha; public net worth fixed; no payoff"
        out.append(_make_initial_condition(model, FAMILY_FIXED_SHARE, REGIME_NONE, direction, dz, dx, None, text, (abs(dz), abs(dx))))
    return out


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------


def propagate(model: AtlasModel, w0: np.ndarray, horizons: np.ndarray) -> np.ndarray:
    return np.vstack([model.expm_cache[repr(float(h))] @ w0 for h in horizons])


def ode_cross_check(model: AtlasModel, w0: np.ndarray, horizons: np.ndarray, path: np.ndarray) -> float:
    F = model.solution.F
    ode = solve_ivp(lambda _t, value: F @ value, (float(horizons[0]), float(horizons[-1])), w0, t_eval=horizons, rtol=1e-12, atol=1e-16, method="DOP853")
    if not ode.success:
        raise RuntimeError(f"Direct ODE cross-check failed: {ode.message}")
    difference = float(np.max(np.abs(path - ode.y.T)))
    scale = float(np.max(np.abs(path)))
    return difference / scale if scale > 0.0 else difference


def discounted_cumulative(model: AtlasModel, w0: np.ndarray) -> tuple[np.ndarray, float]:
    """(rho I - F)^{-1} w0 with the resolvent identity residual evaluated on this probe."""

    cumulative = model.solution.resolvent @ w0
    residual = resolvent_probe_residual(model.rho, model.solution.F, cumulative, w0)
    return cumulative, residual


def discounted_cumulative_expm_integral(model: AtlasModel, w0: np.ndarray, horizon_years: float) -> np.ndarray:
    """Independent route to the same integral: the upper-right block of
    expm([[F - rho I, I], [0, 0]] T) is int_0^T e^{(F - rho I) s} ds."""

    F = model.solution.F
    n = F.shape[0]
    block = np.zeros((2 * n, 2 * n))
    block[:n, :n] = F - model.rho * np.eye(n)
    block[:n, n:] = np.eye(n)
    integral = expm(block * horizon_years)[:n, n:]
    return integral @ w0


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

ROW_STRING_FIELDS = ("model", "family", "experiment_id", "regime", "direction_key", "named_labels", "normalization", "failure_reasons")
ROW_BOOL_FIELDS = ("on_grid", "economic_conditions_ok", "numerical_scaffolding_ok")


def _state_angles(y: np.ndarray, p: PrimitiveParameters, anchor: SteadyState) -> tuple[float, float]:
    standardized = math.degrees(math.atan2(y[1] / p.sigma_x_hat, y[0] / p.sigma_z_hat)) % 360.0
    share_space = math.degrees(math.atan2(anchor.alpha_x_bar * y[1], y[0])) % 360.0
    return standardized, share_space


def _angle_difference(later: float, earlier: float) -> float:
    return ((later - earlier + 180.0) % 360.0) - 180.0


def build_rows(model: AtlasModel, ic: InitialCondition, horizons: np.ndarray, path: np.ndarray, scaffolding: dict[str, Any]) -> list[dict[str, Any]]:
    p = model.parameters
    anchor = model.anchor
    linear = path @ model.outcome_matrix.T  # n_h x n_fields
    impact_angles = _state_angles(ic.y0, p, anchor)
    j = model.local_system.linear_fiscal_wealth
    capital_bar = anchor.capital_bar
    rows = []
    for index, horizon in enumerate(horizons):
        w = path[index]
        y = w[:4]
        values = {name: float(linear[index, LINEAR_INDEX[name]]) for name in LINEAR_FIELDS}
        z = anchor.z_bar + y[0]
        x = anchor.x_bar + y[1]
        capital = capital_bar * math.exp(y[2])
        tax_rate = anchor.tax_rate_bar + y[3]
        exact = evaluate_smooth_branch(z, x, capital, tax_rate, p)
        nu = values["tax_speed"]
        x_level = anchor.comprehensive_resources_bar + values["worker_comprehensive_resources_deviation"]
        consumption_level = anchor.worker_consumption_bar + values["worker_consumption_deviation"]
        transfer_level = consumption_level - exact.wage_income
        transfer_level_linear = anchor.transfer_bar + values["transfer_deviation_linear"]
        tax_revenue_level = anchor.tax_rate_bar * anchor.tax_base_bar + values["tax_revenue_deviation_linear"]
        primary_cash_flow_level = tax_revenue_level - transfer_level_linear
        net_worth_level = anchor.public_net_worth_bar + values["public_net_worth_deviation"]
        risky_level = model.portfolio.leading_unconstrained_position + values["risky_position_deviation"]
        safe_level = net_worth_level - risky_level
        lower_slack = risky_level + float(scaffolding["risky_short_limit"])
        upper_slack = net_worth_level + float(scaffolding["safe_debt_limit"]) - risky_level
        tax_lower = tax_rate - float(scaffolding["tax_min"])
        tax_upper = float(scaffolding["tax_max"]) - tax_rate
        tax_ceiling = 1.0 - tax_rate
        speed_slack = float(scaffolding["tax_speed_abs_max"]) - abs(nu)
        budget_left = float(anchor.comprehensive_resources_bar * (model.local_system.safe_rate_y @ y) - capital_bar * (j @ (model.solution.A_c @ y)))
        budget_right = float(
            p.rho * values["public_net_worth_deviation"]
            + anchor.public_net_worth_bar * (model.local_system.safe_rate_y @ y)
            + values["fiscal_resources_deviation_linear"]
            - values["worker_consumption_deviation"]
        )
        angles = _state_angles(y, p, anchor) if np.max(np.abs(y[:2])) > 0.0 else (float("nan"), float("nan"))
        # Genuine economic conditions of the maintained branch (never "feasibility" of
        # government borrowing, which this calculation does not establish).
        economic_failures = []
        if exact.specialisation_margin_automation_composite <= 0.0 or exact.specialisation_margin_new_task_composite <= 0.0:
            economic_failures.append("economic:off_full_specialisation_branch")
        if exact.output_automation_semielasticity <= 0.0:
            economic_failures.append("economic:automation_output_sign_branch_lost")
        if consumption_level <= 0.0 or x_level <= 0.0:
            economic_failures.append("economic:nonpositive_consumption_or_worker_comprehensive_resources")
        if transfer_level <= 0.0:
            economic_failures.append("economic:transfer_floor_violated")
        if tax_ceiling <= 0.0:
            economic_failures.append("economic:tax_rate_at_or_above_one")
        if not np.all(np.isfinite(w)):
            economic_failures.append("economic:nonfinite_state")
        # Artificial numerical scaffolding (wide portfolio/debt caps, tax box, tax-speed cap):
        # slack here is a numerical-validity statement only, never borrowing capacity.
        scaffolding_failures = []
        if min(lower_slack, upper_slack) <= 0.0:
            scaffolding_failures.append("scaffolding:portfolio_cap")
        if min(tax_lower, tax_upper) <= 0.0:
            scaffolding_failures.append("scaffolding:tax_box")
        if speed_slack <= 0.0:
            scaffolding_failures.append("scaffolding:tax_speed_cap")
        failure_reasons = economic_failures + scaffolding_failures
        row: dict[str, Any] = {
            "model": model.label,
            "family": ic.family,
            "experiment_id": ic.experiment_id,
            "regime": ic.regime,
            "direction_key": ic.direction.key,
            "theta_deg": ic.direction.theta_deg,
            "on_grid": ic.direction.on_grid,
            "named_labels": "|".join(ic.direction.named_labels),
            "dz_per_dalpha": ic.direction.dz_per_dalpha,
            "normalization": ic.normalization,
            "dz_impact": ic.dz,
            "dx_impact": ic.dx,
            "dalpha_impact": anchor.alpha_x_bar * ic.dx,
            "standardized_innovation_z": None if ic.brownian_increment is None else float(ic.brownian_increment[0]),
            "standardized_innovation_x": None if ic.brownian_increment is None else float(ic.brownian_increment[1]),
            "inherited_position_s_minus": ic.inherited_position,
            "claim_payoff_impact": ic.claim_payoff,
            "domestic_planner_resource_wealth_contribution_impact": ic.domestic_planner_resource_wealth_contribution,
            "worker_comprehensive_resources_impact": ic.x0,
            "horizon_years": float(horizon),
        }
        row.update(values)
        row.update(
            {
                "risky_position_level": risky_level,
                "safe_position_level": safe_level,
                "public_net_worth_level": net_worth_level,
                "worker_comprehensive_resources_level": x_level,
                "worker_consumption_level": consumption_level,
                "wage_income_level_exact": exact.wage_income,
                "tax_revenue_level_linear": tax_revenue_level,
                "transfer_level_linear": transfer_level_linear,
                "transfer_level_exact_wage": transfer_level,
                "government_primary_cash_flow_level_linear": primary_cash_flow_level,
                "specialisation_margin_automation_composite": exact.specialisation_margin_automation_composite,
                "specialisation_margin_new_task_composite": exact.specialisation_margin_new_task_composite,
                "output_automation_semielasticity": exact.output_automation_semielasticity,
                "portfolio_lower_slack": lower_slack,
                "portfolio_upper_slack": upper_slack,
                "tax_lower_slack": tax_lower,
                "tax_upper_slack": tax_upper,
                "tax_structural_ceiling_slack": tax_ceiling,
                "tax_speed_slack": speed_slack,
                "transfer_slack": transfer_level,
                "tax_adjustment_cost_quadratic_diagnostic": exact.output * nu * nu / (2.0 * p.tax_adjustment_scale),
                "first_order_budget_residual": budget_left - budget_right,
                "state_angle_standardized_deg": angles[0],
                "state_angle_dz_dalpha_deg": angles[1],
                "state_rotation_from_impact_deg": _angle_difference(angles[0], impact_angles[0]) if not math.isnan(angles[0]) else float("nan"),
                "economic_conditions_ok": not economic_failures,
                "numerical_scaffolding_ok": not scaffolding_failures,
                "failure_reasons": "|".join(failure_reasons),
            }
        )
        rows.append(row)
    return rows


ECONOMIC_SLACK_FIELDS = (
    "specialisation_margin_automation_composite",
    "specialisation_margin_new_task_composite",
    "output_automation_semielasticity",
    "transfer_slack",
    "tax_structural_ceiling_slack",
)
SCAFFOLDING_SLACK_FIELDS = (
    "portfolio_lower_slack",
    "portfolio_upper_slack",
    "tax_lower_slack",
    "tax_upper_slack",
    "tax_speed_slack",
)
SLACK_FIELDS = ECONOMIC_SLACK_FIELDS + SCAFFOLDING_SLACK_FIELDS


# ---------------------------------------------------------------------------
# Path features
# ---------------------------------------------------------------------------


ZERO_DEADBAND = 1e-16  # absolute: first-order responses are O(1e-4); an exactly invariant path is O(1e-20)


def crossings(horizons: np.ndarray, values: np.ndarray) -> list[float]:
    """Times where the piecewise-linear interpolant of `values` crosses zero;
    an exact zero at the first grid point is the starting condition, not a crossing."""

    found: list[float] = []
    values = np.where(np.abs(values) <= ZERO_DEADBAND, 0.0, values)
    for i in range(len(values) - 1):
        v0, v1 = values[i], values[i + 1]
        if v0 == 0.0:
            if i > 0 and v1 != 0.0:
                found.append(float(horizons[i]))
            continue
        if v0 * v1 < 0.0:
            found.append(float(horizons[i] + (horizons[i + 1] - horizons[i]) * (-v0) / (v1 - v0)))
    return found


def _first_sign_change_time(horizons: np.ndarray, values: np.ndarray, impact_scale: float) -> float | None:
    """First time the path's sign differs from its impact sign, ignoring an impact
    value that is numerically zero relative to `impact_scale`."""

    values = np.where(np.abs(values) <= ZERO_DEADBAND, 0.0, values)
    impact = values[0]
    if abs(impact) <= 1e-12 * max(impact_scale, 1e-300):
        nonzero = np.nonzero(np.abs(values) > 1e-12 * max(impact_scale, 1e-300))[0]
        if len(nonzero) == 0:
            return None
        start = int(nonzero[0])
        sign = np.sign(values[start])
        for i in range(start, len(values) - 1):
            if values[i + 1] * sign < 0.0:
                return float(horizons[i] + (horizons[i + 1] - horizons[i]) * values[i] / (values[i] - values[i + 1]))
        return None
    sign = np.sign(impact)
    for i in range(len(values) - 1):
        if values[i + 1] * sign < 0.0:
            return float(horizons[i] + (horizons[i + 1] - horizons[i]) * values[i] / (values[i] - values[i + 1]))
    return None


NEUTRALITY_FIELDS = (
    "output_deviation_linear",
    "wage_income_deviation_linear",
    "fiscal_resources_deviation_linear",
    "rental_rate_deviation_linear",
    "claim_loading_state_functional",
    "planner_resource_wealth_deviation",
)
CUMULATIVE_FIELDS = (
    "output_deviation_linear",
    "wage_income_deviation_linear",
    "fiscal_resources_deviation_linear",
    "tax_revenue_deviation_linear",
    "worker_consumption_deviation",
    "transfer_deviation_linear",
    "capital_deviation_linear",
    "tax_rate_deviation",
)


def _at(horizons: np.ndarray, series: np.ndarray, year: float) -> float:
    index = int(np.argmin(np.abs(horizons - year)))
    if abs(horizons[index] - year) > 1e-9:
        raise ValueError(f"Horizon {year} is not on the reporting grid.")
    return float(series[index])


def path_features(
    model: AtlasModel,
    ic: InitialCondition,
    horizons: np.ndarray,
    path: np.ndarray,
    component_paths: tuple[np.ndarray, np.ndarray],
    rows: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Qualitative features of one propagated path: impact incidence, wage crossing,
    peak capital, tax extrema/reversals, discounted cumulative responses, mixture
    rotation, impact-neutrality unravelling, and minimum slacks."""

    G = model.outcome_matrix
    linear = path @ G.T
    prod_linear = component_paths[0] @ G.T
    auto_linear = component_paths[1] @ G.T
    key_h = [float(h) for h in settings["key_horizons"]]
    frac = float(settings["neutrality_unravel_fraction"])

    def series(name: str) -> np.ndarray:
        return linear[:, LINEAR_INDEX[name]]

    wage = series("wage_income_deviation_linear")
    capital = series("capital_deviation_linear")
    tax = series("tax_rate_deviation")
    speed = series("tax_speed")
    out: dict[str, Any] = {
        "model": ic.model,
        "family": ic.family,
        "experiment_id": ic.experiment_id,
        "regime": ic.regime,
        "direction_key": ic.direction.key,
        "theta_deg": ic.direction.theta_deg,
        "named_labels": "|".join(ic.direction.named_labels),
        "dz_per_dalpha": ic.direction.dz_per_dalpha,
        "dz_impact": ic.dz,
        "dx_impact": ic.dx,
        "dalpha_impact": model.anchor.alpha_x_bar * ic.dx,
        "inherited_position_s_minus": ic.inherited_position,
        "claim_payoff_impact": ic.claim_payoff,
        "domestic_planner_resource_wealth_contribution_impact": ic.domestic_planner_resource_wealth_contribution,
        "worker_comprehensive_resources_impact": ic.x0,
    }
    for name in (
        "output_deviation_linear",
        "rental_rate_deviation_linear",
        "wage_income_deviation_linear",
        "tax_base_deviation_linear",
        "tax_revenue_deviation_linear",
        "fiscal_resources_deviation_linear",
        "planner_resource_wealth_deviation",
        "planner_resource_wealth_wage_component_deviation",
        "planner_resource_wealth_capital_tax_component_deviation",
        "worker_consumption_deviation",
        "transfer_deviation_linear",
        "government_primary_cash_flow_deviation_linear",
        "tax_speed",
        "risky_position_deviation",
        "safe_position_deviation",
        "capital_growth_drift_deviation",
    ):
        out[f"impact_{name}"] = float(series(name)[0])
    for name in ("output_deviation_linear", "wage_income_deviation_linear", "fiscal_resources_deviation_linear", "tax_revenue_deviation_linear", "worker_consumption_deviation", "transfer_deviation_linear", "government_primary_cash_flow_deviation_linear", "capital_deviation_linear", "tax_rate_deviation", "public_net_worth_deviation", "safe_position_deviation"):
        for year in (1.0, 2.0, 5.0, 10.0, 20.0, 40.0):
            out[f"{name}_{year:g}y"] = _at(horizons, series(name), year)

    def component_reference(name: str) -> float:
        return float(abs(prod_linear[0, LINEAR_INDEX[name]]) + abs(auto_linear[0, LINEAR_INDEX[name]]))

    out["wage_sign_change_time_years"] = _first_sign_change_time(horizons, wage, component_reference("wage_income_deviation_linear"))
    out["wage_crossing_count"] = len(crossings(horizons, wage))
    out["wage_sign_at_40y"] = float(np.sign(wage[-1]))
    peak_index = int(np.argmax(np.abs(capital)))
    out["peak_abs_capital_time_years"] = float(horizons[peak_index])
    out["peak_abs_capital_value"] = float(capital[peak_index])
    out["capital_max"] = float(np.max(capital))
    out["capital_max_time_years"] = float(horizons[int(np.argmax(capital))])
    out["capital_min"] = float(np.min(capital))
    out["capital_min_time_years"] = float(horizons[int(np.argmin(capital))])
    out["capital_sign_change_time_years"] = _first_sign_change_time(horizons, capital, float(np.max(np.abs(capital))))
    out["tax_rate_max"] = float(np.max(tax))
    out["tax_rate_max_time_years"] = float(horizons[int(np.argmax(tax))])
    out["tax_rate_min"] = float(np.min(tax))
    out["tax_rate_min_time_years"] = float(horizons[int(np.argmin(tax))])
    tax_crossings = crossings(horizons, tax)
    out["tax_sign_reversal_count"] = len(tax_crossings)
    out["tax_first_sign_reversal_time_years"] = tax_crossings[0] if tax_crossings else None
    out["max_abs_tax_speed"] = float(np.max(np.abs(speed)))
    out["max_abs_tax_speed_time_years"] = float(horizons[int(np.argmax(np.abs(speed)))])

    cumulative, resolvent_residual = discounted_cumulative(model, ic.w0)
    cumulative_outcomes = G @ cumulative
    for name in CUMULATIVE_FIELDS:
        out[f"discounted_cumulative_{name}"] = float(cumulative_outcomes[LINEAR_INDEX[name]])
    out["resolvent_identity_residual"] = float(resolvent_residual)

    # Mixture rotation of the exogenous (z, x) state, in standardized-innovation coordinates.
    p = model.parameters
    anchor = model.anchor
    for year in key_h:
        y = path[int(np.argmin(np.abs(horizons - year)))][:4]
        if np.max(np.abs(y[:2])) > 0.0:
            angle = _state_angles(y, p, anchor)[0]
            out[f"state_angle_standardized_deg_{year:g}y"] = angle
        else:
            out[f"state_angle_standardized_deg_{year:g}y"] = float("nan")
    if np.max(np.abs(ic.y0[:2])) > 0.0:
        base = _state_angles(ic.y0, p, anchor)[0]
        out["state_rotation_deg_5y"] = _angle_difference(out["state_angle_standardized_deg_5y"], base)
        out["state_rotation_deg_10y"] = _angle_difference(out["state_angle_standardized_deg_10y"], base)
    else:
        out["state_rotation_deg_5y"] = float("nan")
        out["state_rotation_deg_10y"] = float("nan")

    # Impact-neutrality unravelling: reference scale = |productivity component| + |automation component|.
    for name in NEUTRALITY_FIELDS:
        total = series(name)
        reference = abs(prod_linear[0, LINEAR_INDEX[name]]) + abs(auto_linear[0, LINEAR_INDEX[name]])
        ratio = np.abs(total) / reference if reference > 0.0 else np.full_like(total, np.nan)
        out[f"cancellation_index_impact_{name}"] = float(ratio[0]) if reference > 0.0 else float("nan")
        out[f"impact_neutral_{name}"] = bool(reference > 0.0 and ratio[0] <= 1e-9)
        exceed = np.nonzero(ratio >= frac)[0] if reference > 0.0 else np.array([], dtype=int)
        out[f"neutrality_unravel_time_years_{name}"] = float(horizons[int(exceed[0])]) if len(exceed) else None
        out[f"max_abs_relative_to_components_{name}"] = float(np.nanmax(ratio)) if reference > 0.0 else float("nan")
        for year in (1.0, 5.0, 10.0):
            out[f"abs_relative_to_components_{name}_{year:g}y"] = float(_at(horizons, ratio, year)) if reference > 0.0 else float("nan")

    for name in SLACK_FIELDS:
        out[f"min_{name}"] = float(min(row[name] for row in rows))
    out["min_slack_overall"] = float(min(out[f"min_{name}"] for name in SLACK_FIELDS))
    out["min_economic_slack"] = float(min(out[f"min_{name}"] for name in ECONOMIC_SLACK_FIELDS))
    out["min_numerical_scaffolding_slack"] = float(min(out[f"min_{name}"] for name in SCAFFOLDING_SLACK_FIELDS))
    out["all_rows_economic_conditions_ok"] = bool(all(row["economic_conditions_ok"] for row in rows))
    out["all_rows_numerical_scaffolding_ok"] = bool(all(row["numerical_scaffolding_ok"] for row in rows))
    out["max_first_order_budget_residual"] = float(max(abs(row["first_order_budget_residual"]) for row in rows))
    return out


# ---------------------------------------------------------------------------
# Computing one family (a "chunk")
# ---------------------------------------------------------------------------


@dataclass
class PathSet:
    ic: InitialCondition
    path: np.ndarray
    component_paths: tuple[np.ndarray, np.ndarray]
    rows: list[dict[str, Any]]
    features: dict[str, Any]
    ode_relative_error: float | None


def _component_initial_conditions(model: AtlasModel, ic: InitialCondition) -> tuple[InitialCondition, InitialCondition]:
    increment = ic.brownian_increment
    prod_inc = None if increment is None else np.array([increment[0], 0.0])
    auto_inc = None if increment is None else np.array([0.0, increment[1]])
    prod = _make_initial_condition(model, ic.family, ic.regime, ic.direction, ic.dz, 0.0, prod_inc, ic.normalization, ic.magnitudes)
    auto = _make_initial_condition(model, ic.family, ic.regime, ic.direction, 0.0, ic.dx, auto_inc, ic.normalization, ic.magnitudes)
    return prod, auto


def compute_path_sets(
    model: AtlasModel,
    initial_conditions: list[InitialCondition],
    horizons: np.ndarray,
    scaffolding: dict[str, Any],
    settings: dict[str, Any],
    ode_check: Callable[[InitialCondition], bool] | None = None,
) -> list[PathSet]:
    out = []
    for ic in initial_conditions:
        path = propagate(model, ic.w0, horizons)
        prod_ic, auto_ic = _component_initial_conditions(model, ic)
        components = (propagate(model, prod_ic.w0, horizons), propagate(model, auto_ic.w0, horizons))
        rows = build_rows(model, ic, horizons, path, scaffolding)
        features = path_features(model, ic, horizons, path, components, rows, settings)
        ode_error = None
        if ode_check is None or ode_check(ic):
            ode_error = ode_cross_check(model, ic.w0, horizons, path)
        features["matrix_exponential_vs_ode_relative_error"] = ode_error
        out.append(PathSet(ic=ic, path=path, component_paths=components, rows=rows, features=features, ode_relative_error=ode_error))
    return out


# ---------------------------------------------------------------------------
# Independent checks
# ---------------------------------------------------------------------------


def _rel(diff: np.ndarray, reference: np.ndarray) -> float:
    scale = float(np.max(np.abs(reference)))
    value = float(np.max(np.abs(diff)))
    return value / scale if scale > 0.0 else value


def superposition_checks(model: AtlasModel, path_sets: list[PathSet]) -> dict[str, Any]:
    """(i) every joint path equals its productivity-component path plus its
    automation-component path, on the augmented state and on every linear
    outcome field; (ii) for theta-parametrized families, the path equals
    cos(theta) * path(0 deg) + sin(theta) * path(90 deg) when both basis
    directions are present in the same family/regime."""

    G = model.outcome_matrix
    worst_component = {"error": 0.0, "locator": None}
    worst_basis = {"error": 0.0, "locator": None, "count": 0}
    by_group: dict[tuple[str, str], dict[float, PathSet]] = {}
    for ps in path_sets:
        by_group.setdefault((ps.ic.family, ps.ic.regime), {})[round(ps.ic.direction.theta_deg, 9)] = ps
        total_state = ps.component_paths[0] + ps.component_paths[1]
        err_state = _rel(ps.path - total_state, ps.path)
        err_outcome = _rel((ps.path - total_state) @ G.T, ps.path @ G.T)
        err = max(err_state, err_outcome)
        if err > worst_component["error"]:
            worst_component = {"error": err, "locator": ps.ic.path_key}
    for (family, regime), group in by_group.items():
        if family == FAMILY_FIXED_SHARE:
            continue
        basis_z = group.get(0.0)
        basis_x = group.get(90.0)
        if basis_z is None or basis_x is None:
            continue
        for theta, ps in group.items():
            u = ps.ic.direction.unit
            predicted = u[0] * basis_z.path + u[1] * basis_x.path
            err = max(_rel(ps.path - predicted, ps.path), _rel((ps.path - predicted) @ G.T, ps.path @ G.T))
            worst_basis["count"] += 1
            if err > worst_basis["error"]:
                worst_basis = {"error": err, "locator": ps.ic.path_key, "count": worst_basis["count"]}
    return {"component_split": worst_component, "cos_sin_basis": worst_basis}


def sign_symmetry_checks(path_sets: list[PathSet]) -> dict[str, Any]:
    index: dict[tuple[str, str, float], PathSet] = {}
    for ps in path_sets:
        index[(ps.ic.family, ps.ic.regime, round(ps.ic.direction.theta_deg, 6))] = ps
    worst = {"error": 0.0, "locator": None, "pairs": 0, "unpaired": []}
    for (family, regime, theta), ps in index.items():
        opposite = index.get((family, regime, round((theta + 180.0) % 360.0, 6)))
        if opposite is None:
            worst["unpaired"].append(ps.ic.path_key)
            continue
        err = _rel(ps.path + opposite.path, ps.path)
        worst["pairs"] += 1
        if err > worst["error"]:
            worst.update({"error": err, "locator": ps.ic.path_key})
    return worst


def scaling_checks(model: AtlasModel, ic: InitialCondition, horizons: np.ndarray, factors: tuple[float, ...] = (0.5, 2.0)) -> dict[str, Any]:
    base = propagate(model, ic.w0, horizons)
    worst = {"error": 0.0, "locator": ic.path_key, "factors": list(factors)}
    for factor in factors:
        scaled = propagate(model, factor * ic.w0, horizons)
        err = _rel(scaled - factor * base, base)
        worst["error"] = max(worst["error"], err)
    return worst


def timing_distinction_checks(brownian: list[PathSet], matched: list[PathSet]) -> dict[str, Any]:
    """Brownian (either inherited position) and matched-state paths share the physical
    state path exactly; the Brownian optimal-position path's comprehensive resources
    exceed the matched path's by the inherited claim payoff at EVERY horizon (F has no
    X self-term); the zero-inherited-position path coincides with the matched path."""

    matched_by_theta = {round(ps.ic.direction.theta_deg, 9): ps for ps in matched}
    out = {
        "physical_state_max_abs_difference": 0.0,
        "optimal_x_gap_minus_payoff_max_abs": 0.0,
        "zero_position_vs_matched_max_abs": 0.0,
        "optimal_x_gap_constancy_max_abs": 0.0,
        "compared_paths": 0,
        "worst_locator": None,
    }
    for ps in brownian:
        ref = matched_by_theta.get(round(ps.ic.direction.theta_deg, 9))
        if ref is None:
            continue
        out["compared_paths"] += 1
        phys = float(np.max(np.abs(ps.path[:, :4] - ref.path[:, :4])))
        if phys > out["physical_state_max_abs_difference"]:
            out["physical_state_max_abs_difference"] = phys
            out["worst_locator"] = ps.ic.path_key
        gap = ps.path[:, 4] - ref.path[:, 4]
        if ps.ic.regime == REGIME_OPTIMAL:
            out["optimal_x_gap_minus_payoff_max_abs"] = max(out["optimal_x_gap_minus_payoff_max_abs"], float(np.max(np.abs(gap - ps.ic.claim_payoff))))
            out["optimal_x_gap_constancy_max_abs"] = max(out["optimal_x_gap_constancy_max_abs"], float(np.max(gap) - np.min(gap)))
        elif ps.ic.regime == REGIME_ZERO:
            out["zero_position_vs_matched_max_abs"] = max(out["zero_position_vs_matched_max_abs"], float(np.max(np.abs(ps.path - ref.path))))
    return out


def no_jump_checks(initial_conditions: list[InitialCondition]) -> dict[str, Any]:
    worst = 0.0
    for ic in initial_conditions:
        worst = max(worst, abs(float(ic.y0[2])), abs(float(ic.y0[3])))
    return {"max_abs_capital_or_tax_impact_displacement": worst, "count": len(initial_conditions)}


def accounting_identity_checks(rows: list[dict[str, Any]], rho: float) -> dict[str, Any]:
    """Impact and path accounting identities on every row:
    dX = domestic planner-resource-wealth contribution + inherited claim payoff (impact);
    dc = rho dX; dT = dc - dW; dX = dN + dJ; and the separate-accounts identity
    F - c = tau B - T (the planner object F contains wages; the government's primary
    cash flow does not)."""

    keys = ("dX_equals_domestic_contribution_plus_payoff_at_impact", "dc_equals_rho_dX", "dT_equals_dc_minus_dW", "dX_equals_dN_plus_dJ", "F_minus_c_equals_tauB_minus_T", "primary_cash_flow_equals_tauB_minus_T_minus_Psi", "dJ_equals_wage_component_plus_capital_tax_component")
    worst: dict[str, Any] = {key: 0.0 for key in keys}
    worst["locator"] = None
    for row in rows:
        errors = {}
        if row["horizon_years"] == 0.0:
            errors["dX_equals_domestic_contribution_plus_payoff_at_impact"] = abs(row["worker_comprehensive_resources_deviation"] - (row["domestic_planner_resource_wealth_contribution_impact"] + row["claim_payoff_impact"]))
        errors["dc_equals_rho_dX"] = abs(row["worker_consumption_deviation"] - rho * row["worker_comprehensive_resources_deviation"])
        errors["dT_equals_dc_minus_dW"] = abs(row["transfer_deviation_linear"] - (row["worker_consumption_deviation"] - row["wage_income_deviation_linear"]))
        errors["dX_equals_dN_plus_dJ"] = abs(row["worker_comprehensive_resources_deviation"] - (row["public_net_worth_deviation"] + row["planner_resource_wealth_deviation"]))
        errors["F_minus_c_equals_tauB_minus_T"] = abs((row["fiscal_resources_deviation_linear"] - row["worker_consumption_deviation"]) - (row["tax_revenue_deviation_linear"] - row["transfer_deviation_linear"]))
        errors["primary_cash_flow_equals_tauB_minus_T_minus_Psi"] = abs(row["government_primary_cash_flow_deviation_linear"] - (row["tax_revenue_deviation_linear"] - row["transfer_deviation_linear"] - row["tax_adjustment_cost_deviation_first_order"]))
        errors["dJ_equals_wage_component_plus_capital_tax_component"] = abs(row["planner_resource_wealth_deviation"] - (row["planner_resource_wealth_wage_component_deviation"] + row["planner_resource_wealth_capital_tax_component_deviation"]))
        for key, value in errors.items():
            if value > worst[key]:
                worst[key] = value
                worst["locator"] = f"{row['model']}::{row['family']}::{row['regime']}::{row['direction_key']}@{row['horizon_years']}:{key}"
    return worst


def anchor_decomposition(model: AtlasModel) -> dict[str, Any]:
    """Constant-flow decomposition of planner-resource wealth at the frozen anchor
    (r_I = rho, zero adjustment cost): J = W/rho + tau B/rho. This is an anchor
    identity only; away from the anchor the state-dependent decomposition differs."""

    anchor = model.anchor
    rho = model.rho
    wage_value = anchor.wage_income_bar / rho
    tax_value = anchor.tax_rate_bar * anchor.tax_base_bar / rho
    return {
        "wage_income_W": anchor.wage_income_bar,
        "capital_tax_receipts_tauB": anchor.tax_rate_bar * anchor.tax_base_bar,
        "transfer_T": anchor.transfer_bar,
        "tax_adjustment_cost_Psi": 0.0,
        "government_primary_cash_flow_tauB_minus_T_minus_Psi": anchor.tax_rate_bar * anchor.tax_base_bar - anchor.transfer_bar,
        "public_net_worth_N": anchor.public_net_worth_bar,
        "worker_wage_endowment_value_W_over_rho": wage_value,
        "capital_tax_resource_value_tauB_over_rho": tax_value,
        "sum_of_values": wage_value + tax_value,
        "planner_resource_wealth_J": anchor.fiscal_wealth_bar,
        "decomposition_residual": anchor.fiscal_wealth_bar - (wage_value + tax_value),
        "worker_comprehensive_resources_X": anchor.comprehensive_resources_bar,
        "unconstrained_leading_position_s_star": model.portfolio.leading_unconstrained_position,
        "unconstrained_safe_position_N_minus_s_star": anchor.public_net_worth_bar - model.portfolio.leading_unconstrained_position,
        "caveat": "Constant-flow anchor decomposition with r_I = rho and Psi = 0; not automatically the full state-dependent decomposition away from the anchor. J includes future worker wages and is not government borrowing capacity.",
        "portfolio_classification": "unconstrained local desired portfolio; genuine fiscal-capacity feasibility unverified",
    }


def impact_functional_coefficients(model: AtlasModel, family: str, regime: str, reporting: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Every impact object as A cos(theta) + B sin(theta) for the given family/regime
    normalization, from the two basis initial conditions (0 deg and 90 deg)."""

    basis = [Direction(key="basis_z", theta_deg=0.0, on_grid=True, named_labels=(), dz_per_dalpha=None), Direction(key="basis_x", theta_deg=90.0, on_grid=True, named_labels=(), dz_per_dalpha=0.0)]
    if family == FAMILY_BROWNIAN:
        ics = [ic for ic in brownian_initial_conditions(model, basis, reporting) if ic.regime == regime]
    elif family == FAMILY_MATCHED_STATE:
        ics = matched_state_initial_conditions(model, basis, reporting)
    elif family == FAMILY_OU_WINDOW:
        ics = ou_window_initial_conditions(model, basis, reporting)
    else:
        raise ValueError(family)
    impacts = [model.outcome_matrix @ ic.w0 for ic in ics]
    out = {}
    for name in LINEAR_FIELDS:
        out[name] = (float(impacts[0][LINEAR_INDEX[name]]), float(impacts[1][LINEAR_INDEX[name]]))
    out["claim_payoff_impact"] = (float(ics[0].claim_payoff), float(ics[1].claim_payoff))
    out["domestic_planner_resource_wealth_contribution_impact"] = (float(ics[0].domestic_planner_resource_wealth_contribution), float(ics[1].domestic_planner_resource_wealth_contribution))
    return out


def zero_impact_table(model: AtlasModel, reporting: dict[str, Any]) -> list[dict[str, Any]]:
    """For each impact object and normalization: the two zero angles, the positive
    arc, the maximizing angle, the amplitude, and the (dz, dalpha)-space slope of
    the zero direction."""

    rows = []
    p = model.parameters
    anchor = model.anchor
    for family, regime in ((FAMILY_BROWNIAN, REGIME_OPTIMAL), (FAMILY_BROWNIAN, REGIME_ZERO), (FAMILY_OU_WINDOW, REGIME_NONE)):
        coefficients = impact_functional_coefficients(model, family, regime, reporting)
        for name, (A, B) in coefficients.items():
            amplitude = math.hypot(A, B)
            if amplitude == 0.0:
                rows.append({"model": model.label, "family": family, "regime": regime, "impact_object": name, "coefficient_cos": A, "coefficient_sin": B, "amplitude": 0.0, "zero_angle_1_deg": None, "zero_angle_2_deg": None, "maximizing_angle_deg": None, "positive_arc_start_deg": None, "positive_arc_end_deg": None, "dz_per_dalpha_at_zero": None, "standardized_productivity_per_standardized_automation_at_zero": None})
                continue
            zero1 = math.degrees(math.atan2(-A, B)) % 360.0
            zero2 = (zero1 + 180.0) % 360.0
            argmax = math.degrees(math.atan2(B, A)) % 360.0
            slope = dz_dalpha_slope_from_theta(zero1, p, anchor)
            ratio = math.cos(math.radians(zero1)) / math.sin(math.radians(zero1)) if abs(math.sin(math.radians(zero1))) > 1e-15 else None
            rows.append(
                {
                    "model": model.label,
                    "family": family,
                    "regime": regime,
                    "impact_object": name,
                    "coefficient_cos": A,
                    "coefficient_sin": B,
                    "amplitude": amplitude,
                    "zero_angle_1_deg": zero1,
                    "zero_angle_2_deg": zero2,
                    "maximizing_angle_deg": argmax,
                    "positive_arc_start_deg": (argmax - 90.0) % 360.0,
                    "positive_arc_end_deg": (argmax + 90.0) % 360.0,
                    "dz_per_dalpha_at_zero": slope,
                    "standardized_productivity_per_standardized_automation_at_zero": ratio,
                }
            )
    return rows


def coordinate_conversion_checks(model: AtlasModel, directions: list[Direction], reporting: dict[str, Any], tolerance_deg: float) -> dict[str, Any]:
    """(i) theta -> (dz, dalpha) -> theta round trip; (ii) each named impact-neutral
    direction's angle equals the analytic zero angle of its impact functional; (iii)
    the claim-neutral direction is orthogonal to lambda_hat in standardized coordinates."""

    p = model.parameters
    anchor = model.anchor
    worst_round_trip = 0.0
    for direction in directions:
        slope = dz_dalpha_slope_from_theta(direction.theta_deg, p, anchor)
        if slope is None:
            continue
        theta_back = theta_from_dz_dalpha_slope(slope, p, anchor)
        if direction.theta_deg >= 180.0:
            theta_back = (theta_back + 180.0) % 360.0
        worst_round_trip = max(worst_round_trip, abs(_angle_difference(theta_back, direction.theta_deg)))
    coefficients = impact_functional_coefficients(model, FAMILY_BROWNIAN, REGIME_OPTIMAL, reporting)
    named_zero_errors = {}
    for direction in directions:
        for label in direction.named_labels:
            base = label.rsplit("_", 1)[0]
            target = NAMED_IMPACT_NEUTRAL_OBJECT.get(base)
            if target is None:
                continue
            A, B = coefficients[target]
            zero1 = math.degrees(math.atan2(-A, B)) % 360.0
            err = min(abs(_angle_difference(zero1, direction.theta_deg)), abs(_angle_difference(zero1 + 180.0, direction.theta_deg)))
            named_zero_errors[label] = err
    lam = model.portfolio.lambda_hat
    claim_orthogonality = None
    for direction in directions:
        if any(label.startswith(NAMED_CLAIM_NEUTRAL) for label in direction.named_labels):
            claim_orthogonality = abs(float(lam @ direction.unit)) / float(np.linalg.norm(lam))
            break
    return {
        "round_trip_max_abs_degrees": worst_round_trip,
        "named_zero_angle_errors_degrees": named_zero_errors,
        "named_zero_angle_max_error_degrees": max(named_zero_errors.values()) if named_zero_errors else 0.0,
        "claim_neutral_orthogonal_to_lambda_hat_relative": claim_orthogonality,
        "tolerance_degrees": tolerance_deg,
    }


def neutral_zero_checks(path_sets: list[PathSet], tolerance: float) -> dict[str, Any]:
    """Each analytically constructed neutral direction must produce the intended
    zero impact object relative to the size of its cancelling components."""

    results = {}
    worst = 0.0
    for ps in path_sets:
        for label in ps.ic.direction.named_labels:
            base = label.rsplit("_", 1)[0]
            target = NAMED_IMPACT_NEUTRAL_OBJECT.get(base)
            if target is None:
                continue
            index = ps.features[f"cancellation_index_impact_{target}"]
            key = f"{ps.ic.model}::{ps.ic.family}::{ps.ic.regime}::{label}"
            results[key] = {"object": target, "cancellation_index_impact": index}
            worst = max(worst, index)
    return {"per_direction": results, "worst_cancellation_index": worst, "tolerance": tolerance}


def fixed_share_reproduction_check(model: AtlasModel, fixed_share_sets: list[PathSet], horizons: np.ndarray) -> dict[str, Any]:
    """The dalpha=+0.01 rows must reproduce the baseline pipeline's own
    'constructed_*' experiments (irfs.build_experiments) field by field."""

    name_map = {
        NAMED_CLAIM_NEUTRAL: "constructed_combined_claim_rental_base_neutral",
        NAMED_RENTAL_BASE_NEUTRAL: "constructed_combined_claim_rental_base_neutral",
        NAMED_WORKER_INCOME_NEUTRAL: "constructed_worker_income_neutral",
        NAMED_OUTPUT_NEUTRAL: "constructed_output_neutral",
        NAMED_PRIMARY_RESOURCE_NEUTRAL: "constructed_primary_resource_neutral",
    }
    field_map = {
        "z_deviation": "z_deviation",
        "x_deviation": "x_deviation",
        "log_capital_deviation": "log_capital_deviation",
        "tax_rate_deviation": "tax_rate_deviation",
        "tax_speed": "tax_speed",
        "output_deviation_linear": "output_deviation_linear",
        "rental_rate_deviation_linear": "rental_rate_deviation_linear",
        "wage_income_deviation_linear": "wage_income_deviation_linear",
        "tax_base_deviation_linear": "tax_base_deviation_linear",
        "tax_revenue_deviation_linear": "tax_revenue_deviation_linear",
        "fiscal_resources_deviation_linear": "fiscal_resources_deviation_linear",
        "public_net_worth_deviation": "public_net_worth_deviation",
        "worker_comprehensive_resources_deviation": "comprehensive_resources_deviation",
        "worker_consumption_deviation": "worker_consumption_deviation",
        "transfer_deviation_linear": "transfer_deviation_linear",
        "specialisation_margin_automation_composite": "specialisation_margin_automation_composite",
        "specialisation_margin_new_task_composite": "specialisation_margin_new_task_composite",
        "transfer_level_exact_wage": "transfer_level",
        "risky_position_deviation": "risky_position_deviation",
    }
    baseline_rows: dict[tuple[str, float], dict[str, Any]] = {}
    for row in model.baseline_irfs["rows"]:
        if row["experiment"].startswith("constructed_") and row["regime"] == "full_access":
            baseline_rows[(row["experiment"], round(row["horizon_years"], 6))] = row
    worst = {"max_abs_difference": 0.0, "locator": None, "compared_rows": 0, "missing": []}
    for ps in fixed_share_sets:
        if not (0.0 < ps.ic.direction.theta_deg < 180.0):
            continue  # the baseline pipeline only runs the +0.01 sign
        targets = {name_map[label.rsplit("_", 1)[0]] for label in ps.ic.direction.named_labels if label.rsplit("_", 1)[0] in name_map}
        for target in targets:
            for row in ps.rows:
                ref = baseline_rows.get((target, round(row["horizon_years"], 6)))
                if ref is None:
                    worst["missing"].append(f"{target}@{row['horizon_years']}")
                    continue
                worst["compared_rows"] += 1
                for mine, theirs in field_map.items():
                    diff = abs(float(row[mine]) - float(ref[theirs]))
                    if diff > worst["max_abs_difference"]:
                        worst.update({"max_abs_difference": diff, "locator": f"{ps.ic.path_key}@{row['horizon_years']}:{mine} vs {target}:{theirs}"})
    return worst


def row_builder_cross_check(model: AtlasModel, path_sets: list[PathSet], scaffolding: dict[str, Any], sample: int = 12) -> dict[str, Any]:
    """Compare this module's row builder with the baseline pipeline's irfs._row on
    optimal-inherited-position rows (the only regime where the two builders are
    meant to agree on every shared field)."""

    from .irfs import Experiment, _row  # baseline pipeline row builder

    shared = {
        "z_deviation": "z_deviation",
        "x_deviation": "x_deviation",
        "log_capital_deviation": "log_capital_deviation",
        "tax_rate_deviation": "tax_rate_deviation",
        "tax_speed": "tax_speed",
        "output_deviation_linear": "output_deviation_linear",
        "rental_rate_deviation_linear": "rental_rate_deviation_linear",
        "wage_income_deviation_linear": "wage_income_deviation_linear",
        "tax_base_deviation_linear": "tax_base_deviation_linear",
        "tax_revenue_deviation_linear": "tax_revenue_deviation_linear",
        "fiscal_resources_deviation_linear": "fiscal_resources_deviation_linear",
        "public_net_worth_deviation": "public_net_worth_deviation",
        "worker_comprehensive_resources_deviation": "comprehensive_resources_deviation",
        "worker_consumption_deviation": "worker_consumption_deviation",
        "transfer_deviation_linear": "transfer_deviation_linear",
        "risky_position_deviation": "risky_position_deviation",
        "safe_position_level": "safe_position_level",
        "specialisation_margin_automation_composite": "specialisation_margin_automation_composite",
        "specialisation_margin_new_task_composite": "specialisation_margin_new_task_composite",
        "worker_consumption_level": "worker_consumption_level",
        "transfer_level_exact_wage": "transfer_level",
        "portfolio_lower_slack": "portfolio_lower_slack",
        "portfolio_upper_slack": "portfolio_upper_slack",
        "tax_speed_slack": "tax_speed_slack",
        "first_order_budget_residual": "first_order_budget_residual",
    }
    eligible = [ps for ps in path_sets if ps.ic.regime in (REGIME_OPTIMAL, REGIME_NONE)]
    step = max(1, len(eligible) // sample)
    worst = {"max_abs_difference": 0.0, "locator": None, "compared_rows": 0}
    for ps in eligible[::step]:
        experiment = Experiment(ps.ic.direction.key, ps.ic.family, ps.ic.normalization, ps.ic.y0, ps.ic.brownian_increment, ps.ic.brownian_increment is not None)
        for row, w in zip(ps.rows, ps.path, strict=True):
            ref = _row(model.local_system, model.solution, model.portfolio, scaffolding, experiment, "full_access", row["horizon_years"], w[:4], float(w[4]))
            worst["compared_rows"] += 1
            for mine, theirs in shared.items():
                diff = abs(float(row[mine]) - float(ref[theirs]))
                if diff > worst["max_abs_difference"]:
                    worst.update({"max_abs_difference": diff, "locator": f"{ps.ic.path_key}@{row['horizon_years']}:{mine}"})
    return worst


def invariant_line_checks(model: AtlasModel) -> dict[str, Any]:
    """Structural 'coincidence' diagnostics at this calibration: the tax-speed feedback
    on the exogenous states and the capital-growth loading are both proportional to the
    rental-rate gradient, so the claim/rental-neutral line is invariant for capital
    and tax whenever the two OU rates coincide."""

    ls = model.local_system
    sol = model.solution
    rental = ls.rental_y[:2]
    rental_ratio = rental[1] / rental[0]
    feedback = model.anchor.chi * sol.H[3, :2]
    growth = ls.A[2, :2]
    h_i_minus_rental = model.anchor.h_international_bar - (model.anchor.eta_output_alpha + 1.0 / model.anchor.alpha_bar)
    return {
        "rental_gradient_x_over_z": float(rental_ratio),
        "tax_speed_feedback_x_over_z": float(feedback[1] / feedback[0]) if feedback[0] != 0.0 else None,
        "capital_growth_loading_x_over_z": float(growth[1] / growth[0]) if growth[0] != 0.0 else None,
        "tax_speed_feedback_alignment_relative_error": float(abs(feedback[1] / feedback[0] - rental_ratio) / abs(rental_ratio)) if feedback[0] != 0.0 else None,
        "capital_growth_alignment_relative_error": float(abs(growth[1] / growth[0] - rental_ratio) / abs(rental_ratio)) if growth[0] != 0.0 else None,
        "h_I_minus_eta_plus_one_over_alpha": float(h_i_minus_rental),
        "k_I_minus_log_capital_labour": float(model.parameters.international_log_capital_labour_ratio - math.log(model.anchor.capital_bar / model.parameters.labour)),
        "kappa_z_minus_kappa_x": float(model.parameters.kappa_z - model.parameters.kappa_x),
    }


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def sign_pattern(features: dict[str, Any]) -> str:
    def s(value: float) -> str:
        if abs(value) <= 1e-13:
            return "0"
        return "+" if value > 0.0 else "-"

    return " ".join(
        f"{label}{s(features[key])}"
        for label, key in (
            ("Y", "impact_output_deviation_linear"),
            ("W", "impact_wage_income_deviation_linear"),
            ("B", "impact_tax_base_deviation_linear"),
            ("F", "impact_fiscal_resources_deviation_linear"),
            ("J", "impact_planner_resource_wealth_deviation"),
            ("pay", "claim_payoff_impact"),
            ("X", "worker_comprehensive_resources_impact"),
            ("c", "impact_worker_consumption_deviation"),
            ("T", "impact_transfer_deviation_linear"),
            ("G", "impact_government_primary_cash_flow_deviation_linear"),
            ("nu", "impact_tax_speed"),
        )
    )


def impact_sign_regions(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Contiguous angular arcs (over the ordered direction set) sharing one impact sign pattern."""

    ordered = sorted(features, key=lambda f: f["theta_deg"])
    regions: list[dict[str, Any]] = []
    for f in ordered:
        pattern = sign_pattern(f)
        if regions and regions[-1]["sign_pattern"] == pattern:
            regions[-1]["theta_end_deg"] = f["theta_deg"]
            regions[-1]["direction_count"] += 1
            regions[-1]["named_labels"] = "|".join(x for x in [regions[-1]["named_labels"], f["named_labels"]] if x)
        else:
            regions.append({"model": f["model"], "family": f["family"], "regime": f["regime"], "sign_pattern": pattern, "theta_start_deg": f["theta_deg"], "theta_end_deg": f["theta_deg"], "direction_count": 1, "named_labels": f["named_labels"]})
    if len(regions) > 1 and regions[0]["sign_pattern"] == regions[-1]["sign_pattern"]:
        last = regions.pop()
        regions[0]["theta_start_deg"] = last["theta_start_deg"]
        regions[0]["direction_count"] += last["direction_count"]
        regions[0]["named_labels"] = "|".join(x for x in [last["named_labels"], regions[0]["named_labels"]] if x)
        regions[0]["wraps_360"] = True
    for region in regions:
        region.setdefault("wraps_360", False)
    return regions


# ---------------------------------------------------------------------------
# Atomic writes, state file, event log
# ---------------------------------------------------------------------------


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def gzip_bytes(text: str) -> bytes:
    """Deterministic gzip (mtime=0, no filename) so identical content gives identical bytes."""

    import gzip
    import io

    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(text.encode("utf-8"))
    return buffer.getvalue()


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(serial(payload), indent=2, sort_keys=True, allow_nan=True) + "\n")


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    """CSV with full-precision floats; a ``.gz`` suffix writes a deterministic gzip stream."""

    import io

    if not rows and fieldnames is None:
        text = ""
    else:
        fieldnames = fieldnames or list(rows[0].keys())
        for row in rows:
            if list(row.keys()) != fieldnames and set(row.keys()) != set(fieldnames):
                raise ValueError("Every row must carry the same fields.")
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})
        text = buffer.getvalue()
    if path.suffix == ".gz":
        atomic_write_bytes(path, gzip_bytes(text))
    else:
        atomic_write_text(path, text)


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(serial(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    if isinstance(value, np.integer):
        return int(value)
    return value


def read_csv_typed(path: Path, string_fields: tuple[str, ...] = (), bool_fields: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    import gzip

    rows = []
    opener = (lambda: gzip.open(path, "rt", encoding="utf-8", newline="")) if path.suffix == ".gz" else (lambda: path.open("r", encoding="utf-8", newline=""))
    with opener() as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key in string_fields:
                    row[key] = value
                elif key in bool_fields:
                    row[key] = value == "True"
                elif value == "":
                    row[key] = None
                else:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        row[key] = value
            rows.append(row)
    return rows


def serial(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {name: serial(getattr(value, name)) for name in value.__dataclass_fields__}
    return value


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class EventLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **payload: Any) -> None:
        record = {"utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"), "event": event, **serial(payload)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class StateFile:
    def __init__(self, path: Path, fingerprint: dict[str, Any]):
        self.path = path
        self.fingerprint = fingerprint
        self.state: dict[str, Any] = {"fingerprint": fingerprint, "completed_chunks": [], "chunk_artifacts": {}, "created_utc": _now(), "updated_utc": _now()}

    def load_or_refuse(self, resume: bool) -> None:
        if not self.path.exists():
            self.save()
            return
        existing = json.loads(self.path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != self.fingerprint:
            raise RuntimeError(
                "Refusing to resume: the existing state file carries a different commit/configuration fingerprint. "
                f"existing={existing.get('fingerprint')} current={self.fingerprint}"
            )
        if not resume:
            raise RuntimeError(f"Output directory already holds a state file ({self.path}); pass resume=True to continue unfinished chunks.")
        self.state = existing

    def is_complete(self, chunk: str) -> bool:
        return chunk in self.state["completed_chunks"]

    def mark_complete(self, chunk: str, artifacts: dict[str, str]) -> None:
        if chunk not in self.state["completed_chunks"]:
            self.state["completed_chunks"].append(chunk)
        self.state["chunk_artifacts"][chunk] = artifacts
        self.save()

    def save(self) -> None:
        self.state["updated_utc"] = _now()
        atomic_write_json(self.path, self.state)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def module_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def atlas_fingerprint(config: Cs001Configuration, settings: dict[str, Any], commit: str | None) -> dict[str, Any]:
    return {
        "commit": commit,
        "complete_input_sha256": config.fingerprints["complete_input_sha256"],
        "atlas_settings_sha256": sha256_of(settings),
        "shock_atlas_source_sha256": module_source_sha256(),
    }


# ---------------------------------------------------------------------------
# The chunked runner
# ---------------------------------------------------------------------------


class AtlasCheckFailure(RuntimeError):
    """A load-bearing independent check failed reproducibly; the run stops with the
    failing diagnostics already written to disk."""


def make_horizons(settings: dict[str, Any]) -> np.ndarray:
    step = float(settings["horizon_step_years"])
    end = float(settings["horizon_end_years"])
    horizons = np.arange(0.0, end + step / 2.0, step)
    for required in REQUIRED_HORIZONS + tuple(settings["key_horizons"]):
        if not np.any(np.abs(horizons - required) < 1e-9):
            raise ValueError(f"Required horizon {required} is not on the grid.")
    return horizons


CHUNK_MODELS = "models"
CHUNK_BROWNIAN = "atlas_brownian_innovation"
CHUNK_MATCHED = "atlas_matched_state_displacement"
CHUNK_OU = "atlas_finite_window_ou_displacement"
CHUNK_FIXED_SHARE = "atlas_fixed_share_check"
CHUNK_PERSISTENCE = "persistence_unravelling"
CHUNK_CHECKS = "independent_checks"
CHUNK_TABLES = "tables"
CHUNK_ORDER = (CHUNK_MODELS, CHUNK_BROWNIAN, CHUNK_MATCHED, CHUNK_OU, CHUNK_FIXED_SHARE, CHUNK_PERSISTENCE, CHUNK_CHECKS, CHUNK_TABLES)


class AtlasRunner:
    """Runs the atlas as resumable chunks. Every chunk writes its raw rows and features
    to parts/ atomically before the state file records it as complete; an interrupted
    run resumes only unfinished chunks under the same commit and configuration."""

    def __init__(self, config: Cs001Configuration, output_dir: Path, settings: dict[str, Any], commit: str | None, resume: bool = False, mode: str = "full"):
        self.config = config
        self.experiment = config.experiment
        self.scaffolding = self.experiment["numerical_scaffolding"]
        self.reporting = self.experiment["reporting"]
        self.settings = {**DEFAULT_SETTINGS, **settings}
        self.mode = mode
        self.output_dir = output_dir
        self.parts_dir = output_dir / "parts"
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.horizons = make_horizons(self.settings)
        self.fingerprint = atlas_fingerprint(config, {**self.settings, "mode": mode}, commit)
        self.events = EventLog(output_dir / "events.log")
        self.state = StateFile(output_dir / "state.json", self.fingerprint)
        self.state.load_or_refuse(resume)
        self.runtime: dict[str, float] = {}
        self.models: dict[str, AtlasModel] = {}
        self.directions: dict[str, list[Direction]] = {}
        self.path_sets: dict[str, list[PathSet]] = {}
        self.checks: dict[str, Any] = {}
        self.events.write("runner_initialized", mode=mode, resume=resume, fingerprint=self.fingerprint, completed_chunks=list(self.state.state["completed_chunks"]))

    # -- helpers --------------------------------------------------------------

    def _timed(self, chunk: str, work: Callable[[], dict[str, str]]) -> None:
        if self.state.is_complete(chunk):
            self.events.write("chunk_skipped_already_complete", chunk=chunk)
            self._reload(chunk)
            return
        self.events.write("chunk_started", chunk=chunk)
        started = time.perf_counter()
        artifacts = work()
        elapsed = time.perf_counter() - started
        self.runtime[chunk] = elapsed
        hashed = {name: sha256_file(self.output_dir / rel) for name, rel in artifacts.items()}
        self.state.mark_complete(chunk, hashed)
        self.events.write("chunk_completed", chunk=chunk, seconds=elapsed, artifacts=hashed)

    def _reload(self, chunk: str) -> None:
        """Chunks that later chunks depend on are recomputed in memory when they were
        completed by an earlier process (the solve is sub-second); their on-disk
        artifacts are the record of what was written."""

        if chunk == CHUNK_MODELS:
            self._solve_models()
        elif chunk in (CHUNK_BROWNIAN, CHUNK_MATCHED, CHUNK_OU, CHUNK_FIXED_SHARE, CHUNK_PERSISTENCE):
            self.path_sets[chunk] = self._compute_chunk_sets(chunk)
        elif chunk == CHUNK_CHECKS:
            self.checks = json.loads((self.output_dir / "numerical_diagnostics.json").read_text(encoding="utf-8"))

    def _directions_for(self, model: AtlasModel, named_only: bool = False) -> list[Direction]:
        step = float(self.settings["angular_step_degrees"])
        if self.mode == "smoke":
            dirs = build_directions(model.parameters, model.anchor, step, named_only=True)
            keep = [d for d in dirs if any(l.startswith(NAMED_PURE_PRODUCTIVITY) or l.startswith(NAMED_PURE_AUTOMATION) for l in d.named_labels)]
            mixed = Direction(key=_direction_key(45.0), theta_deg=45.0, on_grid=True, named_labels=("smoke_mixed_45",), dz_per_dalpha=dz_dalpha_slope_from_theta(45.0, model.parameters, model.anchor))
            opposite = Direction(key=_direction_key(225.0), theta_deg=225.0, on_grid=True, named_labels=("smoke_mixed_225",), dz_per_dalpha=dz_dalpha_slope_from_theta(225.0, model.parameters, model.anchor))
            return sorted(keep + [mixed, opposite], key=lambda d: d.theta_deg)
        return build_directions(model.parameters, model.anchor, step, named_only=named_only)

    # -- chunk bodies -----------------------------------------------------------

    def _solve_models(self) -> None:
        base = self.config.parameters
        self.models["baseline"] = solve_atlas_model("baseline", base, self.experiment, self.horizons)
        for persistence in self.settings["persistence_cases"]:
            # Every persistence case gets its own labelled model, including the one equal to
            # the baseline persistence: its rows are the reference case of the persistence
            # tables and must not be confused with (or duplicate the label of) the atlas rows.
            label = f"automation_persistence_{persistence:.2f}"
            variant = persistence_variant_parameters(base, float(persistence))
            self.models[label] = solve_atlas_model(label, variant, self.experiment, self.horizons, {"automation_persistence_annual": float(persistence)})
        for label, model in self.models.items():
            self.directions[label] = self._directions_for(model)

    def _chunk_models(self) -> dict[str, str]:
        self._solve_models()
        summary = {}
        for label, model in self.models.items():
            summary[label] = {
                "overrides": model.overrides,
                "acceptance_outcome": model.acceptance.outcome,
                "failed_checks": model.acceptance.failed_checks,
                "kappa_z": model.parameters.kappa_z,
                "kappa_x": model.parameters.kappa_x,
                "sigma_z_hat": model.parameters.sigma_z_hat,
                "sigma_x_hat": model.parameters.sigma_x_hat,
                "brownian_magnitudes": brownian_magnitudes(model, self.reporting),
                "ou_window_magnitudes": ou_window_magnitudes(model, self.reporting),
                "leading_unconstrained_position": model.portfolio.leading_unconstrained_position,
                "lambda_hat": model.portfolio.lambda_hat,
                "zeta_j": model.portfolio.zeta_j,
                "zeta_j_perp": model.portfolio.zeta_j_perp,
                "real_closed_loop_eigenvalues": model.diagnostics.closed_loop["real_closed_loop_eigenvalues"],
                "real_closed_loop_hurwitz": model.diagnostics.closed_loop["real_closed_loop_hurwitz"],
                "full_closed_loop_hurwitz": model.diagnostics.closed_loop["full_closed_loop_hurwitz"],
                "diagnostics": model.diagnostics,
                "named_direction_slopes": named_direction_slopes(model.anchor),
                "named_direction_angles_deg": {d.key: {"theta_deg": d.theta_deg, "labels": d.named_labels, "dz_per_dalpha": d.dz_per_dalpha} for d in self.directions[label] if d.named_labels},
                "named_direction_coincidences": named_direction_coincidences(self.directions[label]),
                "invariant_line": invariant_line_checks(model),
                "anchor_separate_accounts_and_planner_resource_wealth_decomposition": anchor_decomposition(model),
                "portfolio_classification": "unconstrained local desired portfolio; genuine fiscal-capacity feasibility unverified",
                "naming": {
                    "J": "planner-resource wealth (worker fiscal-endowment wealth; includes future worker wages); code attribute fiscal_wealth_bar",
                    "X": "worker comprehensive resources N + J; code attribute comprehensive_resources_bar",
                    "s_star": "unconstrained leading small-risk portfolio; code attribute leading_unconstrained_position",
                    "N_minus_s_star": "unconstrained safe position; negative means a desired debt-financed risky holding conditional on slack genuine transfer/solvency boundaries, not established borrowing capacity",
                },
                "anchor": model.anchor,
                "portfolio": model.portfolio,
                "direction_count": len(self.directions[label]),
            }
        atomic_write_json(self.output_dir / "models.json", summary)
        for label, model in self.models.items():
            if model.acceptance.outcome != "pass":
                raise AtlasCheckFailure(f"Baseline pipeline acceptance failed for model {label}: {model.acceptance.failed_checks}")
            if not model.diagnostics.closed_loop["real_closed_loop_hurwitz"]:
                raise AtlasCheckFailure(f"Selected capital-tax block is not Hurwitz for model {label}.")
        return {"models.json": "models.json"}

    def _compute_chunk_sets(self, chunk: str) -> list[PathSet]:
        baseline = self.models["baseline"]
        dirs = self.directions["baseline"]
        if chunk == CHUNK_BROWNIAN:
            ics = brownian_initial_conditions(baseline, dirs, self.reporting)
            return compute_path_sets(baseline, ics, self.horizons, self.scaffolding, self.settings)
        if chunk == CHUNK_MATCHED:
            ics = matched_state_initial_conditions(baseline, dirs, self.reporting)
            return compute_path_sets(baseline, ics, self.horizons, self.scaffolding, self.settings)
        if chunk == CHUNK_OU:
            ics = ou_window_initial_conditions(baseline, dirs, self.reporting)
            return compute_path_sets(baseline, ics, self.horizons, self.scaffolding, self.settings)
        if chunk == CHUNK_FIXED_SHARE:
            ics = fixed_share_initial_conditions(baseline, dirs, self.reporting)
            return compute_path_sets(baseline, ics, self.horizons, self.scaffolding, self.settings)
        if chunk == CHUNK_PERSISTENCE:
            sets: list[PathSet] = []
            base_mags = brownian_magnitudes(baseline, self.reporting)
            named_baseline = [d for d in dirs if d.named_labels]
            for label, model in self.models.items():
                if label == "baseline":
                    continue
                named_here = [d for d in self.directions[label] if d.named_labels]
                fixed = matched_state_initial_conditions(model, named_baseline, self.reporting, magnitudes=base_mags, family=FAMILY_FIXED_ACROSS_PERSISTENCE)
                brownian = brownian_initial_conditions(model, named_here, self.reporting)
                sets.extend(compute_path_sets(model, fixed + brownian, self.horizons, self.scaffolding, self.settings))
            return sets
        raise ValueError(chunk)

    def _write_sets(self, chunk: str, sets: list[PathSet]) -> dict[str, str]:
        rows = [row for ps in sets for row in ps.rows]
        features = [ps.features for ps in sets]
        rows_name = f"parts/{chunk}.rows.csv.gz"
        features_name = f"parts/{chunk}.features.csv"
        atomic_write_csv(self.output_dir / rows_name, rows)
        atomic_write_csv(self.output_dir / features_name, features)
        return {"rows": rows_name, "features": features_name}

    def _chunk_sets(self, chunk: str) -> dict[str, str]:
        sets = self._compute_chunk_sets(chunk)
        self.path_sets[chunk] = sets
        return self._write_sets(chunk, sets)

    def _chunk_checks(self) -> dict[str, str]:
        s = self.settings
        baseline = self.models["baseline"]
        all_sets = [ps for chunk in (CHUNK_BROWNIAN, CHUNK_MATCHED, CHUNK_OU, CHUNK_FIXED_SHARE, CHUNK_PERSISTENCE) for ps in self.path_sets.get(chunk, [])]
        all_rows = [row for ps in all_sets for row in ps.rows]
        checks: dict[str, Any] = {}
        checks["models"] = {
            label: {
                "acceptance_outcome": m.acceptance.outcome,
                "failed_checks": m.acceptance.failed_checks,
                "riccati_full_scaled_residual": m.diagnostics.riccati_full_scaled_residual,
                "riccati_real_block_scaled_residual": m.diagnostics.riccati_real_block_scaled_residual,
                "riccati_symmetry_error": m.diagnostics.riccati_symmetry_error,
                "sylvester_scaled_residual": m.diagnostics.sylvester_scaled_residual,
                "discounted_lyapunov_scaled_residual": m.diagnostics.discounted_lyapunov_scaled_residual,
                "closed_form_vs_invariant_subspace_relative_error": m.diagnostics.closed_form_vs_invariant_subspace_relative_error,
                "hamiltonian_imaginary_axis_distance": m.diagnostics.hamiltonian["imaginary_axis_distance"],
                "real_closed_loop_hurwitz": m.diagnostics.closed_loop["real_closed_loop_hurwitz"],
                "real_closed_loop_stability_margin": m.diagnostics.closed_loop["real_closed_loop_stability_margin"],
                "full_closed_loop_hurwitz": m.diagnostics.closed_loop["full_closed_loop_hurwitz"],
                "finite_difference_checks": m.diagnostics.finite_difference_checks,
                "feedback_construction_errors": m.diagnostics.feedback_construction_errors,
                "resolvent_identity_residual": m.diagnostics.resolvent_identity_residual,
                "baseline_pipeline_max_expm_vs_ode": m.baseline_irfs["max_matrix_exponential_vs_ode_relative_error"],
                "baseline_pipeline_max_budget_residual": m.baseline_irfs["max_first_order_budget_residual"],
                "invariant_line": invariant_line_checks(m),
            }
            for label, m in self.models.items()
        }
        ode_errors = [(ps.ode_relative_error, ps.ic.path_key) for ps in all_sets if ps.ode_relative_error is not None]
        checks["matrix_exponential_vs_ode"] = {"max_relative_error": max(e for e, _ in ode_errors) if ode_errors else None, "locator": max(ode_errors)[1] if ode_errors else None, "paths_checked": len(ode_errors)}
        by_model: dict[str, list[PathSet]] = {}
        for ps in all_sets:
            by_model.setdefault(ps.ic.model, []).append(ps)
        checks["superposition"] = {label: superposition_checks(self.models[label], sets) for label, sets in by_model.items()}
        checks["sign_symmetry"] = {label: sign_symmetry_checks(sets) for label, sets in by_model.items()}
        scaling_targets = [ps for ps in self.path_sets.get(CHUNK_BROWNIAN, []) if ps.ic.regime == REGIME_OPTIMAL and (ps.ic.direction.named_labels or abs(ps.ic.direction.theta_deg - 45.0) < 1e-9)]
        checks["scaling"] = [scaling_checks(baseline, ps.ic, self.horizons) for ps in scaling_targets[:8]]
        checks["timing_distinction"] = timing_distinction_checks(self.path_sets.get(CHUNK_BROWNIAN, []), self.path_sets.get(CHUNK_MATCHED, []))
        checks["no_jump"] = no_jump_checks([ps.ic for ps in all_sets])
        checks["accounting_identities"] = accounting_identity_checks(all_rows, baseline.rho)
        checks["coordinate_conversion"] = {label: coordinate_conversion_checks(m, self.directions[label], self.reporting, float(s["coordinate_conversion_tolerance_degrees"])) for label, m in self.models.items()}
        checks["neutral_zero"] = neutral_zero_checks([ps for ps in all_sets if ps.ic.family != FAMILY_FIXED_SHARE], float(s["neutral_zero_relative_tolerance"]))
        checks["fixed_share_reproduction"] = fixed_share_reproduction_check(baseline, self.path_sets.get(CHUNK_FIXED_SHARE, []), self.horizons) if self.mode != "smoke" else {"skipped": "smoke mode"}
        checks["row_builder_cross_check"] = row_builder_cross_check(baseline, self.path_sets.get(CHUNK_BROWNIAN, []) + self.path_sets.get(CHUNK_MATCHED, []), self.scaffolding)
        budget = max(((abs(row["first_order_budget_residual"]), f"{row['model']}::{row['family']}::{row['regime']}::{row['direction_key']}@{row['horizon_years']}") for row in all_rows), default=(0.0, None))
        checks["first_order_budget_identity"] = {"max_abs_residual": budget[0], "locator": budget[1], "rows": len(all_rows)}
        resolvent = max(((ps.features["resolvent_identity_residual"], ps.ic.path_key) for ps in all_sets), default=(0.0, None))
        checks["discounted_cumulative_resolvent"] = {"max_probe_residual": resolvent[0], "locator": resolvent[1]}
        expm_int = {"max_relative_error": 0.0, "locator": None, "horizon_years": float(s["cumulative_expm_integral_horizon_years"])}
        for ps in [x for x in self.path_sets.get(CHUNK_BROWNIAN, []) if x.ic.direction.named_labels][:16]:
            model = self.models[ps.ic.model]
            resolvent_value = model.solution.resolvent @ ps.ic.w0
            integral_value = discounted_cumulative_expm_integral(model, ps.ic.w0, float(s["cumulative_expm_integral_horizon_years"]))
            err = _rel(resolvent_value - integral_value, resolvent_value)
            if err > expm_int["max_relative_error"]:
                expm_int.update({"max_relative_error": err, "locator": ps.ic.path_key})
        checks["discounted_cumulative_expm_integral"] = expm_int
        economic_fail = [row for row in all_rows if not row["economic_conditions_ok"]]
        scaffolding_fail = [row for row in all_rows if not row["numerical_scaffolding_ok"]]
        checks["feasibility"] = {
            "classification": "genuine economic conditions (specialisation branch, transfer floor, positive consumption and worker comprehensive resources, tau<1) are reported separately from numerical-scaffolding slack; neither establishes government borrowing capacity, which this calculation does not verify",
            "rows": len(all_rows),
            "rows_failing_economic_conditions": len(economic_fail),
            "rows_failing_numerical_scaffolding": len(scaffolding_fail),
            "min_economic_slack": min(ps.features["min_economic_slack"] for ps in all_sets) if all_sets else None,
            "min_economic_slack_locator": min(((ps.features["min_economic_slack"], ps.ic.path_key) for ps in all_sets), default=(None, None))[1],
            "min_numerical_scaffolding_slack": min(ps.features["min_numerical_scaffolding_slack"] for ps in all_sets) if all_sets else None,
            "min_numerical_scaffolding_slack_locator": min(((ps.features["min_numerical_scaffolding_slack"], ps.ic.path_key) for ps in all_sets), default=(None, None))[1],
            "min_slack_overall": min(ps.features["min_slack_overall"] for ps in all_sets) if all_sets else None,
        }
        nonfinite = [row for row in all_rows if any(isinstance(v, float) and not math.isfinite(v) for k, v in row.items() if k in LINEAR_FIELDS)]
        checks["nonfinite_rows"] = len(nonfinite)

        failures: list[str] = []
        tol_expm = float(self.experiment["acceptance_tolerances"]["matrix_exponential_vs_ode_relative_error"])
        if checks["matrix_exponential_vs_ode"]["max_relative_error"] is not None and checks["matrix_exponential_vs_ode"]["max_relative_error"] > tol_expm:
            failures.append("matrix_exponential_vs_ode")
        for label, result in checks["superposition"].items():
            if result["component_split"]["error"] > float(s["superposition_relative_tolerance"]) or result["cos_sin_basis"]["error"] > float(s["superposition_relative_tolerance"]):
                failures.append(f"superposition:{label}")
        for label, result in checks["sign_symmetry"].items():
            if result["error"] > float(s["sign_symmetry_relative_tolerance"]) or result["unpaired"]:
                failures.append(f"sign_symmetry:{label}")
        if any(item["error"] > float(s["scaling_relative_tolerance"]) for item in checks["scaling"]):
            failures.append("scaling")
        td = checks["timing_distinction"]
        tol_t = float(s["timing_distinction_absolute_tolerance"])
        if max(td["physical_state_max_abs_difference"], td["optimal_x_gap_minus_payoff_max_abs"], td["zero_position_vs_matched_max_abs"], td["optimal_x_gap_constancy_max_abs"]) > tol_t:
            failures.append("timing_distinction")
        if checks["no_jump"]["max_abs_capital_or_tax_impact_displacement"] != 0.0:
            failures.append("capital_or_tax_jumps_at_impact")
        acc = checks["accounting_identities"]
        if max(v for k, v in acc.items() if k != "locator") > float(s["accounting_identity_absolute_tolerance"]):
            failures.append("accounting_identities")
        for label, result in checks["coordinate_conversion"].items():
            if result["round_trip_max_abs_degrees"] > float(s["coordinate_conversion_tolerance_degrees"]) or result["named_zero_angle_max_error_degrees"] > float(s["coordinate_conversion_tolerance_degrees"]):
                failures.append(f"coordinate_conversion:{label}")
        if checks["neutral_zero"]["worst_cancellation_index"] > float(s["neutral_zero_relative_tolerance"]):
            failures.append("neutral_zero_impact")
        if isinstance(checks["fixed_share_reproduction"], dict) and checks["fixed_share_reproduction"].get("max_abs_difference", 0.0) > float(s["fixed_share_reproduction_absolute_tolerance"]):
            failures.append("fixed_share_reproduction")
        if checks["fixed_share_reproduction"].get("missing"):
            failures.append("fixed_share_reproduction_missing_rows")
        if checks["row_builder_cross_check"]["max_abs_difference"] > float(s["row_builder_cross_check_absolute_tolerance"]):
            failures.append("row_builder_cross_check")
        if checks["first_order_budget_identity"]["max_abs_residual"] > float(self.experiment["acceptance_tolerances"]["matrix_scaled_residual"]):
            failures.append("first_order_budget_identity")
        if checks["discounted_cumulative_resolvent"]["max_probe_residual"] > float(self.experiment["acceptance_tolerances"]["matrix_scaled_residual"]):
            failures.append("discounted_cumulative_resolvent")
        if expm_int["max_relative_error"] > float(s["cumulative_expm_integral_relative_tolerance"]):
            failures.append("discounted_cumulative_expm_integral")
        if checks["nonfinite_rows"]:
            failures.append("nonfinite_rows")
        checks["failures"] = failures
        checks["outcome"] = "pass" if not failures else "fail"
        self.checks = checks
        atomic_write_json(self.output_dir / "numerical_diagnostics.json", checks)
        if failures:
            raise AtlasCheckFailure(f"Independent checks failed: {failures} (see numerical_diagnostics.json)")
        return {"numerical_diagnostics.json": "numerical_diagnostics.json"}

    def _chunk_tables(self) -> dict[str, str]:
        key_h = {round(float(h), 9) for h in self.settings["key_horizons"]}
        artifacts: dict[str, str] = {}
        # 1. tidy raw table: all quarterly rows (large, hash-referenced) + key horizons (committed)
        quarterly_path = self.output_dir / "atlas_raw_quarterly.csv.gz"
        key_path = self.output_dir / "atlas_raw.csv"
        chunks_present = [c for c in (CHUNK_BROWNIAN, CHUNK_MATCHED, CHUNK_OU, CHUNK_FIXED_SHARE, CHUNK_PERSISTENCE) if c in self.path_sets]
        all_rows = [row for chunk in chunks_present for ps in self.path_sets[chunk] for row in ps.rows]
        fieldnames = list(all_rows[0].keys()) if all_rows else []
        atomic_write_csv(quarterly_path, all_rows, fieldnames)
        atomic_write_csv(key_path, [row for row in all_rows if round(row["horizon_years"], 9) in key_h], fieldnames)
        artifacts["atlas_raw_quarterly.csv.gz"] = "atlas_raw_quarterly.csv.gz"
        artifacts["atlas_raw.csv"] = "atlas_raw.csv"
        # 2. path features (all path-sets)
        features = [ps.features for chunk in chunks_present for ps in self.path_sets[chunk]]
        atomic_write_csv(self.output_dir / "path_features.csv", features)
        artifacts["path_features.csv"] = "path_features.csv"
        # 3. named directions
        named = [f | {"impact_sign_pattern": sign_pattern(f)} for f in features if f["named_labels"]]
        atomic_write_csv(self.output_dir / "named_directions.csv", named)
        artifacts["named_directions.csv"] = "named_directions.csv"
        # 4. zero-impact thresholds and sign regions
        zero_rows = [row for model in self.models.values() for row in zero_impact_table(model, self.reporting)]
        atomic_write_csv(self.output_dir / "zero_impact_thresholds.csv", zero_rows)
        artifacts["zero_impact_thresholds.csv"] = "zero_impact_thresholds.csv"
        regions = []
        for chunk in (CHUNK_BROWNIAN, CHUNK_MATCHED, CHUNK_OU):
            if chunk not in self.path_sets:
                continue
            groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for ps in self.path_sets[chunk]:
                groups.setdefault((ps.ic.family, ps.ic.regime), []).append(ps.features)
            for group in groups.values():
                regions.extend(impact_sign_regions(group))
        atomic_write_csv(self.output_dir / "impact_sign_regions.csv", regions)
        artifacts["impact_sign_regions.csv"] = "impact_sign_regions.csv"
        # 5. persistence tables
        persistence_features = [ps.features for ps in self.path_sets.get(CHUNK_PERSISTENCE, [])]
        atomic_write_csv(self.output_dir / "persistence_unravelling.csv", persistence_features)
        artifacts["persistence_unravelling.csv"] = "persistence_unravelling.csv"
        persistence_rows = [row for ps in self.path_sets.get(CHUNK_PERSISTENCE, []) for row in ps.rows if round(row["horizon_years"], 9) in key_h]
        atomic_write_csv(self.output_dir / "persistence_named_paths.csv", persistence_rows, fieldnames if persistence_rows else None)
        artifacts["persistence_named_paths.csv"] = "persistence_named_paths.csv"
        # 6. failed / infeasible rows (retained)
        failed = [row for row in all_rows if row["failure_reasons"]]
        atomic_write_csv(self.output_dir / "failed_rows.csv", failed, fieldnames)
        artifacts["failed_rows.csv"] = "failed_rows.csv"
        # 7. runtime
        atomic_write_json(self.output_dir / "runtime.json", {"seconds_by_chunk": self.runtime, "total_seconds": sum(self.runtime.values())})
        artifacts["runtime.json"] = "runtime.json"
        return artifacts

    # -- driver -----------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        self._timed(CHUNK_MODELS, self._chunk_models)
        for chunk in (CHUNK_BROWNIAN, CHUNK_MATCHED, CHUNK_OU):
            self._timed(chunk, lambda c=chunk: self._chunk_sets(c))
        if self.mode != "smoke":
            self._timed(CHUNK_FIXED_SHARE, lambda: self._chunk_sets(CHUNK_FIXED_SHARE))
            self._timed(CHUNK_PERSISTENCE, lambda: self._chunk_sets(CHUNK_PERSISTENCE))
        self._timed(CHUNK_CHECKS, self._chunk_checks)
        self._timed(CHUNK_TABLES, self._chunk_tables)
        self.events.write("run_completed", runtime_seconds=self.runtime)
        return {"checks": self.checks, "runtime": self.runtime, "models": {label: m.acceptance.outcome for label, m in self.models.items()}}
