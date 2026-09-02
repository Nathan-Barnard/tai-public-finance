"""Independent off-mesh ODE and boundary residuals.

Nothing here reads solve_bvp's own `rms_residuals` or `success` flag -- both
are reconstructed from the returned path `sol(t)` alone: the ODE residual by
comparing a manual central finite difference of `sol(t)` against the exact
RHS re-evaluated at OFF-MESH points (never the solver's own collocation
nodes), and the boundary residual by re-applying the terminal-condition
formula to `sol(T)`. Reusing the canonical primitive/characteristic
functions themselves is fine and expected (CS002 Block D0/D1 handoff); what
must never be reused is the solver's internal residual arrays or pass flag.

Scaling follows CS001's own convention exactly (imported, not reimplemented):
scaled_norm(residual, terms) = norm(residual) / (1 + sum(norm(term))).

D2 mandatory repair #4 (D0-D1 review finding): `independent_ode_residual`'s
off-mesh check compares a finite difference of the solved path against
`characteristic_rhs_vectorized` -- the SAME composite right-hand side the
solver itself uses. That is real evidence the returned path solves the
system it was given, but it is NOT independent of a bug in how
`characteristic_rates` assembles its four terms (a dropped term or flipped
sign there would appear identically on both sides of that comparison). The
four `manual_*_dot` functions and `componentwise_manual_rhs_check` below are
an algebraically SEPARATE reconstruction: each calls only
`evaluate_smooth_branch`, `capital_derivatives`, and `safe_rate` directly,
never `characteristic_rates`/`characteristic_rhs_vectorized`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..cs001_lq_anchor.diagnostics import scaled_norm
from ..cs001_lq_anchor.equations import LocalSystem
from ..cs001_lq_anchor.solver import LqSolution
from ..primitives import PrimitiveParameters, evaluate_smooth_branch, safe_rate
from .bvp import BvpSolveResult, TerminalConvention
from .exogenous import ExogenousEvaluator
from .model import capital_derivatives, capital_from_log, characteristic_rates, characteristic_rhs_vectorized
from .recovery import ExogenousResourcesPath, OpportunityValueRecovery, r0_along_path
from .terminal import crude_costates, lq_stable_manifold_costates


@dataclass(frozen=True)
class OdeResidualReport:
    max_scaled_residual: float
    per_state_max_scaled_residual: dict[str, float]
    n_off_mesh_points: int
    finite_difference_step: float


@dataclass(frozen=True)
class BoundaryResidualReport:
    initial_scaled_residual: float
    terminal_scaled_residual: float
    max_scaled_residual: float


def independent_ode_residual(
    result: BvpSolveResult,
    local_system: LocalSystem,
    n_off_mesh_points: int = 401,
    fd_step_fraction: float = 1e-4,
    exogenous_path: ExogenousEvaluator | None = None,
) -> OdeResidualReport:
    horizon = float(result.x_mesh[-1] - result.x_mesh[0])
    t0 = float(result.x_mesh[0])
    h = max(fd_step_fraction * horizon, 1e-8)
    # Deliberately off the solver's own mesh: mid-points of a grid that does
    # not share the (possibly non-uniform, adaptively refined) node spacing.
    off_mesh = t0 + (horizon) * (np.arange(n_off_mesh_points) + 0.5) / n_off_mesh_points
    off_mesh = np.clip(off_mesh, t0 + h, t0 + horizon - h)

    y_plus = result.sol(off_mesh + h)
    y_minus = result.sol(off_mesh - h)
    y_mid = result.sol(off_mesh)
    fd_derivative = (y_plus - y_minus) / (2.0 * h)

    anchor = local_system.anchor
    exact_rhs = characteristic_rhs_vectorized(
        y_mid, anchor.z_bar, anchor.x_bar, anchor.capital_bar, local_system.parameters, t=off_mesh, exogenous_path=exogenous_path
    )

    residual = fd_derivative - exact_rhs
    state_names = ("k", "tau", "ell", "m")
    per_state = {
        name: scaled_norm(residual[i, :], [fd_derivative[i, :], exact_rhs[i, :]]) for i, name in enumerate(state_names)
    }
    overall = scaled_norm(residual, [fd_derivative, exact_rhs])
    return OdeResidualReport(
        max_scaled_residual=max(overall, max(per_state.values())),
        per_state_max_scaled_residual=per_state,
        n_off_mesh_points=n_off_mesh_points,
        finite_difference_step=float(h),
    )


def independent_boundary_residual(
    result: BvpSolveResult,
    capital_0: float,
    tau_0: float,
    terminal_convention: TerminalConvention,
    local_system: LocalSystem,
    solution: LqSolution,
    z_T: float | None = None,
    x_T: float | None = None,
) -> BoundaryResidualReport:
    k0_target = np.log(capital_0 / local_system.anchor.capital_bar)
    y0 = result.sol(np.array([result.x_mesh[0]]))[:, 0]
    initial_residual = np.array([y0[0] - k0_target, y0[1] - tau_0])
    initial_scaled = scaled_norm(initial_residual, [y0[:2], np.array([k0_target, tau_0])])

    yT = result.sol(np.array([result.x_mesh[-1]]))[:, 0]
    capital_T = capital_from_log(yT[0], local_system.anchor.capital_bar)
    tau_T = yT[1]
    if terminal_convention == "lq_stable_manifold":
        tail = lq_stable_manifold_costates(capital_T, tau_T, local_system, solution, z=z_T, x=x_T)
    elif terminal_convention == "crude":
        tail = crude_costates()
    else:
        raise ValueError(f"Unknown terminal_convention: {terminal_convention!r}")
    terminal_residual = np.array([yT[2] - tail.ell, yT[3] - tail.m])
    terminal_scaled = scaled_norm(terminal_residual, [yT[2:], np.array([tail.ell, tail.m])])

    return BoundaryResidualReport(
        initial_scaled_residual=initial_scaled,
        terminal_scaled_residual=terminal_scaled,
        max_scaled_residual=max(initial_scaled, terminal_scaled),
    )


# --------------------------------------------------------------------------
# D2 mandatory repair #4: componentwise manual RHS reconstruction,
# algebraically separate from characteristic_rates/characteristic_rhs_vectorized.
# --------------------------------------------------------------------------


def manual_k_dot(z: float, x: float, capital: float, tau: float, p: PrimitiveParameters) -> float:
    """dot(k) = g(state). Calls evaluate_smooth_branch directly."""

    state = evaluate_smooth_branch(z, x, capital, tau, p)
    return state.capital_growth


def manual_tau_dot(z: float, x: float, capital: float, tau: float, m: float, p: PrimitiveParameters) -> float:
    """dot(tau) = nu = kappa_tau*m/Y. Calls evaluate_smooth_branch directly."""

    state = evaluate_smooth_branch(z, x, capital, tau, p)
    return p.tax_adjustment_scale * m / state.output


def manual_ell_dot(
    z: float, x: float, z_bar: float, x_bar: float, capital: float, tau: float, ell: float, m: float, p: PrimitiveParameters
) -> float:
    """dot(ell) = [r0-g-K*g_K]*ell - [F_K - nu^2*Y_K/(2*kappa_tau)]. Calls
    evaluate_smooth_branch, capital_derivatives, and safe_rate directly."""

    state = evaluate_smooth_branch(z, x, capital, tau, p)
    derivs = capital_derivatives(state, p)
    r0 = safe_rate(z, x, z_bar, x_bar, p)
    kappa_tau = p.tax_adjustment_scale
    nu = kappa_tau * m / state.output
    return (r0 - state.capital_growth - capital * derivs.capital_growth_K) * ell - (
        derivs.fiscal_resources_K - (nu * nu) * derivs.output_K / (2.0 * kappa_tau)
    )


def manual_m_dot(
    z: float, x: float, z_bar: float, x_bar: float, capital: float, tau: float, ell: float, m: float, p: PrimitiveParameters
) -> float:
    """dot(m) = r0*m + B*(ell-1). Calls evaluate_smooth_branch and safe_rate directly."""

    state = evaluate_smooth_branch(z, x, capital, tau, p)
    r0 = safe_rate(z, x, z_bar, x_bar, p)
    return r0 * m + state.tax_base * (ell - 1.0)


@dataclass(frozen=True)
class ComponentwiseRhsReport:
    manual: dict[str, float]
    composite: dict[str, float]
    max_abs_difference: dict[str, float]
    all_agree: bool
    tolerance: float


def componentwise_manual_rhs_check(
    k: float,
    tau: float,
    ell: float,
    m: float,
    z: float,
    x: float,
    z_bar: float,
    x_bar: float,
    capital_bar: float,
    p: PrimitiveParameters,
    tolerance: float = 1e-9,
) -> ComponentwiseRhsReport:
    """Compare the four standalone `manual_*_dot` reconstructions above
    against `characteristic_rates`' composite RHS at ONE economic point.
    Unlike `independent_ode_residual` (which also uses
    `characteristic_rhs_vectorized` for its comparator), the manual side
    here never calls `characteristic_rates`/`characteristic_rhs_vectorized`
    at all -- see test_residuals.py's perturbation tests, which corrupt one
    manual formula at a time and confirm `all_agree` goes False."""

    capital = capital_from_log(k, capital_bar)
    composite = characteristic_rates(k, tau, ell, m, z_bar, x_bar, capital_bar, p, z=z, x=x)
    composite_values = {"k_dot": composite.k_dot, "tau_dot": composite.tau_dot, "ell_dot": composite.ell_dot, "m_dot": composite.m_dot}
    manual_values = {
        "k_dot": manual_k_dot(z, x, capital, tau, p),
        "tau_dot": manual_tau_dot(z, x, capital, tau, m, p),
        "ell_dot": manual_ell_dot(z, x, z_bar, x_bar, capital, tau, ell, m, p),
        "m_dot": manual_m_dot(z, x, z_bar, x_bar, capital, tau, ell, m, p),
    }
    max_abs_difference = {name: abs(manual_values[name] - composite_values[name]) for name in manual_values}
    return ComponentwiseRhsReport(
        manual=manual_values,
        composite=composite_values,
        max_abs_difference=max_abs_difference,
        all_agree=all(value <= tolerance for value in max_abs_difference.values()),
        tolerance=tolerance,
    )


# --------------------------------------------------------------------------
# D2: independent varpi along-path residual and the budget-separation
# accounting identity from the falsification/checking matrix.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VarpiAlongPathResidualReport:
    max_scaled_residual: float
    n_off_mesh_points: int
    finite_difference_step: float


def independent_varpi_along_path_residual(
    varpi_recovery: OpportunityValueRecovery,
    local_system: LocalSystem,
    exogenous_path: ExogenousEvaluator,
    n_off_mesh_points: int = 401,
    fd_step_fraction: float = 1e-4,
) -> VarpiAlongPathResidualReport:
    """Independent off-mesh check of varpi_dot = rho*varpi - (r0-rho)/rho:
    reconstructed here from `r0_along_path`/rho directly, never by calling
    `recover_varpi`'s own internal ODE right-hand side, and evaluated at
    points off `varpi_recovery.t_grid` via its dense (solve_ivp
    dense_output) interpolant `varpi_of_t`."""

    rho = local_system.parameters.rho
    horizon = varpi_recovery.tail_horizon
    h = max(fd_step_fraction * horizon, 1e-8)
    off_mesh = horizon * (np.arange(n_off_mesh_points) + 0.5) / n_off_mesh_points
    off_mesh = np.clip(off_mesh, h, horizon - h)

    varpi_plus = varpi_recovery.varpi_of_t(off_mesh + h)
    varpi_minus = varpi_recovery.varpi_of_t(off_mesh - h)
    varpi_mid = varpi_recovery.varpi_of_t(off_mesh)
    fd_derivative = (varpi_plus - varpi_minus) / (2.0 * h)

    r0_mid = r0_along_path(off_mesh, local_system, exogenous_path)
    exact_rhs = rho * varpi_mid - (r0_mid - rho) / rho

    residual = fd_derivative - exact_rhs
    return VarpiAlongPathResidualReport(
        max_scaled_residual=scaled_norm(residual, [fd_derivative, exact_rhs]),
        n_off_mesh_points=n_off_mesh_points,
        finite_difference_step=float(h),
    )


@dataclass(frozen=True)
class BudgetSeparationResidualReport:
    max_scaled_residual: float


def budget_separation_residual(exogenous_resources: ExogenousResourcesPath) -> BudgetSeparationResidualReport:
    """Falsification/checking-matrix row 'Budget separation': dot(N)+dot(J)
    = r0*(N+J)-c pointwise (Route 1 section 1.3; R08 dossier proof roadmap
    step 3). Central finite difference (np.gradient) of the ALREADY-
    recovered N (budget-ODE route) and J arrays on their shared t_grid,
    compared against the identity's right-hand side built from the
    ALSO-already-recovered r0 and consumption paths -- an accounting cross-
    check on recovered outputs, distinct from (and less tight than) the
    dedicated off-mesh characteristic-BVP residual above."""

    t_grid = exogenous_resources.t_grid
    n_path = exogenous_resources.n_path_budget_ode
    j_path = exogenous_resources.j_path
    r0_path = exogenous_resources.r0_path
    c_path = exogenous_resources.consumption_path

    n_dot = np.gradient(n_path, t_grid)
    j_dot = np.gradient(j_path, t_grid)
    lhs = n_dot + j_dot
    rhs = r0_path * (n_path + j_path) - c_path
    residual = lhs - rhs
    return BudgetSeparationResidualReport(max_scaled_residual=scaled_norm(residual, [lhs, rhs]))
