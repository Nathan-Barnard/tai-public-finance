"""Post-processing: recover J two independent ways, then X, c, N, and (D2)
the opportunity-value term varpi.

Both J routes use the SAME declared terminal tail value (CS002 handoff: "The
discounted-flow and along-path-HJB reconstructions must use the same
declared terminal value; agreement produced by different hidden tail
conventions is not a pass.") -- `recover_j` takes one `tail_value` and feeds
it to both.

- Discounted flow integral: J_0 = int_0^T D_0t*[F_t - Y_t*nu_t^2/(2*kappa_tau)]dt
  + D_0T*tail_value, D_0t=exp(-int_0^t r0(u)du) (D0-D1 frozen states:
  r0=rho identically, so this is exactly exp(-rho*t); trapezoidal
  quadrature integrates a constant exactly, so the D2 generalization below
  reproduces the D0-D1 numbers bit-for-bit when exogenous_path=None).
- Along-path HJB ODE (R08 dossier, eq. after D1): J_dot = r0*J - F +
  Y/(2*kappa_tau)*nu^2, integrated BACKWARD from J(T)=tail_value.

CS002 D2 extension: every recovery function below takes an OPTIONAL
`exogenous_path` (a `t -> (z(t), x(t))` evaluator, e.g. an
`exogenous.ExogenousPath`); omitting it (the D0-D1 call signature)
evaluates everywhere at the frozen anchor (z_bar, x_bar) exactly as before,
since `safe_rate(z_bar, x_bar, z_bar, x_bar, p) == rho` exactly (both
`z_bar-z_bar` and `x_bar-x_bar` are exactly 0.0). D2 mandatory repair #2
(D0-D1 review finding): `recover_comprehensive_resources`'s N ODE
previously selected the NEXT precomputed node via `np.searchsorted` rather
than evaluating the BVP's own continuous interpolant `result.sol(t)`,
producing an unnecessary O(mesh spacing) step-function error (~6.7e-6
relative in the frozen D1 fixture). It and every new D2 ODE right-hand side
below call `result.sol(t)` directly at each adaptive solver step.

In the frozen-common-state case X=N+J is exactly constant along the path
(X_dot=(r0-rho)X=0 since r0=rho identically), so c=rho*X_0 is constant too;
N(t) still varies, recovered by forward-integrating the public budget. That
X(t) computed as N(t)+J(t) stays at X_0 is itself an independent internal-
consistency check, not an assumption -- see `recover_comprehensive_resources`.
D2 does NOT impose X constancy (task: "Do not impose constancy of X, which
was special to D1") -- see `recover_exogenous_resources`, whose two X routes
(ODE and exponential/quadrature integral) and two N routes (budget ODE and
the algebraic N=X-J identity) generalize D1's "X stays constant" check into
its non-degenerate form: the same underlying budget-separation identity
dot(N)+dot(J)=r0*(N+J)-c.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

from ..cs001_lq_anchor.equations import LocalSystem
from .bvp import BvpSolveResult
from .exogenous import ExogenousEvaluator, current_state
from .model import capital_from_log
from ..primitives import evaluate_smooth_branch, safe_rate


def r0_along_path(t_grid: np.ndarray, local_system: LocalSystem, exogenous_path: ExogenousEvaluator | None = None) -> np.ndarray:
    """r0(t) along the exogenous path. Constant at rho (bit-for-bit) when
    exogenous_path is None, since safe_rate(z_bar, x_bar, z_bar, x_bar, p)
    is exactly rho; generally time-varying otherwise (D2)."""

    p = local_system.parameters
    anchor = local_system.anchor
    z_t, x_t = current_state(t_grid, anchor.z_bar, anchor.x_bar, exogenous_path)
    return np.array([safe_rate(z, x, anchor.z_bar, anchor.x_bar, p) for z, x in zip(z_t, x_t)])


def cumulative_discount_factor(r0_grid: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """D_0t = exp(-integral_0^t r0(u) du), via cumulative trapezoidal
    quadrature anchored at D_{t_grid[0]}=1. Exact (not merely accurate) when
    r0_grid is constant, since the trapezoidal rule integrates a degree-0
    polynomial without discretization error."""

    cumulative_integral = cumulative_trapezoid(r0_grid, t_grid, initial=0.0)
    return np.exp(-cumulative_integral)


def _nu_and_state(k: float, tau: float, m: float, local_system: LocalSystem, *, z: float | None = None, x: float | None = None):
    anchor = local_system.anchor
    p = local_system.parameters
    current_z = anchor.z_bar if z is None else z
    current_x = anchor.x_bar if x is None else x
    capital = capital_from_log(k, anchor.capital_bar)
    state = evaluate_smooth_branch(current_z, current_x, capital, tau, p)
    nu = p.tax_adjustment_scale * m / state.output
    return nu, state


def flow_integrand(
    path: np.ndarray, local_system: LocalSystem, t_grid: np.ndarray | None = None, exogenous_path: ExogenousEvaluator | None = None
) -> np.ndarray:
    """F_t - Y_t*nu_t**2/(2*kappa_tau), vectorized over path columns (4, n)."""

    anchor = local_system.anchor
    kappa_tau = local_system.parameters.tax_adjustment_scale
    n = path.shape[1]
    t_for_zx = t_grid if t_grid is not None else np.zeros(n)
    z_at_t, x_at_t = current_state(t_for_zx, anchor.z_bar, anchor.x_bar, exogenous_path)
    out = np.empty(n)
    for i in range(n):
        k, tau, _ell, m = path[:, i]
        nu, state = _nu_and_state(k, tau, m, local_system, z=float(z_at_t[i]), x=float(x_at_t[i]))
        out[i] = state.fiscal_resources - state.output * nu * nu / (2.0 * kappa_tau)
    return out


def recover_j0_via_flow_integral(
    result: BvpSolveResult,
    local_system: LocalSystem,
    tail_value: float,
    horizon: float,
    exogenous_path: ExogenousEvaluator | None = None,
    n_points: int = 4001,
) -> float:
    t_grid = np.linspace(0.0, horizon, n_points)
    path = result.sol(t_grid)
    flow = flow_integrand(path, local_system, t_grid=t_grid, exogenous_path=exogenous_path)
    r0_grid = r0_along_path(t_grid, local_system, exogenous_path)
    discount = cumulative_discount_factor(r0_grid, t_grid)
    integral = float(np.trapezoid(discount * flow, t_grid))
    return integral + float(discount[-1]) * tail_value


def recover_j_path_via_hjb_ode(
    result: BvpSolveResult,
    local_system: LocalSystem,
    tail_value: float,
    horizon: float,
    t_grid: np.ndarray,
    exogenous_path: ExogenousEvaluator | None = None,
) -> np.ndarray:
    """Backward integration of J_dot=r0*J-F+Y/(2*kappa_tau)*nu**2 from J(T)=tail_value."""

    p = local_system.parameters
    anchor = local_system.anchor
    kappa_tau = p.tax_adjustment_scale

    def rhs(t: float, j: np.ndarray) -> list[float]:
        k, tau, _ell, m = result.sol(np.array([t]))[:, 0]
        z_t, x_t = current_state(np.array([t]), anchor.z_bar, anchor.x_bar, exogenous_path)
        z_i, x_i = float(z_t[0]), float(x_t[0])
        nu, state = _nu_and_state(k, tau, m, local_system, z=z_i, x=x_i)
        r0 = safe_rate(z_i, x_i, anchor.z_bar, anchor.x_bar, p)
        return [r0 * j[0] - state.fiscal_resources + state.output * nu * nu / (2.0 * kappa_tau)]

    solution = solve_ivp(rhs, (horizon, 0.0), [tail_value], t_eval=t_grid[::-1], rtol=1e-11, atol=1e-13, method="DOP853")
    if not solution.success:
        raise RuntimeError(f"Along-path HJB backward integration for J failed: {solution.message}")
    return solution.y[0, ::-1].copy()


@dataclass(frozen=True)
class JRecovery:
    tail_kind: str  # "quadratic" | "anchor"
    tail_value: float
    t_grid: np.ndarray
    j_path_hjb: np.ndarray
    j0_flow_integral: float

    @property
    def j0_hjb(self) -> float:
        return float(self.j_path_hjb[0])

    @property
    def route_disagreement(self) -> float:
        return abs(self.j0_flow_integral - self.j0_hjb)

    @property
    def route_disagreement_relative(self) -> float:
        return self.route_disagreement / (1.0 + abs(self.j0_hjb))


def recover_j(
    result: BvpSolveResult,
    local_system: LocalSystem,
    tail_kind: str,
    tail_value: float,
    horizon: float,
    exogenous_path: ExogenousEvaluator | None = None,
    n_grid_points: int = 401,
) -> JRecovery:
    t_grid = np.linspace(0.0, horizon, n_grid_points)
    j_path_hjb = recover_j_path_via_hjb_ode(result, local_system, tail_value, horizon, t_grid, exogenous_path=exogenous_path)
    j0_flow = recover_j0_via_flow_integral(result, local_system, tail_value, horizon, exogenous_path=exogenous_path)
    return JRecovery(tail_kind=tail_kind, tail_value=tail_value, t_grid=t_grid, j_path_hjb=j_path_hjb, j0_flow_integral=j0_flow)


@dataclass(frozen=True)
class ComprehensiveResourcesPath:
    net_worth_0: float
    x_0: float
    consumption: float  # constant, = rho * x_0 -- exact only at frozen common states (D0-D1)
    t_grid: np.ndarray
    j_path: np.ndarray
    n_path: np.ndarray
    x_path: np.ndarray  # n_path + j_path; should be ~constant at x_0
    x_constancy_max_abs_deviation: float
    x_constancy_max_rel_deviation: float


def recover_comprehensive_resources(result: BvpSolveResult, local_system: LocalSystem, j_recovery: JRecovery, net_worth_0: float) -> ComprehensiveResourcesPath:
    """D0-D1 frozen-common-state recovery: X=N+J is exactly constant here
    because r0=rho identically, so a single scalar `consumption` and an
    "X stays constant" check are the right objects. For D2's state-dependent
    r0 (X genuinely time-varying), use `recover_exogenous_resources` instead
    -- this function is NOT a special case reachable by passing an
    exogenous_path here; it has no such parameter, by design, since a
    "constant consumption" field would silently mislabel a D2 result."""

    p = local_system.parameters
    rho = p.rho
    kappa_tau = p.tax_adjustment_scale
    t_grid = j_recovery.t_grid
    j_path = j_recovery.j_path_hjb
    x_0 = net_worth_0 + float(j_path[0])
    consumption = rho * x_0

    def rhs(t: float, n: np.ndarray) -> list[float]:
        # D2 mandatory repair #2: evaluate the BVP's own continuous
        # interpolant at the solver's actual query time, not the nearest
        # precomputed node (np.searchsorted against a fixed dense grid).
        k, tau, _ell, m = result.sol(np.array([t]))[:, 0]
        nu, state = _nu_and_state(k, tau, m, local_system)
        return [rho * n[0] + state.fiscal_resources - consumption - state.output * nu * nu / (2.0 * kappa_tau)]

    solution = solve_ivp(rhs, (0.0, t_grid[-1]), [net_worth_0], t_eval=t_grid, rtol=1e-11, atol=1e-13, method="DOP853", max_step=max(t_grid[-1] / 200.0, 1e-3))
    if not solution.success:
        raise RuntimeError(f"Public-budget forward integration for N failed: {solution.message}")
    n_path = solution.y[0].copy()
    x_path = n_path + j_path

    abs_dev = np.abs(x_path - x_0)
    return ComprehensiveResourcesPath(
        net_worth_0=net_worth_0,
        x_0=x_0,
        consumption=consumption,
        t_grid=t_grid,
        j_path=j_path,
        n_path=n_path,
        x_path=x_path,
        x_constancy_max_abs_deviation=float(abs_dev.max()),
        x_constancy_max_rel_deviation=float(abs_dev.max() / (1.0 + abs(x_0))),
    )


@dataclass(frozen=True)
class ExogenousResourcesPath:
    """CS002 D2 two-route recovery of X (and, from it, c and N) along a path
    with state-dependent r0(t) -- the non-degenerate generalization of
    `ComprehensiveResourcesPath`, whose "X is exactly constant" special case
    held only because r0=rho identically at frozen common states.

    X is recovered both from its scalar ODE (X_dot=(r0-rho)X) and from its
    exponential/quadrature integral (X_t=X_0*exp(int_0^t(r0-rho)du));
    `x_routes_max_rel_deviation` is their disagreement. N is recovered both
    from the forward public-budget ODE (as in D1, continuous-path-repaired,
    now with time-varying r0(t) and c(t)=rho*X(t)) and ALGEBRAICALLY as
    X(t)-J(t); `n_routes_max_rel_deviation` is their disagreement -- the D2
    analogue of D1's "X stays constant" check, testing the same underlying
    budget-separation identity dot(N)+dot(J)=r0*(N+J)-c in its genuinely
    time-varying form."""

    net_worth_0: float
    x_0: float
    t_grid: np.ndarray
    r0_path: np.ndarray
    j_path: np.ndarray
    x_path_ode: np.ndarray
    x_path_exponential: np.ndarray
    x_routes_max_abs_deviation: float
    x_routes_max_rel_deviation: float
    consumption_path: np.ndarray
    n_path_budget_ode: np.ndarray
    n_path_algebraic: np.ndarray
    n_routes_max_abs_deviation: float
    n_routes_max_rel_deviation: float


def recover_exogenous_resources(
    result: BvpSolveResult,
    local_system: LocalSystem,
    j_recovery: JRecovery,
    net_worth_0: float,
    exogenous_path: ExogenousEvaluator,
) -> ExogenousResourcesPath:
    p = local_system.parameters
    anchor = local_system.anchor
    rho = p.rho
    kappa_tau = p.tax_adjustment_scale
    t_grid = j_recovery.t_grid
    horizon = float(t_grid[-1])
    j_path = j_recovery.j_path_hjb
    x_0 = net_worth_0 + float(j_path[0])

    r0_grid = r0_along_path(t_grid, local_system, exogenous_path)

    def x_rhs(t: float, x: np.ndarray) -> list[float]:
        r0_t = float(r0_along_path(np.array([t]), local_system, exogenous_path)[0])
        return [(r0_t - rho) * x[0]]

    x_solution = solve_ivp(x_rhs, (0.0, horizon), [x_0], t_eval=t_grid, dense_output=True, rtol=1e-12, atol=1e-14, method="DOP853")
    if not x_solution.success:
        raise RuntimeError(f"Forward integration for X failed: {x_solution.message}")
    x_path_ode = x_solution.y[0].copy()
    x_continuous = x_solution.sol

    cumulative_excess = cumulative_trapezoid(r0_grid - rho, t_grid, initial=0.0)
    x_path_exponential = x_0 * np.exp(cumulative_excess)

    x_abs_dev = np.abs(x_path_ode - x_path_exponential)
    x_routes_max_abs_deviation = float(x_abs_dev.max())
    x_routes_max_rel_deviation = float(x_abs_dev.max() / (1.0 + np.max(np.abs(x_path_ode))))

    consumption_path = rho * x_path_ode

    def n_rhs(t: float, n: np.ndarray) -> list[float]:
        k, tau, _ell, m = result.sol(np.array([t]))[:, 0]
        z_t, x_t = current_state(np.array([t]), anchor.z_bar, anchor.x_bar, exogenous_path)
        z_i, x_i = float(z_t[0]), float(x_t[0])
        nu, state = _nu_and_state(k, tau, m, local_system, z=z_i, x=x_i)
        r0_t = safe_rate(z_i, x_i, anchor.z_bar, anchor.x_bar, p)
        c_t = float(rho * x_continuous(t)[0])
        return [r0_t * n[0] + state.fiscal_resources - c_t - state.output * nu * nu / (2.0 * kappa_tau)]

    n_solution = solve_ivp(n_rhs, (0.0, horizon), [net_worth_0], t_eval=t_grid, rtol=1e-11, atol=1e-13, method="DOP853", max_step=max(horizon / 200.0, 1e-3))
    if not n_solution.success:
        raise RuntimeError(f"Public-budget forward integration for N failed: {n_solution.message}")
    n_path_budget_ode = n_solution.y[0].copy()
    n_path_algebraic = x_path_ode - j_path

    n_abs_dev = np.abs(n_path_budget_ode - n_path_algebraic)
    n_routes_max_abs_deviation = float(n_abs_dev.max())
    n_routes_max_rel_deviation = float(n_abs_dev.max() / (1.0 + np.max(np.abs(n_path_algebraic))))

    return ExogenousResourcesPath(
        net_worth_0=net_worth_0,
        x_0=x_0,
        t_grid=t_grid,
        r0_path=r0_grid,
        j_path=j_path,
        x_path_ode=x_path_ode,
        x_path_exponential=x_path_exponential,
        x_routes_max_abs_deviation=x_routes_max_abs_deviation,
        x_routes_max_rel_deviation=x_routes_max_rel_deviation,
        consumption_path=consumption_path,
        n_path_budget_ode=n_path_budget_ode,
        n_path_algebraic=n_path_algebraic,
        n_routes_max_abs_deviation=n_routes_max_abs_deviation,
        n_routes_max_rel_deviation=n_routes_max_rel_deviation,
    )


@dataclass(frozen=True)
class OpportunityValueRecovery:
    """CS002 D2: the opportunity-value term varpi solves
    varpi_dot = rho*varpi - (r0-rho)/rho along the exogenous path (R08/R09
    dossier, eq. after (D1); research-notes Route 1 section 1.3). The
    2-state PDE (rho - mu_z d_z - mu_x d_x) varpi = (r0-rho)/rho reduces
    EXACTLY to this 1-D along-path ODE once (z(t), x(t)) are known in closed
    form, because (r0(z(t),x(t))-rho)/rho is then a known function of t
    alone -- no 2-D PDE solve is performed or needed; the along-path
    residual (residuals.py) is the independent evidence that this reduction
    was applied correctly.

    Both routes below share the SAME declared terminal tail_value at the
    SAME horizon (mirroring recover_j's two routes sharing one declared
    terminal value):

    - route A (`varpi_path_ode`): backward ODE integration from
      varpi(T)=tail_value;
    - route B (`varpi_path_quadrature`): forward discounted quadrature
      varpi(t) = int_t^T exp(-rho(u-t))*(r0(u)-rho)/rho du +
      exp(-rho(T-t))*tail_value, using the identical tail_value/T.

    The declared tail convention used in the D2 material run is
    tail_value=0 at the run's own horizon, justified (not silently assumed)
    by a horizon-sensitivity study: the source (r0-rho)/rho has already
    decayed by several OU half-lives at that horizon, so varpi(0) is checked
    to stabilize as the horizon is extended further (see experiment_d2.py's
    horizon/tail sensitivity report)."""

    t_grid: np.ndarray
    tail_horizon: float
    tail_value: float
    varpi_path_ode: np.ndarray
    varpi_path_quadrature: np.ndarray
    route_disagreement_max_abs: float
    route_disagreement_max_rel: float
    varpi_of_t: Callable[[np.ndarray], np.ndarray]  # route A's own continuous (dense_output) interpolant

    @property
    def varpi_0(self) -> float:
        return float(self.varpi_path_ode[0])


def recover_varpi(
    local_system: LocalSystem,
    exogenous_path: ExogenousEvaluator,
    horizon: float,
    tail_value: float,
    n_grid_points: int = 401,
) -> OpportunityValueRecovery:
    rho = local_system.parameters.rho
    t_grid = np.linspace(0.0, horizon, n_grid_points)
    r0_grid = r0_along_path(t_grid, local_system, exogenous_path)
    source = (r0_grid - rho) / rho

    def rhs(t: float, v: np.ndarray) -> list[float]:
        r0_t = float(r0_along_path(np.array([t]), local_system, exogenous_path)[0])
        return [rho * v[0] - (r0_t - rho) / rho]

    ode_solution = solve_ivp(rhs, (horizon, 0.0), [tail_value], t_eval=t_grid[::-1], dense_output=True, rtol=1e-12, atol=1e-14, method="DOP853")
    if not ode_solution.success:
        raise RuntimeError(f"Backward ODE integration for varpi failed: {ode_solution.message}")
    varpi_path_ode = ode_solution.y[0, ::-1].copy()
    varpi_dense = ode_solution.sol

    def varpi_of_t(t: np.ndarray) -> np.ndarray:
        """Always takes and returns a 1-D array, however many points."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        return np.asarray(varpi_dense(t_arr)).reshape(-1)

    varpi_path_quadrature = np.empty(n_grid_points)
    for i, t_i in enumerate(t_grid):
        u = t_grid[i:]
        if u.size > 1:
            integrand = np.exp(-rho * (u - t_i)) * source[i:]
            integral = float(np.trapezoid(integrand, u))
        else:
            integral = 0.0
        varpi_path_quadrature[i] = integral + math.exp(-rho * (horizon - t_i)) * tail_value

    disagreement = np.abs(varpi_path_ode - varpi_path_quadrature)
    return OpportunityValueRecovery(
        t_grid=t_grid,
        tail_horizon=horizon,
        tail_value=tail_value,
        varpi_path_ode=varpi_path_ode,
        varpi_path_quadrature=varpi_path_quadrature,
        route_disagreement_max_abs=float(disagreement.max()),
        route_disagreement_max_rel=float(disagreement.max() / (1.0 + np.max(np.abs(varpi_path_ode)))),
        varpi_of_t=varpi_of_t,
    )
