"""Post-processing: recover J two independent ways, then X, c, and N.

Both J routes use the SAME declared terminal tail value (CS002 handoff: "The
discounted-flow and along-path-HJB reconstructions must use the same
declared terminal value; agreement produced by different hidden tail
conventions is not a pass.") -- `recover_j` takes one `tail_value` and feeds
it to both.

- Discounted flow integral: J_0 = int_0^T D_0t*[F_t - Y_t*nu_t^2/(2*kappa_tau)]dt
  + D_0T*tail_value, D_0t=exp(-rho*t) (frozen states: r0=rho exactly).
- Along-path HJB ODE (R08 dossier, eq. after D1): J_dot = r0*J - F +
  Y/(2*kappa_tau)*nu^2, integrated BACKWARD from J(T)=tail_value.

In the frozen-common-state case X=N+J is exactly constant along the path
(X_dot=(r0-rho)X=0 since r0=rho identically), so c=rho*X_0 is constant too;
N(t) still varies, recovered by forward-integrating the public budget. That
X(t) computed as N(t)+J(t) stays at X_0 is itself an independent internal-
consistency check, not an assumption -- see `recover_comprehensive_resources`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..cs001_lq_anchor.equations import LocalSystem
from .bvp import BvpSolveResult
from .model import capital_from_log
from ..primitives import evaluate_smooth_branch


def _nu_and_state(k: float, tau: float, m: float, local_system: LocalSystem):
    anchor = local_system.anchor
    p = local_system.parameters
    capital = capital_from_log(k, anchor.capital_bar)
    state = evaluate_smooth_branch(anchor.z_bar, anchor.x_bar, capital, tau, p)
    nu = p.tax_adjustment_scale * m / state.output
    return nu, state


def flow_integrand(path: np.ndarray, local_system: LocalSystem) -> np.ndarray:
    """F_t - Y_t*nu_t**2/(2*kappa_tau), vectorized over path columns (4, n)."""

    kappa_tau = local_system.parameters.tax_adjustment_scale
    n = path.shape[1]
    out = np.empty(n)
    for i in range(n):
        k, tau, _ell, m = path[:, i]
        nu, state = _nu_and_state(k, tau, m, local_system)
        out[i] = state.fiscal_resources - state.output * nu * nu / (2.0 * kappa_tau)
    return out


def recover_j0_via_flow_integral(result: BvpSolveResult, local_system: LocalSystem, tail_value: float, horizon: float, n_points: int = 4001) -> float:
    rho = local_system.parameters.rho
    t_grid = np.linspace(0.0, horizon, n_points)
    path = result.sol(t_grid)
    flow = flow_integrand(path, local_system)
    discount = np.exp(-rho * t_grid)
    integral = float(np.trapezoid(discount * flow, t_grid))
    return integral + float(np.exp(-rho * horizon)) * tail_value


def recover_j_path_via_hjb_ode(result: BvpSolveResult, local_system: LocalSystem, tail_value: float, horizon: float, t_grid: np.ndarray) -> np.ndarray:
    """Backward integration of J_dot=r0*J-F+Y/(2*kappa_tau)*nu**2 from J(T)=tail_value."""

    rho = local_system.parameters.rho  # = r0 exactly at frozen common states
    kappa_tau = local_system.parameters.tax_adjustment_scale

    def rhs(t: float, j: np.ndarray) -> list[float]:
        k, tau, _ell, m = result.sol(np.array([t]))[:, 0]
        nu, state = _nu_and_state(k, tau, m, local_system)
        return [rho * j[0] - state.fiscal_resources + state.output * nu * nu / (2.0 * kappa_tau)]

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


def recover_j(result: BvpSolveResult, local_system: LocalSystem, tail_kind: str, tail_value: float, horizon: float, n_grid_points: int = 401) -> JRecovery:
    t_grid = np.linspace(0.0, horizon, n_grid_points)
    j_path_hjb = recover_j_path_via_hjb_ode(result, local_system, tail_value, horizon, t_grid)
    j0_flow = recover_j0_via_flow_integral(result, local_system, tail_value, horizon)
    return JRecovery(tail_kind=tail_kind, tail_value=tail_value, t_grid=t_grid, j_path_hjb=j_path_hjb, j0_flow_integral=j0_flow)


@dataclass(frozen=True)
class ComprehensiveResourcesPath:
    net_worth_0: float
    x_0: float
    consumption: float  # constant, = rho * x_0
    t_grid: np.ndarray
    j_path: np.ndarray
    n_path: np.ndarray
    x_path: np.ndarray  # n_path + j_path; should be ~constant at x_0
    x_constancy_max_abs_deviation: float
    x_constancy_max_rel_deviation: float


def recover_comprehensive_resources(result: BvpSolveResult, local_system: LocalSystem, j_recovery: JRecovery, net_worth_0: float) -> ComprehensiveResourcesPath:
    p = local_system.parameters
    rho = p.rho
    kappa_tau = p.tax_adjustment_scale
    t_grid = j_recovery.t_grid
    j_path = j_recovery.j_path_hjb
    x_0 = net_worth_0 + float(j_path[0])
    consumption = rho * x_0

    path = result.sol(t_grid)

    def rhs(t: float, n: np.ndarray) -> list[float]:
        idx = np.searchsorted(t_grid, t)
        idx = int(np.clip(idx, 0, t_grid.size - 1))
        k, tau, _ell, m = path[:, idx]
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
