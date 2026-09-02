"""Two-point BVP scaffolding: a generic scipy.solve_bvp wrapper, shared
verbatim by the manufactured-solution interface check and the real
economic characteristic system (frozen-state D0-D1 or D2's deterministic
mean-reverting exogenous path), plus the economic RHS/boundary-condition
construction and an LQ-path initial guess.

Using the SAME `solve_two_point_bvp` wrapper for both the manufactured test
and the economic solve is what makes the manufactured check real evidence
about this code path (CS002 Block D0 requirement 3), rather than a
disconnected toy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.integrate import solve_bvp

from ..cs001_lq_anchor.equations import LocalSystem
from ..cs001_lq_anchor.solver import LqSolution
from .exogenous import ExogenousEvaluator
from .model import capital_from_log, characteristic_rhs_vectorized, log_from_capital
from .terminal import crude_costates, lq_linear_kt_path, lq_stable_manifold_costates

TerminalConvention = Literal["lq_stable_manifold", "crude"]


@dataclass(frozen=True)
class BvpSolveResult:
    success: bool
    message: str
    sol: Callable[[np.ndarray], np.ndarray]
    x_mesh: np.ndarray
    y_mesh: np.ndarray
    solver_max_rms_residual: float
    n_mesh_nodes: int
    status: int


def solve_two_point_bvp(
    fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    bc: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_mesh: np.ndarray,
    y_guess: np.ndarray,
    tol: float = 1e-10,
    max_nodes: int = 100_000,
) -> BvpSolveResult:
    """Thin, deliberately dumb wrapper: scipy does the collocation, this
    function only reshapes the result. `solver_max_rms_residual` and
    `success` are reported for logging only -- CS002 acceptance #10 requires
    that solver-reported convergence alone never sets the aggregate outcome,
    so nothing downstream may treat these two fields as a pass condition on
    their own (see residuals.py for the independent check that does)."""

    result = solve_bvp(fun, bc, x_mesh, y_guess, tol=tol, max_nodes=max_nodes, verbose=0)
    residuals = np.asarray(result.rms_residuals)
    return BvpSolveResult(
        success=bool(result.success),
        message=str(result.message),
        sol=result.sol,
        x_mesh=result.x,
        y_mesh=result.y,
        solver_max_rms_residual=float(residuals.max()) if residuals.size else float("nan"),
        n_mesh_nodes=int(result.x.size),
        status=int(result.status),
    )


def economic_rhs(local_system: LocalSystem) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    anchor = local_system.anchor
    p = local_system.parameters

    def fun(t: np.ndarray, y: np.ndarray) -> np.ndarray:
        del t  # autonomous under frozen common states
        return characteristic_rhs_vectorized(y, anchor.z_bar, anchor.x_bar, anchor.capital_bar, p)

    return fun


def economic_bc(
    capital_0: float,
    tau_0: float,
    terminal_convention: TerminalConvention,
    local_system: LocalSystem,
    solution: LqSolution,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    k0 = log_from_capital(capital_0, local_system.anchor.capital_bar)

    def bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        capital_T = capital_from_log(yb[0], local_system.anchor.capital_bar)
        tau_T = yb[1]
        if terminal_convention == "lq_stable_manifold":
            tail = lq_stable_manifold_costates(capital_T, tau_T, local_system, solution)
        elif terminal_convention == "crude":
            tail = crude_costates()
        else:
            raise ValueError(f"Unknown terminal_convention: {terminal_convention!r}")
        return np.array([ya[0] - k0, ya[1] - tau_0, yb[2] - tail.ell, yb[3] - tail.m])

    return bc


def lq_path_initial_guess(x_mesh: np.ndarray, k0: float, t0: float, local_system: LocalSystem, solution: LqSolution) -> np.ndarray:
    """The LQ-approximate (k, tau) path from displacement (k0, t0=tau0-1/2),
    propagated by the CS001 real-block closed loop A_rc (dot(k,t)=A_rc.(k,t) --
    valid because frozen z,x deviations are exactly decoupled and stay at 0
    under the closed loop: A's z,x rows are diagonal-only and chi*B*B^T*H only
    ever adds to the tau row), with costates read off the same LQ
    stable-manifold mapping used for the terminal condition itself so the
    guess and the boundary condition it must satisfy are consistent."""

    anchor = local_system.anchor
    kt_path = lq_linear_kt_path(x_mesh, k0, t0, solution)
    y_guess = np.empty((4, x_mesh.size))
    for i in range(x_mesh.size):
        k_t, tdev_t = kt_path[:, i]
        capital_t = capital_from_log(k_t, anchor.capital_bar)
        tau_t = tdev_t + anchor.tax_rate_bar
        tail = lq_stable_manifold_costates(capital_t, tau_t, local_system, solution)
        y_guess[:, i] = [k_t, tau_t, tail.ell, tail.m]
    return y_guess


def economic_rhs_with_exogenous_path(local_system: LocalSystem, exogenous_path: ExogenousEvaluator) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """CS002 D2: the exogenous (z, x) follow a KNOWN closed-form path in t
    rather than sitting frozen at (z_bar, x_bar), so the system is genuinely
    non-autonomous (`t` is no longer discarded). Passing an `exogenous_path`
    that returns (z_bar, x_bar) at every t reproduces `economic_rhs` exactly."""

    anchor = local_system.anchor
    p = local_system.parameters

    def fun(t: np.ndarray, y: np.ndarray) -> np.ndarray:
        return characteristic_rhs_vectorized(y, anchor.z_bar, anchor.x_bar, anchor.capital_bar, p, t=t, exogenous_path=exogenous_path)

    return fun


def economic_bc_with_exogenous_path(
    capital_0: float,
    tau_0: float,
    horizon: float,
    terminal_convention: TerminalConvention,
    local_system: LocalSystem,
    solution: LqSolution,
    exogenous_path: ExogenousEvaluator,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """CS002 D2 generalization of `economic_bc`: the terminal LQ tail is read
    at the ACTUAL terminal exogenous state (z(T), x(T)) from `exogenous_path`
    -- generally nonzero, since T is finite -- rather than the D0-D1 (0, 0)
    deviation. `crude_costates()` is unaffected (it has no state dependence
    to generalize)."""

    k0 = log_from_capital(capital_0, local_system.anchor.capital_bar)
    z_T_arr, x_T_arr = exogenous_path(np.array([horizon]))
    z_T = float(np.asarray(z_T_arr).reshape(-1)[0])
    x_T = float(np.asarray(x_T_arr).reshape(-1)[0])

    def bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        capital_T = capital_from_log(yb[0], local_system.anchor.capital_bar)
        tau_T = yb[1]
        if terminal_convention == "lq_stable_manifold":
            tail = lq_stable_manifold_costates(capital_T, tau_T, local_system, solution, z=z_T, x=x_T)
        elif terminal_convention == "crude":
            tail = crude_costates()
        else:
            raise ValueError(f"Unknown terminal_convention: {terminal_convention!r}")
        return np.array([ya[0] - k0, ya[1] - tau_0, yb[2] - tail.ell, yb[3] - tail.m])

    return bc


def crude_constant_initial_guess(x_mesh: np.ndarray, k0: float, tau0: float) -> np.ndarray:
    """The alternative, deliberately cruder initialization route: the
    displaced (k, tau) held CONSTANT at their initial values with the
    anchor's costates (ell=1, m=0) everywhere -- ignores the LQ dynamics
    entirely. Used only to give continuation a materially different starting
    point for the branch-sensitivity check (CS002 acceptance #8)."""

    y_guess = np.empty((4, x_mesh.size))
    y_guess[0, :] = k0
    y_guess[1, :] = tau0
    y_guess[2, :] = 1.0
    y_guess[3, :] = 0.0
    return y_guess
