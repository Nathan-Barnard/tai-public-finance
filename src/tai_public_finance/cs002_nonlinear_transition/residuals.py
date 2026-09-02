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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..cs001_lq_anchor.diagnostics import scaled_norm
from ..cs001_lq_anchor.equations import LocalSystem
from ..cs001_lq_anchor.solver import LqSolution
from .bvp import BvpSolveResult, TerminalConvention
from .model import capital_from_log, characteristic_rhs_vectorized
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
    result: BvpSolveResult, local_system: LocalSystem, n_off_mesh_points: int = 401, fd_step_fraction: float = 1e-4
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
    exact_rhs = characteristic_rhs_vectorized(y_mid, anchor.z_bar, anchor.x_bar, anchor.capital_bar, local_system.parameters)

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
) -> BoundaryResidualReport:
    k0_target = np.log(capital_0 / local_system.anchor.capital_bar)
    y0 = result.sol(np.array([result.x_mesh[0]]))[:, 0]
    initial_residual = np.array([y0[0] - k0_target, y0[1] - tau_0])
    initial_scaled = scaled_norm(initial_residual, [y0[:2], np.array([k0_target, tau_0])])

    yT = result.sol(np.array([result.x_mesh[-1]]))[:, 0]
    capital_T = capital_from_log(yT[0], local_system.anchor.capital_bar)
    tau_T = yT[1]
    if terminal_convention == "lq_stable_manifold":
        tail = lq_stable_manifold_costates(capital_T, tau_T, local_system, solution)
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
