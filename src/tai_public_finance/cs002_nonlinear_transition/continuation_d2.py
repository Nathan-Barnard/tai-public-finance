"""Continuation in shock amplitude for CS002 D2's exogenous-state shocks
(pure productivity: z(0) displaced; pure automation: x(0) displaced via its
implied automation_share target), along two materially different routes --
mirroring continuation.py's D0-D1 structure so branch-sensitivity is a
comparison between genuinely different numerics, not the same solve twice:

- "warm_start" (route A, preferred): at each amplitude step, warm-start from
  the PREVIOUS accepted checkpoint's solution; at the first step (always
  amplitude 0, the exact undisplaced anchor) the crude constant-anchor guess
  IS the exact solution, so no separate LQ-informed guess is needed the way
  D1's lq_path_initial_guess was for a directly-displaced (k, tau).
- "crude_direct" (route B): at every amplitude, solve directly from the
  crude constant-anchor guess, never warm-started from a neighbouring
  amplitude.

D2's fiscal state starts undisplaced at every amplitude -- K(0)=K_bar,
tau(0)=tau_bar always; only z(0) or x(0) moves between the anchor and the
shock's target level as `amplitude` goes from 0 to 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..cs001_lq_anchor.equations import LocalSystem
from ..cs001_lq_anchor.solver import LqSolution
from .bvp import (
    BvpSolveResult,
    TerminalConvention,
    crude_constant_initial_guess,
    economic_bc_with_exogenous_path,
    economic_rhs_with_exogenous_path,
    solve_two_point_bvp,
)
from .exogenous import ExogenousPath

ShockDirection = Literal["productivity", "automation"]
ExogenousContinuationRoute = Literal["warm_start", "crude_direct"]


@dataclass(frozen=True)
class ExogenousCheckpoint:
    amplitude: float
    shock_direction: ShockDirection
    z0: float
    x0: float
    exogenous_path: ExogenousPath
    accepted: bool
    result: BvpSolveResult | None
    failure_message: str | None

    def path_at(self, t: np.ndarray) -> np.ndarray:
        if not self.accepted or self.result is None:
            raise RuntimeError(f"Checkpoint at amplitude={self.amplitude} was not accepted; no path to evaluate.")
        return self.result.sol(t)


@dataclass(frozen=True)
class ExogenousContinuationRun:
    route: ExogenousContinuationRoute
    shock_direction: ShockDirection
    terminal_convention: TerminalConvention
    horizon: float
    z0_target: float
    x0_target: float
    checkpoints: list[ExogenousCheckpoint]

    @property
    def all_accepted(self) -> bool:
        return all(c.accepted for c in self.checkpoints)

    @property
    def final(self) -> ExogenousCheckpoint:
        return self.checkpoints[-1]


def run_exogenous_shock_continuation(
    route: ExogenousContinuationRoute,
    shock_direction: ShockDirection,
    z0_target: float,
    x0_target: float,
    amplitudes: list[float],
    horizon: float,
    n_mesh_points: int,
    terminal_convention: TerminalConvention,
    local_system: LocalSystem,
    solution: LqSolution,
    tol: float = 1e-10,
    max_nodes: int = 100_000,
) -> ExogenousContinuationRun:
    if route not in ("warm_start", "crude_direct"):
        raise ValueError(f"Unknown continuation route: {route!r}")

    anchor = local_system.anchor
    p = local_system.parameters
    z_bar, x_bar = anchor.z_bar, anchor.x_bar
    kappa_z, kappa_x = p.kappa_z, p.kappa_x
    capital_0, tau_0 = anchor.capital_bar, anchor.tax_rate_bar

    x_mesh = np.linspace(0.0, horizon, n_mesh_points)
    checkpoints: list[ExogenousCheckpoint] = []
    previous_result: BvpSolveResult | None = None

    for amplitude in amplitudes:
        z0 = z_bar + amplitude * (z0_target - z_bar)
        x0 = x_bar + amplitude * (x0_target - x_bar)
        exogenous_path = ExogenousPath(z0=z0, x0=x0, z_bar=z_bar, x_bar=x_bar, kappa_z=kappa_z, kappa_x=kappa_x)

        if route == "warm_start" and previous_result is not None:
            y_guess = previous_result.sol(x_mesh)
        else:  # route B always, or route A's first (amplitude=0, exact) step
            y_guess = crude_constant_initial_guess(x_mesh, 0.0, tau_0)

        fun = economic_rhs_with_exogenous_path(local_system, exogenous_path)
        bc = economic_bc_with_exogenous_path(capital_0, tau_0, horizon, terminal_convention, local_system, solution, exogenous_path)
        try:
            result = solve_two_point_bvp(fun, bc, x_mesh, y_guess, tol=tol, max_nodes=max_nodes)
        except Exception as error:  # noqa: BLE001 -- retained as a failed checkpoint, not a crash
            checkpoints.append(
                ExogenousCheckpoint(
                    amplitude=amplitude, shock_direction=shock_direction, z0=z0, x0=x0, exogenous_path=exogenous_path,
                    accepted=False, result=None, failure_message=repr(error),
                )
            )
            previous_result = None
            continue

        accepted = result.success
        checkpoints.append(
            ExogenousCheckpoint(
                amplitude=amplitude, shock_direction=shock_direction, z0=z0, x0=x0, exogenous_path=exogenous_path,
                accepted=accepted, result=result, failure_message=None if accepted else result.message,
            )
        )
        previous_result = result if accepted else None

    return ExogenousContinuationRun(
        route=route, shock_direction=shock_direction, terminal_convention=terminal_convention, horizon=horizon,
        z0_target=z0_target, x0_target=x0_target, checkpoints=checkpoints,
    )
