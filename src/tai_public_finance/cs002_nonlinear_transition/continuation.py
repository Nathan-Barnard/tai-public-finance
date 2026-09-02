"""Continuation in displacement amplitude, along two materially different
routes, so branch-sensitivity (CS002 acceptance #8) is a comparison between
genuinely different numerics rather than the same solve run twice.

- "lq_path_continuation" (route A, preferred): at each amplitude step, warm-
  start from the PREVIOUS accepted checkpoint's solution (or, at the first
  step, from the closed-form LQ path); this is standard continuation.
- "crude_direct" (route B): at every amplitude, solve directly from the
  crude constant-anchor guess with no warm start from a neighbouring
  amplitude at all -- deliberately not continuation, so its agreement with
  route A is not guaranteed by construction.

Every accepted checkpoint is retained (not just the final one), in compact
form (mesh + solved values, not the solver's internal iteration history).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..cs001_lq_anchor.equations import LocalSystem
from ..cs001_lq_anchor.solver import LqSolution
from .bvp import BvpSolveResult, TerminalConvention, crude_constant_initial_guess, economic_bc, economic_rhs, lq_path_initial_guess, solve_two_point_bvp
from .model import capital_from_log

ContinuationRoute = str  # "lq_path_continuation" | "crude_direct"


@dataclass(frozen=True)
class Checkpoint:
    amplitude: float
    capital_0: float
    tau_0: float
    accepted: bool
    result: BvpSolveResult | None
    failure_message: str | None

    def path_at(self, t: np.ndarray) -> np.ndarray:
        if not self.accepted or self.result is None:
            raise RuntimeError(f"Checkpoint at amplitude={self.amplitude} was not accepted; no path to evaluate.")
        return self.result.sol(t)


@dataclass(frozen=True)
class ContinuationRun:
    route: ContinuationRoute
    terminal_convention: TerminalConvention
    horizon: float
    delta_k: float
    delta_tau: float
    checkpoints: list[Checkpoint]

    @property
    def all_accepted(self) -> bool:
        return all(c.accepted for c in self.checkpoints)

    @property
    def final(self) -> Checkpoint:
        return self.checkpoints[-1]


def _initial_state(amplitude: float, delta_k: float, delta_tau: float, local_system: LocalSystem) -> tuple[float, float, float]:
    k0 = amplitude * delta_k
    dtau0 = amplitude * delta_tau
    tau0 = local_system.anchor.tax_rate_bar + dtau0
    capital0 = capital_from_log(k0, local_system.anchor.capital_bar)
    return capital0, tau0, dtau0


def run_continuation(
    route: ContinuationRoute,
    delta_k: float,
    delta_tau: float,
    amplitudes: list[float],
    horizon: float,
    n_mesh_points: int,
    terminal_convention: TerminalConvention,
    local_system: LocalSystem,
    solution: LqSolution,
    tol: float = 1e-10,
    max_nodes: int = 100_000,
) -> ContinuationRun:
    if route not in ("lq_path_continuation", "crude_direct"):
        raise ValueError(f"Unknown continuation route: {route!r}")

    x_mesh = np.linspace(0.0, horizon, n_mesh_points)
    fun = economic_rhs(local_system)
    checkpoints: list[Checkpoint] = []
    previous_result: BvpSolveResult | None = None

    for amplitude in amplitudes:
        capital0, tau0, dtau0 = _initial_state(amplitude, delta_k, delta_tau, local_system)
        k0 = amplitude * delta_k

        if route == "lq_path_continuation" and previous_result is not None:
            y_guess = previous_result.sol(x_mesh)
            y_guess[0, 0] = k0
            y_guess[1, 0] = tau0
        elif route == "lq_path_continuation":
            y_guess = lq_path_initial_guess(x_mesh, k0, dtau0, local_system, solution)
        else:  # crude_direct: always from the crude constant guess, never warm-started
            y_guess = crude_constant_initial_guess(x_mesh, k0, tau0)

        bc = economic_bc(capital0, tau0, terminal_convention, local_system, solution)
        try:
            result = solve_two_point_bvp(fun, bc, x_mesh, y_guess, tol=tol, max_nodes=max_nodes)
        except Exception as error:  # noqa: BLE001 -- retained as a failed checkpoint, not a crash
            checkpoints.append(
                Checkpoint(amplitude=amplitude, capital_0=capital0, tau_0=tau0, accepted=False, result=None, failure_message=repr(error))
            )
            previous_result = None
            continue

        accepted = result.success
        checkpoints.append(
            Checkpoint(
                amplitude=amplitude,
                capital_0=capital0,
                tau_0=tau0,
                accepted=accepted,
                result=result,
                failure_message=None if accepted else result.message,
            )
        )
        previous_result = result if accepted else None

    return ContinuationRun(
        route=route, terminal_convention=terminal_convention, horizon=horizon, delta_k=delta_k, delta_tau=delta_tau, checkpoints=checkpoints
    )
