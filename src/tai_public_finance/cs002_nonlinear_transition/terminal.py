"""Terminal-tail conventions for the frozen-state characteristic BVP.

Two costate tails (CS002 D0-D1 handoff):

- "lq_stable_manifold" (preferred): read the costates off CS001's quadratic
  local value function J(y) ~= J_bar + j.y + 0.5 y'Hy, y=(z-z_bar, x-x_bar,
  k, t), t=tau-1/2. Its k- and t-partials are USABLE DIRECTLY as J_k and
  J_tau -- t=tau-1/2 is a pure shift so J_tau=J_t exactly -- but ell=J_K is
  a derivative with respect to REAL capital, not log-capital: by the chain
  rule k=log(K/K_bar) gives dJ/dk = K * dJ/dK, so

      ell_lq(K, tau) = J_k(y) / K,        m_lq(K, tau) = J_t(y).

  At the anchor (K=K_bar, tau=tau_bar) this is exactly CS001's own
  j[2]==1, j[3]==0 diagnostic (test_equations.py
  test_linear_fiscal_wealth_capital_and_tax_coefficients_are_exact) -- the
  K-division is a no-op there (K=K_bar=1) but load-bearing away from it.
- "crude": ell(T)=1, m(T)=0, the naive interior-fixed-point tail with no
  state dependence.

Both are boundary CONDITIONS on the costates (used inside the BVP's `bc`
callback). For post-processing J itself, the analogous two TAIL VALUES are:

- "quadratic" (preferred): the same J(y) approximation, evaluated in levels
  (not just its gradient), at the terminal state.
- "anchor": the cruder zeroth-order tail J_bar alone, ignoring the terminal
  displacement entirely.

Every function here takes the same (local_system, solution) pair CS001's own
portfolio.py takes: local_system carries the anchor and the linear fiscal-
wealth vector `j`, solution carries the quadratic coefficient `H`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from ..cs001_lq_anchor.equations import COORDINATES, LocalSystem
from ..cs001_lq_anchor.solver import LqSolution
from .model import log_from_capital

assert COORDINATES == ("z", "x", "k", "tau_deviation"), (
    "CS002's terminal-condition mapping hard-codes indices 2 (k) and 3 (tau_deviation); "
    "CS001's coordinate order changed and this module must be updated to match."
)
_K_INDEX = 2
_TAU_INDEX = 3


@dataclass(frozen=True)
class TerminalCostates:
    ell: float
    m: float


def lq_deviation_vector(capital: float, tau: float, local_system: LocalSystem) -> np.ndarray:
    """y=(z-z_bar, x-x_bar, k, tau-1/2) at frozen common states: only the
    capital/tax components are ever nonzero here."""

    k = log_from_capital(capital, local_system.anchor.capital_bar)
    return np.array([0.0, 0.0, k, tau - 0.5])


def lq_stable_manifold_costates(capital: float, tau: float, local_system: LocalSystem, solution: LqSolution) -> TerminalCostates:
    y = lq_deviation_vector(capital, tau, local_system)
    gradient = local_system.linear_fiscal_wealth + solution.H @ y
    j_k = gradient[_K_INDEX]
    j_tau = gradient[_TAU_INDEX]
    return TerminalCostates(ell=j_k / capital, m=j_tau)


def crude_costates() -> TerminalCostates:
    return TerminalCostates(ell=1.0, m=0.0)


def lq_quadratic_value_tail(capital: float, tau: float, local_system: LocalSystem, solution: LqSolution) -> float:
    """J(y) ~= J_bar + j.y + 0.5 y'Hy, evaluated in LEVEL units of fiscal wealth."""

    y = lq_deviation_vector(capital, tau, local_system)
    j = local_system.linear_fiscal_wealth
    return local_system.anchor.fiscal_wealth_bar + float(j @ y) + 0.5 * float(y @ solution.H @ y)


def anchor_value_tail(local_system: LocalSystem) -> float:
    return local_system.anchor.fiscal_wealth_bar


def lq_linear_kt_path(t_grid: np.ndarray, k0: float, t0: float, solution: LqSolution) -> np.ndarray:
    """The pure LQ-approximate (k, tau-1/2) path from displacement (k0, t0),
    dot(k,t) = A_rc.(k,t): valid under frozen common states because the
    (z, x) rows of the CS001 closed loop are diagonal-only (A's z,x rows
    have zero k/tau columns, and the control only ever enters the tau row),
    so a zero (z,x) deviation stays exactly zero and the 2-state real block
    A_rc alone governs (k, t). Returns shape (2, n): row 0 is k, row 1 is
    tau-1/2, matching `t_grid`'s length."""

    a_rc = solution.A_rc
    y0 = np.array([k0, t0])
    return np.stack([expm(a_rc * t) @ y0 for t in t_grid], axis=1)
