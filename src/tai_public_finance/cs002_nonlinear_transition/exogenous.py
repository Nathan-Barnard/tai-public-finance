"""Analytic deterministic OU paths for the exogenous states (z, x), and the
automation-share inversion needed to construct a pure-automation shock.

CS002 D2 handoff, Route 1 section 1.1: mu_z(z) = kappa_z*(z_bar-z),
mu_x(x) = kappa_x*(x_bar-x) -- exactly the OU drift already implicit in
primitives.international_pricing.safe_rate (r0 = rho + kappa_z*(z_bar-z) +
kappa_x*(x_bar-x)*ell_x(x)); this module never re-derives that drift, only
propagates it forward in time in closed form and cross-checks the closed
form against a direct numerical integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from ..primitives import PrimitiveParameters
from ..primitives.production import automation_share

ExogenousEvaluator = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def z_path(t: np.ndarray | float, z0: float, z_bar: float, kappa_z: float) -> np.ndarray:
    return z_bar + (z0 - z_bar) * np.exp(-kappa_z * np.asarray(t, dtype=float))


def x_path(t: np.ndarray | float, x0: float, x_bar: float, kappa_x: float) -> np.ndarray:
    return x_bar + (x0 - x_bar) * np.exp(-kappa_x * np.asarray(t, dtype=float))


@dataclass(frozen=True)
class ExogenousPath:
    """One deterministic exogenous path pair (z(t), x(t)), bundled with the
    displacement/persistence parameters that generated it so every caller
    (BVP right-hand side, terminal condition, recovery, reporting) reads the
    SAME closed form rather than independent re-derivations. Calling the
    instance directly returns (z(t), x(t)), matching the
    `t -> (z, x)` shape the BVP/terminal/recovery code expects."""

    z0: float
    x0: float
    z_bar: float
    x_bar: float
    kappa_z: float
    kappa_x: float

    def z(self, t: np.ndarray | float) -> np.ndarray:
        return z_path(t, self.z0, self.z_bar, self.kappa_z)

    def x(self, t: np.ndarray | float) -> np.ndarray:
        return x_path(t, self.x0, self.x_bar, self.kappa_x)

    def __call__(self, t: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        return self.z(t), self.x(t)


def propagate_exogenous_numerically(
    t_grid: np.ndarray, z0: float, x0: float, z_bar: float, x_bar: float, kappa_z: float, kappa_x: float
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-check: integrate dz/dt=kappa_z*(z_bar-z), dx/dt=kappa_x*(x_bar-x)
    with an independent ODE solve and return (z_numeric, x_numeric) at
    `t_grid`, to be compared against the analytic z_path/x_path -- CS002 D2
    handoff: "Check these against a numerical propagation." """

    def rhs(t: float, y: np.ndarray) -> list[float]:
        del t
        z, x = y
        return [kappa_z * (z_bar - z), kappa_x * (x_bar - x)]

    t_grid = np.asarray(t_grid, dtype=float)
    solution = solve_ivp(
        rhs, (float(t_grid[0]), float(t_grid[-1])), [z0, x0], t_eval=t_grid, rtol=1e-12, atol=1e-14, method="DOP853"
    )
    if not solution.success:
        raise RuntimeError(f"Numerical propagation of the exogenous OU paths failed: {solution.message}")
    return solution.y[0].copy(), solution.y[1].copy()


def invert_automation_share(alpha_target: float, p: PrimitiveParameters) -> float:
    """Numerically solve automation_share(x, p) == alpha_target for x, via a
    bracketed root-find on the CANONICAL automation_share function itself --
    CS002 D2 handoff: "Do not approximate the share-to-state conversion by
    hand in the production run."

    automation_share is alpha_lower + (alpha_upper-alpha_lower)*logistic(x): a
    strictly increasing bijection of x onto the OPEN interval
    (alpha_lower, alpha_upper). logistic(+-50) is already within machine
    epsilon of {1, 0}, so automation_share(-50) ~= alpha_lower and
    automation_share(+50) ~= alpha_upper for every primitive set; the fixed
    bracket [-50, 50] therefore always brackets a root for any target
    strictly inside (alpha_lower, alpha_upper), with no expansion loop
    needed.
    """

    if not p.alpha_lower < alpha_target < p.alpha_upper:
        raise ValueError(
            f"alpha_target={alpha_target} is outside the feasible open interval "
            f"({p.alpha_lower}, {p.alpha_upper})."
        )

    def objective(x: float) -> float:
        return automation_share(x, p) - alpha_target

    lo, hi = -50.0, 50.0
    assert objective(lo) < 0.0 < objective(hi), (
        "invert_automation_share's fixed bracket [-50, 50] failed to bracket a root; "
        f"objective(lo)={objective(lo)!r}, objective(hi)={objective(hi)!r}."
    )
    return float(brentq(objective, lo, hi, xtol=1e-14))
