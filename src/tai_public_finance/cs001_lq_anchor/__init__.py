"""CS001: LQ anchor, matrix equations, and impulse responses.

Local deterministic and first-order stochastic 4x4 LQ system for the
Brownian Version 5.1, q_D=1 Ramsey economy around its illustrative interior
steady state, plus the leading small-risk public portfolio and welfare
objects. See the computational specification (CS001) in the codex research
workspace for the full contract; this package implements its stated scope.
"""

from .anchor import SteadyState, compute_steady_state
from .config import Cs001Configuration, load_cs001_configuration
from .equations import COORDINATES, LocalSystem, build_local_system
from .portfolio import LeadingPortfolio, leading_portfolio_and_welfare, net_worth_grid
from .solver import LqSolution, solve_lq_system

__all__ = [
    "SteadyState",
    "compute_steady_state",
    "Cs001Configuration",
    "load_cs001_configuration",
    "COORDINATES",
    "LocalSystem",
    "build_local_system",
    "LeadingPortfolio",
    "leading_portfolio_and_welfare",
    "net_worth_grid",
    "LqSolution",
    "solve_lq_system",
]
