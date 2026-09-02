"""CS002 D0-D1: exploratory nonlinear deterministic Ramsey transition prototype.

Bounded feasibility prototype under CS002 v0.2's draft-specification
exception (see the codex workspace's nonlinear-deterministic-ramsey-
transition--CS002.md, "Immediate implementation handoff"). Every output of
this package is exploratory: it does not make CS002 approved, and it
implements only Blocks D0 (primitives, anchor, residual interface) and D1
(frozen-common-state nonlinear transition) -- not D2 (mean reversion), D3
(derivative service), or anything stochastic.
"""

from .config import Cs002Configuration, load_cs002_configuration
from .continuation import Checkpoint, ContinuationRun, run_continuation
from .experiment import ExperimentReport, run_d0_d1_experiment
from .model import CapitalDerivatives, CharacteristicRates, capital_derivatives, capital_from_log, characteristic_rates, characteristic_rhs_vectorized, log_from_capital
from .outcome import OUTCOME_TAXONOMY, AggregateOutcome, determine_outcome
from .terminal import TerminalCostates, anchor_value_tail, crude_costates, lq_quadratic_value_tail, lq_stable_manifold_costates

__all__ = [
    "Cs002Configuration",
    "load_cs002_configuration",
    "Checkpoint",
    "ContinuationRun",
    "run_continuation",
    "ExperimentReport",
    "run_d0_d1_experiment",
    "CapitalDerivatives",
    "CharacteristicRates",
    "capital_derivatives",
    "capital_from_log",
    "characteristic_rates",
    "characteristic_rhs_vectorized",
    "log_from_capital",
    "OUTCOME_TAXONOMY",
    "AggregateOutcome",
    "determine_outcome",
    "TerminalCostates",
    "anchor_value_tail",
    "crude_costates",
    "lq_quadratic_value_tail",
    "lq_stable_manifold_costates",
]
