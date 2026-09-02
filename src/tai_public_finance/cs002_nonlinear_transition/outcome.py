"""Aggregate outcome taxonomy (CS002 D0-D1 handoff, acceptance #10).

Priority order when several categories could apply at once:
derivation_blocked > numerical_failure > branch_sensitive > boundary_reaching
> computational_pass. Rationale: a derivation gap or a numerical-machinery
failure makes every other check's result unreliable, so those are reported
first; branch sensitivity (does the solution even mean one thing) is
reported ahead of a specific path's economic boundary outcome, since a
branch-sensitive result has no single path whose boundary margins would be
authoritative.

Solver-reported convergence is never itself one of the checks this module
aggregates -- callers pass only INDEPENDENTLY verified checks (residuals.py,
margins.py, the LQ-convergence and horizon/mesh-stability tests), so
"solver convergence alone never sets the aggregate outcome to pass" holds by
construction: there is no code path here that can even see the solver's own
success flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field

OUTCOME_TAXONOMY = ("computational_pass", "branch_sensitive", "boundary_reaching", "numerical_failure", "derivation_blocked")


@dataclass(frozen=True)
class AggregateOutcome:
    outcome: str
    checks: dict[str, bool]
    failed_checks: list[str] = field(default_factory=list)
    conclusion: str = ""

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOME_TAXONOMY:
            raise ValueError(f"outcome={self.outcome!r} is not in the CS002 taxonomy {OUTCOME_TAXONOMY}.")


def determine_outcome(
    checks: dict[str, bool],
    derivation_blocked: bool = False,
    derivation_blocked_reason: str = "",
    numerical_failure_check_names: tuple[str, ...] = (),
    branch_sensitivity_check_names: tuple[str, ...] = (),
    boundary_check_names: tuple[str, ...] = (),
) -> AggregateOutcome:
    failed = [name for name, ok in checks.items() if not ok]

    if derivation_blocked:
        outcome = "derivation_blocked"
    elif any(name in failed for name in numerical_failure_check_names):
        outcome = "numerical_failure"
    elif any(name in failed for name in branch_sensitivity_check_names):
        outcome = "branch_sensitive"
    elif any(name in failed for name in boundary_check_names):
        outcome = "boundary_reaching"
    elif failed:
        # An uncategorised failed check is conservatively treated as a numerical
        # failure rather than silently passing -- every check callers pass in
        # should be assigned to one of the three named categories above.
        outcome = "numerical_failure"
    else:
        outcome = "computational_pass"

    if derivation_blocked:
        conclusion = derivation_blocked_reason or "Derivation gap; see the returned checks for detail."
    elif outcome == "computational_pass":
        conclusion = "All D0-D1 acceptance checks passed. Exploratory prototype evidence only -- not an approved CS002 result."
    else:
        conclusion = f"outcome={outcome}; failed checks: {failed}."

    return AggregateOutcome(outcome=outcome, checks=checks, failed_checks=failed, conclusion=conclusion)
