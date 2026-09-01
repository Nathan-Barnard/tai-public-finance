"""The independent residual/diagnostic evaluator.

Nothing in this module reuses an intermediate variable computed inside
solver.py: every residual, eigenvalue, and finite-difference check below is
recomputed here from primitives, local_system, and the solution's final
matrices only. This is deliberate — CS001 requires that "an independent
residual evaluator does not share the solver's internal result path," so
that a bug in solver.py's construction of, say, the Hamiltonian cannot also
be silently reproduced in the check that is supposed to catch it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import eigvals

from ..primitives.international_pricing import safe_rate
from ..primitives.production import evaluate_smooth_branch
from .anchor import SteadyState
from .equations import LocalSystem
from .portfolio import LeadingPortfolio
from .solver import LqSolution


def scaled_norm(residual: np.ndarray, terms: list[np.ndarray]) -> float:
    """norm(residual) / (1 + sum(norm(term) for term in terms)).

    The default norm (Frobenius for 2-D, Euclidean for 1-D) so this works
    uniformly for both matrix residuals and vector residuals (e.g. one
    experiment's discounted cumulative-response vector).
    """

    return float(np.linalg.norm(residual) / (1.0 + sum(np.linalg.norm(term) for term in terms)))


# --------------------------------------------------------------------------
# Anchor identities (exact, closed-form benchmarks independent of the LQ solve)
# --------------------------------------------------------------------------


def anchor_identity_errors(p, anchor: SteadyState) -> dict[str, float]:
    """Exact closed-form identities the anchor must satisfy, checked via the
    independent nonlinear evaluator rather than the anchor's own construction."""

    exact = evaluate_smooth_branch(anchor.z_bar, anchor.x_bar, anchor.capital_bar, anchor.tax_rate_bar, p)
    net_rental = exact.rental_rate - p.depreciation_rate
    return {
        # Both steady-state equations, evaluated against exact.rental_rate (the
        # nonlinear evaluator applied to z_bar) rather than anchor.rental_rate_bar
        # (which anchor.py sets to depreciation_rate + 2*rho BY DEFINITION before
        # solving for z_bar, so comparing it to "2*rho" again would be checking a
        # value against its own construction and could never fail regardless of
        # whether z_bar itself was solved correctly).
        "capital_law_vs_rho": abs((1.0 - anchor.tax_rate_bar) * net_rental - p.rho),
        "fiscal_envelope_vs_rho": abs(anchor.tax_rate_bar * net_rental - p.rho),
        "capital_growth_vs_zero": abs(exact.capital_growth),
        "fiscal_resources_vs_consumption": abs(anchor.fiscal_resources_bar - anchor.worker_consumption_bar),
        "specialisation_margins_positive": min(
            exact.specialisation_margin_automation_composite,
            exact.specialisation_margin_new_task_composite,
        ),
    }


def linear_fiscal_wealth_identity_errors(local_system: LocalSystem) -> dict[str, float]:
    p = local_system.parameters
    anchor = local_system.anchor
    j = local_system.linear_fiscal_wealth
    output_x_bar = anchor.output_bar * anchor.alpha_x_bar * anchor.eta_output_alpha
    j_z_formula = (anchor.output_bar / anchor.capital_bar + p.kappa_z * anchor.fiscal_wealth_normalized_bar) / (
        p.rho + p.kappa_z
    )
    j_x_formula = (output_x_bar / anchor.capital_bar + p.kappa_x * anchor.ell_x_bar * anchor.fiscal_wealth_normalized_bar) / (
        p.rho + p.kappa_x
    )
    return {
        "j_k_vs_one": abs(j[2] - 1.0),
        "j_t_vs_zero": abs(j[3]),
        "j_z_closed_form_relative_error": abs(j[0] - j_z_formula) / (1.0 + abs(j_z_formula)),
        "j_x_closed_form_relative_error": abs(j[1] - j_x_formula) / (1.0 + abs(j_x_formula)),
    }


def q_rr_cancellation_relative_error(local_system: LocalSystem) -> float:
    Q_rr = local_system.Q[2:, 2:]
    closed = local_system.Q_rr_closed_form
    return float(np.linalg.norm(Q_rr - closed) / (1.0 + np.linalg.norm(closed)))


# --------------------------------------------------------------------------
# Finite-difference cross-check of every primitive derivative feeding Q
# --------------------------------------------------------------------------


def _perturbed_state(y: np.ndarray, anchor: SteadyState) -> tuple[float, float, float, float]:
    z = anchor.z_bar + y[0]
    x = anchor.x_bar + y[1]
    capital = anchor.capital_bar * math.exp(y[2])
    tax_rate = anchor.tax_rate_bar + y[3]
    return z, x, capital, tax_rate


def _finite_difference_gradient_hessian(f, step: float) -> tuple[np.ndarray, np.ndarray]:
    origin = f(np.zeros(4))
    gradient = np.zeros(4)
    hessian = np.zeros((4, 4))
    for i in range(4):
        direction = np.zeros(4)
        direction[i] = step
        plus = f(direction)
        minus = f(-direction)
        gradient[i] = (plus - minus) / (2.0 * step)
        hessian[i, i] = (plus - 2.0 * origin + minus) / step**2
        for j in range(i):
            other = np.zeros(4)
            other[j] = step
            cross = (
                f(direction + other) - f(direction - other) - f(-direction + other) + f(-direction - other)
            ) / (4.0 * step**2)
            hessian[i, j] = hessian[j, i] = cross
    return gradient, hessian


def finite_difference_primitive_checks(local_system: LocalSystem, step: float = 1e-4) -> dict[str, dict[str, float]]:
    p = local_system.parameters
    anchor = local_system.anchor

    def output_at(y: np.ndarray) -> float:
        z, x, capital, tax_rate = _perturbed_state(y, anchor)
        return evaluate_smooth_branch(z, x, capital, tax_rate, p).output

    def rental_at(y: np.ndarray) -> float:
        z, x, capital, tax_rate = _perturbed_state(y, anchor)
        return evaluate_smooth_branch(z, x, capital, tax_rate, p).rental_rate

    def wage_at(y: np.ndarray) -> float:
        z, x, capital, tax_rate = _perturbed_state(y, anchor)
        return evaluate_smooth_branch(z, x, capital, tax_rate, p).wage_income

    def safe_rate_at(y: np.ndarray) -> float:
        z, x, _capital, _tax_rate = _perturbed_state(y, anchor)
        return safe_rate(z, x, anchor.z_bar, anchor.x_bar, p)

    targets = {
        "output": (output_at, local_system.output_y, local_system.output_yy),
        "rental_rate": (rental_at, local_system.rental_y, local_system.rental_yy),
        "wage_income": (wage_at, local_system.wage_y, local_system.wage_yy),
        "safe_rate": (safe_rate_at, local_system.safe_rate_y, local_system.safe_rate_yy),
    }
    results = {}
    for name, (f, analytic_gradient, analytic_hessian) in targets.items():
        gradient, hessian = _finite_difference_gradient_hessian(f, step)
        results[name] = {
            "gradient_relative_error": float(
                np.linalg.norm(gradient - analytic_gradient) / (1.0 + np.linalg.norm(analytic_gradient))
            ),
            "hessian_relative_error": float(
                np.linalg.norm(hessian - analytic_hessian) / (1.0 + np.linalg.norm(analytic_hessian))
            ),
        }
    return results


# --------------------------------------------------------------------------
# Matrix-equation residuals, spectra, and covariance/resolvent identities
# --------------------------------------------------------------------------


def riccati_full_residual(local_system: LocalSystem, solution: LqSolution) -> float:
    A, Q, B = local_system.A, local_system.Q, local_system.B.reshape(-1, 1)
    H = solution.H
    rho, chi = local_system.parameters.rho, local_system.anchor.chi
    term = Q + A.T @ H + H @ A - rho * H + chi * H @ B @ B.T @ H
    terms = [Q, A.T @ H, H @ A, rho * H, chi * H @ B @ B.T @ H]
    return scaled_norm(term, terms)


def riccati_real_block_residual(local_system: LocalSystem, solution: LqSolution) -> float:
    A_r = local_system.A[2:, 2:]
    Q_rr = local_system.Q[2:, 2:]
    H_rr = solution.H_rr
    rho, chi = local_system.parameters.rho, local_system.anchor.chi
    b = np.array([[0.0], [1.0]])
    term = Q_rr + A_r.T @ H_rr + H_rr @ A_r - rho * H_rr + chi * H_rr @ b @ b.T @ H_rr
    terms = [Q_rr, A_r.T @ H_rr, H_rr @ A_r, rho * H_rr, chi * H_rr @ b @ b.T @ H_rr]
    return scaled_norm(term, terms)


def real_block_scalar_residuals(local_system: LocalSystem, solution: LqSolution) -> np.ndarray:
    """The three scalar Riccati identities, each scaled by 1 + the sum of its
    own term magnitudes (matching scaled_norm's convention) rather than left
    as raw residuals — a raw residual is only comparable to
    matrix_scaled_residual by coincidence of this calibration's magnitudes,
    not in general."""

    rho, gamma, chi = local_system.parameters.rho, local_system.anchor.gamma, local_system.anchor.chi
    h, ell, m = solution.H_rr[0, 0], solution.H_rr[0, 1], solution.H_rr[1, 1]
    equations = [
        (rho - (2.0 * gamma + rho) * h + chi * ell * ell, [rho, (2.0 * gamma + rho) * h, chi * ell * ell]),
        (
            2.0 * rho - 2.0 * rho * h - (gamma + rho) * ell + chi * ell * m,
            [2.0 * rho, 2.0 * rho * h, (gamma + rho) * ell, chi * ell * m],
        ),
        (-4.0 * rho * ell - rho * m + chi * m * m, [4.0 * rho * ell, rho * m, chi * m * m]),
    ]
    return np.array([residual / (1.0 + sum(abs(term) for term in terms)) for residual, terms in equations])


def riccati_symmetry_error(solution: LqSolution) -> float:
    return scaled_norm(solution.H - solution.H.T, [solution.H])


def raw_solve_symmetry_errors(solution: LqSolution) -> dict[str, float]:
    """Asymmetry of H_rr, H_ee, and the stationary covariance BEFORE solver.py
    force-symmetrizes them. riccati_symmetry_error above checks solution.H
    AFTER symmetrization, which is triviailly ~0 by construction regardless
    of whether the raw solve was any good — this is the check that would
    actually catch a raw solve producing a substantially asymmetric result
    that force-symmetrization would otherwise silently paper over."""

    return {"H_rr": solution.invariant_subspace.raw_real_part_asymmetry, **solution.raw_symmetry_errors}


def sylvester_residual(local_system: LocalSystem, solution: LqSolution) -> float:
    A_e = local_system.A[:2, :2]
    D = local_system.D
    Q_er = local_system.Q[:2, 2:]
    rho = local_system.parameters.rho
    H_rr, H_er, A_rc = solution.H_rr, solution.H_er, solution.A_rc
    term = (A_e.T - rho * np.eye(2)) @ H_er + H_er @ A_rc + Q_er + D.T @ H_rr
    terms = [(A_e.T - rho * np.eye(2)) @ H_er, H_er @ A_rc, Q_er, D.T @ H_rr]
    return scaled_norm(term, terms)


def discounted_lyapunov_residual(local_system: LocalSystem, solution: LqSolution) -> float:
    A_e = local_system.A[:2, :2]
    D = local_system.D
    Q_ee = local_system.Q[:2, :2]
    rho, chi = local_system.parameters.rho, local_system.anchor.chi
    H_ee, H_er = solution.H_ee, solution.H_er
    b = np.array([[0.0], [1.0]])
    L_e = A_e - rho * np.eye(2) / 2.0
    K_ee = Q_ee + D.T @ H_er.T + H_er @ D + chi * H_er @ b @ b.T @ H_er.T
    term = L_e.T @ H_ee + H_ee @ L_e + K_ee
    terms = [L_e.T @ H_ee, H_ee @ L_e, K_ee]
    return scaled_norm(term, terms)


def hamiltonian_hyperbolicity(local_system: LocalSystem) -> dict[str, Any]:
    """Rebuild the Hamiltonian from scratch (A_r, Q_rr, rho, chi only) and check its spectrum."""

    A_r = local_system.A[2:, 2:]
    Q_rr = local_system.Q[2:, 2:]
    rho, chi = local_system.parameters.rho, local_system.anchor.chi
    b = np.array([[0.0], [1.0]])
    A_tilde = A_r - rho * np.eye(2) / 2.0
    hamiltonian = np.block([[A_tilde, chi * b @ b.T], [-Q_rr, -A_tilde.T]])
    eigenvalues = eigvals(hamiltonian)
    scale = 1.0 + np.linalg.norm(hamiltonian, ord=2)
    tolerance = 1e-12 * scale
    distance = float(np.min(np.abs(eigenvalues.real)))
    return {"eigenvalues": eigenvalues, "imaginary_axis_distance": distance, "tolerance": tolerance, "hyperbolic": distance > tolerance}


def popov_function_positive(local_system: LocalSystem, omega_grid: np.ndarray) -> bool:
    """Pi(omega) = 1/chi - G(i omega)^* Q_rr G(i omega) > 0 for all real omega (spot check)."""

    rho, gamma, chi = local_system.parameters.rho, local_system.anchor.gamma, local_system.anchor.chi
    a0 = gamma + rho / 2.0
    b0 = rho / 2.0
    values = 1.0 / chi + 8.0 * gamma * rho**2 / ((omega_grid**2 + a0**2) * (omega_grid**2 + b0**2))
    return bool(np.all(values > 0.0))


def closed_loop_spectra(local_system: LocalSystem, solution: LqSolution) -> dict[str, Any]:
    real_closed_loop_eigenvalues = eigvals(solution.A_rc)
    full_closed_loop_eigenvalues = eigvals(solution.A_c)
    return {
        "real_closed_loop_eigenvalues": real_closed_loop_eigenvalues,
        "full_closed_loop_eigenvalues": full_closed_loop_eigenvalues,
        "real_closed_loop_hurwitz": bool(np.max(real_closed_loop_eigenvalues.real) < 0.0),
        "full_closed_loop_hurwitz": bool(np.max(full_closed_loop_eigenvalues.real) < 0.0),
        "real_closed_loop_stability_margin": float(-np.max(real_closed_loop_eigenvalues.real)),
        "full_closed_loop_stability_margin": float(-np.max(full_closed_loop_eigenvalues.real)),
    }


def stationary_covariance_checks(local_system: LocalSystem, solution: LqSolution) -> dict[str, float]:
    A_c = solution.A_c
    Sigma = local_system.Sigma_e_hat
    covariance = solution.stationary_covariance_y
    residual = A_c @ covariance + covariance @ A_c.T + Sigma @ Sigma.T
    terms = [A_c @ covariance, covariance @ A_c.T, Sigma @ Sigma.T]
    p = local_system.parameters
    expected_exogenous = np.diag([p.productivity_stationary_sd**2, p.automation_stationary_sd**2])
    return {
        "scaled_residual": scaled_norm(residual, terms),
        "min_eigenvalue": float(np.min(np.linalg.eigvalsh(covariance))),
        "exogenous_block_closed_form_relative_error": scaled_norm(covariance[:2, :2] - expected_exogenous, [expected_exogenous]),
    }


def resolvent_probe_residual(rho: float, F: np.ndarray, probe_result: np.ndarray, probe_input: np.ndarray) -> float:
    """norm((rho I - F) @ probe_result - probe_input) scaled by (1 + norm(probe_input)).

    The same resolvent identity checked at two different probes: the full
    identity matrix (resolvent_identity_residual, below — checks the stored
    resolvent itself) and, in irfs.py, one experiment's discounted
    cumulative-response vector (checks that vector without needing to trust
    that the matrix-level check implies every individual probe).
    """

    identity = np.eye(F.shape[0])
    residual = (rho * identity - F) @ probe_result - probe_input
    return scaled_norm(residual, [probe_input])


def resolvent_identity_residual(local_system: LocalSystem, solution: LqSolution) -> float:
    rho = local_system.parameters.rho
    return resolvent_probe_residual(rho, solution.F, solution.resolvent, np.eye(5))


def closed_form_vs_invariant_subspace_relative_error(solution: LqSolution) -> float:
    return scaled_norm(solution.H_rr - solution.closed_form.H_rr, [solution.H_rr])


def feedback_construction_errors(local_system: LocalSystem, solution: LqSolution) -> dict[str, float]:
    """Independently rebuild A_rc, A_c, and F from primitives + H and diff them
    against the solver's stored values.

    The residual checks above (Riccati/Sylvester/Lyapunov/covariance/resolvent)
    all verify that solution.H, solution.A_c, etc. are mutually self-consistent
    with EACH OTHER — none of them checks that A_c was actually assembled as
    A + chi*B*B^T*H in the first place, so a construction bug in that one line
    (e.g. a dropped factor, or A_rc built with the wrong sign) can still leave
    every other residual near zero and the closed loop Hurwitz, because H,
    A_c, and F would all be internally consistent with the WRONG A_c. This is
    the one check in this module that reconstructs a solver output from
    nothing but primitives and H, rather than checking H/A_c/F against each
    other.
    """

    p = local_system.parameters
    anchor = local_system.anchor
    A, B, D = local_system.A, local_system.B.reshape(-1, 1), local_system.D
    A_r = A[2:, 2:]
    b = np.array([[0.0], [1.0]])
    chi = anchor.chi

    A_rc_expected = A_r + chi * b @ b.T @ solution.H_rr
    A_c_expected = A + chi * B @ B.T @ solution.H
    F_expected = np.block(
        [
            [A_c_expected, np.zeros((4, 1))],
            [anchor.comprehensive_resources_bar * local_system.safe_rate_y.reshape(1, 4), np.zeros((1, 1))],
        ]
    )
    return {
        "A_rc_relative_error": scaled_norm(solution.A_rc - A_rc_expected, [A_rc_expected]),
        "A_c_relative_error": scaled_norm(solution.A_c - A_c_expected, [A_c_expected]),
        "F_relative_error": scaled_norm(solution.F - F_expected, [F_expected]),
    }


def portfolio_algebraic_identity_errors(portfolio: LeadingPortfolio) -> dict[str, float]:
    return {
        "zeta_perp_orthogonal_to_lambda_hat": abs(float(portfolio.zeta_j_perp @ portfolio.lambda_hat)),
    }


@dataclass(frozen=True)
class DiagnosticsReport:
    anchor_identity_errors: dict[str, float]
    linear_fiscal_wealth_identity_errors: dict[str, float]
    q_rr_cancellation_relative_error: float
    finite_difference_checks: dict[str, dict[str, float]]
    riccati_full_scaled_residual: float
    riccati_real_block_scaled_residual: float
    riccati_symmetry_error: float
    raw_solve_symmetry_errors: dict[str, float]
    real_block_scalar_residuals: np.ndarray
    sylvester_scaled_residual: float
    discounted_lyapunov_scaled_residual: float
    closed_form_vs_invariant_subspace_relative_error: float
    hamiltonian: dict[str, Any]
    popov_strict: bool
    closed_loop: dict[str, Any]
    feedback_construction_errors: dict[str, float]
    stationary_covariance: dict[str, float]
    resolvent_identity_residual: float
    portfolio_identity_errors: dict[str, float]


def run_diagnostics(local_system: LocalSystem, solution: LqSolution, portfolio: LeadingPortfolio) -> DiagnosticsReport:
    omega_grid = np.linspace(-50.0, 50.0, 2001)
    return DiagnosticsReport(
        anchor_identity_errors=anchor_identity_errors(local_system.parameters, local_system.anchor),
        linear_fiscal_wealth_identity_errors=linear_fiscal_wealth_identity_errors(local_system),
        q_rr_cancellation_relative_error=q_rr_cancellation_relative_error(local_system),
        finite_difference_checks=finite_difference_primitive_checks(local_system),
        riccati_full_scaled_residual=riccati_full_residual(local_system, solution),
        riccati_real_block_scaled_residual=riccati_real_block_residual(local_system, solution),
        riccati_symmetry_error=riccati_symmetry_error(solution),
        raw_solve_symmetry_errors=raw_solve_symmetry_errors(solution),
        real_block_scalar_residuals=real_block_scalar_residuals(local_system, solution),
        sylvester_scaled_residual=sylvester_residual(local_system, solution),
        discounted_lyapunov_scaled_residual=discounted_lyapunov_residual(local_system, solution),
        closed_form_vs_invariant_subspace_relative_error=closed_form_vs_invariant_subspace_relative_error(solution),
        hamiltonian=hamiltonian_hyperbolicity(local_system),
        popov_strict=popov_function_positive(local_system, omega_grid)
        and local_system.anchor.chi > 0.0
        and local_system.anchor.gamma > 0.0
        and local_system.parameters.rho > 0.0,
        closed_loop=closed_loop_spectra(local_system, solution),
        feedback_construction_errors=feedback_construction_errors(local_system, solution),
        stationary_covariance=stationary_covariance_checks(local_system, solution),
        resolvent_identity_residual=resolvent_identity_residual(local_system, solution),
        portfolio_identity_errors=portfolio_algebraic_identity_errors(portfolio),
    )


@dataclass(frozen=True)
class AcceptanceReport:
    outcome: str
    checks: dict[str, bool]
    failed_checks: list[str] = field(default_factory=list)
    conclusion: str = ""


def acceptance(
    local_system: LocalSystem,
    solution: LqSolution,
    diagnostics: DiagnosticsReport,
    portfolio: LeadingPortfolio,
    irf_boundary_summary: dict[str, float],
    max_matrix_exponential_vs_ode_relative_error: float,
    max_first_order_budget_residual: float,
    tolerances: dict[str, float],
) -> AcceptanceReport:
    anchor_scale = 1.0 + max(abs(v) for v in local_system.anchor.__dict__.values() if isinstance(v, (int, float)))
    anchor_error = max(diagnostics.anchor_identity_errors["capital_law_vs_rho"],
                        diagnostics.anchor_identity_errors["fiscal_envelope_vs_rho"],
                        diagnostics.anchor_identity_errors["capital_growth_vs_zero"],
                        diagnostics.anchor_identity_errors["fiscal_resources_vs_consumption"]) / anchor_scale
    checks = {
        "anchor_identities": anchor_error <= tolerances["anchor_relative_error"],
        "anchor_specialisation_margins_positive": diagnostics.anchor_identity_errors["specialisation_margins_positive"] > 0.0,
        "linear_fiscal_wealth_identities": max(diagnostics.linear_fiscal_wealth_identity_errors.values())
        <= tolerances["anchor_relative_error"],
        "Q_rr_cancellation": diagnostics.q_rr_cancellation_relative_error <= tolerances["anchor_relative_error"],
        "finite_difference_primitive_checks": all(
            entry["gradient_relative_error"] <= tolerances["finite_difference_relative_error"]
            and entry["hessian_relative_error"] <= tolerances["finite_difference_relative_error"]
            for entry in diagnostics.finite_difference_checks.values()
        ),
        "riccati_residual": diagnostics.riccati_full_scaled_residual <= tolerances["matrix_scaled_residual"],
        "riccati_real_block_residual": diagnostics.riccati_real_block_scaled_residual <= tolerances["matrix_scaled_residual"],
        "riccati_scalar_equations": bool(np.max(np.abs(diagnostics.real_block_scalar_residuals)) <= tolerances["matrix_scaled_residual"]),
        "riccati_symmetry": diagnostics.riccati_symmetry_error <= tolerances["riccati_symmetry_error"],
        "raw_solve_symmetry": max(diagnostics.raw_solve_symmetry_errors.values()) <= tolerances["riccati_symmetry_error"],
        "closed_form_agreement": diagnostics.closed_form_vs_invariant_subspace_relative_error <= tolerances["closed_form_relative_error"],
        "sylvester_residual": diagnostics.sylvester_scaled_residual <= tolerances["matrix_scaled_residual"],
        "discounted_lyapunov_residual": diagnostics.discounted_lyapunov_scaled_residual <= tolerances["matrix_scaled_residual"],
        "stationary_covariance_residual": diagnostics.stationary_covariance["scaled_residual"] <= tolerances["matrix_scaled_residual"],
        "stationary_exogenous_covariance_identity": diagnostics.stationary_covariance["exogenous_block_closed_form_relative_error"] <= tolerances["matrix_scaled_residual"],
        "hamiltonian_hyperbolic": diagnostics.hamiltonian["hyperbolic"],
        "popov_strict": diagnostics.popov_strict,
        "real_closed_loop_hurwitz": diagnostics.closed_loop["real_closed_loop_hurwitz"],
        "full_closed_loop_hurwitz": diagnostics.closed_loop["full_closed_loop_hurwitz"],
        "A_rc_construction": diagnostics.feedback_construction_errors["A_rc_relative_error"] <= tolerances["matrix_scaled_residual"],
        "A_c_construction": diagnostics.feedback_construction_errors["A_c_relative_error"] <= tolerances["matrix_scaled_residual"],
        "F_construction": diagnostics.feedback_construction_errors["F_relative_error"] <= tolerances["matrix_scaled_residual"],
        "resolvent_identity": diagnostics.resolvent_identity_residual <= tolerances["matrix_scaled_residual"],
        "matrix_exponential_ode_agreement": max_matrix_exponential_vs_ode_relative_error <= tolerances["matrix_exponential_vs_ode_relative_error"],
        "first_order_budget_identity": max_first_order_budget_residual <= tolerances["matrix_scaled_residual"],
        "portfolio_concavity": portfolio.portfolio_curvature < 0.0,
        "portfolio_zeta_perp_orthogonality": diagnostics.portfolio_identity_errors["zeta_perp_orthogonal_to_lambda_hat"]
        <= tolerances["riccati_symmetry_error"],
        "portfolio_interior": portfolio.portfolio_lower_slack > 0.0 and portfolio.portfolio_upper_slack > 0.0,
        "same_state_zero_position_comparator_feasible": portfolio.zero_position_feasible,
        "same_state_merton_comparator_feasible": portfolio.merton_comparator_feasible,
        "reported_paths_specialisation": irf_boundary_summary["specialisation_margin_automation_composite"] > 0.0
        and irf_boundary_summary["specialisation_margin_new_task_composite"] > 0.0,
        "reported_paths_output_sign_branch": irf_boundary_summary["output_automation_semielasticity"] > 0.0,
        "reported_paths_positive_consumption_and_X": irf_boundary_summary["worker_consumption_level"] > 0.0
        and irf_boundary_summary["comprehensive_resources_level"] > 0.0,
        "reported_paths_transfer_floor": irf_boundary_summary["transfer_level"] > 0.0,
        "reported_paths_numerical_scaffolding_slack": min(
            irf_boundary_summary[name]
            for name in (
                "portfolio_lower_slack",
                "portfolio_upper_slack",
                "tax_lower_slack",
                "tax_upper_slack",
                "tax_speed_slack",
                "tax_structural_ceiling_slack",
            )
        )
        > 0.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    conclusion = (
        "The configured Stage 1 (deterministic/first-order) and Stage 2A (leading small-risk) calculations "
        "passed their numerical and local-applicability checks."
        if not failed
        else "The run is retained as a failed smoke test; see failed_checks before interpreting outputs."
    )
    return AcceptanceReport(outcome="pass" if not failed else "fail", checks=checks, failed_checks=failed, conclusion=conclusion)
