from __future__ import annotations

import dataclasses

import numpy as np

from tai_public_finance.cs001_lq_anchor.diagnostics import acceptance, feedback_construction_errors


def test_baseline_passes_every_cs001_acceptance_check(baseline):
    tolerances = baseline["config"].experiment["acceptance_tolerances"]
    report = acceptance(
        baseline["local_system"],
        baseline["solution"],
        baseline["diagnostics"],
        baseline["portfolio"],
        baseline["irfs"]["boundary_summary"],
        baseline["irfs"]["max_matrix_exponential_vs_ode_relative_error"],
        baseline["irfs"]["max_first_order_budget_residual"],
        tolerances,
    )
    assert report.outcome == "pass", report.failed_checks


def test_riccati_residuals_are_near_machine_precision(baseline):
    diagnostics = baseline["diagnostics"]
    assert diagnostics.riccati_full_scaled_residual < 1e-12
    assert diagnostics.riccati_real_block_scaled_residual < 1e-12
    assert diagnostics.sylvester_scaled_residual < 1e-12
    assert diagnostics.discounted_lyapunov_scaled_residual < 1e-12
    assert diagnostics.riccati_symmetry_error < 1e-14


def test_finite_difference_checks_confirm_every_matrix_entry_traces_to_primitives(baseline):
    for name, entry in baseline["diagnostics"].finite_difference_checks.items():
        assert entry["gradient_relative_error"] < 1e-6, name
        assert entry["hessian_relative_error"] < 1e-4, name


def test_stationary_exogenous_covariance_matches_the_calibrated_ou_variance(baseline):
    # An exact, solver-independent identity: two uncoupled OU states have
    # stationary variance sigma_hat^2 / (2 kappa) = (stationary_sd)^2 by
    # construction of sigma_hat itself.
    assert baseline["diagnostics"].stationary_covariance["exogenous_block_closed_form_relative_error"] < 1e-10


def test_popov_condition_holds_on_a_wide_frequency_grid(baseline):
    assert baseline["diagnostics"].popov_strict


def test_portfolio_hedge_is_exactly_orthogonal_to_the_traded_payoff(baseline):
    assert baseline["diagnostics"].portfolio_identity_errors["zeta_perp_orthogonal_to_lambda_hat"] < 1e-10


def test_feedback_construction_check_passes_on_the_correctly_solved_baseline(baseline):
    errors = baseline["diagnostics"].feedback_construction_errors
    assert errors["A_rc_relative_error"] < 1e-10
    assert errors["A_c_relative_error"] < 1e-10
    assert errors["F_relative_error"] < 1e-10


def test_feedback_construction_check_catches_a_halved_feedback_gain(baseline):
    # Regression guard for a real gap this diagnostic was added to close: a
    # bug in how A_c is assembled from A, B, chi, and H (e.g. a dropped
    # factor, or a wrong sign) leaves the Riccati/Sylvester/Lyapunov/
    # covariance/resolvent residuals and the Hurwitz check ALL near zero,
    # because H, A_c, and F stay mutually self-consistent with each other —
    # just not with the correct feedback formula. Only a check that rebuilds
    # A_c independently from primitives + H catches this.
    solution = baseline["solution"]
    local_system = baseline["local_system"]
    B = local_system.B.reshape(-1, 1)
    halved_A_c = local_system.A + 0.5 * local_system.anchor.chi * B @ B.T @ solution.H
    # A real construction bug would feed the wrong A_c into F too, not leave F
    # correctly built from the right A_c while only A_c itself is wrong.
    halved_F = np.block(
        [
            [halved_A_c, np.zeros((4, 1))],
            [
                local_system.anchor.comprehensive_resources_bar * local_system.safe_rate_y.reshape(1, 4),
                np.zeros((1, 1)),
            ],
        ]
    )
    mutated_solution = dataclasses.replace(solution, A_c=halved_A_c, F=halved_F)

    errors = feedback_construction_errors(local_system, mutated_solution)
    assert errors["A_c_relative_error"] > 1e-3
    assert errors["F_relative_error"] > 1e-3
