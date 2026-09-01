from __future__ import annotations

import numpy as np
import pytest

from tai_public_finance.cs001_lq_anchor.solver import (
    closed_form_real_block,
    invariant_subspace_real_block,
)


def test_invariant_subspace_and_closed_form_agree_on_the_baseline(baseline):
    solution = baseline["solution"]
    np.testing.assert_allclose(solution.H_rr, solution.closed_form.H_rr, atol=1e-8)


def test_baseline_real_block_is_hyperbolic_and_hurwitz(baseline):
    diagnostics = baseline["diagnostics"]
    assert diagnostics.hamiltonian["hyperbolic"]
    assert diagnostics.closed_loop["real_closed_loop_hurwitz"]
    assert diagnostics.closed_loop["full_closed_loop_hurwitz"]


def test_nonhyperbolic_hamiltonian_is_refused_not_silently_solved():
    # rho=gamma=0 collapses the Hamiltonian's stable/unstable split; the
    # invariant-subspace solver must refuse rather than return a wrong branch.
    with pytest.raises(RuntimeError):
        invariant_subspace_real_block(np.zeros((2, 2)), np.zeros((2, 2)), rho=0.0, chi=1.0)


def test_ordered_schur_remains_accurate_at_the_oscillation_threshold(baseline):
    # chi_* = gamma(gamma+rho)^2 / (32 rho^2) is where the capital-tax roots
    # transition from real to complex; the two stable roots nearly collide
    # there, which is exactly where naive eigenvector selection is fragile.
    p = baseline["local_system"].parameters
    anchor = baseline["local_system"].anchor
    rho, gamma = p.rho, anchor.gamma
    chi_critical = gamma * (gamma + rho) ** 2 / (32.0 * rho**2)
    A_r = np.array([[-gamma, -2.0 * rho], [0.0, 0.0]])
    Q_rr = np.array([[rho, 2.0 * rho], [2.0 * rho, 0.0]])

    result = invariant_subspace_real_block(A_r, Q_rr, rho, chi_critical)
    H = result.H_rr
    b = np.array([[0.0], [1.0]])
    residual = Q_rr + A_r.T @ H + H @ A_r - rho * H + chi_critical * H @ b @ b.T @ H
    assert np.linalg.norm(residual) < 1e-10
    assert result.imaginary_axis_distance > result.selection_tolerance


def test_closed_form_matches_invariant_subspace_across_a_chi_sweep(baseline):
    # Independent cross-check away from the baseline chi, spanning both the
    # real-root (chi < chi_*) and complex-root (chi > chi_*) regimes.
    p = baseline["local_system"].parameters
    anchor = baseline["local_system"].anchor
    rho, gamma = p.rho, anchor.gamma
    chi_critical = gamma * (gamma + rho) ** 2 / (32.0 * rho**2)
    A_r = np.array([[-gamma, -2.0 * rho], [0.0, 0.0]])
    Q_rr = np.array([[rho, 2.0 * rho], [2.0 * rho, 0.0]])
    for multiplier in (0.2, 0.8, 1.25, 5.0):
        chi = multiplier * chi_critical
        closed = closed_form_real_block(rho, gamma, chi)
        invariant = invariant_subspace_real_block(A_r, Q_rr, rho, chi)
        np.testing.assert_allclose(closed.H_rr, invariant.H_rr, atol=1e-7, err_msg=f"multiplier={multiplier}")
