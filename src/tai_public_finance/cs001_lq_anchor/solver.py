"""Riccati, Sylvester, Lyapunov, and stationary-covariance solves.

This module only solves; it computes no residuals or acceptance checks
(diagnostics.py recomputes every identity independently, without reusing
any intermediate variable defined here). Two independent methods are used
for the 2x2 real (capital-tax) Riccati block, as the local LQ system and
computation plan requires: closed-form Hamiltonian roots, and a stable
invariant subspace via an ordered complex Schur decomposition. The
invariant-subspace solution is primary; the closed-form solution is the
cross-check diagnostics.py compares it against.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigvals, schur, solve_sylvester

from .equations import LocalSystem


@dataclass(frozen=True)
class ClosedFormRealBlock:
    H_rr: np.ndarray
    hamiltonian_roots: np.ndarray  # the four roots +-sqrt(u_plus), +-sqrt(u_minus)
    closed_loop_roots: np.ndarray  # mu_i = lambda_i + rho/2, the two stable roots
    u_plus: complex
    u_minus: complex
    discriminant: complex
    imaginary_reconstruction_norm: float


@dataclass(frozen=True)
class InvariantSubspaceRealBlock:
    H_rr: np.ndarray
    hamiltonian: np.ndarray
    hamiltonian_eigenvalues: np.ndarray
    stable_dimension: int
    stable_graph_condition: float
    stable_graph_imaginary_norm: float
    imaginary_axis_distance: float
    selection_tolerance: float
    raw_real_part_asymmetry: float  # ||Re(H) - Re(H)^T|| before forcing symmetry; large values mean the
    # graph solve itself misbehaved, not just floating-point noise a post-hoc symmetrize should paper over.


@dataclass(frozen=True)
class LqSolution:
    H: np.ndarray
    H_ee: np.ndarray
    H_er: np.ndarray
    H_rr: np.ndarray
    A_rc: np.ndarray  # A_r + chi * b b^T H_rr: the actual capital-tax closed loop
    A_c: np.ndarray  # the full 4x4 closed-loop state matrix
    F: np.ndarray  # 5x5 augmented drift matrix for (y, X)
    resolvent: np.ndarray  # (rho I - F)^-1
    stationary_covariance_y: np.ndarray
    closed_form: ClosedFormRealBlock
    invariant_subspace: InvariantSubspaceRealBlock
    raw_symmetry_errors: dict[str, float]  # H_ee/stationary_covariance_y asymmetry BEFORE forced symmetrization


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _asymmetry(matrix: np.ndarray) -> float:
    """Scaled asymmetry BEFORE symmetrizing — large values mean the solve itself
    produced a non-symmetric result (a real problem), not floating-point noise
    that _symmetrize is entitled to average away."""

    return float(np.linalg.norm(matrix - matrix.T) / (1.0 + np.linalg.norm(matrix)))


def _positive_real_sqrt(value: complex) -> complex:
    root = cmath.sqrt(value)
    if root.real < 0.0 or (abs(root.real) < 1e-15 and root.imag < 0.0):
        root = -root
    return root


def closed_form_real_block(rho: float, gamma: float, chi: float) -> ClosedFormRealBlock:
    """Solve the 2x2 real Riccati block via the closed-form Hamiltonian-root formulas.

    p(lambda) = lambda^4 - (a0^2+b0^2) lambda^2 + a0^2 b0^2 + 8 chi rho^2 gamma,
    a0 = gamma + rho/2, b0 = rho/2. Roots never touch the imaginary axis because
    p(i*omega) = omega^4 + (a0^2+b0^2) omega^2 + a0^2 b0^2 + 8 chi rho^2 gamma > 0
    for chi, rho, gamma > 0.
    """

    a0 = gamma + rho / 2.0
    b0 = rho / 2.0
    discriminant = (a0 * a0 - b0 * b0) ** 2 - 32.0 * chi * rho * rho * gamma
    root = cmath.sqrt(discriminant)
    u_plus = (a0 * a0 + b0 * b0 + root) / 2.0
    u_minus = (a0 * a0 + b0 * b0 - root) / 2.0
    lambdas = [-_positive_real_sqrt(u_plus), -_positive_real_sqrt(u_minus)]
    mus = [value + b0 for value in lambdas]

    # Recover H_rr = [[h, ell], [ell, m]] from the stable closed-loop roots via
    # the elementary-symmetric-function relations behind the scalar Riccati
    # equations (m from the closed-loop trace/product, then ell, then h).
    m = (mus[0] + mus[1] + gamma) / chi
    ell = (mus[0] * mus[1] / chi + gamma * m) / (2.0 * rho)
    h = (rho + chi * ell * ell) / (2.0 * gamma + rho)
    matrix_complex = np.array([[h, ell], [ell, m]], dtype=complex)
    imaginary_norm = float(np.linalg.norm(matrix_complex.imag, ord="fro"))
    matrix = matrix_complex.real
    return ClosedFormRealBlock(
        H_rr=matrix,
        hamiltonian_roots=np.array([lambdas[0], lambdas[1], -lambdas[0], -lambdas[1]], dtype=complex),
        closed_loop_roots=np.array(mus, dtype=complex),
        u_plus=u_plus,
        u_minus=u_minus,
        discriminant=discriminant,
        imaginary_reconstruction_norm=imaginary_norm,
    )


def invariant_subspace_real_block(A_r: np.ndarray, Q_rr: np.ndarray, rho: float, chi: float) -> InvariantSubspaceRealBlock:
    """Solve the same 2x2 block from the stable invariant subspace of the Hamiltonian.

    Near the monotone/oscillatory transition chi_* the two stable roots nearly
    collide, so an ordered real-Schur/spectral-projector construction is used
    rather than selecting individual eigenvectors (which becomes ill-conditioned
    exactly there).
    """

    b = np.array([[0.0], [1.0]])
    A_tilde = A_r - rho * np.eye(2) / 2.0
    hamiltonian = np.block([[A_tilde, chi * b @ b.T], [-Q_rr, -A_tilde.T]])
    scale = 1.0 + np.linalg.norm(hamiltonian, ord=2)
    selection_tolerance = 1e-12 * scale

    ordered, vectors, stable_dimension = schur(
        hamiltonian,
        output="complex",
        sort=lambda value: value.real < -selection_tolerance,
    )
    if stable_dimension != 2:
        raise RuntimeError(f"Expected a two-dimensional stable Hamiltonian subspace; found {stable_dimension}.")
    U = vectors[:2, :2]
    V = vectors[2:, :2]
    condition_u = float(np.linalg.cond(U))
    if not np.isfinite(condition_u) or condition_u > 1e12:
        raise RuntimeError(f"Stable Hamiltonian graph is ill-conditioned: cond(U)={condition_u:.3e}.")
    H_complex = np.linalg.solve(U.T, V.T).T
    graph_imaginary_norm = float(np.linalg.norm(H_complex.imag, ord="fro"))
    H = H_complex.real
    raw_real_part_asymmetry = _asymmetry(H)
    H = _symmetrize(H)
    eigenvalues = eigvals(hamiltonian)
    imaginary_axis_distance = float(np.min(np.abs(eigenvalues.real)))
    return InvariantSubspaceRealBlock(
        H_rr=H,
        hamiltonian=hamiltonian,
        hamiltonian_eigenvalues=eigenvalues,
        stable_dimension=int(stable_dimension),
        stable_graph_condition=condition_u,
        stable_graph_imaginary_norm=graph_imaginary_norm,
        imaginary_axis_distance=imaginary_axis_distance,
        selection_tolerance=selection_tolerance,
        raw_real_part_asymmetry=raw_real_part_asymmetry,
    )


def solve_lq_system(local_system: LocalSystem) -> LqSolution:
    p = local_system.parameters
    anchor = local_system.anchor
    A = local_system.A
    B = local_system.B.reshape(-1, 1)
    Q = local_system.Q
    rho = p.rho
    gamma = anchor.gamma
    chi = anchor.chi

    A_e = A[:2, :2]
    A_r = A[2:, 2:]
    D = local_system.D
    b = np.array([[0.0], [1.0]])
    Q_rr = Q[2:, 2:]
    Q_er = Q[:2, 2:]
    Q_ee = Q[:2, :2]

    invariant = invariant_subspace_real_block(A_r, Q_rr, rho, chi)
    closed = closed_form_real_block(rho, gamma, chi)
    H_rr = invariant.H_rr
    A_rc = A_r + chi * b @ b.T @ H_rr

    # Sylvester block: (A_e^T - rho I) H_er + H_er A_rc = -(Q_er + D^T H_rr).
    H_er = solve_sylvester(A_e.T - rho * np.eye(2), A_rc, -(Q_er + D.T @ H_rr))

    # Discounted-Lyapunov block: (A_e - rho I/2)^T H_ee + H_ee (A_e - rho I/2) = -K_ee.
    K_ee = Q_ee + D.T @ H_er.T + H_er @ D + chi * H_er @ b @ b.T @ H_er.T
    L_e = A_e - rho * np.eye(2) / 2.0
    H_ee_raw = solve_sylvester(L_e.T, L_e, -K_ee)
    H_ee = _symmetrize(H_ee_raw)

    H = np.block([[H_ee, H_er], [H_er.T, H_rr]])
    H = _symmetrize(H)
    A_c = A + chi * B @ B.T @ H

    Sigma_e_hat = local_system.Sigma_e_hat
    stationary_covariance_raw = solve_sylvester(A_c, A_c.T, -(Sigma_e_hat @ Sigma_e_hat.T))
    stationary_covariance_y = _symmetrize(stationary_covariance_raw)
    raw_symmetry_errors = {
        "H_ee": _asymmetry(H_ee_raw),
        "stationary_covariance_y": _asymmetry(stationary_covariance_raw),
    }

    F = np.block(
        [
            [A_c, np.zeros((4, 1))],
            [anchor.comprehensive_resources_bar * local_system.safe_rate_y.reshape(1, 4), np.zeros((1, 1))],
        ]
    )
    resolvent = np.linalg.inv(rho * np.eye(5) - F)

    return LqSolution(
        H=H,
        H_ee=H_ee,
        H_er=H_er,
        H_rr=H_rr,
        A_rc=A_rc,
        A_c=A_c,
        F=F,
        resolvent=resolvent,
        stationary_covariance_y=stationary_covariance_y,
        raw_symmetry_errors=raw_symmetry_errors,
        closed_form=closed,
        invariant_subspace=invariant,
    )
