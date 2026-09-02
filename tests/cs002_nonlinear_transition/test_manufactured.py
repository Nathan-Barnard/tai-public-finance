"""CS002 Block D0 requirement 3: a manufactured BVP, solved through the exact
same `solve_two_point_bvp` interface the real economic problem uses, checked
against a closed-form matrix-exponential solution that never calls
scipy.integrate.solve_bvp. Zero economic content -- this validates the
generic two-point-BVP scaffolding on its own."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from tai_public_finance.cs002_nonlinear_transition.bvp import solve_two_point_bvp

# Mixed stable/unstable spectrum (a genuine two-point BVP, not a one-sided IVP):
# a lightly-coupled saddle-like 4x4 system, deliberately unrelated to the
# economic A_rc/A_c matrices.
_M = np.array(
    [
        [-0.5, 0.1, 0.0, 0.0],
        [0.05, -0.3, 0.02, 0.0],
        [0.0, 0.03, 0.4, 0.01],
        [0.0, 0.0, 0.02, 0.6],
    ]
)
_T = 3.0
_YA_TARGET = np.array([1.0, -0.5])
_YB_TARGET = np.array([0.2, 0.3])


def _exact_solution(t: np.ndarray) -> np.ndarray:
    e_full = expm(_M * _T)
    block = e_full[2:4, 2:4]
    rhs = _YB_TARGET - e_full[2:4, 0:2] @ _YA_TARGET
    c23 = np.linalg.solve(block, rhs)
    c = np.array([_YA_TARGET[0], _YA_TARGET[1], c23[0], c23[1]])
    return np.stack([expm(_M * ti) @ c for ti in np.atleast_1d(t)], axis=1)


def _fun(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    del t
    return _M @ y


def _bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
    return np.array([ya[0] - _YA_TARGET[0], ya[1] - _YA_TARGET[1], yb[2] - _YB_TARGET[0], yb[3] - _YB_TARGET[1]])


def test_manufactured_bvp_recovers_the_matrix_exponential_solution():
    x_mesh = np.linspace(0.0, _T, 11)
    y_guess = np.zeros((4, x_mesh.size))
    y_guess[0, :] = _YA_TARGET[0]
    y_guess[1, :] = _YA_TARGET[1]

    result = solve_two_point_bvp(_fun, _bc, x_mesh, y_guess, tol=1e-11)
    assert result.success, result.message

    off_mesh_t = np.linspace(0.0, _T, 37)  # deliberately not the solver's own mesh
    numeric = result.sol(off_mesh_t)
    exact = _exact_solution(off_mesh_t)
    np.testing.assert_allclose(numeric, exact, atol=1e-8, rtol=1e-6)


def test_manufactured_bvp_boundary_conditions_are_met():
    x_mesh = np.linspace(0.0, _T, 11)
    y_guess = np.zeros((4, x_mesh.size))
    result = solve_two_point_bvp(_fun, _bc, x_mesh, y_guess, tol=1e-11)
    assert result.success

    ya = result.sol(np.array([0.0]))[:, 0]
    yb = result.sol(np.array([_T]))[:, 0]
    assert ya[0] == pytest.approx(_YA_TARGET[0], abs=1e-8)
    assert ya[1] == pytest.approx(_YA_TARGET[1], abs=1e-8)
    assert yb[2] == pytest.approx(_YB_TARGET[0], abs=1e-8)
    assert yb[3] == pytest.approx(_YB_TARGET[1], abs=1e-8)


def test_manufactured_bvp_is_robust_to_a_different_naive_guess():
    """A materially different (linear-interpolation) initial guess must
    converge to the SAME solution -- a cheap negative control for the
    branch-sensitivity check used later on the real economic problem."""

    x_mesh = np.linspace(0.0, _T, 15)
    y_guess = np.zeros((4, x_mesh.size))
    for row, (start, end) in enumerate(zip([*_YA_TARGET, 0.0, 0.0], [0.0, 0.0, *_YB_TARGET])):
        y_guess[row, :] = np.linspace(start, end, x_mesh.size)

    result = solve_two_point_bvp(_fun, _bc, x_mesh, y_guess, tol=1e-11)
    assert result.success

    off_mesh_t = np.linspace(0.0, _T, 37)
    numeric = result.sol(off_mesh_t)
    exact = _exact_solution(off_mesh_t)
    np.testing.assert_allclose(numeric, exact, atol=1e-7, rtol=1e-5)
