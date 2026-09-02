from __future__ import annotations

import numpy as np
import pytest

from tai_public_finance.cs002_nonlinear_transition.continuation import run_continuation

AMPLITUDES = [0.0, 0.25, 0.5, 1.0]
DELTA_K = 0.01
DELTA_TAU = 0.01


@pytest.fixture(scope="module")
def route_a(cs001_local_system, cs001_solution):
    return run_continuation(
        "lq_path_continuation", DELTA_K, DELTA_TAU, AMPLITUDES, horizon=40.0, n_mesh_points=161,
        terminal_convention="lq_stable_manifold", local_system=cs001_local_system, solution=cs001_solution,
    )


@pytest.fixture(scope="module")
def route_b(cs001_local_system, cs001_solution):
    return run_continuation(
        "crude_direct", DELTA_K, DELTA_TAU, AMPLITUDES, horizon=40.0, n_mesh_points=161,
        terminal_convention="lq_stable_manifold", local_system=cs001_local_system, solution=cs001_solution,
    )


def test_every_amplitude_checkpoint_is_accepted(route_a, route_b):
    assert route_a.all_accepted, [(c.amplitude, c.failure_message) for c in route_a.checkpoints if not c.accepted]
    assert route_b.all_accepted, [(c.amplitude, c.failure_message) for c in route_b.checkpoints if not c.accepted]
    assert len(route_a.checkpoints) == len(AMPLITUDES)
    assert len(route_b.checkpoints) == len(AMPLITUDES)


def test_zero_amplitude_checkpoint_is_the_anchor(route_a):
    zero = route_a.checkpoints[0]
    assert zero.amplitude == 0.0
    path = zero.path_at(np.array([0.0, 20.0, 40.0]))
    np.testing.assert_allclose(path[0, :], 0.0, atol=1e-9)
    np.testing.assert_allclose(path[2, :], 1.0, atol=1e-8)
    np.testing.assert_allclose(path[3, :], 0.0, atol=1e-8)


def test_two_continuation_routes_agree_at_every_amplitude(route_a, route_b):
    """Route A (LQ-path warm-started continuation) and route B (crude direct
    solve, no warm start) are materially different numerical paths to the
    same BVP; agreement here is real evidence against branch sensitivity,
    not a tautology."""

    check_t = np.linspace(0.0, 40.0, 21)
    for ckpt_a, ckpt_b in zip(route_a.checkpoints, route_b.checkpoints):
        assert ckpt_a.amplitude == ckpt_b.amplitude
        path_a = ckpt_a.path_at(check_t)
        path_b = ckpt_b.path_at(check_t)
        np.testing.assert_allclose(path_a, path_b, atol=5e-7, rtol=1e-6, err_msg=f"routes disagree at amplitude={ckpt_a.amplitude}")


def test_continuation_checkpoints_are_ordered_by_amplitude(route_a):
    amplitudes = [c.amplitude for c in route_a.checkpoints]
    assert amplitudes == sorted(amplitudes)
    assert amplitudes == AMPLITUDES
