from __future__ import annotations

import numpy as np
import pytest

from tai_public_finance.cs002_nonlinear_transition.continuation_d2 import run_exogenous_shock_continuation
from tai_public_finance.cs002_nonlinear_transition.exogenous import invert_automation_share
from tai_public_finance.primitives.production import automation_share

AMPLITUDES = [0.0, 0.25, 0.5, 1.0]
HORIZON = 40.0
MESH_POINTS = 161


@pytest.fixture(scope="module")
def productivity_target(anchor):
    return anchor.z_bar + 0.01


@pytest.fixture(scope="module")
def automation_target(primitives, anchor):
    alpha_target = automation_share(anchor.x_bar, primitives) + 0.01
    return invert_automation_share(alpha_target, primitives)


@pytest.fixture(scope="module")
def productivity_warm_start(cs001_local_system, cs001_solution, productivity_target, anchor):
    return run_exogenous_shock_continuation(
        "warm_start", "productivity", productivity_target, anchor.x_bar, AMPLITUDES, HORIZON, MESH_POINTS,
        "lq_stable_manifold", cs001_local_system, cs001_solution,
    )


@pytest.fixture(scope="module")
def productivity_crude(cs001_local_system, cs001_solution, productivity_target, anchor):
    return run_exogenous_shock_continuation(
        "crude_direct", "productivity", productivity_target, anchor.x_bar, AMPLITUDES, HORIZON, MESH_POINTS,
        "lq_stable_manifold", cs001_local_system, cs001_solution,
    )


@pytest.fixture(scope="module")
def automation_warm_start(cs001_local_system, cs001_solution, automation_target, anchor):
    return run_exogenous_shock_continuation(
        "warm_start", "automation", anchor.z_bar, automation_target, AMPLITUDES, HORIZON, MESH_POINTS,
        "lq_stable_manifold", cs001_local_system, cs001_solution,
    )


def test_every_amplitude_checkpoint_is_accepted(productivity_warm_start, productivity_crude, automation_warm_start):
    for run in (productivity_warm_start, productivity_crude, automation_warm_start):
        assert run.all_accepted, [(c.amplitude, c.failure_message) for c in run.checkpoints if not c.accepted]
        assert len(run.checkpoints) == len(AMPLITUDES)


def test_zero_amplitude_checkpoint_is_the_undisplaced_anchor(productivity_warm_start, automation_warm_start):
    """CS002 D2 required check #3: zero exogenous displacement must collapse
    exactly to the D1 anchor fixed point (k=0, tau=tau_bar, ell=1, m=0),
    constant over the whole horizon -- independent of shock direction."""

    for run in (productivity_warm_start, automation_warm_start):
        zero = run.checkpoints[0]
        assert zero.amplitude == 0.0
        assert zero.z0 == pytest.approx(run.checkpoints[0].exogenous_path.z_bar, abs=1e-14)
        assert zero.x0 == pytest.approx(run.checkpoints[0].exogenous_path.x_bar, abs=1e-14)
        path = zero.path_at(np.array([0.0, 20.0, 40.0]))
        np.testing.assert_allclose(path[0, :], 0.0, atol=1e-9)  # k
        np.testing.assert_allclose(path[2, :], 1.0, atol=1e-8)  # ell
        np.testing.assert_allclose(path[3, :], 0.0, atol=1e-8)  # m


def test_two_continuation_routes_agree_at_every_amplitude(productivity_warm_start, productivity_crude):
    """Route A (warm-started continuation) and route B (crude direct solve,
    no warm start) are materially different numerical paths to the same
    BVP; agreement here is real evidence against branch sensitivity."""

    check_t = np.linspace(0.0, HORIZON, 21)
    for ckpt_a, ckpt_b in zip(productivity_warm_start.checkpoints, productivity_crude.checkpoints):
        assert ckpt_a.amplitude == ckpt_b.amplitude
        path_a = ckpt_a.path_at(check_t)
        path_b = ckpt_b.path_at(check_t)
        np.testing.assert_allclose(path_a, path_b, atol=5e-7, rtol=1e-6, err_msg=f"routes disagree at amplitude={ckpt_a.amplitude}")


def test_nonzero_amplitude_checkpoints_are_not_the_anchor(productivity_warm_start, automation_warm_start):
    """Negative control: a shock that never moves the path would make every
    other check in this module vacuous."""

    for run in (productivity_warm_start, automation_warm_start):
        displaced = run.checkpoints[-1]  # amplitude=1.0
        path = displaced.path_at(np.array([1.0]))
        moved = abs(path[0, 0]) > 1e-6 or abs(path[1, 0] - 0.5) > 1e-6 or abs(path[2, 0] - 1.0) > 1e-6 or abs(path[3, 0]) > 1e-6
        assert moved, "full-amplitude shock produced no visible response"


def test_automation_target_inverts_to_the_correct_alpha(primitives, anchor, automation_target):
    alpha_bar = automation_share(anchor.x_bar, primitives)
    assert automation_share(automation_target, primitives) == pytest.approx(alpha_bar + 0.01, abs=1e-12)
