from __future__ import annotations

import numpy as np
import pytest

from tai_public_finance.cs002_nonlinear_transition.exogenous import (
    ExogenousPath,
    invert_automation_share,
    propagate_exogenous_numerically,
    x_path,
    z_path,
)
from tai_public_finance.primitives.production import automation_share


def test_analytic_ou_paths_agree_with_numerical_propagation(primitives):
    """CS002 D2 handoff: 'Check these against a numerical propagation.'"""

    kappa_z, kappa_x = primitives.kappa_z, primitives.kappa_x
    z_bar, x_bar = -1.5, 0.2
    z0, x0 = z_bar + 0.01, x_bar - 0.03
    t_grid = np.linspace(0.0, 80.0, 401)

    z_analytic = z_path(t_grid, z0, z_bar, kappa_z)
    x_analytic = x_path(t_grid, x0, x_bar, kappa_x)
    z_numeric, x_numeric = propagate_exogenous_numerically(t_grid, z0, x0, z_bar, x_bar, kappa_z, kappa_x)

    np.testing.assert_allclose(z_analytic, z_numeric, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(x_analytic, x_numeric, rtol=1e-9, atol=1e-11)


def test_ou_path_decays_to_the_mean_and_matches_the_closed_form_at_t0():
    kappa = 0.3
    bar, x0 = -2.0, -1.5
    assert z_path(0.0, x0, bar, kappa) == pytest.approx(x0, abs=1e-14)
    assert z_path(1000.0, x0, bar, kappa) == pytest.approx(bar, abs=1e-6)
    # Closed-form derivative check: d/dt z(t) at t=0 must equal kappa*(bar-x0).
    h = 1e-6
    numeric_derivative = (z_path(h, x0, bar, kappa) - z_path(-h, x0, bar, kappa)) / (2.0 * h)
    assert numeric_derivative == pytest.approx(kappa * (bar - x0), rel=1e-4)


def test_exogenous_path_dataclass_matches_the_module_functions():
    path = ExogenousPath(z0=0.01, x0=-0.02, z_bar=0.0, x_bar=0.0, kappa_z=0.2, kappa_x=0.3)
    t = np.linspace(0.0, 10.0, 5)
    z, x = path(t)
    np.testing.assert_allclose(z, z_path(t, 0.01, 0.0, 0.2))
    np.testing.assert_allclose(x, x_path(t, -0.02, 0.0, 0.3))
    np.testing.assert_allclose(path.z(t), z)
    np.testing.assert_allclose(path.x(t), x)


def test_zero_displacement_path_is_constant_at_the_bar():
    t = np.linspace(0.0, 50.0, 11)
    path = ExogenousPath(z0=-1.0, x0=0.5, z_bar=-1.0, x_bar=0.5, kappa_z=0.2, kappa_x=0.3)
    z, x = path(t)
    np.testing.assert_allclose(z, -1.0, atol=1e-14)
    np.testing.assert_allclose(x, 0.5, atol=1e-14)


def test_invert_automation_share_recovers_the_target_share(primitives):
    alpha_bar = automation_share(0.0, primitives)
    for delta in (0.01, -0.01, 0.1, -0.1):
        target = alpha_bar + delta
        x0 = invert_automation_share(target, primitives)
        assert automation_share(x0, primitives) == pytest.approx(target, abs=1e-12)


def test_invert_automation_share_rejects_out_of_range_targets(primitives):
    with pytest.raises(ValueError):
        invert_automation_share(primitives.alpha_lower - 0.001, primitives)
    with pytest.raises(ValueError):
        invert_automation_share(primitives.alpha_upper + 0.001, primitives)
    with pytest.raises(ValueError):
        invert_automation_share(primitives.alpha_lower, primitives)  # boundary itself, not strictly interior
