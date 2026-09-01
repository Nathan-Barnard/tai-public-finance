from __future__ import annotations

import numpy as np
import pytest

from tai_public_finance.cs001_lq_anchor.equations import COORDINATES


def test_coordinate_order_is_z_x_k_tau(baseline):
    assert COORDINATES == ("z", "x", "k", "tau_deviation")


def test_A_matrix_structure_matches_the_documented_state_law(baseline):
    local_system = baseline["local_system"]
    p = local_system.parameters
    anchor = local_system.anchor
    A = local_system.A
    # z and x mean-revert exogenously; only capital's row couples to them.
    np.testing.assert_allclose(A[0, :], [-p.kappa_z, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(A[1, :], [0.0, -p.kappa_x, 0.0, 0.0])
    # tau is a pure integrator of tax speed (control enters only through B).
    np.testing.assert_allclose(A[3, :], [0.0, 0.0, 0.0, 0.0])
    assert A[2, 0] == pytest.approx(anchor.rental_rate_bar / 2.0)
    assert A[2, 2] == pytest.approx(-anchor.gamma)
    assert A[2, 3] == pytest.approx(-2.0 * p.rho)


def test_B_moves_only_the_tax_state(baseline):
    B = baseline["local_system"].B
    np.testing.assert_allclose(B, [0.0, 0.0, 0.0, 1.0])


def test_D_is_the_lower_left_block_of_A(baseline):
    local_system = baseline["local_system"]
    np.testing.assert_allclose(local_system.D, local_system.A[2:, :2])


def test_Q_rr_cancels_to_the_closed_form_matrix(baseline):
    local_system = baseline["local_system"]
    p = local_system.parameters
    Q_rr = local_system.Q[2:, 2:]
    expected = np.array([[p.rho, 2.0 * p.rho], [2.0 * p.rho, 0.0]])
    np.testing.assert_allclose(Q_rr, expected, atol=1e-12)
    np.testing.assert_allclose(local_system.Q_rr_closed_form, expected, atol=1e-14)


def test_Q_is_symmetric(baseline):
    Q = baseline["local_system"].Q
    np.testing.assert_allclose(Q, Q.T, atol=1e-12)


def test_linear_fiscal_wealth_capital_and_tax_coefficients_are_exact(baseline):
    j = baseline["local_system"].linear_fiscal_wealth
    # j_k = 1 (a unit of capital is worth a unit of fiscal wealth at the margin
    # under q_D=1); j_t = 0 (the tax RATE has no first-order fiscal-wealth
    # effect at tau_bar=1/2, since the envelope condition is first order there).
    assert j[2] == pytest.approx(1.0, abs=1e-12)
    assert j[3] == pytest.approx(0.0, abs=1e-12)


def test_price_of_risk_beta_hat_is_positive(baseline):
    assert baseline["local_system"].price_of_risk.beta_hat > 0.0


def test_output_log_gradient_has_zero_tax_component(baseline):
    # Output depends on (z, x, K) only; tau enters the model only through the
    # tax base/fiscal-resources channel, not the production function.
    assert baseline["local_system"].output_y[3] == 0.0
    assert baseline["local_system"].rental_y[3] == 0.0
    assert baseline["local_system"].wage_y[3] == 0.0
