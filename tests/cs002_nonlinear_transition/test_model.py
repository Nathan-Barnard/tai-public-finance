from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tai_public_finance.cs002_nonlinear_transition.model import capital_derivatives, capital_from_log, characteristic_rates, log_from_capital
from tai_public_finance.primitives import evaluate_smooth_branch


def test_log_capital_round_trip():
    capital_bar = 1.3
    for k in (-0.5, -0.01, 0.0, 0.01, 0.7):
        capital = capital_from_log(k, capital_bar)
        assert capital > 0.0
        assert log_from_capital(capital, capital_bar) == pytest.approx(k, abs=1e-12)


def test_log_from_capital_rejects_nonpositive():
    with pytest.raises(ValueError):
        log_from_capital(0.0, 1.0)
    with pytest.raises(ValueError):
        log_from_capital(-1.0, 1.0)


@pytest.mark.parametrize("z, x, capital, tau", [(-1.8, 0.0, 1.0, 0.5), (-1.5, 0.3, 1.4, 0.2), (-2.1, -0.4, 0.6, 0.7)])
def test_capital_derivatives_match_finite_differences(primitives, z, x, capital, tau):
    """Independent check of the closed-form K-derivatives against central
    finite differences of evaluate_smooth_branch -- the same cross-check
    discipline CS001's diagnostics.finite_difference_primitive_checks uses
    for its own local-system gradients."""

    state = evaluate_smooth_branch(z, x, capital, tau, primitives)
    derivs = capital_derivatives(state, primitives)

    step = 1e-6 * capital
    plus = evaluate_smooth_branch(z, x, capital + step, tau, primitives)
    minus = evaluate_smooth_branch(z, x, capital - step, tau, primitives)

    fd_output_K = (plus.output - minus.output) / (2.0 * step)
    fd_rental_K = (plus.rental_rate - minus.rental_rate) / (2.0 * step)
    fd_tax_base_K = (plus.tax_base - minus.tax_base) / (2.0 * step)
    fd_fiscal_K = (plus.fiscal_resources - minus.fiscal_resources) / (2.0 * step)
    fd_growth_K = (plus.capital_growth - minus.capital_growth) / (2.0 * step)

    assert derivs.output_K == pytest.approx(fd_output_K, rel=1e-6)
    assert derivs.rental_rate_K == pytest.approx(fd_rental_K, rel=1e-6)
    assert derivs.tax_base_K == pytest.approx(fd_tax_base_K, rel=1e-6)
    assert derivs.fiscal_resources_K == pytest.approx(fd_fiscal_K, rel=1e-6)
    assert derivs.capital_growth_K == pytest.approx(fd_growth_K, rel=1e-6)


def test_anchor_is_a_fixed_point_of_the_characteristic_system(primitives, anchor):
    """Zero-displacement checking-matrix row: at the anchor the maintained
    interior branch requires ell=1, m=0, nu=0, and the whole state should be
    an exact fixed point of the characteristic RHS (k_dot=tau_dot=ell_dot=m_dot=0)."""

    rates = characteristic_rates(
        k=0.0, tau=anchor.tax_rate_bar, ell=1.0, m=0.0, z_bar=anchor.z_bar, x_bar=anchor.x_bar, capital_bar=anchor.capital_bar, p=primitives
    )
    assert rates.nu == pytest.approx(0.0, abs=1e-12)
    assert rates.k_dot == pytest.approx(0.0, abs=1e-12)
    assert rates.tau_dot == pytest.approx(0.0, abs=1e-12)
    assert rates.ell_dot == pytest.approx(0.0, abs=1e-10)
    assert rates.m_dot == pytest.approx(0.0, abs=1e-12)
    assert rates.r0 == pytest.approx(primitives.rho, abs=1e-12)
    assert rates.state.rental_rate - primitives.depreciation_rate == pytest.approx(2.0 * primitives.rho, rel=1e-10)


def test_r16_fixed_tax_capital_transition_matches_closed_form(primitives, anchor):
    """R16 (model-tracker): holding (z, alpha, tau) fixed, u=(K/K*)**(1-alpha)
    linearises the capital law to u_dot = kappa_K*(1-u). Re-derived here from
    K_dot=K*g with g=(1-tau)(R^K-delta)-rho and R^K=alpha*Y/K (so R^K=C*K**(alpha-1)):
    at the fixed point K*, (1-tau)(C*(K*)**(alpha-1)-delta)=rho, and substituting
    u=(K/K*)**(1-alpha) gives exactly kappa_K=(1-alpha)*(rho+(1-tau)*delta).

    This is an analytic closed-form solution wholly independent of the BVP
    machinery; it is checked here only against a direct `solve_ivp` integration
    of k_dot=g (tau frozen, i.e. nu forced to 0), which itself calls
    characteristic_rates -- the same RHS the BVP solver will use -- so this
    doubles as the required fixed-tax analytic-transition check (CS002 Block D0)."""

    tau = 0.6  # deliberately off the anchor tax rate, to exercise a nontrivial transition
    alpha = anchor.alpha_bar  # alpha depends only on x, frozen at x_bar here
    delta = primitives.depreciation_rate
    rho = primitives.rho
    capital_bar = anchor.capital_bar

    # K* solves R^K(K*) = rho/(1-tau) + delta; back it out from the smooth branch at K=capital_bar.
    state_bar = evaluate_smooth_branch(anchor.z_bar, anchor.x_bar, capital_bar, tau, primitives)
    target_rental = rho / (1.0 - tau) + delta
    capital_star = capital_bar * (state_bar.rental_rate / target_rental) ** (1.0 / (1.0 - alpha))
    state_star = evaluate_smooth_branch(anchor.z_bar, anchor.x_bar, capital_star, tau, primitives)
    assert state_star.rental_rate == pytest.approx(target_rental, rel=1e-10)
    assert (1.0 - tau) * (state_star.rental_rate - delta) - rho == pytest.approx(0.0, abs=1e-9)

    kappa_K = (1.0 - alpha) * (rho + (1.0 - tau) * delta)

    k0 = math.log(0.6 * capital_star / capital_bar)  # start well displaced from K*
    u0 = (math.exp(k0) * capital_bar / capital_star) ** (1.0 - alpha)

    def rhs(t, y):
        rates = characteristic_rates(y[0], tau, ell=1.0, m=0.0, z_bar=anchor.z_bar, x_bar=anchor.x_bar, capital_bar=capital_bar, p=primitives)
        return [rates.k_dot]

    horizon = 60.0
    grid = np.linspace(0.0, horizon, 25)
    solution = solve_ivp(rhs, (0.0, horizon), [k0], t_eval=grid, method="DOP853", rtol=1e-12, atol=1e-14)
    assert solution.success

    capital_path = capital_bar * np.exp(solution.y[0])
    u_numeric = (capital_path / capital_star) ** (1.0 - alpha)
    u_closed_form = 1.0 - (1.0 - u0) * np.exp(-kappa_K * grid)

    np.testing.assert_allclose(u_numeric, u_closed_form, rtol=1e-6, atol=1e-8)
    assert np.all(np.diff(u_numeric) > 0.0)  # R16: monotone convergence to K*
    assert u_numeric[-1] > u_numeric[0]
    assert 1.0 - (1.0 - u0) * math.exp(-kappa_K * horizon) == pytest.approx(u_numeric[-1], rel=1e-6)
