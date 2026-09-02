"""CS002 acceptance #7: nonlinear-minus-LQ path error must shrink at second
order as the displacement amplitude shrinks. Reports the observed ratios
explicitly rather than only asserting a boolean."""

from __future__ import annotations

import numpy as np
import pytest

from tai_public_finance.cs002_nonlinear_transition.bvp import economic_bc, economic_rhs, lq_path_initial_guess, solve_two_point_bvp
from tai_public_finance.cs002_nonlinear_transition.model import capital_from_log
from tai_public_finance.cs002_nonlinear_transition.terminal import lq_linear_kt_path

HORIZON = 40.0
AMPLITUDES = [1.0, 0.5, 0.25, 0.125, 0.0625]  # includes the frozen protocol's {1.0, 0.5, 0.25} plus two finer points
EXPECTED_ORDER = 2.0


@pytest.fixture(scope="module")
def path_errors(cs001_local_system, cs001_solution):
    anchor = cs001_local_system.anchor
    x_mesh = np.linspace(0.0, HORIZON, 161)
    check_t = np.linspace(0.0, HORIZON, 401)
    fun = economic_rhs(cs001_local_system)

    errors = {}
    for amplitude in AMPLITUDES:
        dk = 0.01 * amplitude
        dtau = 0.01 * amplitude
        tau0 = anchor.tax_rate_bar + dtau
        capital0 = capital_from_log(dk, anchor.capital_bar)
        y_guess = lq_path_initial_guess(x_mesh, dk, dtau, cs001_local_system, cs001_solution)
        bc = economic_bc(capital0, tau0, "lq_stable_manifold", cs001_local_system, cs001_solution)
        result = solve_two_point_bvp(fun, bc, x_mesh, y_guess, tol=1e-12)
        assert result.success

        nonlinear = result.sol(check_t)
        lq = lq_linear_kt_path(check_t, dk, dtau, cs001_solution)
        err_k = float(np.max(np.abs(nonlinear[0, :] - lq[0, :])))
        err_tau = float(np.max(np.abs((nonlinear[1, :] - 0.5) - lq[1, :])))
        errors[amplitude] = (err_k, err_tau)
    return errors


def test_lq_minus_nonlinear_error_shrinks_monotonically(path_errors):
    amps_desc = sorted(path_errors, reverse=True)
    err_k = [path_errors[a][0] for a in amps_desc]
    err_tau = [path_errors[a][1] for a in amps_desc]
    assert all(a > b for a, b in zip(err_k, err_k[1:])), err_k
    assert all(a > b for a, b in zip(err_tau, err_tau[1:])), err_tau


def test_lq_minus_nonlinear_error_shrinks_at_second_order(path_errors):
    amps_desc = sorted(path_errors, reverse=True)
    ratios_k = []
    ratios_tau = []
    for a1, a2 in zip(amps_desc, amps_desc[1:]):  # a1 = 2 * a2
        ratios_k.append(path_errors[a1][0] / path_errors[a2][0])
        ratios_tau.append(path_errors[a1][1] / path_errors[a2][1])

    expected_ratio = 2.0**EXPECTED_ORDER  # halving amplitude -> error / 4 for second order
    print("\nLQ-vs-nonlinear error ratios (halving amplitude), expected ~4.0 for second order:")
    for (a1, a2), rk, rt in zip(zip(amps_desc, amps_desc[1:]), ratios_k, ratios_tau):
        print(f"  amplitude {a1} -> {a2}: ratio_k={rk:.4f}  ratio_tau={rt:.4f}")

    for ratio in ratios_k + ratios_tau:
        assert ratio == pytest.approx(expected_ratio, rel=0.05), (ratios_k, ratios_tau)
