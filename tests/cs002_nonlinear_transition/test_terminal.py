from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tai_public_finance.cs002_nonlinear_transition.terminal import (
    anchor_value_tail,
    crude_costates,
    lq_deviation_vector,
    lq_quadratic_value_tail,
    lq_stable_manifold_costates,
)


def test_lq_tail_at_the_anchor_matches_the_crude_tail(cs001_local_system, cs001_solution):
    anchor = cs001_local_system.anchor
    lq = lq_stable_manifold_costates(anchor.capital_bar, anchor.tax_rate_bar, cs001_local_system, cs001_solution)
    crude = crude_costates()
    assert lq.ell == pytest.approx(crude.ell, abs=1e-12)
    assert lq.m == pytest.approx(crude.m, abs=1e-12)


def test_lq_and_anchor_value_tails_agree_at_the_anchor(cs001_local_system, cs001_solution):
    anchor = cs001_local_system.anchor
    quadratic = lq_quadratic_value_tail(anchor.capital_bar, anchor.tax_rate_bar, cs001_local_system, cs001_solution)
    crude = anchor_value_tail(cs001_local_system)
    assert quadratic == pytest.approx(crude, abs=1e-12)
    assert quadratic == pytest.approx(anchor.fiscal_wealth_bar, abs=1e-12)


def test_ell_lq_uses_j_K_over_K_not_j_k_directly(cs001_local_system, cs001_solution):
    """Away from the anchor the chain-rule K-division is load-bearing: ell_lq
    must equal J_k/K, and since K != K_bar here the two differ measurably."""

    anchor = cs001_local_system.anchor
    capital = anchor.capital_bar * 1.05  # k = log(1.05) != 0
    tau = anchor.tax_rate_bar + 0.01

    lq = lq_stable_manifold_costates(capital, tau, cs001_local_system, cs001_solution)

    k = np.log(capital / anchor.capital_bar)
    y = np.array([0.0, 0.0, k, tau - 0.5])
    j_k_raw = (cs001_local_system.linear_fiscal_wealth + cs001_solution.H @ y)[2]  # dJ/dk, NOT ell

    assert lq.ell == pytest.approx(j_k_raw / capital, rel=1e-12)
    assert lq.ell != pytest.approx(j_k_raw, rel=1e-3)  # dividing by K != 1 must actually move the number


def test_lq_costates_are_continuous_in_displacement(cs001_local_system, cs001_solution):
    """A small displacement should produce a small costate change -- catches
    an accidental sign flip or unit error in the mapping."""

    anchor = cs001_local_system.anchor
    base = lq_stable_manifold_costates(anchor.capital_bar, anchor.tax_rate_bar, cs001_local_system, cs001_solution)
    bumped = lq_stable_manifold_costates(anchor.capital_bar * 1.001, anchor.tax_rate_bar + 0.001, cs001_local_system, cs001_solution)
    assert abs(bumped.ell - base.ell) < 0.1
    assert abs(bumped.m - base.m) < 0.1


def test_terminal_generalizes_to_arbitrary_tau_bar_and_capital_bar():
    """CS002 D2 mandatory generalization/repair: the terminal-gradient
    helper previously hard-coded `tau - 0.5` (D0-D1's recorded calibration
    always has tax_rate_bar=0.5, so the bug was numerically invisible there
    -- compute_steady_state currently always sets tax_rate_bar=0.5, so this
    test uses a SYNTHETIC anchor/local_system/solution, duck-typed via
    SimpleNamespace to expose only the attributes lq_deviation_vector /
    lq_stable_manifold_costates / lq_quadratic_value_tail actually read, to
    prove the helper itself generalizes -- independent of whether today's
    calibration pipeline can produce tau_bar != 0.5."""

    tau_bar = 0.3
    capital_bar = 2.5
    z_bar, x_bar = -1.5, 0.2
    fiscal_wealth_bar = 3.0
    # j_k = 1, NORMALIZED (CS001's own j[2]==1 anchor diagnostic is
    # capital_bar-INDEPENDENT -- it is q_D=1's "marginal fiscal value of
    # capital is 1" in NORMALIZED units, not j_k=capital_bar). Physical-unit
    # costates require the caller to convert with the K_bar multiplier
    # (terminal.py's D2 review repair); j_tau=0 at the anchor.
    linear_fiscal_wealth = np.array([0.1, -0.05, 1.0, 0.0])
    H = np.diag([0.01, 0.02, 0.5, 0.3])

    anchor = SimpleNamespace(z_bar=z_bar, x_bar=x_bar, capital_bar=capital_bar, tax_rate_bar=tau_bar, fiscal_wealth_bar=fiscal_wealth_bar)
    local_system = SimpleNamespace(anchor=anchor, linear_fiscal_wealth=linear_fiscal_wealth)
    solution = SimpleNamespace(H=H)

    # Zero displacement (K=capital_bar, tau=tau_bar, z=z_bar, x=x_bar) must
    # return the anchor's own costates (j_k=1 -> ell=1, j_tau=0 -> m=0) and
    # fiscal-wealth level, exactly CS001's own j[2]==1, j[3]==0 diagnostic --
    # only true if tau-tau_bar and k=log(K/capital_bar) are BOTH exactly
    # zero at (capital_bar, tau_bar), which requires reading them off the
    # anchor rather than off literals 1 and 0.5.
    zero_costates = lq_stable_manifold_costates(capital_bar, tau_bar, local_system, solution)
    assert zero_costates.ell == pytest.approx(1.0, abs=1e-12)
    assert zero_costates.m == pytest.approx(0.0, abs=1e-12)

    zero_tail = lq_quadratic_value_tail(capital_bar, tau_bar, local_system, solution)
    assert zero_tail == pytest.approx(fiscal_wealth_bar, abs=1e-12)

    # Away from the anchor: the deviation vector must use tau-tau_bar (NOT
    # tau-0.5) and log(K/capital_bar) (NOT log(K/1)).
    displaced_tau = tau_bar + 0.02  # = 0.32; a hard-coded "-0.5" would instead read -0.18
    displaced_capital = capital_bar * 1.05
    y = lq_deviation_vector(displaced_capital, displaced_tau, local_system)
    assert y[3] == pytest.approx(0.02, abs=1e-12)
    assert y[2] == pytest.approx(np.log(1.05), abs=1e-12)


def test_terminal_generalizes_to_a_displaced_exogenous_state():
    """CS002 D2: the (z, x) components of the deviation vector must be
    current-state minus the ANCHOR's own (z_bar, x_bar), not always zero --
    needed for a finite-horizon terminal state where z, x have not yet fully
    mean-reverted. Omitting z/x reproduces the D0-D1 zero-deviation case."""

    anchor = SimpleNamespace(z_bar=-1.5, x_bar=0.2, capital_bar=1.0, tax_rate_bar=0.5, fiscal_wealth_bar=3.0)
    local_system = SimpleNamespace(anchor=anchor, linear_fiscal_wealth=np.array([0.1, -0.05, 1.0, 0.0]))

    y = lq_deviation_vector(1.0, 0.5, local_system, z=-1.4, x=0.15)
    np.testing.assert_allclose(y, [0.1, -0.05, 0.0, 0.0], atol=1e-12)

    y_default = lq_deviation_vector(1.0, 0.5, local_system)
    np.testing.assert_allclose(y_default, [0.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_terminal_map_converts_normalized_coefficients_to_physical_units_when_capital_bar_is_not_one():
    """CS002 D2 review repair (finding 1): CS001 stores NORMALIZED
    fiscal-wealth coefficients (j, H) -- j_k=1 at the anchor regardless of
    K_bar, not j_k=K_bar (see the corrected fixture in
    test_terminal_generalizes_to_arbitrary_tau_bar_and_capital_bar above).
    Converting to PHYSICAL units requires an explicit K_bar multiplier:

        J_y^L = K_bar*(j + H@y),   J^L = J_bar + K_bar*(j@y + 0.5 y'Hy).

    `lq_stable_manifold_costates`/`lq_quadratic_value_tail` previously
    omitted this factor entirely -- silently a no-op at K_bar=1 (every
    calibration run so far), silently wrong otherwise. This uses K_bar=2.5,
    NORMALIZED j_k=1 (per the review's explicit instruction: "use normalized
    j_k=1, not j_k=K_bar")."""

    capital_bar = 2.5
    tau_bar = 0.3
    z_bar, x_bar = -1.5, 0.2
    fiscal_wealth_bar = 3.0
    j = np.array([0.1, -0.05, 1.0, 0.0])  # normalized: j_k=1, j_tau=0 at the anchor
    H = np.diag([0.01, 0.02, 0.5, 0.3])

    anchor = SimpleNamespace(z_bar=z_bar, x_bar=x_bar, capital_bar=capital_bar, tax_rate_bar=tau_bar, fiscal_wealth_bar=fiscal_wealth_bar)
    local_system = SimpleNamespace(anchor=anchor, linear_fiscal_wealth=j)
    solution = SimpleNamespace(H=H)

    # 1. The anchor gives ell=1 (m=0): the K_bar multiplier and the 1/K_T
    # chain-rule division exactly cancel at K_T=K_bar, same as CS001's own
    # j[2]==1 diagnostic, but now via physical units rather than by accident.
    anchor_costates = lq_stable_manifold_costates(capital_bar, tau_bar, local_system, solution)
    assert anchor_costates.ell == pytest.approx(1.0, abs=1e-12)
    assert anchor_costates.m == pytest.approx(0.0, abs=1e-12)

    # Displaced point, away from the anchor in every coordinate.
    capital, tau, z, x = 2.6, 0.32, -1.49, 0.18
    y = lq_deviation_vector(capital, tau, local_system, z=z, x=x)

    # 2 & 3. Displaced (k, tau, z, x) produce the analytic K_bar-scaled
    # gradient, and the physical value tail carries the same multiplier.
    expected_gradient = capital_bar * (j + H @ y)
    expected_ell = expected_gradient[2] / capital
    expected_m = expected_gradient[3]
    actual_costates = lq_stable_manifold_costates(capital, tau, local_system, solution, z=z, x=x)
    assert actual_costates.ell == pytest.approx(expected_ell, rel=1e-12)
    assert actual_costates.m == pytest.approx(expected_m, rel=1e-12)

    expected_tail = fiscal_wealth_bar + capital_bar * (float(j @ y) + 0.5 * float(y @ H @ y))
    actual_tail = lq_quadratic_value_tail(capital, tau, local_system, solution, z=z, x=x)
    assert actual_tail == pytest.approx(expected_tail, rel=1e-12)

    # 4. An independent finite-difference derivative of the physical value
    # tail (central difference in K holding tau/z/x fixed, and in tau
    # holding K/z/x fixed) agrees with the returned costates -- this checks
    # the whole formula (including the K_bar factor) by a completely
    # different method than re-deriving the same closed form.
    h_k = 1e-6
    tail_plus_k = lq_quadratic_value_tail(capital + h_k, tau, local_system, solution, z=z, x=x)
    tail_minus_k = lq_quadratic_value_tail(capital - h_k, tau, local_system, solution, z=z, x=x)
    fd_ell = (tail_plus_k - tail_minus_k) / (2.0 * h_k)
    assert fd_ell == pytest.approx(actual_costates.ell, rel=1e-5)

    h_tau = 1e-6
    tail_plus_tau = lq_quadratic_value_tail(capital, tau + h_tau, local_system, solution, z=z, x=x)
    tail_minus_tau = lq_quadratic_value_tail(capital, tau - h_tau, local_system, solution, z=z, x=x)
    fd_m = (tail_plus_tau - tail_minus_tau) / (2.0 * h_tau)
    assert fd_m == pytest.approx(actual_costates.m, rel=1e-6)

    # 5. Negative control: the old unscaled implementation (j+H@y with no
    # K_bar factor, and J_bar+j@y+0.5y'Hy with no K_bar factor) would fail
    # this test -- it disagrees with the correct physical-unit result by a
    # whole factor of K_bar=2.5, far beyond any numerical tolerance.
    old_buggy_gradient = j + H @ y
    old_buggy_ell = old_buggy_gradient[2] / capital
    old_buggy_m = old_buggy_gradient[3]
    assert old_buggy_ell == pytest.approx(actual_costates.ell / capital_bar, rel=1e-12)
    assert abs(old_buggy_ell - actual_costates.ell) > 0.3
    assert old_buggy_m == pytest.approx(actual_costates.m / capital_bar, rel=1e-12)

    old_buggy_tail = fiscal_wealth_bar + float(j @ y) + 0.5 * float(y @ H @ y)
    assert abs(old_buggy_tail - actual_tail) > 0.05
