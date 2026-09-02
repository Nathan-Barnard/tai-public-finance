from __future__ import annotations

import numpy as np
import pytest

from tai_public_finance.cs002_nonlinear_transition.terminal import (
    anchor_value_tail,
    crude_costates,
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
