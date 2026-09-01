"""Frozen regression snapshot for the lq_farhi_illustrative_smoke_v1 baseline.

These values were confirmed against an independently authored reference
implementation (the pre-existing, separately written prototype at
../TAI public finnace codex/implementation/lq-ramsey-numerics, itself
tested against the same closed-form identities) — both implementations
agree to float64 precision despite sharing no code, which is the
independent-cross-check CS001 asks for beyond this repository's own
finite-difference and closed-form checks. A future change to a primitive,
equation, or tolerance must update this snapshot deliberately, not
silently.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_H_rr_matches_the_frozen_snapshot(baseline):
    expected = np.array([[0.443803780453, 0.130463054911], [0.130463054911, -0.079553816607]])
    np.testing.assert_allclose(baseline["solution"].H_rr, expected, atol=2e-11)


def test_linear_fiscal_wealth_matches_the_frozen_snapshot(baseline):
    expected = np.array([13.003071526358, 3.542617109345, 1.0, 0.0])
    np.testing.assert_allclose(baseline["local_system"].linear_fiscal_wealth, expected, atol=2e-11)


def test_real_closed_loop_roots_match_the_frozen_snapshot(baseline):
    roots = np.sort_complex(baseline["diagnostics"].closed_loop["real_closed_loop_eigenvalues"])
    expected = np.sort_complex(np.array([-0.076027776479 - 0.078265175983j, -0.076027776479 + 0.078265175983j]))
    np.testing.assert_allclose(roots, expected, atol=2e-11)


def test_leading_portfolio_position_matches_the_frozen_snapshot(baseline):
    assert baseline["portfolio"].leading_unconstrained_position == pytest.approx(0.4464893494362929, abs=2e-11)


def test_portfolio_curvature_matches_the_frozen_snapshot(baseline):
    assert baseline["portfolio"].portfolio_curvature == pytest.approx(-0.000946408610616, abs=2e-13)


def test_access_and_hedge_consumption_equivalents_match_the_frozen_snapshot(baseline):
    assert baseline["portfolio"].access_consumption_equivalent_leading == pytest.approx(9.43345744455e-05, abs=2e-14)
    assert baseline["portfolio"].hedge_consumption_equivalent_leading == pytest.approx(0.069541588442, abs=2e-11)


def test_anchor_levels_match_the_documented_illustrative_calibration(baseline):
    anchor = baseline["anchor"]
    assert anchor.rental_rate_bar == pytest.approx(0.1204054, abs=1e-6)
    assert anchor.output_bar == pytest.approx(0.3541336, abs=1e-6)
    assert anchor.wage_income_bar == pytest.approx(0.2337282, abs=1e-6)
    assert anchor.tax_base_bar == pytest.approx(0.0404054, abs=1e-6)
    assert anchor.fiscal_resources_bar == pytest.approx(0.2539309, abs=1e-6)
    assert anchor.comprehensive_resources_bar == pytest.approx(12.56915, abs=1e-4)


def test_minimum_boundary_slack_matches_the_frozen_snapshot(baseline):
    boundary = baseline["irfs"]["boundary_summary"]
    assert boundary["specialisation_margin_automation_composite"] == pytest.approx(0.989752255812, abs=1e-9)
    assert boundary["specialisation_margin_new_task_composite"] == pytest.approx(0.57670578259, abs=1e-9)
    assert boundary["transfer_level"] == pytest.approx(0.0191960261842, abs=1e-9)
