from __future__ import annotations

import csv

from tai_public_finance.cs001_lq_anchor.reporting import _write_csv, serializable


def test_csv_round_trip_does_not_rescale_tidy_irf_fields(tmp_path, baseline):
    rows = baseline["irfs"]["rows"][:5]
    path = tmp_path / "irfs.csv"
    _write_csv(path, rows)
    with path.open(newline="", encoding="utf-8") as handle:
        read_back = list(csv.DictReader(handle))
    for original, restored in zip(rows, read_back, strict=True):
        assert float(restored["worker_consumption_deviation"]) == original["worker_consumption_deviation"]
        assert float(restored["horizon_years"]) == original["horizon_years"]
        assert restored["experiment"] == original["experiment"]
        assert restored["regime"] == original["regime"]


def test_serializable_round_trips_dataclasses_and_complex_eigenvalues(baseline):
    payload = serializable(baseline["solution"])
    assert isinstance(payload, dict)
    assert isinstance(payload["A_c"], list)
    closed_form = payload["closed_form"]
    assert isinstance(closed_form["hamiltonian_roots"], list)
    for entry in closed_form["hamiltonian_roots"]:
        assert set(entry) == {"real", "imag"}
