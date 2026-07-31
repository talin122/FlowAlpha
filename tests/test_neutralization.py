"""Cross-sectional treatment: winsorisation, z-scoring, sector neutralisation."""

from __future__ import annotations

import datetime as _dt

import numpy as np
import polars as pl
import pytest

from flowalpha.validation.neutralization import (
    sector_neutralize,
    treat_cross_section,
    winsorize,
    zscore,
)

D1, D2 = _dt.date(2022, 1, 3), _dt.date(2022, 1, 4)


def _panel(rows) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64})


def test_winsorize_clips_rather_than_drops():
    """Dropping tails would make the universe a function of factor value."""
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    panel = _panel([{"date": D1, "symbol": f"S{i}", "value": v} for i, v in enumerate(values)])
    got = winsorize(panel, lower=0.0, upper=0.75)
    assert got.height == 5
    assert max(got["value"].to_list()) == pytest.approx(float(np.quantile(values, 0.75)))


def test_winsorize_is_per_date():
    panel = _panel(
        [{"date": D1, "symbol": f"S{i}", "value": float(i)} for i in range(5)]
        + [{"date": D2, "symbol": f"S{i}", "value": float(i) * 100.0} for i in range(5)]
    )
    got = winsorize(panel, lower=0.0, upper=0.5)
    d1_max = max(got.filter(pl.col("date") == D1)["value"].to_list())
    d2_max = max(got.filter(pl.col("date") == D2)["value"].to_list())
    assert d1_max == pytest.approx(2.0)
    assert d2_max == pytest.approx(200.0)


def test_winsorize_rejects_invalid_bounds():
    panel = _panel([{"date": D1, "symbol": "A", "value": 1.0}])
    with pytest.raises(ValueError):
        winsorize(panel, lower=0.9, upper=0.1)
    with pytest.raises(ValueError):
        winsorize(panel, lower=-0.1, upper=0.9)


def test_winsorize_requires_the_value_column():
    with pytest.raises(ValueError, match="missing required column"):
        winsorize(pl.DataFrame({"date": [D1]}), value_col="value")


def test_zscore_standardises_within_date():
    panel = _panel([{"date": D1, "symbol": f"S{i}", "value": float(i)} for i in range(5)])
    got = zscore(panel)
    vals = np.array(got["value"].to_list())
    assert vals.mean() == pytest.approx(0.0)
    assert vals.std(ddof=1) == pytest.approx(1.0)


def test_zscore_nulls_a_degenerate_cross_section():
    """Zero dispersion carries no cross-sectional information; nulls beat infinities."""
    panel = _panel([{"date": D1, "symbol": f"S{i}", "value": 5.0} for i in range(5)])
    got = zscore(panel)
    assert all(v is None for v in got["value"].to_list())


def test_zscore_nulls_a_single_name_date():
    panel = _panel([{"date": D1, "symbol": "A", "value": 1.0}])
    assert zscore(panel)["value"].to_list() == [None]


def test_zscore_is_independent_across_dates():
    panel = _panel(
        [{"date": D1, "symbol": f"S{i}", "value": float(i)} for i in range(5)]
        + [{"date": D2, "symbol": f"S{i}", "value": float(i) * 1000.0 + 5.0} for i in range(5)]
    )
    got = zscore(panel)
    a = got.filter(pl.col("date") == D1)["value"].to_list()
    b = got.filter(pl.col("date") == D2)["value"].to_list()
    assert a == pytest.approx(b)


def test_sector_neutralize_demeans_within_sector():
    panel = _panel(
        [
            {"date": D1, "symbol": "A", "value": 1.0},
            {"date": D1, "symbol": "B", "value": 3.0},
            {"date": D1, "symbol": "C", "value": 10.0},
            {"date": D1, "symbol": "D", "value": 20.0},
        ]
    )
    sectors = {"A": "IT", "B": "IT", "C": "Bank", "D": "Bank"}
    got = sector_neutralize(panel, sectors).sort("symbol")
    assert got["value"].to_list() == pytest.approx([-1.0, 1.0, -5.0, 5.0])


def test_sector_neutralize_pools_unmapped_names_as_NA():
    """Dropping them would shrink the universe; calling them one real sector lies."""
    panel = _panel(
        [
            {"date": D1, "symbol": "A", "value": 1.0},
            {"date": D1, "symbol": "B", "value": 3.0},
            {"date": D1, "symbol": "Z", "value": 100.0},
        ]
    )
    got = sector_neutralize(panel, {"A": "IT", "B": "IT"}).sort("symbol")
    assert got.height == 3
    assert got.filter(pl.col("symbol") == "Z")["value"].to_list() == pytest.approx([0.0])


def test_sector_neutralize_with_one_sector_is_a_demean():
    """Which is why it does nothing when sector data is unavailable."""
    panel = _panel([{"date": D1, "symbol": s, "value": v} for s, v in zip("ABC", [1.0, 2.0, 3.0])])
    got = sector_neutralize(panel, {"A": "NA", "B": "NA", "C": "NA"}).sort("symbol")
    assert got["value"].to_list() == pytest.approx([-1.0, 0.0, 1.0])


def test_sector_neutralize_accepts_a_frame():
    panel = _panel([{"date": D1, "symbol": s, "value": v} for s, v in zip("AB", [1.0, 3.0])])
    mapping = pl.DataFrame({"symbol": ["A", "B"], "sector": ["IT", "IT"]})
    got = sector_neutralize(panel, mapping).sort("symbol")
    assert got["value"].to_list() == pytest.approx([-1.0, 1.0])


def test_sector_neutralize_rejects_a_frame_without_sector():
    panel = _panel([{"date": D1, "symbol": "A", "value": 1.0}])
    with pytest.raises(ValueError, match="sector"):
        sector_neutralize(panel, pl.DataFrame({"symbol": ["A"]}))


def test_treat_cross_section_pipeline():
    values = [1.0, 2.0, 3.0, 4.0, 500.0]
    panel = _panel([{"date": D1, "symbol": f"S{i}", "value": v} for i, v in enumerate(values)])
    got = treat_cross_section(panel, winsor=(0.0, 0.75))
    vals = np.array(got["value"].to_list())
    assert vals.mean() == pytest.approx(0.0)
    assert vals.max() < 2.0  # the 500 outlier was clipped before standardising


def test_treat_cross_section_refuses_to_silently_skip_neutralisation():
    """A run that believes it is sector-neutral and is not misattributes an industry
    bet to a factor."""
    panel = _panel([{"date": D1, "symbol": "A", "value": 1.0}])
    with pytest.raises(ValueError, match="Refusing to silently skip"):
        treat_cross_section(panel, sector_neutral=True, sectors=None)


def test_treat_cross_section_neutralises_and_restandardises():
    panel = _panel(
        [
            {"date": D1, "symbol": "A", "value": 1.0},
            {"date": D1, "symbol": "B", "value": 2.0},
            {"date": D1, "symbol": "C", "value": 10.0},
            {"date": D1, "symbol": "D", "value": 11.0},
        ]
    )
    sectors = {"A": "IT", "B": "IT", "C": "Bank", "D": "Bank"}
    got = treat_cross_section(panel, sector_neutral=True, sectors=sectors)
    vals = np.array(got["value"].to_list())
    assert vals.mean() == pytest.approx(0.0)
    assert vals.std(ddof=1) == pytest.approx(1.0)
    # The large between-sector spread is gone; only within-sector spread remains, so
    # every name sits the same distance from zero. Four points at +/-d standardised
    # with ddof=1 give +/-sqrt(3)/2, not +/-1.
    assert sorted(np.abs(vals)) == pytest.approx([np.sqrt(3) / 2] * 4)
    assert set(np.sign(vals)) == {-1.0, 1.0}


def test_all_treatments_pass_empty_panels_through_unchanged():
    empty = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64})
    assert winsorize(empty).is_empty()
    assert zscore(empty).is_empty()
    assert sector_neutralize(empty, {}).is_empty()
    assert treat_cross_section(empty).is_empty()
    assert treat_cross_section(empty).schema == empty.schema
