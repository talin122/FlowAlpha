"""Forward returns, rank IC, and Newey-West adjustment."""

from __future__ import annotations

import datetime as _dt
import math

import numpy as np
import polars as pl
import pytest
from scipy import stats

from flowalpha.validation.ic import (
    forward_returns,
    ic_table,
    newey_west_se,
    rank_ic,
    summarize_ic,
    trailing_ic_series,
)

D = _dt.date
DAYS = [D(2022, 1, 3) + _dt.timedelta(days=i) for i in range(12) if (D(2022, 1, 3) + _dt.timedelta(days=i)).weekday() < 5]


def _prices(levels: dict[str, list[float]]) -> pl.DataFrame:
    rows = []
    for sym, series in levels.items():
        for day, px in zip(DAYS, series):
            rows.append({"date": day, "symbol": sym, "adj_close": float(px)})
    return pl.DataFrame(rows, schema={"date": pl.Date, "symbol": pl.Utf8, "adj_close": pl.Float64})


# --- forward returns -------------------------------------------------------

def test_forward_return_is_labelled_by_the_entry_date():
    prices = _prices({"AAA": [100.0 * (1.1 ** i) for i in range(len(DAYS))]})
    fwd = forward_returns(prices, [1, 2])
    row = fwd.filter(pl.col("date") == DAYS[0])
    assert float(row["fwd_1"][0]) == pytest.approx(0.1)
    assert float(row["fwd_2"][0]) == pytest.approx(1.1 ** 2 - 1)


def test_forward_return_is_null_at_the_end_of_the_sample():
    """No future to measure. Clipping to the last price would fabricate a shorter
    horizon under a longer horizon's name."""
    prices = _prices({"AAA": [100.0] * len(DAYS)})
    fwd = forward_returns(prices, [1, 5])
    last = fwd.filter(pl.col("date") == DAYS[-1])
    assert last["fwd_1"][0] is None
    assert fwd.filter(pl.col("date") == DAYS[-3])["fwd_5"][0] is None


def test_forward_returns_are_per_symbol():
    prices = _prices({"AAA": [100.0] * len(DAYS), "BBB": [10.0 * (i + 1) for i in range(len(DAYS))]})
    fwd = forward_returns(prices, [1])
    aaa = fwd.filter((pl.col("symbol") == "AAA") & (pl.col("date") == DAYS[0]))["fwd_1"][0]
    bbb = fwd.filter((pl.col("symbol") == "BBB") & (pl.col("date") == DAYS[0]))["fwd_1"][0]
    assert aaa == pytest.approx(0.0)
    assert bbb == pytest.approx(1.0)


def test_forward_returns_empty_input_keeps_schema():
    empty = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "adj_close": pl.Float64})
    fwd = forward_returns(empty, [1, 5])
    assert fwd.is_empty()
    assert fwd.schema["date"] == pl.Date
    assert "fwd_5" in fwd.columns


def test_forward_returns_rejects_nonpositive_horizon():
    prices = _prices({"AAA": [100.0] * len(DAYS)})
    with pytest.raises(ValueError):
        forward_returns(prices, [0])


def test_forward_returns_rejects_missing_price_column():
    prices = _prices({"AAA": [100.0] * len(DAYS)})
    with pytest.raises(ValueError, match="no column"):
        forward_returns(prices, [1], price_col="nope")


# --- rank IC ---------------------------------------------------------------

def test_rank_ic_matches_scipy_spearman():
    rng = np.random.default_rng(1)
    n = 40
    values = rng.normal(size=n)
    labels = 0.5 * values + rng.normal(size=n)
    day = DAYS[0]
    panel = pl.DataFrame(
        {"date": [day] * n, "symbol": [f"S{i}" for i in range(n)], "value": values},
        schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64},
    )
    fwd = pl.DataFrame(
        {"date": [day] * n, "symbol": [f"S{i}" for i in range(n)], "fwd_1": labels},
        schema={"date": pl.Date, "symbol": pl.Utf8, "fwd_1": pl.Float64},
    )
    got = rank_ic(panel, fwd, 1)
    expected = stats.spearmanr(values, labels).statistic
    assert float(got["ic"][0]) == pytest.approx(expected, abs=1e-10)
    assert int(got["n"][0]) == n


def test_rank_ic_perfect_monotone_relationship_is_one():
    n = 20
    day = DAYS[0]
    panel = pl.DataFrame(
        {"date": [day] * n, "symbol": [f"S{i}" for i in range(n)],
         "value": [float(i) for i in range(n)]},
        schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64},
    )
    fwd = pl.DataFrame(
        {"date": [day] * n, "symbol": [f"S{i}" for i in range(n)],
         "fwd_1": [float(i) ** 3 for i in range(n)]},
        schema={"date": pl.Date, "symbol": pl.Utf8, "fwd_1": pl.Float64},
    )
    assert float(rank_ic(panel, fwd, 1)["ic"][0]) == pytest.approx(1.0)


def test_rank_ic_handles_ties_via_average_ranks():
    n = 12
    day = DAYS[0]
    values = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0]
    labels = [0.1 * i for i in range(n)]
    panel = pl.DataFrame(
        {"date": [day] * n, "symbol": [f"S{i}" for i in range(n)], "value": values},
        schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64},
    )
    fwd = pl.DataFrame(
        {"date": [day] * n, "symbol": [f"S{i}" for i in range(n)], "fwd_1": labels},
        schema={"date": pl.Date, "symbol": pl.Utf8, "fwd_1": pl.Float64},
    )
    got = float(rank_ic(panel, fwd, 1, min_names=5)["ic"][0])
    assert got == pytest.approx(stats.spearmanr(values, labels).statistic, abs=1e-10)


def test_rank_ic_drops_thin_cross_sections():
    day = DAYS[0]
    panel = pl.DataFrame(
        {"date": [day] * 4, "symbol": list("ABCD"), "value": [1.0, 2.0, 3.0, 4.0]},
        schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64},
    )
    fwd = pl.DataFrame(
        {"date": [day] * 4, "symbol": list("ABCD"), "fwd_1": [1.0, 2.0, 3.0, 4.0]},
        schema={"date": pl.Date, "symbol": pl.Utf8, "fwd_1": pl.Float64},
    )
    assert rank_ic(panel, fwd, 1, min_names=10).is_empty()
    assert not rank_ic(panel, fwd, 1, min_names=4).is_empty()


def test_rank_ic_on_empty_panel_keeps_schema():
    empty = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64})
    fwd = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "fwd_1": pl.Float64})
    got = rank_ic(empty, fwd, 1)
    assert got.is_empty() and got.schema["date"] == pl.Date


def test_rank_ic_requires_the_horizon_column():
    empty = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64})
    fwd = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "fwd_1": pl.Float64})
    with pytest.raises(ValueError, match="fwd_5"):
        rank_ic(empty, fwd, 5)


# --- Newey-West ------------------------------------------------------------

def test_newey_west_equals_plain_se_at_zero_lags():
    rng = np.random.default_rng(7)
    x = rng.normal(size=500)
    se, lags = newey_west_se(x, lags=0)
    assert lags == 0
    assert se == pytest.approx(float(np.std(x, ddof=0) / math.sqrt(x.size)), rel=1e-12)


def test_newey_west_inflates_se_for_positively_autocorrelated_series():
    """The whole point: overlapping labels make daily ICs autocorrelated, and the
    naive t-stat then overstates significance."""
    rng = np.random.default_rng(11)
    n = 2000
    eps = rng.normal(size=n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.8 * ar[i - 1] + eps[i]
    plain = float(np.std(ar, ddof=1) / math.sqrt(n))
    nw, lags = newey_west_se(ar)
    assert lags > 0
    assert nw > 1.5 * plain


def test_newey_west_handles_short_and_degenerate_input():
    assert math.isnan(newey_west_se(np.array([1.0]))[0])
    assert math.isnan(newey_west_se(np.array([]))[0])
    se, _ = newey_west_se(np.array([2.0, 2.0, 2.0, 2.0]))
    assert se == pytest.approx(0.0)


def test_newey_west_ignores_non_finite_values():
    a = newey_west_se(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), lags=1)[0]
    b = newey_west_se(np.array([1.0, np.nan, 2.0, 3.0, np.inf, 4.0, 5.0]), lags=1)[0]
    assert a == pytest.approx(b)


def test_newey_west_lags_are_clamped_to_n_minus_one():
    _, lags = newey_west_se(np.array([1.0, 2.0, 3.0]), lags=99)
    assert lags == 2


# --- summaries -------------------------------------------------------------

def _ic_series(values) -> pl.DataFrame:
    days = [D(2022, 1, 3) + _dt.timedelta(days=i) for i in range(len(values))]
    return pl.DataFrame(
        {"date": days, "ic": list(values), "n": [50] * len(values)},
        schema={"date": pl.Date, "ic": pl.Float64, "n": pl.UInt32},
    )


def test_summarize_ic_reports_mean_ir_and_hit_rate():
    s = summarize_ic(_ic_series([0.1, -0.05, 0.2, 0.0, 0.15]), factor="f", horizon=1)
    assert s.n_dates == 5
    assert s.mean == pytest.approx(0.08)
    assert s.hit_rate == pytest.approx(0.6)
    assert s.mean_names == pytest.approx(50.0)


def test_summarize_ic_marks_itself_not_deflated():
    """Raw IC t-stats must never be presented as multiple-testing adjusted."""
    d = summarize_ic(_ic_series([0.05] * 300), factor="f", horizon=5).to_dict()
    assert d["deflated"] is False
    assert "t_stat_newey_west" in d


def test_summarize_ic_default_lags_cover_label_overlap():
    """A 5-day horizon shares four days of return between consecutive labels."""
    s = summarize_ic(_ic_series(np.full(50, 0.03)), factor="f", horizon=5)
    assert s.nw_lags >= 4


def test_summarize_ic_on_empty_series():
    s = summarize_ic(_ic_series([]), factor="f", horizon=1)
    assert s.n_dates == 0 and math.isnan(s.mean)


def test_ic_table_covers_every_factor_horizon_pair():
    rng = np.random.default_rng(5)
    days = DAYS
    syms = [f"S{i}" for i in range(30)]
    rows_p, rows_f = [], []
    for day in days:
        vals = rng.normal(size=len(syms))
        for sym, v in zip(syms, vals):
            rows_p.append({"date": day, "symbol": sym, "value": float(v)})
            rows_f.append({"date": day, "symbol": sym, "fwd_1": float(v * 0.1 + rng.normal() * 0.5),
                           "fwd_5": float(rng.normal())})
    panel = pl.DataFrame(rows_p, schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64})
    fwd = pl.DataFrame(rows_f, schema={"date": pl.Date, "symbol": pl.Utf8,
                                       "fwd_1": pl.Float64, "fwd_5": pl.Float64})
    table = ic_table({"alpha": panel, "beta": panel}, fwd, [1, 5])
    assert table.height == 4
    assert set(table["factor"].to_list()) == {"alpha", "beta"}
    assert table["deflated"].to_list() == [False] * 4


def test_ic_table_empty_input_keeps_schema():
    table = ic_table({}, pl.DataFrame(schema={"date": pl.Date}), [1])
    assert table.is_empty() and "mean_ic" in table.columns


def test_trailing_ic_series_attaches_label_availability_date():
    """An IC for date d is not knowable until the label realises, h sessions later.

    Weighting today's factors by an IC dated today is the commonest way a
    "point-in-time" composite turns out not to be.
    """
    prices = _prices({f"S{i}": [100.0 + i + j for j in range(len(DAYS))] for i in range(15)})
    fwd = forward_returns(prices, [2])
    panel = prices.select("date", "symbol").with_columns(
        (pl.col("symbol").str.slice(1).cast(pl.Float64)).alias("value")
    )
    series = trailing_ic_series(panel, fwd, 2, min_names=5)
    assert "usable_from" in series.columns
    for row in series.iter_rows(named=True):
        assert row["usable_from"] > row["date"]
