"""Data quality checks.

The universe-cardinality check gets particular attention: its band must come from
config, and it must make the survivorship-driven growth visible rather than averaging it
away.
"""

from __future__ import annotations

import datetime as _dt

import polars as pl
import pytest

from flowalpha.data.calendar import TradingCalendar
from flowalpha.data.prices import process_prices
from flowalpha.data.quality import (
    FAIL,
    PASS,
    WARN,
    check_adjustment_sanity,
    check_flow_gaps,
    check_missing_trading_days,
    check_per_stock_coverage,
    check_return_outliers,
    check_universe_cardinality,
    overall_status,
    rebalance_dates,
    run_all_checks,
    universe_cardinality,
)

from conftest import make_config, make_tiny_flows, make_tiny_prices

D = _dt.date


def _sessions(n: int) -> list[_dt.date]:
    out, day = [], D(2021, 1, 4)
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += _dt.timedelta(days=1)
    return out


SESSIONS = _sessions(300)
SYMBOLS = tuple(f"S{i:02d}" for i in range(20))


@pytest.fixture(scope="module")
def calendar():
    return TradingCalendar(SESSIONS)


@pytest.fixture(scope="module")
def prices():
    return make_tiny_prices(SESSIONS, symbols=SYMBOLS)


@pytest.fixture
def cfg(tmp_path):
    base = make_config(tmp_path, SESSIONS)
    return base.with_overrides(
        universe={
            "name": "T", "size": len(SYMBOLS), "cardinality_band": [10, 30],
            "reconstruction": {"method": "mktcap_rank", "rebalance_freq": "quarterly",
                               "min_history_days": 20},
        }
    )


# --- missing sessions ------------------------------------------------------

def test_no_missing_sessions_passes(prices, calendar):
    r = check_missing_trading_days(prices, calendar, start=SESSIONS[0], end=SESSIONS[-1])
    assert r.status == PASS
    assert r.details["n_missing"] == 0


def test_many_missing_sessions_fails(prices, calendar):
    trimmed = prices.filter(pl.col("date") <= SESSIONS[100])
    r = check_missing_trading_days(trimmed, calendar, start=SESSIONS[0], end=SESSIONS[-1])
    assert r.status == FAIL
    assert r.details["n_missing"] > 0
    assert r.details["first_missing"]


def test_empty_calendar_window_fails(prices, calendar):
    r = check_missing_trading_days(prices, calendar, start=D(2030, 1, 1), end=D(2030, 2, 1))
    assert r.status == FAIL


# --- per-stock coverage ----------------------------------------------------

def test_full_coverage_passes(prices, calendar):
    assert check_per_stock_coverage(prices, calendar).status == PASS


def test_coverage_is_measured_over_each_name_s_own_span(prices, calendar):
    """A 2024 listing judged against a 2019 start would look 80% missing, which says
    nothing about data quality and would bury the names that really are broken."""
    late = prices.filter(
        (pl.col("symbol") != "S00") | (pl.col("date") >= SESSIONS[250])
    )
    r = check_per_stock_coverage(late, calendar)
    assert r.status == PASS
    worst = {row["symbol"]: row["gap_fraction"] for row in r.details["worst"]}
    assert worst.get("S00", 0.0) == 0.0


def test_holey_names_are_flagged(prices, calendar):
    import numpy as np

    rng = np.random.default_rng(0)
    drop = set(rng.choice(len(SESSIONS), size=100, replace=False).tolist())
    holey = prices.filter(
        ~(pl.col("symbol").is_in(list(SYMBOLS[:5]))
          & pl.col("date").is_in([SESSIONS[i] for i in drop]))
    )
    r = check_per_stock_coverage(holey, calendar)
    assert r.details["n_bad"] == 5
    assert r.status in (WARN, FAIL)


def test_empty_panel_fails_coverage(calendar):
    from flowalpha.data.prices import empty_prices

    assert check_per_stock_coverage(empty_prices(), calendar).status == FAIL


# --- return outliers -------------------------------------------------------

def test_clean_returns_pass(prices):
    assert check_return_outliers(prices).status == PASS


def test_unadjusted_split_is_caught():
    """A 5-for-1 split with no adjustment shows up as a -80% day on every date."""
    rows = []
    for i, day in enumerate(SESSIONS[:100]):
        for s in SYMBOLS:
            px = 100.0 if i < 50 else 20.0
            rows.append({"date": day, "symbol": s, "open": px, "high": px, "low": px,
                         "close": px, "adj_close": px, "volume": 1000.0})
    r = check_return_outliers(process_prices(pl.DataFrame(rows)))
    assert r.status in (WARN, FAIL)
    assert r.details["n_outliers"] >= len(SYMBOLS)
    assert r.details["worst"]


def test_outlier_check_on_empty_panel():
    from flowalpha.data.prices import empty_prices

    assert check_return_outliers(empty_prices()).status == FAIL


# --- flow gaps -------------------------------------------------------------

def test_full_flow_coverage_passes(calendar):
    daily, _ = make_tiny_flows(SESSIONS)
    r = check_flow_gaps(daily, calendar, start=SESSIONS[0], end=SESSIONS[-1])
    assert r.status == PASS


def test_a_few_flow_gaps_still_pass(calendar):
    """Real NSE data has a handful of unpublished sessions; that is not a failure."""
    daily, _ = make_tiny_flows(SESSIONS, drop_sessions={SESSIONS[10], SESSIONS[50]})
    r = check_flow_gaps(daily, calendar, start=SESSIONS[0], end=SESSIONS[-1])
    assert r.status == PASS
    assert r.details["n_missing"] == 2


def test_many_flow_gaps_fail(calendar):
    daily, _ = make_tiny_flows(SESSIONS, drop_sessions=set(SESSIONS[:100]))
    r = check_flow_gaps(daily, calendar, start=SESSIONS[0], end=SESSIONS[-1])
    assert r.status == FAIL


def test_flow_check_labels_itself(calendar):
    daily, _ = make_tiny_flows(SESSIONS)
    r = check_flow_gaps(daily, calendar, start=SESSIONS[0], end=SESSIONS[-1],
                        label="fii_dii_cash")
    assert r.name == "fii_dii_cash_gaps"


# --- universe cardinality --------------------------------------------------

def test_rebalance_dates_are_period_ends(calendar):
    quarters = rebalance_dates(calendar, SESSIONS[0], SESSIONS[-1], "quarterly")
    months = rebalance_dates(calendar, SESSIONS[0], SESSIONS[-1], "monthly")
    assert len(months) > len(quarters) > 0
    assert quarters[-1] == SESSIONS[-1]


def test_rebalance_freq_must_be_known(calendar):
    with pytest.raises(ValueError, match="unsupported rebalance_freq"):
        rebalance_dates(calendar, SESSIONS[0], SESSIONS[-1], "fortnightly")


def test_cardinality_band_is_read_from_config_not_hardcoded(prices, calendar, cfg):
    """The band must come from config; a hardcoded constant could never accommodate the
    survivorship-driven growth this project's data actually shows."""
    r = check_universe_cardinality(prices, calendar, cfg)
    assert r.details["band"] == [10, 30]

    narrow = cfg.with_overrides(
        universe={**cfg["universe"], "cardinality_band": [1, 2]}
    )
    assert check_universe_cardinality(prices, calendar, narrow).status == FAIL


def test_cardinality_reports_the_first_to_last_span(prices, calendar, cfg):
    """Growth must be visible, not averaged away by the median."""
    r = check_universe_cardinality(prices, calendar, cfg)
    assert "first" in r.details and "last" in r.details
    assert "span" in r.message
    assert len(r.details["series"]) == r.details["n_rebalances"]


def test_cardinality_growth_is_detected(calendar, cfg):
    """Names that list later push the count up over time, exactly as with a
    current-constituent list applied historically."""
    frames = []
    for i, sym in enumerate(SYMBOLS):
        start = i * 12
        frames.append(make_tiny_prices(SESSIONS[start:], symbols=(sym,), seed=i))
    prices = pl.concat(frames, how="vertical")
    frame = universe_cardinality(prices, calendar, cfg)
    counts = frame["n_members"].to_list()
    assert counts[0] < counts[-1]


def test_cardinality_respects_min_history(prices, calendar, cfg):
    strict = cfg.with_overrides(
        universe={**cfg["universe"],
                  "reconstruction": {"method": "mktcap_rank", "rebalance_freq": "quarterly",
                                     "min_history_days": 10_000}}
    )
    frame = universe_cardinality(prices, calendar, strict)
    assert set(frame["n_members"].to_list()) == {0}


def test_cardinality_is_capped_at_universe_size(prices, calendar, cfg):
    small = cfg.with_overrides(
        universe={**cfg["universe"], "size": 5, "cardinality_band": [1, 10]}
    )
    frame = universe_cardinality(prices, calendar, small)
    assert max(frame["n_members"].to_list()) <= 5


def test_cardinality_note_explains_the_bias(prices, calendar, cfg):
    r = check_universe_cardinality(prices, calendar, cfg)
    assert "survivorship" in r.details["note"].lower()


def test_cardinality_on_empty_panel_fails(calendar, cfg):
    from flowalpha.data.prices import empty_prices

    assert check_universe_cardinality(empty_prices(), calendar, cfg).status == FAIL


# --- adjustment sanity -----------------------------------------------------

def test_clean_adjustment_passes(prices):
    assert check_adjustment_sanity(prices).status == PASS


def test_non_positive_adj_factor_fails_on_a_single_row(prices):
    """One bad factor turns a whole symbol's return series into nonsense, so any
    occurrence is a FAIL rather than a fraction."""
    broken = prices.with_columns(
        pl.when(pl.col("date") == SESSIONS[5]).then(pl.lit(0.0))
        .otherwise(pl.col("adj_factor")).alias("adj_factor")
    )
    r = check_adjustment_sanity(broken)
    assert r.status == FAIL
    assert r.details["n_bad_factor"] > 0
    assert r.details["examples"]


def test_null_adj_close_fails(prices):
    broken = prices.with_columns(
        pl.when(pl.col("date") == SESSIONS[5]).then(pl.lit(None, pl.Float64))
        .otherwise(pl.col("adj_close")).alias("adj_close")
    )
    assert check_adjustment_sanity(broken).status == FAIL


# --- runner ----------------------------------------------------------------

def test_run_all_checks_covers_every_check(prices, calendar, cfg):
    daily, _ = make_tiny_flows(SESSIONS)
    results = run_all_checks(cfg, prices, calendar, flows=daily)
    names = {r.name for r in results}
    assert names == {
        "missing_trading_days", "per_stock_coverage", "return_outliers",
        "flows_daily_gaps", "universe_cardinality", "adjustment_sanity",
    }


def test_run_all_checks_without_flows_skips_the_flow_check(prices, calendar, cfg):
    names = {r.name for r in run_all_checks(cfg, prices, calendar)}
    assert "flows_daily_gaps" not in names


def test_overall_status_precedence():
    from flowalpha.data.quality import CheckResult

    assert overall_status([CheckResult("a", PASS, "")]) == PASS
    assert overall_status([CheckResult("a", PASS, ""), CheckResult("b", WARN, "")]) == WARN
    assert overall_status([CheckResult("a", WARN, ""), CheckResult("b", FAIL, "")]) == FAIL


def test_check_results_serialise(prices, calendar, cfg):
    import json

    results = run_all_checks(cfg, prices, calendar)
    json.dumps([r.to_dict() for r in results], default=str)


def test_a_clean_tree_passes_overall(prices, calendar, cfg):
    daily, _ = make_tiny_flows(SESSIONS)
    assert overall_status(run_all_checks(cfg, prices, calendar, flows=daily)) == PASS
