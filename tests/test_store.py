"""Point-in-time store: availability arithmetic and the look-ahead injection test.

GATE 3 lives here. The injection test is the project's warrant: it demonstrates that
research code *cannot* read data that was not public as of the query date, rather than
asserting that the authors were careful.
"""

from __future__ import annotations

import datetime as _dt

import polars as pl
import pytest

from flowalpha.data.calendar import TradingCalendar
from flowalpha.data.store import (
    DATASET_SPECS,
    DatasetSpec,
    LookAheadError,
    PointInTimeStore,
)
from flowalpha.factors.flow_features import build_flow_features

from conftest import make_tiny_flows, make_tiny_prices, write_tree

D = _dt.date


# --- availability arithmetic ----------------------------------------------

def test_zero_lag_is_same_session(tiny_store, tiny_sessions):
    day = tiny_sessions[10]
    assert tiny_store.available_from("prices", day) == day


def test_one_trading_day_lag_moves_to_next_session(tiny_store, tiny_sessions):
    day = tiny_sessions[10]
    assert tiny_store.available_from("flows_daily", day) == tiny_sessions[11]


def test_one_trading_day_lag_spans_a_weekend(tiny_store, tiny_calendar):
    """A one-TRADING-day lag over a weekend is three calendar days."""
    fridays = [d for d in tiny_calendar.sessions if d.weekday() == 4]
    friday = fridays[3]
    got = tiny_store.available_from("flows_daily", friday)
    assert got > friday + _dt.timedelta(days=1)
    assert got.weekday() == 0


def test_one_trading_day_lag_spans_the_holiday(tiny_store, tiny_calendar):
    """The fixture calendar has a mid-series holiday; the lag must step over it."""
    from conftest import TINY_HOLIDAY

    before = tiny_calendar.prev_session(TINY_HOLIDAY)
    got = tiny_store.available_from("flows_daily", before)
    assert got != TINY_HOLIDAY
    assert got == tiny_calendar.next_session(TINY_HOLIDAY, inclusive=True)


def test_calendar_day_lag_is_wall_clock(tiny_store):
    day = D(2021, 3, 31)
    assert tiny_store.available_from("shareholding", day) == day + _dt.timedelta(days=21)


def test_availability_describe(tiny_store):
    assert tiny_store.availability_of("prices").describe() == "0 trading day(s)"
    assert tiny_store.availability_of("shareholding").describe() == "21 calendar day(s)"


def test_unknown_dataset_raises(tiny_store):
    with pytest.raises(LookAheadError, match="unknown dataset"):
        tiny_store.get("not_a_dataset", as_of_date=D(2021, 6, 1))


def test_unregistered_lag_raises_rather_than_defaulting_to_zero(tiny_tree, tiny_calendar):
    """An unregistered lag is an unenforced lag, so it must be an error."""
    specs = dict(DATASET_SPECS)
    specs["mystery"] = DatasetSpec(
        filename="prices.parquet", event_date_col="date", availability_key="mystery"
    )
    store = PointInTimeStore(
        tiny_tree.path("processed"), tiny_calendar, tiny_tree["availability"], specs=specs
    )
    with pytest.raises(LookAheadError, match="no availability entry"):
        store.availability_of("mystery")


def test_flow_features_is_registered_at_zero_lag():
    """Documented on purpose: its construction already carries the flow lag."""
    spec = DATASET_SPECS["flow_features"]
    assert spec.availability_key == "flow_features"
    assert "double-lag" in spec.note


def test_flow_features_zero_lag_is_assumed_and_documented(tiny_store, tiny_sessions):
    """config.yaml has no availability entry for the derived dataset; lag 0 applies."""
    assert tiny_store.availability_of("flow_features").trading_days == 0
    day = tiny_sessions[5]
    assert tiny_store.available_from("flow_features", day) == day


# --- get() ----------------------------------------------------------------

def test_get_respects_zero_lag(tiny_store, tiny_sessions):
    day = tiny_sessions[20]
    got = tiny_store.get("prices", as_of_date=day)
    assert max(got["date"].to_list()) == day


def test_get_respects_one_day_lag(tiny_store, tiny_sessions):
    day = tiny_sessions[20]
    got = tiny_store.get("flows_daily", as_of_date=day)
    assert max(got["date"].to_list()) == tiny_sessions[19]
    assert day not in set(got["date"].to_list())


def test_get_on_the_first_session_of_a_lagged_dataset_is_empty(tiny_store, tiny_sessions):
    """No data is available yet -- a normal condition, not an error."""
    got = tiny_store.get("flows_daily", as_of_date=tiny_sessions[0])
    assert got.is_empty()


def test_empty_result_keeps_real_dtypes(tiny_store, tiny_sessions):
    """An untyped empty frame would raise SchemaError when joined on date."""
    got = tiny_store.get("flows_daily", as_of_date=tiny_sessions[0])
    assert got.schema["date"] == pl.Date
    real = tiny_store.get("prices", as_of_date=tiny_sessions[5]).select("date")
    joined = real.join(got.select("date"), on="date", how="left")
    assert joined.height >= 0


def test_max_event_date_matches_available_from_exactly(tiny_store, tiny_calendar):
    """The cheap inversion must agree with the definition on every session."""
    for dataset in ("prices", "flows_daily", "shareholding"):
        for day in tiny_calendar.sessions[5:40]:
            cutoff = tiny_store.max_event_date(dataset, day)
            assert cutoff is not None
            assert tiny_store.available_from(dataset, cutoff) <= day
            later = tiny_calendar.next_session(cutoff)
            assert tiny_store.available_from(dataset, later) > day


def test_fields_projection_always_includes_keys(tiny_store, tiny_sessions):
    got = tiny_store.get("prices", fields=["close"], as_of_date=tiny_sessions[30])
    assert set(got.columns) == {"symbol", "date", "close"}


def test_unknown_field_raises(tiny_store, tiny_sessions):
    with pytest.raises(LookAheadError, match="unknown field"):
        tiny_store.get("prices", fields=["nope"], as_of_date=tiny_sessions[30])


def test_lookback_trims_to_n_sessions(tiny_store, tiny_sessions):
    day = tiny_sessions[60]
    got = tiny_store.get("prices", as_of_date=day, lookback=10)
    assert got["date"].n_unique() == 10
    assert max(got["date"].to_list()) == day


def test_lookback_must_be_positive(tiny_store, tiny_sessions):
    with pytest.raises(ValueError):
        tiny_store.get("prices", as_of_date=tiny_sessions[30], lookback=0)


def test_symbol_filter(tiny_store, tiny_sessions):
    got = tiny_store.get("prices", as_of_date=tiny_sessions[30], symbols=["AAA", "BBB"])
    assert set(got["symbol"].to_list()) == {"AAA", "BBB"}


def test_missing_dataset_file_raises_filenotfound(tiny_store, tiny_sessions):
    with pytest.raises(FileNotFoundError, match="bulk_deals"):
        tiny_store.get("bulk_deals", as_of_date=tiny_sessions[30])


def test_as_of_date_accepts_iso_string(tiny_store, tiny_sessions):
    day = tiny_sessions[20]
    a = tiny_store.get("prices", as_of_date=day)
    b = tiny_store.get("prices", as_of_date=day.isoformat())
    assert a.equals(b)


def test_as_of_date_rejects_nonsense(tiny_store):
    with pytest.raises(TypeError):
        tiny_store.get("prices", as_of_date=12345)


def test_describe_lists_lags_and_presence(tiny_store):
    lines = "\n".join(tiny_store.describe())
    assert "prices" in lines and "0 trading day(s)" in lines
    assert "21 calendar day(s)" in lines
    assert "absent" in lines  # bulk_deals is not written by the fixture


# --- the quarterly-filing boundary ---------------------------------------

def test_quarterly_filing_invisible_on_day_20_visible_on_day_21(tiny_cfg, tiny_calendar):
    """A 21-calendar-day lag must be exactly that: not 20, not 22."""
    filing_date = D(2021, 3, 31)
    shareholding = pl.DataFrame(
        {"date": [filing_date], "symbol": ["AAA"], "promoter_pct": [55.0]},
        schema={"date": pl.Date, "symbol": pl.Utf8, "promoter_pct": pl.Float64},
    )
    write_tree(tiny_cfg, shareholding=shareholding)
    store = PointInTimeStore(
        tiny_cfg.path("processed"), tiny_calendar, tiny_cfg["availability"]
    )

    day20 = filing_date + _dt.timedelta(days=20)
    day21 = filing_date + _dt.timedelta(days=21)
    assert store.get("shareholding", as_of_date=day20).is_empty()
    visible = store.get("shareholding", as_of_date=day21)
    assert visible.height == 1
    assert visible["promoter_pct"].to_list() == [55.0]


# --- GATE 3: the look-ahead injection test -------------------------------

def test_injecting_future_flows_does_not_move_past_features(tiny_cfg, tiny_calendar, tiny_sessions):
    """Mutate flow values AFTER a cutoff; every feature up to the cutoff must be
    bit-identical.

    This is the test the whole project's credibility rests on. It does not check that
    the feature code looks careful -- it injects data from the future and demands that
    the past does not move.
    """
    cutoff_idx = 110
    cutoff = tiny_sessions[cutoff_idx]

    daily, part = make_tiny_flows(tiny_sessions)
    prices = make_tiny_prices(tiny_sessions)
    write_tree(tiny_cfg, prices=prices, flows_daily=daily, participant_flows=part)

    store = PointInTimeStore(tiny_cfg.path("processed"), tiny_calendar, tiny_cfg["availability"])
    before = build_flow_features(store, tiny_cfg, calendar=tiny_calendar)
    before_upto = before.filter(pl.col("date") <= cutoff)

    # Inject implausibly large flows strictly after the cutoff.
    poisoned_daily = daily.with_columns(
        pl.when(pl.col("date") > cutoff)
        .then(pl.col("net_futures") * 1_000_000.0 + 5e9)
        .otherwise(pl.col("net_futures"))
        .alias("net_futures")
    )
    poisoned_part = part.with_columns(
        pl.when(pl.col("date") > cutoff)
        .then(pl.col("net_futures") * -1_000_000.0 - 5e9)
        .otherwise(pl.col("net_futures"))
        .alias("net_futures")
    )
    write_tree(tiny_cfg, flows_daily=poisoned_daily, participant_flows=poisoned_part)

    store2 = PointInTimeStore(tiny_cfg.path("processed"), tiny_calendar, tiny_cfg["availability"])
    after = build_flow_features(store2, tiny_cfg, calendar=tiny_calendar)
    after_upto = after.filter(pl.col("date") <= cutoff)

    assert before_upto.height == after_upto.height > 0
    assert before_upto.equals(after_upto), "future flow data leaked into past features"

    # And the injection must actually have changed something, or the test is vacuous.
    assert not before.equals(after)


def test_a_full_sample_quantile_would_fail_the_injection_test(tiny_cfg, tiny_calendar, tiny_sessions):
    """Guard on the guard.

    If regime bucketing used a FULL-SAMPLE quantile instead of an expanding one, the
    injection above would move past labels. This test constructs that comparison
    explicitly so the injection test cannot pass for the wrong reason (e.g. because
    the labels are all null).
    """
    import numpy as np

    from flowalpha.factors.flow_features import expanding_bucket_labels

    rng = np.random.default_rng(3)
    values = rng.normal(size=200)
    expanding = expanding_bucket_labels(values, n_buckets=3, min_obs=20)
    assert sum(1 for x in expanding if x is not None) > 100

    poisoned = values.copy()
    poisoned[150:] = 1e9
    expanding_poisoned = expanding_bucket_labels(poisoned, n_buckets=3, min_obs=20)
    assert expanding[:150] == expanding_poisoned[:150]

    # The full-sample alternative does move, which is why it is banned.
    def full_sample_labels(arr):
        lo, hi = np.quantile(arr, [1 / 3, 2 / 3])
        return ["low" if v <= lo else ("high" if v > hi else "mid") for v in arr]

    assert full_sample_labels(values)[:150] != full_sample_labels(poisoned)[:150]


def test_injecting_future_prices_does_not_move_past_price_reads(tiny_cfg, tiny_calendar, tiny_sessions):
    """Same discipline for the zero-lag dataset: as-of reads must truncate."""
    prices = make_tiny_prices(tiny_sessions)
    write_tree(tiny_cfg, prices=prices)
    store = PointInTimeStore(tiny_cfg.path("processed"), tiny_calendar, tiny_cfg["availability"])
    cutoff = tiny_sessions[80]
    before = store.get("prices", as_of_date=cutoff)

    poisoned = prices.with_columns(
        pl.when(pl.col("date") > cutoff).then(pl.lit(1e6)).otherwise(pl.col("close")).alias("close")
    )
    write_tree(tiny_cfg, prices=poisoned)
    store2 = PointInTimeStore(tiny_cfg.path("processed"), tiny_calendar, tiny_cfg["availability"])
    after = store2.get("prices", as_of_date=cutoff)
    assert before.equals(after)


def test_cache_invalidation_is_explicit(tiny_store, tiny_sessions, tiny_tree):
    """Frames are cached in memory; tests that rewrite parquet must say so."""
    day = tiny_sessions[30]
    first = tiny_store.get("prices", as_of_date=day)
    write_tree(tiny_tree, prices=make_tiny_prices(tiny_sessions, seed=999))
    assert tiny_store.get("prices", as_of_date=day).equals(first)
    tiny_store.invalidate("prices")
    assert not tiny_store.get("prices", as_of_date=day).equals(first)


def test_from_config_derives_calendar_from_prices(tiny_cfg, tiny_sessions):
    write_tree(tiny_cfg, prices=make_tiny_prices(tiny_sessions))
    store = PointInTimeStore.from_config(tiny_cfg)
    assert len(store.calendar) == len(tiny_sessions)


def test_from_config_without_prices_raises(tiny_cfg):
    with pytest.raises(FileNotFoundError, match="cannot build a calendar"):
        PointInTimeStore.from_config(tiny_cfg)


def test_from_config_prefers_the_sessions_file(tiny_cfg, tiny_sessions):
    from flowalpha.data.calendar import ensure_holidays_file

    write_tree(tiny_cfg, prices=make_tiny_prices(tiny_sessions))
    subset = TradingCalendar(tiny_sessions[:50])
    ensure_holidays_file(subset, tiny_cfg.path("reference"))
    store = PointInTimeStore.from_config(tiny_cfg)
    assert len(store.calendar) == 50
