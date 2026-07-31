"""Price panel construction and bhavcopy parsing."""

from __future__ import annotations

import datetime as _dt

import polars as pl
import pytest

from flowalpha.data.prices import (
    PRICES_SCHEMA,
    PriceParseError,
    daily_returns,
    empty_prices,
    parse_bhavcopy,
    process_prices,
    read_prices,
    write_prices,
)

D = _dt.date

BHAVCOPY = (
    "SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE,"
    " CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER\n"
    "RELIANCE, EQ, 29-Jul-2026, 2490.00, 2500.00, 2520.00, 2480.00, 2510.00, 2515.00,"
    " 2505.00, 1000000, 25050.00, 50000, 400000, 40.00\n"
    "TCS, EQ, 29-Jul-2026, 3790.00, 3800.00, 3850.00, 3780.00, 3840.00, 3845.00,"
    " 3820.00, 500000, 19100.00, 30000, -, -\n"
    "SOMEBOND, BE, 29-Jul-2026, 100.00, 100.00, 101.00, 99.00, 100.50, 100.50,"
    " 100.00, 1000, 1.00, 10, 500, 50.00\n"
)


def _raw(rows) -> pl.DataFrame:
    return pl.DataFrame(rows)


# --- empty / schema --------------------------------------------------------

def test_empty_prices_has_full_typed_schema():
    """Built from the schema, not from empty lists: a Null date column raises
    SchemaError when joined against a real Date key, far from the mistake."""
    empty = empty_prices()
    assert empty.is_empty()
    assert empty.schema == PRICES_SCHEMA
    assert empty.schema["date"] == pl.Date


def test_empty_prices_joins_against_a_real_date_key():
    real = pl.DataFrame({"date": [D(2026, 7, 29)]}, schema={"date": pl.Date})
    joined = real.join(empty_prices().select("date", "close"), on="date", how="left")
    assert joined.height == 1


# --- process_prices --------------------------------------------------------

def test_adj_factor_and_adjusted_ohl_hand_computed():
    """close 100, adj_close 90 -> factor 0.9, applied to open/high/low."""
    got = process_prices(_raw([
        {"date": D(2026, 1, 1), "symbol": "A", "open": 100.0, "high": 110.0,
         "low": 90.0, "close": 100.0, "adj_close": 90.0, "volume": 1000.0},
    ]))
    row = got.to_dicts()[0]
    assert row["adj_factor"] == pytest.approx(0.9)
    assert row["adj_open"] == pytest.approx(90.0)
    assert row["adj_high"] == pytest.approx(99.0)
    assert row["adj_low"] == pytest.approx(81.0)


def test_turnover_uses_raw_close_and_volume():
    """close x volume is split-invariant, so it is the actual rupee value traded.
    Adjusting both would double-count the split."""
    got = process_prices(_raw([
        {"date": D(2026, 1, 1), "symbol": "A", "open": 100.0, "high": 100.0, "low": 100.0,
         "close": 100.0, "adj_close": 50.0, "volume": 1000.0},
    ]))
    assert got["turnover"][0] == pytest.approx(100_000.0)


def test_volume_is_left_unadjusted():
    got = process_prices(_raw([
        {"date": D(2026, 1, 1), "symbol": "A", "open": 100.0, "high": 100.0, "low": 100.0,
         "close": 100.0, "adj_close": 50.0, "volume": 1234.0},
    ]))
    assert got["volume"][0] == pytest.approx(1234.0)


def test_delivery_pct_absent_becomes_null_not_zero():
    """Yahoo has no delivery field. Null is honest; zero would be a claim."""
    got = process_prices(_raw([
        {"date": D(2026, 1, 1), "symbol": "A", "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 1.0, "adj_close": 1.0, "volume": 1.0},
    ]))
    assert got["delivery_pct"][0] is None


def test_non_positive_close_rows_are_dropped():
    """A zero close cannot produce a usable adjustment factor, and keeping it would
    propagate an inf through the whole symbol's history."""
    got = process_prices(_raw([
        {"date": D(2026, 1, 1), "symbol": "A", "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 0.0, "adj_close": 1.0, "volume": 1.0},
        {"date": D(2026, 1, 2), "symbol": "A", "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 2.0, "adj_close": 2.0, "volume": 1.0},
    ]))
    assert got.height == 1
    assert got["date"].to_list() == [D(2026, 1, 2)]


def test_null_adj_close_rows_are_dropped():
    got = process_prices(_raw([
        {"date": D(2026, 1, 1), "symbol": "A", "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 1.0, "adj_close": None, "volume": 1.0},
    ]))
    assert got.is_empty()


def test_duplicates_are_deduped_keeping_last():
    got = process_prices(_raw([
        {"date": D(2026, 1, 1), "symbol": "A", "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 1.0, "adj_close": 1.0, "volume": 1.0},
        {"date": D(2026, 1, 1), "symbol": "A", "open": 2.0, "high": 2.0, "low": 2.0,
         "close": 2.0, "adj_close": 2.0, "volume": 2.0},
    ]))
    assert got.height == 1
    assert got["close"][0] == pytest.approx(2.0)


def test_output_is_sorted_by_symbol_then_date():
    got = process_prices(_raw([
        {"date": D(2026, 1, 2), "symbol": "B", "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 1.0, "adj_close": 1.0, "volume": 1.0},
        {"date": D(2026, 1, 1), "symbol": "A", "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 1.0, "adj_close": 1.0, "volume": 1.0},
        {"date": D(2026, 1, 2), "symbol": "A", "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 1.0, "adj_close": 1.0, "volume": 1.0},
    ]))
    assert got["symbol"].to_list() == ["A", "A", "B"]


def test_symbols_are_stripped():
    got = process_prices(_raw([
        {"date": D(2026, 1, 1), "symbol": "  A  ", "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 1.0, "adj_close": 1.0, "volume": 1.0},
    ]))
    assert got["symbol"][0] == "A"


def test_missing_columns_raise():
    with pytest.raises(PriceParseError, match="missing columns"):
        process_prices(_raw([{"date": D(2026, 1, 1), "symbol": "A"}]))


def test_missing_adj_close_raises_unless_waived():
    rows = [{"date": D(2026, 1, 1), "symbol": "A", "open": 1.0, "high": 1.0,
             "low": 1.0, "close": 5.0, "volume": 1.0}]
    with pytest.raises(PriceParseError, match="adj_close"):
        process_prices(_raw(rows))
    got = process_prices(_raw(rows), require_adj_close=False)
    assert got["adj_close"][0] == pytest.approx(5.0)
    assert got["adj_factor"][0] == pytest.approx(1.0)


def test_schema_is_exactly_canonical():
    got = process_prices(_raw([
        {"date": D(2026, 1, 1), "symbol": "A", "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 1.0, "adj_close": 1.0, "volume": 1.0},
    ]))
    assert got.schema == PRICES_SCHEMA


# --- bhavcopy --------------------------------------------------------------

def test_bhavcopy_parses_and_filters_to_eq():
    got = parse_bhavcopy(BHAVCOPY)
    assert set(got["symbol"].to_list()) == {"RELIANCE", "TCS"}
    assert "SOMEBOND" not in got["symbol"].to_list()


def test_bhavcopy_leading_space_headers_are_handled():
    got = parse_bhavcopy(BHAVCOPY)
    rel = got.filter(pl.col("symbol") == "RELIANCE")
    # CLOSE_PRICE (2515), not LAST_PRICE (2510): the settlement close is the one that
    # drives returns.
    assert float(rel["close"][0]) == pytest.approx(2515.00)
    assert float(rel["delivery_pct"][0]) == pytest.approx(40.00)


def test_bhavcopy_carries_delivery_which_yahoo_lacks():
    """The reason this parser exists at all: the delivery factor is inert without it."""
    got = parse_bhavcopy(BHAVCOPY)
    assert got.filter(pl.col("symbol") == "RELIANCE")["delivery_pct"][0] is not None


def test_bhavcopy_dash_delivery_becomes_null():
    got = parse_bhavcopy(BHAVCOPY)
    assert got.filter(pl.col("symbol") == "TCS")["delivery_pct"][0] is None


def test_bhavcopy_turnover_lacs_is_converted_to_rupees():
    got = parse_bhavcopy(BHAVCOPY)
    rel = got.filter(pl.col("symbol") == "RELIANCE")
    assert float(rel["turnover"][0]) == pytest.approx(25050.00 * 1e5)


def test_bhavcopy_parses_the_date_column():
    got = parse_bhavcopy(BHAVCOPY)
    assert set(got["date"].to_list()) == {D(2026, 7, 29)}


def test_bhavcopy_session_date_override():
    got = parse_bhavcopy(BHAVCOPY, session_date=D(2020, 1, 1))
    assert set(got["date"].to_list()) == {D(2020, 1, 1)}


def test_bhavcopy_rejects_html():
    with pytest.raises(PriceParseError, match="HTML error page"):
        parse_bhavcopy("<!DOCTYPE html>\n<html><head></head><body>nope</body></html>")


def test_bhavcopy_rejects_empty():
    with pytest.raises(PriceParseError, match="empty"):
        parse_bhavcopy("   \n ")


def test_bhavcopy_rejects_missing_columns():
    with pytest.raises(PriceParseError, match="missing columns"):
        parse_bhavcopy("SYMBOL,SERIES\nRELIANCE,EQ\n")


def test_bhavcopy_accepts_bytes():
    a = parse_bhavcopy(BHAVCOPY)
    b = parse_bhavcopy(BHAVCOPY.encode("utf-8"))
    assert a.equals(b)


def test_bhavcopy_keeps_other_series_when_asked():
    got = parse_bhavcopy(BHAVCOPY, series=("EQ", "BE"))
    assert "SOMEBOND" in got["symbol"].to_list()


# --- returns ---------------------------------------------------------------

def test_daily_returns_hand_computed():
    prices = process_prices(_raw([
        {"date": D(2026, 1, 1), "symbol": "A", "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 100.0, "adj_close": 100.0, "volume": 1.0},
        {"date": D(2026, 1, 2), "symbol": "A", "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 110.0, "adj_close": 110.0, "volume": 1.0},
    ]))
    rets = daily_returns(prices)
    assert rets["ret"][0] is None
    assert float(rets["ret"][1]) == pytest.approx(0.10)


def test_daily_returns_are_per_symbol():
    prices = process_prices(_raw([
        {"date": D(2026, 1, 1), "symbol": s, "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 100.0, "adj_close": 100.0, "volume": 1.0} for s in ("A", "B")
    ] + [
        {"date": D(2026, 1, 2), "symbol": "A", "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 200.0, "adj_close": 200.0, "volume": 1.0},
    ]))
    rets = daily_returns(prices).filter(pl.col("ret").is_not_null())
    assert rets.height == 1
    assert float(rets["ret"][0]) == pytest.approx(1.0)


def test_daily_returns_of_empty_panel_keeps_schema():
    got = daily_returns(empty_prices())
    assert got.is_empty() and got.schema["ret"] == pl.Float64


def test_daily_returns_use_the_adjusted_series():
    """A dividend shows up in close but not in adj_close, so the adjusted series must be
    the one used or every ex-date becomes a fake return."""
    prices = process_prices(_raw([
        {"date": D(2026, 1, 1), "symbol": "A", "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 100.0, "adj_close": 99.0, "volume": 1.0},
        {"date": D(2026, 1, 2), "symbol": "A", "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 99.0, "adj_close": 99.0, "volume": 1.0},
    ]))
    rets = daily_returns(prices)
    assert float(rets["ret"][1]) == pytest.approx(0.0)


# --- io --------------------------------------------------------------------

def test_write_and_read_roundtrip(tmp_path):
    prices = process_prices(_raw([
        {"date": D(2026, 1, 1), "symbol": "A", "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 1.0, "adj_close": 1.0, "volume": 1.0},
    ]))
    path = write_prices(prices, tmp_path)
    assert path.exists()
    assert read_prices(tmp_path).equals(prices)


def test_read_prices_missing_file_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_yahoo"):
        read_prices(tmp_path)
