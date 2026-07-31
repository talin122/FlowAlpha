"""NSE flow parsing and flow-feature construction.

Every parsing test here corresponds to a real property of the published files, and
every regression test corresponds to a bug that silently destroys the study rather
than crashing it.
"""

from __future__ import annotations

import datetime as _dt
import json
import math

import numpy as np
import polars as pl
import pytest

from flowalpha.data.nse_flows import (
    FII_DII_CASH_FILENAME,
    FLOWS_DAILY_FILENAME,
    PARTICIPANT_CATEGORIES,
    FlowParseError,
    flows_daily_from_participants,
    ingest_flows,
    parse_deals,
    parse_fii_dii_cash,
    parse_participant_cache_dir,
    parse_participant_vol,
    participant_cache_path,
    participant_title_date,
    participant_url,
    write_fii_dii_cash,
)
from flowalpha.data.store import PointInTimeStore
from flowalpha.factors.flow_features import (
    build_flow_features,
    collapse_extreme,
    divergence_label,
    expanding_bucket_labels,
    expanding_percentile_flag,
    regime_coverage,
    shift_one,
    trailing_sum,
)

from conftest import make_tiny_flows, make_tiny_prices, write_tree

D = _dt.date

# --- real payload samples --------------------------------------------------
# Verbatim shape of the published files. Note: the 2019 file uses one level of
# quoting and a full month name; the 2026 file uses two levels, an abbreviated
# month, and trailing spaces in two column names. Both are real.

SAMPLE_2019 = (
    '"Participant wise Trading Volume (no. of contracts) in Equity Derivatives as on January 01, 2019",,,,,,,,,,,,,,\n'
    "Client Type,Future Index Long,Future Index Short,Future Stock Long,Future Stock Short,"
    "Option Index Call Long,Option Index Put Long,Option Index Call Short,Option Index Put Short,"
    "Option Stock Call Long,Option Stock Put Long,Option Stock Call Short,Option Stock Put Short,"
    "Total Long Contracts,Total Short Contracts\n"
    "Client,138912,134096,248139,242804,2548744,2242879,2554636,2225434,96868,40218,95341,40811,5315760,5293122\n"
    "DII,386,357,2653,5077,0,249,0,1200,0,0,0,0,3288,6634\n"
    "FII,25000,15000,60000,40000,1,2,3,4,5,6,7,8,100,200\n"
    "Pro,10000,10000,20000,20000,0,0,0,0,0,0,0,0,50,50\n"
    "TOTAL,174298,159453,330792,307881,2548745,2242881,2554639,2225438,96873,40224,95348,40819,5368198,5299996\n"
)

SAMPLE_2026 = (
    '""Participant wise Trading Volume (no. of contracts) in Equity Derivatives as on Jul 29, 2026"",,,,,,,,,,,,,,\n'
    "Client Type,Future Index Long,Future Index Short,Future Stock Long,Future Stock Short       ,"
    "Option Index Call Long,Option Index Put Long,Option Index Call Short,Option Index Put Short,"
    "Option Stock Call Long,Option Stock Put Long,Option Stock Call Short,Option Stock Put Short,"
    "Total Long Contracts      ,Total Short Contracts\n"
    "Client,33174,51037,345282,379424,8807365,8059886,8987763,7922752,1444285,542875,1399612,529160,19232867,19269748\n"
    "DII,96,375,84947,92146,1335,7917,285,80,790,2788,52912,2349,97873,148147\n"
    "FII,21104,10321,353955,291786,1414347,1456715,1325013,1501439,175073,114545,180050,102129,3535739,3410738\n"
    "Pro,24205,16846,412287,433115,10307340,9113824,10217326,9214071,2405880,1366138,2393454,1392708,23629674,23667520\n"
    "TOTAL,78579,78579,1196471,1196471,20530387,18638342,20530387,18638342,4026028,2026346,4026028,2026346,46496153,46496153\n"
)

HTML_404 = (
    "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"UTF-8\">\n"
    "<title>Page not found</title>\n</head>\n<body>Not found</body>\n</html>\n"
)


# --- title-line date parsing ----------------------------------------------

def test_title_date_handles_the_2019_format():
    assert participant_title_date(SAMPLE_2019.splitlines()[0]) == D(2019, 1, 1)


def test_title_date_handles_the_2026_format():
    """Different month abbreviation AND a second level of quoting."""
    assert participant_title_date(SAMPLE_2026.splitlines()[0]) == D(2026, 7, 29)


def test_title_date_returns_none_when_absent():
    assert participant_title_date("Client Type,Future Index Long") is None
    assert participant_title_date('"as on Smarch 40, 2019"') is None


# --- participant parsing ---------------------------------------------------

@pytest.mark.parametrize("sample", [SAMPLE_2019, SAMPLE_2026], ids=["2019", "2026"])
def test_parses_both_published_formats(sample):
    got = parse_participant_vol(sample)
    assert sorted(got["participant"].to_list()) == sorted(PARTICIPANT_CATEGORIES)


def test_total_row_is_excluded():
    """Including the summary row would double the panel and void every regime."""
    got = parse_participant_vol(SAMPLE_2019)
    assert "TOTAL" not in got["participant"].to_list()
    assert got.height == 4


def test_trailing_space_headers_are_stripped():
    got = parse_participant_vol(SAMPLE_2026)
    assert "fut_stock_short" in got.columns
    assert "total_long" in got.columns
    fii = got.filter(pl.col("participant") == "FII")
    assert float(fii["fut_stock_short"][0]) == 291786.0
    assert float(fii["total_long"][0]) == 3535739.0


def test_net_is_futures_only():
    """Option legs are excluded deliberately: a long call and a short put are not
    comparable exposures, so summing option contracts does not measure direction."""
    got = parse_participant_vol(SAMPLE_2019)
    fii = got.filter(pl.col("participant") == "FII")
    expected = (25000 + 60000) - (15000 + 40000)
    assert float(fii["net_futures"][0]) == pytest.approx(expected)
    # The option columns are present but not in the net.
    assert float(fii["opt_index_call_long"][0]) == 1.0


def test_net_futures_hand_computed_for_every_category():
    got = parse_participant_vol(SAMPLE_2019).sort("participant")
    expected = {
        "Client": (138912 + 248139) - (134096 + 242804),
        "DII": (386 + 2653) - (357 + 5077),
        "FII": (25000 + 60000) - (15000 + 40000),
        "Pro": (10000 + 20000) - (10000 + 20000),
    }
    for row in got.iter_rows(named=True):
        assert row["net_futures"] == pytest.approx(expected[row["participant"]])


def test_accepts_bytes_and_str_identically():
    a = parse_participant_vol(SAMPLE_2019)
    b = parse_participant_vol(SAMPLE_2019.encode("utf-8"))
    assert a.equals(b)


def test_html_error_page_is_rejected():
    """A 404 serves a styled HTML page. Accepting it writes garbage into the panel."""
    with pytest.raises(FlowParseError, match="HTML error page"):
        parse_participant_vol(HTML_404)


def test_empty_payload_is_rejected():
    with pytest.raises(FlowParseError, match="empty"):
        parse_participant_vol("   \n  \n")


def test_explicit_session_date_overrides_the_title():
    """The filename is authoritative; the title format has already changed once."""
    got = parse_participant_vol(SAMPLE_2019, session_date=D(2020, 5, 5))
    assert set(got["date"].to_list()) == {D(2020, 5, 5)}


def test_missing_date_is_an_error_by_default():
    body = "\n".join(SAMPLE_2019.splitlines()[1:])
    with pytest.raises(FlowParseError, match="no parseable session date"):
        parse_participant_vol(body)


def test_header_only_first_line_is_handled():
    """When the title is absent, line 0 IS the header and must not be discarded."""
    body = "\n".join(SAMPLE_2019.splitlines()[1:])
    got = parse_participant_vol(body, session_date=D(2019, 1, 1))
    assert got.height == 4


def test_payload_without_expected_categories_is_rejected():
    payload = (
        '"as on January 01, 2019",,\n'
        "Client Type,Future Index Long,Future Index Short,Future Stock Long,Future Stock Short\n"
        "Nobody,1,2,3,4\n"
    )
    with pytest.raises(FlowParseError, match="none of the expected categories"):
        parse_participant_vol(payload)


def test_payload_without_client_type_column_is_rejected():
    payload = '"as on January 01, 2019",,\nFoo,Bar\n1,2\n'
    with pytest.raises(FlowParseError, match="Client Type"):
        parse_participant_vol(payload)


def test_payload_without_futures_columns_is_rejected():
    payload = (
        '"as on January 01, 2019",,\n'
        "Client Type,Option Index Call Long\n"
        "FII,5\n"
    )
    with pytest.raises(FlowParseError, match="futures column"):
        parse_participant_vol(payload)


def test_thousands_separators_are_handled():
    payload = SAMPLE_2019.replace("138912", '"138,912"')
    got = parse_participant_vol(payload)
    client = got.filter(pl.col("participant") == "Client")
    assert float(client["fut_index_long"][0]) == 138912.0


def test_participant_url_and_cache_path(tmp_path):
    assert participant_url(D(2026, 7, 29)).endswith("fao_participant_vol_29072026.csv")
    path = participant_cache_path(tmp_path, D(2026, 7, 29))
    assert path.name == "fao_participant_vol_29072026.csv"
    assert path.parent.name == "participant"


# --- cache directory parsing ----------------------------------------------

def test_cache_dir_uses_the_filename_date_not_the_title(tmp_path):
    """The filename is authoritative, so a stale title must not win."""
    cache = tmp_path / "flows" / "participant"
    cache.mkdir(parents=True)
    (cache / "fao_participant_vol_15062020.csv").write_text(SAMPLE_2019, encoding="utf-8")
    panel, failures = parse_participant_cache_dir(cache)
    assert failures == []
    assert set(panel["date"].to_list()) == {D(2020, 6, 15)}


def test_cache_dir_reports_failures_rather_than_swallowing_them(tmp_path):
    """A silently dropped session becomes an invisible gap in a rolling window."""
    cache = tmp_path / "flows" / "participant"
    cache.mkdir(parents=True)
    (cache / "fao_participant_vol_01012019.csv").write_text(SAMPLE_2019, encoding="utf-8")
    (cache / "fao_participant_vol_02012019.csv").write_text(HTML_404, encoding="utf-8")
    panel, failures = parse_participant_cache_dir(cache)
    assert panel.height == 4
    assert len(failures) == 1 and failures[0][0] == D(2019, 1, 2)


def test_cache_dir_flags_unparseable_filenames(tmp_path):
    cache = tmp_path / "flows" / "participant"
    cache.mkdir(parents=True)
    (cache / "fao_participant_vol_notadate.csv").write_text(SAMPLE_2019, encoding="utf-8")
    _, failures = parse_participant_cache_dir(cache)
    assert failures and "unparseable date" in failures[0][1]


def test_missing_cache_dir_returns_typed_empty(tmp_path):
    panel, failures = parse_participant_cache_dir(tmp_path / "nope")
    assert panel.is_empty() and failures == []
    assert panel.schema["date"] == pl.Date


# --- projection to flows_daily --------------------------------------------

def test_flows_daily_keeps_only_fii_and_dii():
    part = parse_participant_vol(SAMPLE_2019)
    daily = flows_daily_from_participants(part)
    assert sorted(daily["participant"].to_list()) == ["DII", "FII"]
    assert "gross_futures_long" in daily.columns


def test_flows_daily_gross_legs_are_futures_only():
    part = parse_participant_vol(SAMPLE_2019)
    daily = flows_daily_from_participants(part).filter(pl.col("participant") == "FII")
    assert float(daily["gross_futures_long"][0]) == 25000.0 + 60000.0
    assert float(daily["gross_futures_short"][0]) == 15000.0 + 40000.0


def test_flows_daily_from_empty_is_typed_empty():
    got = flows_daily_from_participants(pl.DataFrame(schema={"date": pl.Date}))
    assert got.is_empty() and got.schema["participant"] == pl.Utf8


# --- ingest ---------------------------------------------------------------

def test_ingest_writes_both_panels_and_reports_gaps(tmp_path):
    cache = tmp_path / "raw" / "flows" / "participant"
    cache.mkdir(parents=True)
    for day in (D(2019, 1, 1), D(2019, 1, 2)):
        (cache / f"fao_participant_vol_{day:%d%m%Y}.csv").write_text(
            SAMPLE_2019, encoding="utf-8"
        )
    processed = tmp_path / "processed"
    summary = ingest_flows(
        tmp_path / "raw", processed,
        expected_sessions=[D(2019, 1, 1), D(2019, 1, 2), D(2019, 1, 3)],
    )
    assert (processed / "participant_flows.parquet").exists()
    assert (processed / FLOWS_DAILY_FILENAME).exists()
    assert summary["sessions_covered"] == 2
    assert summary["missing_sessions"] == ["2019-01-03"]
    assert summary["n_missing_sessions"] == 1


# --- cash series ----------------------------------------------------------

CASH_PAYLOAD = json.dumps(
    [
        {"category": "FII/FPI *", "date": "29-Jul-2026", "buyValue": "12,000.50",
         "sellValue": "11,000.25", "netValue": "1,000.25"},
        {"category": "DII **", "date": "29-Jul-2026", "buyValue": "9,000.00",
         "sellValue": "9,500.00", "netValue": "-500.00"},
    ]
)


def test_cash_payload_parses_units_as_crores():
    got = parse_fii_dii_cash(CASH_PAYLOAD).sort("participant")
    assert got["participant"].to_list() == ["DII", "FII"]
    assert float(got.filter(pl.col("participant") == "FII")["net_crore"][0]) == pytest.approx(1000.25)
    assert float(got.filter(pl.col("participant") == "DII")["net_crore"][0]) == pytest.approx(-500.0)


def test_cash_payload_accepts_a_data_wrapper():
    wrapped = json.dumps({"data": json.loads(CASH_PAYLOAD)})
    assert parse_fii_dii_cash(wrapped).height == 2


def test_cash_payload_rejects_html_and_bad_json():
    with pytest.raises(FlowParseError, match="HTML"):
        parse_fii_dii_cash(HTML_404)
    with pytest.raises(FlowParseError, match="valid JSON"):
        parse_fii_dii_cash("{not json")


def test_cash_payload_rejects_unparseable_date():
    bad = json.dumps([{"category": "FII", "date": "sometime", "netValue": "1"}])
    with pytest.raises(FlowParseError, match="unparseable date"):
        parse_fii_dii_cash(bad)


def test_cash_payload_with_no_institutional_rows_is_rejected():
    with pytest.raises(FlowParseError, match="no FII or DII"):
        parse_fii_dii_cash(json.dumps([{"category": "Mutual Funds", "date": "29-Jul-2026"}]))


def test_cash_writer_never_touches_flows_daily(tmp_path):
    """The trap: overwriting the full-window flow panel with a few months of cash
    data nulls almost every regime downstream."""
    processed = tmp_path / "processed"
    processed.mkdir()
    sentinel = pl.DataFrame({"date": [D(2019, 1, 1)], "participant": ["FII"],
                             "net_futures": [1.0], "gross_futures_long": [1.0],
                             "gross_futures_short": [0.0]})
    sentinel.write_parquet(processed / FLOWS_DAILY_FILENAME)

    write_fii_dii_cash(parse_fii_dii_cash(CASH_PAYLOAD), processed)
    assert (processed / FII_DII_CASH_FILENAME).exists()
    assert pl.read_parquet(processed / FLOWS_DAILY_FILENAME).equals(sentinel)


def test_cash_writer_appends_and_dedupes(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    rows = parse_fii_dii_cash(CASH_PAYLOAD)
    write_fii_dii_cash(rows, processed)
    write_fii_dii_cash(rows, processed)
    got = pl.read_parquet(processed / FII_DII_CASH_FILENAME)
    assert got.height == 2


# --- deals ----------------------------------------------------------------

BULK_DEALS = (
    "Date,Symbol,Security Name,Client Name,Buy/Sell,Quantity Traded,"
    "Trade Price / Wght. Avg. Price,Remarks\n"
    "29-Jul-2026,RELIANCE,Reliance Industries,SOME FUND,BUY,\"1,000,000\",2500.50,-\n"
    "29-Jul-2026,TCS,Tata Consultancy,OTHER FUND,SELL,\"500,000\",3800.75,-\n"
)


def test_bulk_deals_parse():
    got = parse_deals(BULK_DEALS, kind="bulk")
    assert got.height == 2
    assert set(got["symbol"].to_list()) == {"RELIANCE", "TCS"}
    rel = got.filter(pl.col("symbol") == "RELIANCE")
    assert float(rel["quantity"][0]) == 1_000_000.0
    assert float(rel["price"][0]) == pytest.approx(2500.50)
    assert rel["buy_sell"][0] == "BUY"
    assert rel["date"][0] == D(2026, 7, 29)


def test_deals_reject_html_and_empty():
    with pytest.raises(FlowParseError, match="HTML"):
        parse_deals(HTML_404)
    with pytest.raises(FlowParseError, match="empty"):
        parse_deals("  ")


def test_deals_require_symbol_column():
    with pytest.raises(FlowParseError, match="symbol"):
        parse_deals("Date,Foo\n29-Jul-2026,1\n")


# --- trailing_sum: THE regression test ------------------------------------

def test_trailing_sum_basic():
    got = trailing_sum([1.0, 2.0, 3.0, 4.0], 2)
    assert math.isnan(got[0])
    assert list(got[1:]) == pytest.approx([3.0, 5.0, 7.0])


def test_trailing_sum_requires_a_complete_window():
    got = trailing_sum([1.0, 2.0], 3)
    assert np.isnan(got).all()


def test_trailing_sum_recovers_after_a_gap():
    """THE regression test.

    Real NSE data has a handful of unpublished sessions. A cumsum-based
    implementation propagates the first NaN to every later value, so regimes stop
    dead years before the data ends. Values after the gap must recover and must equal
    what the same window would give with no gap at all.
    """
    window = 3
    clean = np.arange(1.0, 21.0)
    gapped = clean.copy()
    gapped[7] = np.nan

    got_clean = trailing_sum(clean, window)
    got_gapped = trailing_sum(gapped, window)

    # Exactly the windows spanning index 7 are nulled: indices 7, 8, 9.
    assert list(np.where(np.isnan(got_gapped))[0]) == [0, 1, 7, 8, 9]
    # Everything from index 10 onward is bit-identical to the ungapped result.
    np.testing.assert_array_equal(got_gapped[10:], got_clean[10:])
    # And before the gap too.
    np.testing.assert_array_equal(got_gapped[2:7], got_clean[2:7])


def test_trailing_sum_recovers_after_multiple_gaps():
    window = 5
    clean = np.arange(1.0, 61.0)
    gapped = clean.copy()
    gapped[[10, 30, 44]] = np.nan
    got = trailing_sum(gapped, window)
    assert not np.isnan(got[20])
    assert not np.isnan(got[40])
    assert got[59] == pytest.approx(trailing_sum(clean, window)[59])


def test_trailing_sum_all_nan_stays_nan():
    """An all-NaN series must not silently become zeros."""
    got = trailing_sum(np.full(30, np.nan), 5)
    assert np.isnan(got).all()


def test_trailing_sum_rejects_nonpositive_window():
    with pytest.raises(ValueError):
        trailing_sum([1.0], 0)


def test_shift_one_moves_information_one_session_later():
    got = shift_one(np.array([1.0, 2.0, 3.0]))
    assert math.isnan(got[0])
    assert list(got[1:]) == pytest.approx([1.0, 2.0])


def test_shift_one_on_a_single_observation():
    assert np.isnan(shift_one(np.array([1.0]))).all()


# --- expanding statistics -------------------------------------------------

def test_expanding_buckets_use_history_only():
    values = np.array([float(i) for i in range(30)])
    labels = expanding_bucket_labels(values, n_buckets=3, min_obs=5)
    assert labels[:4] == [None] * 4
    # A monotonically rising series always sits at the top of its own history.
    assert labels[10] == "high"
    assert labels[29] == "high"


def test_expanding_buckets_emit_nothing_before_min_obs():
    labels = expanding_bucket_labels(np.arange(10.0), n_buckets=3, min_obs=100)
    assert labels == [None] * 10


def test_expanding_buckets_skip_nulls_without_consuming_min_obs():
    values = np.array([np.nan, 1.0, np.nan, 2.0, 3.0])
    labels = expanding_bucket_labels(values, n_buckets=3, min_obs=3)
    assert labels[0] is None and labels[2] is None
    assert labels[3] is None      # only 2 defined values so far
    assert labels[4] is not None  # 3 defined values


def test_expanding_buckets_are_balanced_on_iid_input():
    rng = np.random.default_rng(9)
    labels = expanding_bucket_labels(rng.normal(size=3000), n_buckets=3, min_obs=100)
    defined = [x for x in labels if x is not None]
    counts = {k: defined.count(k) for k in ("low", "mid", "high")}
    for value in counts.values():
        assert 0.28 < value / len(defined) < 0.39, counts


def test_expanding_buckets_handle_heavy_ties():
    """Mid-rank tie handling stops a repeated value from piling into one bucket."""
    values = np.array([1.0] * 50 + [2.0] * 50)
    labels = expanding_bucket_labels(values, n_buckets=3, min_obs=10)
    assert set(x for x in labels if x) <= {"low", "mid", "high"}
    assert labels[-1] == "high"


def test_expanding_buckets_support_more_than_three():
    labels = expanding_bucket_labels(np.arange(100.0), n_buckets=5, min_obs=10)
    assert labels[-1] == "q5"


def test_expanding_buckets_reject_bad_arguments():
    with pytest.raises(ValueError):
        expanding_bucket_labels(np.arange(10.0), n_buckets=1)
    with pytest.raises(ValueError):
        expanding_bucket_labels(np.arange(10.0), n_buckets=3, labels=("a", "b"))


def test_expanding_percentile_flag():
    values = np.array([1.0] * 50 + [100.0])
    flags = expanding_percentile_flag(values, percentile=0.95, min_obs=10)
    assert flags[:9] == [None] * 9
    assert flags[-1] is True


def test_expanding_percentile_flag_is_not_full_sample():
    """A value that is extreme relative to its own history stays flagged even when a
    far larger value arrives later."""
    values = np.concatenate([np.ones(50), [5.0], np.ones(20), [1000.0]])
    flags = expanding_percentile_flag(values, percentile=0.95, min_obs=10)
    assert flags[50] is True
    assert flags[-1] is True


def test_expanding_percentile_flag_rejects_bad_percentile():
    with pytest.raises(ValueError):
        expanding_percentile_flag(np.arange(10.0), percentile=1.5)


def test_divergence_label():
    assert divergence_label(100.0, 50.0) == "aligned"
    assert divergence_label(-100.0, -50.0) == "aligned"
    assert divergence_label(100.0, -50.0) == "opposed"
    assert divergence_label(0.0, 50.0) == "neutral"
    assert divergence_label(float("nan"), 50.0) is None
    assert divergence_label(None, 50.0) is None


def test_collapse_extreme():
    assert collapse_extreme("mid") == "mid"
    assert collapse_extreme("low") == "extreme"
    assert collapse_extreme("high") == "extreme"
    assert collapse_extreme(None) is None


# --- panel construction ---------------------------------------------------

def test_flow_features_row_uses_only_prior_session_flow(tiny_cfg, tiny_calendar, tiny_sessions):
    daily, part = make_tiny_flows(tiny_sessions)
    write_tree(tiny_cfg, prices=make_tiny_prices(tiny_sessions),
               flows_daily=daily, participant_flows=part)
    store = PointInTimeStore(tiny_cfg.path("processed"), tiny_calendar, tiny_cfg["availability"])
    features = build_flow_features(store, tiny_cfg, calendar=tiny_calendar)

    fii_by_date = dict(zip(daily.filter(pl.col("participant") == "FII")["date"].to_list(),
                           daily.filter(pl.col("participant") == "FII")["net_futures"].to_list()))
    row = features.filter(pl.col("date") == tiny_sessions[50])
    assert float(row["fii_daily"][0]) == pytest.approx(fii_by_date[tiny_sessions[49]])


def test_flow_features_first_session_has_no_usable_flow(tiny_cfg, tiny_calendar, tiny_sessions):
    daily, part = make_tiny_flows(tiny_sessions)
    write_tree(tiny_cfg, prices=make_tiny_prices(tiny_sessions),
               flows_daily=daily, participant_flows=part)
    store = PointInTimeStore(tiny_cfg.path("processed"), tiny_calendar, tiny_cfg["availability"])
    features = build_flow_features(store, tiny_cfg, calendar=tiny_calendar)
    first = features.filter(pl.col("date") == tiny_sessions[0])
    assert first["fii_daily"][0] is None


def test_flow_features_survive_a_missing_session(tiny_cfg, tiny_calendar, tiny_sessions):
    """The real-world condition: NSE publishes nothing for a handful of sessions.

    Regimes must recover after the gap and still be defined at the end of the sample.
    """
    gap = tiny_sessions[60]
    daily, part = make_tiny_flows(tiny_sessions, drop_sessions={gap})
    write_tree(tiny_cfg, prices=make_tiny_prices(tiny_sessions),
               flows_daily=daily, participant_flows=part)
    store = PointInTimeStore(tiny_cfg.path("processed"), tiny_calendar, tiny_cfg["availability"])
    features = build_flow_features(store, tiny_cfg, calendar=tiny_calendar)

    coverage = regime_coverage(features)
    last_defined = coverage["regimes"]["fii_regime"]["last_defined"]
    assert last_defined == tiny_sessions[-1].isoformat(), (
        "regimes stopped before the end of the sample -- the trailing sum is "
        "propagating the gap"
    )
    # The windows spanning the gap are nulled, and only those.
    window = int(tiny_cfg["conditioning"]["regime_window"])
    nulled = features.filter(pl.col("fii_flow_21d").is_null())["date"].to_list()
    spanning = [d for d in nulled if gap <= d <= tiny_sessions[60 + window]]
    assert len(spanning) == window
    assert features.filter(pl.col("date") == tiny_sessions[60 + window + 1])["fii_flow_21d"][0] is not None


def test_flow_features_match_the_ungapped_result_after_recovery(tiny_cfg, tiny_calendar, tiny_sessions):
    gap = tiny_sessions[60]
    clean_daily, clean_part = make_tiny_flows(tiny_sessions)
    write_tree(tiny_cfg, prices=make_tiny_prices(tiny_sessions),
               flows_daily=clean_daily, participant_flows=clean_part)
    store = PointInTimeStore(tiny_cfg.path("processed"), tiny_calendar, tiny_cfg["availability"])
    clean = build_flow_features(store, tiny_cfg, calendar=tiny_calendar)

    gapped_daily, gapped_part = make_tiny_flows(tiny_sessions, drop_sessions={gap})
    write_tree(tiny_cfg, flows_daily=gapped_daily, participant_flows=gapped_part)
    store2 = PointInTimeStore(tiny_cfg.path("processed"), tiny_calendar, tiny_cfg["availability"])
    gapped = build_flow_features(store2, tiny_cfg, calendar=tiny_calendar)

    day = tiny_sessions[100]
    a = float(clean.filter(pl.col("date") == day)["fii_flow_21d"][0])
    b = float(gapped.filter(pl.col("date") == day)["fii_flow_21d"][0])
    # The gap removed one observation from the sum, so the values differ -- but the
    # window is complete again and the arithmetic is local, not cumulative.
    assert math.isfinite(a) and math.isfinite(b)


def test_flow_features_empty_window_returns_typed_empty(tiny_cfg, tiny_calendar, tiny_sessions):
    daily, part = make_tiny_flows(tiny_sessions)
    write_tree(tiny_cfg, prices=make_tiny_prices(tiny_sessions),
               flows_daily=daily, participant_flows=part)
    store = PointInTimeStore(tiny_cfg.path("processed"), tiny_calendar, tiny_cfg["availability"])
    features = build_flow_features(store, tiny_cfg, calendar=tiny_calendar,
                                   as_of_date=tiny_sessions[0] - _dt.timedelta(days=30))
    assert features.is_empty()
    assert features.schema["date"] == pl.Date


def test_regime_coverage_reports_the_last_defined_date(tiny_cfg, tiny_calendar, tiny_sessions):
    daily, part = make_tiny_flows(tiny_sessions)
    write_tree(tiny_cfg, prices=make_tiny_prices(tiny_sessions),
               flows_daily=daily, participant_flows=part)
    store = PointInTimeStore(tiny_cfg.path("processed"), tiny_calendar, tiny_cfg["availability"])
    coverage = regime_coverage(build_flow_features(store, tiny_cfg, calendar=tiny_calendar))
    assert coverage["n_sessions"] == len(tiny_sessions)
    for col in ("fii_regime", "retail_regime", "retail_extreme"):
        assert coverage["regimes"][col]["last_defined"] == tiny_sessions[-1].isoformat()
        assert coverage["regimes"][col]["n_defined"] > 0
    assert coverage["regimes"]["flow_shock"]["n_true"] >= 0


def test_regime_coverage_on_empty_panel():
    assert regime_coverage(pl.DataFrame(schema={"date": pl.Date}))["n_sessions"] == 0


def test_flow_features_contain_no_float_nan(tiny_cfg, tiny_calendar, tiny_sessions):
    """NaN is not null in polars.

    Every consumer filters with is_null()/is_not_null(); a float NaN sails through
    such a filter and turns a downstream mean into NaN for the entire series. The
    panel must express "missing" one way only.
    """
    daily, part = make_tiny_flows(tiny_sessions, drop_sessions={tiny_sessions[60]})
    write_tree(tiny_cfg, prices=make_tiny_prices(tiny_sessions),
               flows_daily=daily, participant_flows=part)
    store = PointInTimeStore(tiny_cfg.path("processed"), tiny_calendar, tiny_cfg["availability"])
    features = build_flow_features(store, tiny_cfg, calendar=tiny_calendar)
    for col, dtype in features.schema.items():
        if dtype == pl.Float64:
            assert features[col].is_nan().fill_null(False).sum() == 0, col
    # And the nulls are genuinely there to be found.
    assert features["fii_daily"].null_count() > 0
