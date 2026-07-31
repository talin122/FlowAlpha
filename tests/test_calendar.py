"""Trading-calendar arithmetic.

Availability lags are declared in trading days, so every off-by-one here becomes a
look-ahead bug downstream. The fixture deliberately contains a holiday and a long
weekend.
"""

from __future__ import annotations

import datetime as _dt

import polars as pl
import pytest

from flowalpha.data.calendar import (
    CalendarError,
    TradingCalendar,
    ensure_holidays_file,
)

D = _dt.date

# Mon 2024-01-01 .. Fri 2024-01-12, with Wed 2024-01-03 a holiday.
SESSIONS = [
    D(2024, 1, 1), D(2024, 1, 2), D(2024, 1, 4), D(2024, 1, 5),
    D(2024, 1, 8), D(2024, 1, 9), D(2024, 1, 10), D(2024, 1, 11), D(2024, 1, 12),
]


@pytest.fixture
def cal() -> TradingCalendar:
    return TradingCalendar(SESSIONS)


def test_dedupes_and_sorts():
    c = TradingCalendar([D(2024, 1, 5), D(2024, 1, 2), D(2024, 1, 5)])
    assert list(c.sessions) == [D(2024, 1, 2), D(2024, 1, 5)]
    assert len(c) == 2


def test_empty_calendar_rejected():
    with pytest.raises(CalendarError):
        TradingCalendar([])


def test_accepts_iso_strings_and_datetimes():
    c = TradingCalendar(["2024-01-02", _dt.datetime(2024, 1, 4, 15, 30)])
    assert list(c.sessions) == [D(2024, 1, 2), D(2024, 1, 4)]


def test_membership(cal):
    assert cal.is_session(D(2024, 1, 2))
    assert not cal.is_session(D(2024, 1, 3))     # holiday
    assert not cal.is_session(D(2024, 1, 6))     # Saturday
    assert D(2024, 1, 2) in cal
    assert "not-a-date" not in cal
    assert None not in cal


def test_start_end(cal):
    assert cal.start == D(2024, 1, 1)
    assert cal.end == D(2024, 1, 12)


def test_index_of(cal):
    assert cal.index_of(D(2024, 1, 1)) == 0
    assert cal.index_of(D(2024, 1, 4)) == 2
    with pytest.raises(CalendarError):
        cal.index_of(D(2024, 1, 3))


def test_shift_zero_is_identity(cal):
    """A declared lag of zero trading days means same-session availability."""
    assert cal.shift(D(2024, 1, 4), 0) == D(2024, 1, 4)
    assert cal.shift(D(2024, 1, 3), 0) == D(2024, 1, 3)


def test_shift_forward_skips_holiday(cal):
    # 1 trading day after Tue 2024-01-02 is Thu 2024-01-04, not Wed the 3rd.
    assert cal.shift(D(2024, 1, 2), 1) == D(2024, 1, 4)


def test_shift_forward_skips_weekend(cal):
    # 1 trading day after Fri 2024-01-05 is Mon 2024-01-08.
    assert cal.shift(D(2024, 1, 5), 1) == D(2024, 1, 8)


def test_shift_forward_multiple(cal):
    assert cal.shift(D(2024, 1, 2), 2) == D(2024, 1, 5)
    assert cal.shift(D(2024, 1, 2), 3) == D(2024, 1, 8)


def test_shift_forward_from_a_non_session(cal):
    """An event dated on a holiday still becomes available on a real session."""
    assert cal.shift(D(2024, 1, 3), 1) == D(2024, 1, 4)
    assert cal.shift(D(2024, 1, 6), 1) == D(2024, 1, 8)


def test_shift_backward(cal):
    assert cal.shift(D(2024, 1, 4), -1) == D(2024, 1, 2)
    assert cal.shift(D(2024, 1, 8), -2) == D(2024, 1, 4)


def test_shift_backward_past_start_raises(cal):
    with pytest.raises(CalendarError, match="before the calendar start"):
        cal.shift(D(2024, 1, 1), -1)


def test_shift_past_end_extrapolates_strictly_after_end(cal):
    """The final session's lagged availability must land beyond the known window.

    That is what keeps the last row of a lagged dataset invisible to an as-of query
    made on the last real session.
    """
    out = cal.shift(cal.end, 1)
    assert out > cal.end
    assert out.weekday() < 5
    assert cal.shift(D(2024, 1, 12), 1) == D(2024, 1, 15)  # Fri -> Mon


def test_shift_past_end_from_beyond_calendar(cal):
    assert cal.shift(D(2024, 2, 1), 1) == D(2024, 2, 2)
    assert cal.shift(D(2024, 2, 2), 1) == D(2024, 2, 5)  # Fri -> Mon


def test_next_prev_session(cal):
    assert cal.next_session(D(2024, 1, 2)) == D(2024, 1, 4)
    assert cal.next_session(D(2024, 1, 2), inclusive=True) == D(2024, 1, 2)
    assert cal.next_session(D(2024, 1, 3), inclusive=True) == D(2024, 1, 4)
    assert cal.prev_session(D(2024, 1, 4)) == D(2024, 1, 2)
    assert cal.prev_session(D(2024, 1, 4), inclusive=True) == D(2024, 1, 4)
    with pytest.raises(CalendarError):
        cal.prev_session(D(2023, 12, 31))


def test_sessions_between_is_inclusive(cal):
    got = cal.sessions_between(D(2024, 1, 2), D(2024, 1, 8))
    assert got == [D(2024, 1, 2), D(2024, 1, 4), D(2024, 1, 5), D(2024, 1, 8)]
    assert cal.n_sessions_between(D(2024, 1, 2), D(2024, 1, 8)) == 4
    assert cal.n_sessions_between(D(2024, 2, 1), D(2024, 2, 8)) == 0


def test_holidays_are_weekdays_that_never_traded(cal):
    assert cal.holidays() == [D(2024, 1, 3)]


def test_from_prices_derives_the_calendar():
    prices = pl.DataFrame(
        {"date": [D(2024, 1, 2), D(2024, 1, 2), D(2024, 1, 4)], "symbol": ["A", "B", "A"]}
    )
    c = TradingCalendar.from_prices(prices)
    assert list(c.sessions) == [D(2024, 1, 2), D(2024, 1, 4)]


def test_from_prices_on_empty_panel_raises():
    with pytest.raises(CalendarError):
        TradingCalendar.from_prices(pl.DataFrame({"date": []}, schema={"date": pl.Date}))


def test_roundtrip_through_file(tmp_path, cal):
    sessions_path, holidays_path = ensure_holidays_file(cal, tmp_path)
    assert sessions_path.exists() and holidays_path.exists()
    reloaded = TradingCalendar.from_file(sessions_path)
    assert list(reloaded.sessions) == list(cal.sessions)
    hol = pl.read_csv(holidays_path, try_parse_dates=True)
    assert hol["date"].to_list() == [D(2024, 1, 3)]


def test_from_file_requires_date_column(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("session\n2024-01-02\n", encoding="utf-8")
    with pytest.raises(CalendarError, match="date"):
        TradingCalendar.from_file(p)


def test_to_frame_has_date_dtype(cal):
    """Typed schema matters: a Null-typed date column breaks joins downstream."""
    assert cal.to_frame().schema["date"] == pl.Date
