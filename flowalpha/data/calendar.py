"""Trading calendar for NSE.

There is no free, authoritative, machine-readable NSE holiday history covering
2019->present. Rather than hardcode a list that will silently rot, the calendar is
*derived from observed price data*: any weekday inside the data window on which no
symbol in a broad universe traded is a market holiday. This has the useful property
that the calendar can never disagree with the price panel it was built from.

The calendar exists because availability lags are declared in *trading* days. A lag
of one trading day over a long weekend is three calendar days, and getting that
wrong is a look-ahead bug of exactly the kind this project exists to prevent.
"""

from __future__ import annotations

import bisect
import datetime as _dt
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl


class CalendarError(Exception):
    """Raised when the calendar is empty or asked about an out-of-range date."""


class TradingCalendar:
    """An ordered set of trading sessions, with trading-day arithmetic.

    Parameters
    ----------
    sessions:
        Dates on which the market traded. Duplicates and ordering are handled here
        so callers can pass raw observed dates.
    """

    def __init__(self, sessions: Iterable[_dt.date]) -> None:
        uniq = sorted({_coerce_date(d) for d in sessions})
        if not uniq:
            raise CalendarError("cannot build a TradingCalendar with no sessions")
        self._sessions: list[_dt.date] = uniq
        self._index: dict[_dt.date, int] = {d: i for i, d in enumerate(uniq)}

    # -- construction -------------------------------------------------------
    @classmethod
    def from_prices(cls, prices: pl.DataFrame, date_col: str = "date") -> "TradingCalendar":
        """Build from the distinct dates present in a price panel."""
        if prices.is_empty():
            raise CalendarError("cannot derive a calendar from an empty price panel")
        dates = prices.select(pl.col(date_col)).unique().to_series().to_list()
        return cls(dates)

    @classmethod
    def from_file(cls, path: str | Path) -> "TradingCalendar":
        """Read a one-column ``date`` CSV of sessions."""
        frame = pl.read_csv(path, try_parse_dates=True)
        if "date" not in frame.columns:
            raise CalendarError(f"{path}: expected a 'date' column, got {frame.columns}")
        return cls(frame["date"].to_list())

    # -- membership ---------------------------------------------------------
    @property
    def sessions(self) -> Sequence[_dt.date]:
        return tuple(self._sessions)

    @property
    def start(self) -> _dt.date:
        return self._sessions[0]

    @property
    def end(self) -> _dt.date:
        return self._sessions[-1]

    def __len__(self) -> int:
        return len(self._sessions)

    def __contains__(self, d: object) -> bool:
        try:
            return _coerce_date(d) in self._index  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    def is_session(self, d: _dt.date) -> bool:
        return _coerce_date(d) in self._index

    def index_of(self, d: _dt.date) -> int:
        d = _coerce_date(d)
        try:
            return self._index[d]
        except KeyError as exc:
            raise CalendarError(f"{d} is not a trading session") from exc

    # -- navigation ---------------------------------------------------------
    def next_session(self, d: _dt.date, *, inclusive: bool = False) -> _dt.date:
        """First session on (``inclusive``) or after ``d``.

        Extrapolates past the end of the known calendar by skipping weekends; see
        the module docstring for why that is safe in this codebase.
        """
        d = _coerce_date(d)
        pos = bisect.bisect_left(self._sessions, d) if inclusive else bisect.bisect_right(self._sessions, d)
        if pos < len(self._sessions):
            return self._sessions[pos]
        return _next_weekday(max(d, self.end), strictly_after=not (inclusive and d > self.end))

    def prev_session(self, d: _dt.date, *, inclusive: bool = False) -> _dt.date:
        """Last session on (``inclusive``) or before ``d``."""
        d = _coerce_date(d)
        pos = (bisect.bisect_right(self._sessions, d) if inclusive else bisect.bisect_left(self._sessions, d)) - 1
        if pos < 0:
            raise CalendarError(f"no session on or before {d} (calendar starts {self.start})")
        return self._sessions[pos]

    def shift(self, d: _dt.date, n: int) -> _dt.date:
        """Move ``n`` trading days from ``d``.

        ``n == 0`` returns ``d`` unchanged -- deliberately, so that a declared lag
        of zero trading days means "available the same session", not "available at
        the next session".

        For ``n > 0`` the result is the ``n``-th session strictly after ``d``, which
        is the semantics availability lags need: an event dated ``d`` with a
        one-trading-day lag becomes readable on the following session.
        """
        d = _coerce_date(d)
        if n == 0:
            return d
        if n > 0:
            pos = bisect.bisect_right(self._sessions, d)  # first session strictly after d
            target = pos + n - 1
            if target < len(self._sessions):
                return self._sessions[target]
            # Past the end of observed data: extrapolate over weekdays. Any date
            # produced here is strictly after the calendar end, so a store query
            # made as of a real session excludes the row regardless of the exact
            # extrapolated value.
            if d >= self.end:
                out, steps = d, n
            else:
                out, steps = self.end, target - (len(self._sessions) - 1)
            for _ in range(steps):
                out = _next_weekday(out, strictly_after=True)
            return out
        pos = bisect.bisect_left(self._sessions, d)
        target = pos + n
        if target < 0:
            raise CalendarError(
                f"shifting {d} by {n} trading days falls before the calendar start {self.start}"
            )
        return self._sessions[target]

    def sessions_between(self, start: _dt.date, end: _dt.date) -> list[_dt.date]:
        """Sessions in ``[start, end]`` inclusive."""
        start, end = _coerce_date(start), _coerce_date(end)
        lo = bisect.bisect_left(self._sessions, start)
        hi = bisect.bisect_right(self._sessions, end)
        return self._sessions[lo:hi]

    def n_sessions_between(self, start: _dt.date, end: _dt.date) -> int:
        return len(self.sessions_between(start, end))

    def holidays(self) -> list[_dt.date]:
        """Weekdays inside the calendar span on which the market did not trade."""
        traded = self._index
        out: list[_dt.date] = []
        d = self.start
        one = _dt.timedelta(days=1)
        while d <= self.end:
            if d.weekday() < 5 and d not in traded:
                out.append(d)
            d += one
        return out

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame({"date": self._sessions}, schema={"date": pl.Date})


def _coerce_date(d) -> _dt.date:
    if isinstance(d, _dt.datetime):
        return d.date()
    if isinstance(d, _dt.date):
        return d
    if isinstance(d, str):
        return _dt.date.fromisoformat(d)
    raise TypeError(f"cannot interpret {d!r} as a date")


def _next_weekday(d: _dt.date, *, strictly_after: bool = True) -> _dt.date:
    out = d + _dt.timedelta(days=1) if strictly_after else d
    while out.weekday() >= 5:
        out += _dt.timedelta(days=1)
    return out


def ensure_holidays_file(
    calendar: TradingCalendar,
    reference_dir: str | Path,
    *,
    overwrite: bool = True,
) -> tuple[Path, Path]:
    """Persist the derived calendar and its implied holidays under ``reference_dir``.

    Returns ``(sessions_path, holidays_path)``. Both are regenerable artefacts kept
    in the reference directory so that downstream stages (and a human auditing a
    lag calculation) can see exactly which sessions the run believed in.
    """
    ref = Path(reference_dir)
    ref.mkdir(parents=True, exist_ok=True)
    sessions_path = ref / "trading_sessions.csv"
    holidays_path = ref / "holidays.csv"
    if overwrite or not sessions_path.exists():
        calendar.to_frame().write_csv(sessions_path)
    if overwrite or not holidays_path.exists():
        hol = calendar.holidays()
        pl.DataFrame({"date": hol}, schema={"date": pl.Date}).write_csv(holidays_path)
    return sessions_path, holidays_path
