"""Point-in-time data access.

:class:`PointInTimeStore` is the **only** sanctioned way research code reads data.
The invariant it enforces is not "we were careful": research code is *unable* to see
a row whose declared availability date is after the query's ``as_of_date``, because
the filter happens below every consumer.

How a lag becomes a date
------------------------
Each dataset declares its lag in ``config.yaml`` under ``availability``:

* ``lag_days: n`` -- ``n`` **trading** days. Resolved along the trading calendar, so a
  one-trading-day lag over a long weekend is three calendar days. ``n = 0`` means the
  row is readable on its own event date.
* ``lag_calendar_days: n`` -- ``n`` wall-clock days, for filings whose disclosure
  deadline is expressed that way.

Given an event date ``d``, ``available_from(dataset, d)`` is the date on which that
row first became public. ``get`` refuses to return any row with
``available_from > as_of_date``.

Why ``flow_features`` is registered at lag 0
--------------------------------------------
``flow_features`` is a *derived* dataset. Its construction already bakes in the
underlying flow lag: the row stamped ``d`` is built from flow data through ``d-1``
and is the as-of-``d`` snapshot. Registering it with a further lag would double-lag
it, silently discarding a day of genuinely available information and making every
regime label one session staler than it should be. This is stated here, in
``DATASET_SPECS`` and in ``factors/flow_features.py``, because it is exactly the kind
of thing a later reader "fixes".
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import polars as pl

from ..config import Config
from .calendar import TradingCalendar


class LookAheadError(Exception):
    """Raised when a query would require data that was not yet public.

    Only raised for explicit programming errors -- an unknown dataset, or a request
    for a field that does not exist. Rows that are merely not-yet-available are
    filtered out silently, because "no data yet" is a normal condition at the start
    of a backtest, not an error.
    """


@dataclass(frozen=True)
class DatasetSpec:
    """Registration for one dataset.

    Attributes
    ----------
    filename:
        Parquet file inside ``paths.processed``.
    event_date_col:
        Column holding the date the observation *refers to* (not the date it became
        public). Availability is computed from this.
    availability_key:
        Key under ``availability`` in ``config.yaml``. A dataset with no entry there
        cannot have its lag enforced, so registration requires one.
    entity_col:
        Per-name key, when the dataset is a cross-section. ``None`` for aggregates
        like the daily flow series.
    note:
        Why the spec looks the way it does. Surfaced by ``describe()``.
    """

    filename: str
    event_date_col: str
    availability_key: str
    entity_col: str | None = None
    note: str = ""


#: The dataset registry. Adding a dataset here without an ``availability`` entry in
#: ``config.yaml`` is a hard error -- an unregistered lag is an unenforced lag.
DATASET_SPECS: dict[str, DatasetSpec] = {
    "prices": DatasetSpec(
        filename="prices.parquet",
        event_date_col="date",
        availability_key="prices",
        entity_col="symbol",
        note="EOD bars are public the same session they refer to, hence lag 0.",
    ),
    "flows_daily": DatasetSpec(
        filename="flows_daily.parquet",
        event_date_col="date",
        availability_key="flows_daily",
        note="NSE publishes the participant archive after the close, so a session's "
             "flow is first usable the NEXT trading day.",
    ),
    "participant_flows": DatasetSpec(
        filename="participant_flows.parquet",
        event_date_col="date",
        availability_key="participant_flows",
        note="Same publication timing as flows_daily; all four categories.",
    ),
    "fii_dii_cash": DatasetSpec(
        filename="fii_dii_cash.parquet",
        event_date_col="date",
        availability_key="fii_dii_cash",
        note="Separate from flows_daily: different units (Rs crores) and a much "
             "shorter span, because the source API is current-day only.",
    ),
    "bulk_deals": DatasetSpec(
        filename="bulk_deals.parquet",
        event_date_col="date",
        availability_key="bulk_deals",
        entity_col="symbol",
        note="Disclosed after the close of the deal date.",
    ),
    "block_deals": DatasetSpec(
        filename="block_deals.parquet",
        event_date_col="date",
        availability_key="block_deals",
        entity_col="symbol",
        note="Disclosed after the close of the deal date.",
    ),
    "shareholding": DatasetSpec(
        filename="shareholding.parquet",
        event_date_col="date",
        availability_key="shareholding",
        entity_col="symbol",
        note="Quarterly filing. The lag is in CALENDAR days because the disclosure "
             "deadline is expressed that way, not in sessions.",
    ),
    "flow_features": DatasetSpec(
        filename="flow_features.parquet",
        event_date_col="date",
        availability_key="flow_features",
        note="DERIVED, and registered at lag 0 ON PURPOSE. The row stamped d is the "
             "as-of-d snapshot: its construction already shifted the underlying flow "
             "by one session. Adding a lag here would double-lag it and make every "
             "regime label a session staler than the data allows.",
    ),
}

#: Datasets that are derived rather than ingested. Their availability entry may be
#: absent from config.yaml, in which case lag 0 is assumed *and documented*.
_DERIVED_ZERO_LAG = {"flow_features"}


@dataclass(frozen=True)
class Availability:
    """Resolved lag for one dataset."""

    trading_days: int | None
    calendar_days: int | None

    def describe(self) -> str:
        if self.trading_days is not None:
            return f"{self.trading_days} trading day(s)"
        return f"{self.calendar_days} calendar day(s)"


class PointInTimeStore:
    """Lag-enforcing reader over the processed parquet tree.

    Parameters
    ----------
    processed_dir:
        Directory containing the dataset parquet files.
    calendar:
        Trading calendar used to resolve trading-day lags.
    availability:
        Mapping ``dataset -> {"lag_days": n}`` or ``{"lag_calendar_days": n}``, i.e.
        the ``availability`` block of ``config.yaml``.
    specs:
        Dataset registry. Overridable for tests.

    Examples
    --------
    >>> store.get("flows_daily", as_of_date=date(2024, 1, 4))  # doctest: +SKIP
    Rows whose event date is 2024-01-03 or earlier, given a one-trading-day lag.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        calendar: TradingCalendar,
        availability: Mapping[str, Mapping[str, int]],
        *,
        specs: Mapping[str, DatasetSpec] | None = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.calendar = calendar
        self._availability_cfg = dict(availability)
        self.specs: dict[str, DatasetSpec] = dict(specs or DATASET_SPECS)
        self._frames: dict[str, pl.DataFrame] = {}

    # -- construction -------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        cfg: Config,
        *,
        calendar: TradingCalendar | None = None,
        specs: Mapping[str, DatasetSpec] | None = None,
    ) -> "PointInTimeStore":
        """Build a store from config, deriving the calendar if not supplied."""
        if calendar is None:
            sessions_path = cfg.path("reference") / "trading_sessions.csv"
            if sessions_path.exists():
                calendar = TradingCalendar.from_file(sessions_path)
            else:
                prices_path = cfg.path("processed") / DATASET_SPECS["prices"].filename
                if not prices_path.exists():
                    raise FileNotFoundError(
                        "cannot build a calendar: neither reference/trading_sessions.csv "
                        f"nor {prices_path} exists. Run scripts/download_yahoo.py first."
                    )
                calendar = TradingCalendar.from_prices(
                    pl.read_parquet(prices_path, columns=["date"])
                )
        return cls(cfg.path("processed"), calendar, cfg["availability"], specs=specs)

    # -- availability -------------------------------------------------------
    def availability_of(self, dataset: str) -> Availability:
        """Resolve the declared lag for ``dataset``."""
        spec = self._spec(dataset)
        entry = self._availability_cfg.get(spec.availability_key)
        if entry is None:
            if dataset in _DERIVED_ZERO_LAG:
                # Documented in DATASET_SPECS: derived features already carry the
                # underlying lag inside their construction.
                return Availability(trading_days=0, calendar_days=None)
            raise LookAheadError(
                f"dataset {dataset!r} has no availability entry under "
                f"'{spec.availability_key}' in config.yaml. An unregistered lag is an "
                "unenforced lag; add one rather than defaulting to zero."
            )
        if "lag_days" in entry:
            return Availability(trading_days=int(entry["lag_days"]), calendar_days=None)
        if "lag_calendar_days" in entry:
            return Availability(trading_days=None, calendar_days=int(entry["lag_calendar_days"]))
        raise LookAheadError(
            f"availability.{spec.availability_key} declares neither lag_days nor "
            "lag_calendar_days"
        )

    def available_from(self, dataset: str, event_date: _dt.date) -> _dt.date:
        """Date on which an observation dated ``event_date`` first became public."""
        av = self.availability_of(dataset)
        if av.calendar_days is not None:
            return event_date + _dt.timedelta(days=av.calendar_days)
        return self.calendar.shift(event_date, int(av.trading_days or 0))

    def max_event_date(self, dataset: str, as_of_date: _dt.date) -> _dt.date | None:
        """Latest event date whose availability date is on or before ``as_of_date``.

        ``None`` when nothing is available yet. Computed by inverting the lag rather
        than by scanning, so it is cheap and matches :meth:`get` exactly.
        """
        av = self.availability_of(dataset)
        if av.calendar_days is not None:
            return as_of_date - _dt.timedelta(days=av.calendar_days)
        n = int(av.trading_days or 0)
        if n == 0:
            # Lag zero: everything dated on or before today is readable today. Note
            # this is `as_of_date` itself, not the previous session, so a query made
            # on a non-session still sees that session's data.
            return as_of_date
        return self._invert_trading_lag(as_of_date, n)

    def _invert_trading_lag(self, as_of_date: _dt.date, n: int) -> _dt.date | None:
        """Largest event date ``d`` with ``shift(d, n) <= as_of_date``.

        Equivalently: step back ``n`` sessions from the last session at or before
        ``as_of_date``. Returns ``None`` when that falls before the calendar starts.
        """
        from .calendar import CalendarError

        try:
            anchor = self.calendar.prev_session(as_of_date, inclusive=True)
        except CalendarError:
            return None
        try:
            return self.calendar.shift(anchor, -n)
        except CalendarError:
            return None

    # -- reads --------------------------------------------------------------
    def _spec(self, dataset: str) -> DatasetSpec:
        try:
            return self.specs[dataset]
        except KeyError as exc:
            raise LookAheadError(
                f"unknown dataset {dataset!r}. Register it in DATASET_SPECS with an "
                f"availability key. Known: {sorted(self.specs)}"
            ) from exc

    def path_of(self, dataset: str) -> Path:
        return self.processed_dir / self._spec(dataset).filename

    def exists(self, dataset: str) -> bool:
        return self.path_of(dataset).exists()

    def _load(self, dataset: str) -> pl.DataFrame:
        if dataset in self._frames:
            return self._frames[dataset]
        path = self.path_of(dataset)
        if not path.exists():
            raise FileNotFoundError(
                f"dataset {dataset!r} not found at {path}. Run the relevant "
                "scripts/download_*.py or scripts/build_*.py stage first."
            )
        frame = pl.read_parquet(path)
        spec = self._spec(dataset)
        if spec.event_date_col not in frame.columns:
            raise LookAheadError(
                f"dataset {dataset!r} has no event-date column "
                f"{spec.event_date_col!r}; found {frame.columns}"
            )
        frame = frame.with_columns(pl.col(spec.event_date_col).cast(pl.Date))
        self._frames[dataset] = frame
        return frame

    def invalidate(self, dataset: str | None = None) -> None:
        """Drop cached frames. Used by tests that mutate parquet on disk."""
        if dataset is None:
            self._frames.clear()
        else:
            self._frames.pop(dataset, None)

    def get(
        self,
        dataset: str,
        fields: Sequence[str] | None = None,
        *,
        as_of_date: _dt.date,
        lookback: int | None = None,
        symbols: Iterable[str] | None = None,
    ) -> pl.DataFrame:
        """Read ``dataset`` as it stood on ``as_of_date``.

        Parameters
        ----------
        dataset:
            Registered dataset name.
        fields:
            Columns to return, in addition to the event-date and entity columns
            (which are always included -- a frame without its keys is not usable).
        as_of_date:
            The query date. **No row that became public after this date is returned.**
        lookback:
            Keep only the most recent ``lookback`` *trading sessions* of event dates
            at or before the availability cutoff. Trimming here rather than in caller
            code means a rolling-window factor cannot accidentally widen its own
            window.
        symbols:
            Restrict to these entities, for datasets with an entity column.

        Returns
        -------
        A frame sorted by entity then event date (or just event date for aggregates).
        Empty is a legitimate result: early in a backtest nothing is available yet.
        """
        spec = self._spec(dataset)
        as_of_date = _as_date(as_of_date)
        frame = self._load(dataset)
        date_col = spec.event_date_col

        cutoff = self.max_event_date(dataset, as_of_date)
        if cutoff is None:
            return _empty_like(frame, self._projection(spec, fields, frame.columns))
        out = frame.filter(pl.col(date_col) <= cutoff)

        if lookback is not None:
            if lookback <= 0:
                raise ValueError("lookback must be a positive number of sessions")
            window = self.calendar.sessions_between(self.calendar.start, cutoff)[-lookback:]
            if window:
                out = out.filter(pl.col(date_col) >= window[0])

        if symbols is not None:
            if spec.entity_col:
                out = out.filter(pl.col(spec.entity_col).is_in(list(symbols)))
            else:
                # An aggregate series has no per-name dimension to filter on. Silently
                # ignoring the argument would let a caller believe it had restricted the
                # universe when it had not.
                raise LookAheadError(
                    f"dataset {dataset!r} is an aggregate series with no entity column, so "
                    "a `symbols` filter cannot be applied. Drop the argument rather than "
                    "relying on it being ignored."
                )

        cols = self._projection(spec, fields, frame.columns)
        out = out.select(cols)
        sort_keys = [spec.entity_col, date_col] if spec.entity_col else [date_col]
        return out.sort([k for k in sort_keys if k in out.columns])

    def _projection(
        self, spec: DatasetSpec, fields: Sequence[str] | None, available: Sequence[str]
    ) -> list[str]:
        if fields is None:
            return list(available)
        requested = list(fields)
        unknown = [f for f in requested if f not in available]
        if unknown:
            raise LookAheadError(
                f"dataset {spec.filename}: unknown field(s) {unknown}; available {list(available)}"
            )
        keys = [c for c in (spec.entity_col, spec.event_date_col) if c]
        return list(dict.fromkeys(keys + requested))

    # -- introspection ------------------------------------------------------
    def describe(self) -> list[str]:
        """ASCII lines describing every registered dataset and its enforced lag."""
        out = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            try:
                lag = self.availability_of(name).describe()
            except LookAheadError as exc:
                lag = f"UNREGISTERED ({exc.args[0].splitlines()[0]})"
            present = "present" if self.exists(name) else "absent"
            out.append(f"  {name:<20} lag={lag:<20} {present:<8} {spec.filename}")
            if spec.note:
                out.append(f"      {spec.note}")
        return out


def _as_date(value) -> _dt.date:
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str):
        return _dt.date.fromisoformat(value)
    raise TypeError(f"as_of_date must be a date, got {type(value).__name__}")


def _empty_like(frame: pl.DataFrame, cols: Sequence[str]) -> pl.DataFrame:
    """Empty frame carrying the *real* dtypes of ``frame``.

    Built from the source schema, never from empty Python lists: a ``Null``-typed
    date column raises ``SchemaError`` the moment it is joined against a real
    ``Date`` key, in a module far from where the mistake was made.
    """
    schema = {c: frame.schema[c] for c in cols if c in frame.schema}
    return pl.DataFrame(schema=schema)
