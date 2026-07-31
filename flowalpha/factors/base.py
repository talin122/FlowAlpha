"""Factor abstraction and the shared computation context.

Design
------
Factors are computed from a :class:`FactorContext`, which loads price data **once**
through the point-in-time store and exposes it as date-by-symbol matrices. Two
reasons:

1. Every factor then reads the same as-of snapshot. A factor that reached for the
   parquet file directly would bypass lag enforcement, and the invariant would hold
   only by convention.
2. A rolling window over a dense matrix is straightforward to verify by hand, which
   is what the per-factor unit tests do.

A factor with no usable input returns an **empty panel with an explicit schema** --
never a frame built from empty Python lists, which would carry ``Null``-typed columns
and raise ``SchemaError`` when joined against a real ``Date`` key somewhere else
entirely.
"""

from __future__ import annotations

import datetime as _dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from ..config import Config
from ..data.calendar import TradingCalendar
from ..data.store import PointInTimeStore
from ..validation.neutralization import treat_cross_section

#: Canonical schema of a factor panel. Consumers may rely on it.
PANEL_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "symbol": pl.Utf8,
    "value": pl.Float64,
}


def empty_panel() -> pl.DataFrame:
    """An empty factor panel with real dtypes.

    Built from :data:`PANEL_SCHEMA`, so joining it against a real date key is safe.
    """
    return pl.DataFrame(schema=PANEL_SCHEMA)


class FactorError(Exception):
    """Raised when a factor is asked for something structurally impossible."""


@dataclass
class FactorContext:
    """Date-by-symbol matrices for one as-of snapshot.

    All matrices are ``(n_sessions, n_symbols)`` float arrays with ``NaN`` where the
    observation is absent. ``NaN`` rather than zero throughout: a name that had not
    listed yet has *unknown* volume, and zero is a specific wrong claim about it.
    """

    sessions: list[_dt.date]
    symbols: list[str]
    adj_close: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    turnover: np.ndarray
    delivery_pct: np.ndarray
    cfg: Config
    as_of_date: _dt.date
    sectors: dict[str, str] = field(default_factory=dict)
    shares: dict[str, float] = field(default_factory=dict)
    shares_available: bool = False
    calendar: TradingCalendar | None = None
    _returns: np.ndarray | None = field(default=None, repr=False)

    # -- derived ------------------------------------------------------------
    @property
    def n_sessions(self) -> int:
        return len(self.sessions)

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    @property
    def returns(self) -> np.ndarray:
        """Simple daily returns from the adjusted close, ``NaN`` on the first row."""
        if self._returns is None:
            prev = np.vstack([np.full((1, self.n_symbols), np.nan), self.adj_close[:-1]])
            with np.errstate(divide="ignore", invalid="ignore"):
                self._returns = self.adj_close / prev - 1.0
        return self._returns

    @property
    def shares_vector(self) -> np.ndarray:
        return np.array([float(self.shares.get(s, np.nan)) for s in self.symbols], dtype=float)

    def has_any(self, matrix: np.ndarray) -> bool:
        """True when a matrix carries at least one finite observation."""
        return bool(np.isfinite(matrix).any())

    def to_long(self, matrix: np.ndarray, *, drop_nulls: bool = True) -> pl.DataFrame:
        """Melt a matrix into a typed ``(date, symbol, value)`` panel."""
        if matrix.shape != (self.n_sessions, self.n_symbols):
            raise FactorError(
                f"matrix shape {matrix.shape} does not match context "
                f"({self.n_sessions}, {self.n_symbols})"
            )
        if not self.n_sessions or not self.n_symbols:
            return empty_panel()
        dates = np.repeat(np.array(self.sessions, dtype="object"), self.n_symbols)
        syms = np.tile(np.array(self.symbols, dtype="object"), self.n_sessions)
        out = pl.DataFrame(
            {"date": dates.tolist(), "symbol": syms.tolist(), "value": matrix.reshape(-1)},
            schema=PANEL_SCHEMA,
        )
        if drop_nulls:
            out = out.filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
        return out.sort(["symbol", "date"])

    # -- construction -------------------------------------------------------
    @classmethod
    def from_store(
        cls,
        store: PointInTimeStore,
        cfg: Config,
        *,
        as_of_date: _dt.date | None = None,
        symbols: Sequence[str] | None = None,
        start_date: _dt.date | None = None,
    ) -> "FactorContext":
        """Load one as-of snapshot through the store.

        Every read goes through ``store.get``, so factors inherit lag enforcement for
        free and cannot opt out of it.
        """
        as_of = as_of_date or cfg.end_date
        prices = store.get("prices", as_of_date=as_of, symbols=symbols)
        if prices.is_empty():
            return cls(
                sessions=[], symbols=[],
                adj_close=np.empty((0, 0)), close=np.empty((0, 0)),
                volume=np.empty((0, 0)), turnover=np.empty((0, 0)),
                delivery_pct=np.empty((0, 0)),
                cfg=cfg, as_of_date=as_of, calendar=store.calendar,
            )
        start = start_date or cfg.start_date
        prices = prices.filter(pl.col("date") >= start)

        sessions = sorted(set(prices["date"].to_list()))
        syms = sorted(set(prices["symbol"].to_list()))
        matrices = {
            name: _pivot(prices, name, sessions, syms)
            for name in ("adj_close", "close", "volume", "turnover", "delivery_pct")
        }
        sectors = _load_sectors(cfg.path("reference"))
        shares, shares_available = _load_shares(cfg.path("reference"))
        return cls(
            sessions=sessions, symbols=syms,
            cfg=cfg, as_of_date=as_of, calendar=store.calendar,
            sectors=sectors, shares=shares, shares_available=shares_available,
            **matrices,
        )


def _pivot(
    prices: pl.DataFrame, column: str, sessions: list[_dt.date], symbols: list[str]
) -> np.ndarray:
    """Pivot one price column into a dense (dates x symbols) float matrix."""
    if column not in prices.columns:
        return np.full((len(sessions), len(symbols)), np.nan)
    wide = (
        prices.select("date", "symbol", column)
        .pivot(on="symbol", index="date", values=column, aggregate_function="first")
        .sort("date")
    )
    out = np.full((len(sessions), len(symbols)), np.nan)
    date_index = {d: i for i, d in enumerate(sessions)}
    rows = [date_index[d] for d in wide["date"].to_list()]
    for j, sym in enumerate(symbols):
        if sym in wide.columns:
            col = wide[sym].cast(pl.Float64).to_numpy()
            out[rows, j] = col
    return out


def _load_sectors(reference_dir: Path) -> dict[str, str]:
    """Read ``sectors.csv`` if present. Absent means every name is ``NA``."""
    path = Path(reference_dir) / "sectors.csv"
    if not path.exists():
        return {}
    frame = pl.read_csv(path)
    cols = {c.lower(): c for c in frame.columns}
    sym_col = cols.get("symbol")
    sec_col = cols.get("sector") or cols.get("industry")
    if not sym_col or not sec_col:
        return {}
    return dict(zip(frame[sym_col].to_list(), frame[sec_col].to_list()))


def _load_shares(reference_dir: Path) -> tuple[dict[str, float], bool]:
    """Read ``shares_outstanding.csv``.

    Returns ``(mapping, available)``. ``available`` is False when the file is missing
    **or** when every share count is identical, because a constant share count carries
    no cross-sectional information and turns market cap into price. The size factor
    renames itself on the strength of this flag so the degradation is visible.
    """
    path = Path(reference_dir) / "shares_outstanding.csv"
    if not path.exists():
        return {}, False
    frame = pl.read_csv(path)
    cols = {c.lower(): c for c in frame.columns}
    sym_col, share_col = cols.get("symbol"), cols.get("shares")
    if not sym_col or not share_col:
        return {}, False
    mapping = {
        s: float(v) for s, v in zip(frame[sym_col].to_list(), frame[share_col].to_list())
        if v is not None
    }
    distinct = len(set(mapping.values()))
    return mapping, distinct > 1


class Factor(ABC):
    """Base class for a cross-sectional factor.

    Subclasses implement :meth:`compute`, returning a *raw* ``(date, symbol, value)``
    panel. :meth:`panel` then applies the configured cross-sectional treatment. The
    split matters: unit tests assert exact raw values, while everything downstream
    consumes the standardised panel.
    """

    #: Stable identifier. Used as the trial-registry key, so renaming a factor
    #: resets its multiple-testing history -- do it deliberately.
    name: str = "factor"

    #: Short statement of what the factor is betting on, surfaced in reports.
    hypothesis: str = ""

    #: Datasets or context matrices required. Reported when a factor is inert.
    requires: tuple[str, ...] = ("adj_close",)

    @abstractmethod
    def compute(self, ctx: FactorContext) -> pl.DataFrame:
        """Raw factor values as ``(date, symbol, value)``. May be empty."""

    def is_available(self, ctx: FactorContext) -> bool:
        """Whether the context carries the inputs this factor needs."""
        for req in self.requires:
            matrix = getattr(ctx, req, None)
            if matrix is None or not isinstance(matrix, np.ndarray) or not ctx.has_any(matrix):
                return False
        return True

    def unavailable_reason(self, ctx: FactorContext) -> str:
        missing = [
            r for r in self.requires
            if not isinstance(getattr(ctx, r, None), np.ndarray)
            or not ctx.has_any(getattr(ctx, r))
        ]
        return f"{self.name}: no usable data for {missing}"

    def panel(self, ctx: FactorContext) -> pl.DataFrame:
        """Standardised panel, per ``factors.winsorize`` / ``factors.sector_neutralize``."""
        raw = self.compute(ctx)
        if raw.is_empty():
            return empty_panel()
        fcfg = ctx.cfg["factors"]
        lower, upper = fcfg.get("winsorize", [0.01, 0.99])
        return treat_cross_section(
            raw,
            winsor=(float(lower), float(upper)),
            sector_neutral=bool(fcfg.get("sector_neutralize", False)),
            sectors=ctx.sectors or None,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"


# ---------------------------------------------------------------------------
# Rolling-window primitives, shared by the library
# ---------------------------------------------------------------------------

def rolling_mean(matrix: np.ndarray, window: int, *, min_valid: int | None = None) -> np.ndarray:
    """Trailing mean over ``window`` rows, per column.

    ``min_valid`` observations must be present in the window, else ``NaN``. Defaults
    to the full window: a "21-day average" computed from four observations is not the
    quantity the factor definition claims.

    Implemented with an explicit sliding window rather than a cumulative sum, for the
    same reason as :func:`flowalpha.factors.flow_features.trailing_sum` -- a
    cumulative sum propagates a single ``NaN`` to every later row.
    """
    return _rolling_reduce(matrix, window, "mean", min_valid)


def rolling_sum(matrix: np.ndarray, window: int, *, min_valid: int | None = None) -> np.ndarray:
    return _rolling_reduce(matrix, window, "sum", min_valid)


def rolling_std(matrix: np.ndarray, window: int, *, min_valid: int | None = None) -> np.ndarray:
    """Trailing sample standard deviation (ddof=1)."""
    return _rolling_reduce(matrix, window, "std", min_valid)


def _rolling_reduce(
    matrix: np.ndarray, window: int, how: str, min_valid: int | None
) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    arr = np.asarray(matrix, dtype=float)
    n, m = arr.shape
    out = np.full((n, m), np.nan)
    if n < window:
        return out
    need = window if min_valid is None else int(min_valid)
    view = np.lib.stride_tricks.sliding_window_view(arr, window, axis=0)  # (n-w+1, m, w)
    valid = np.isfinite(view)
    counts = valid.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        if how == "sum":
            agg = np.nansum(np.where(valid, view, np.nan), axis=-1)
        elif how == "mean":
            agg = np.nansum(np.where(valid, view, np.nan), axis=-1) / np.where(counts > 0, counts, np.nan)
        elif how == "std":
            filled = np.where(valid, view, np.nan)
            mean = np.nansum(filled, axis=-1) / np.where(counts > 0, counts, np.nan)
            dev = filled - mean[..., None]
            ss = np.nansum(dev * dev, axis=-1)
            agg = np.sqrt(ss / np.where(counts > 1, counts - 1, np.nan))
        else:  # pragma: no cover - guarded by callers
            raise ValueError(f"unknown reduction {how!r}")
    agg = np.where(counts >= need, agg, np.nan)
    out[window - 1 :] = agg
    return out


def shift_rows(matrix: np.ndarray, n: int) -> np.ndarray:
    """Shift a matrix down by ``n`` rows, padding the top with ``NaN``.

    ``shift_rows(x, 1)[t] == x[t-1]``, i.e. positive ``n`` moves information *later*,
    which is the only direction a point-in-time factor may use.
    """
    arr = np.asarray(matrix, dtype=float)
    if n == 0:
        return arr.copy()
    if n < 0:
        raise ValueError(
            "shift_rows only shifts information forward in time; a negative shift "
            "would pull future values into the present"
        )
    out = np.full_like(arr, np.nan)
    if n < arr.shape[0]:
        out[n:] = arr[:-n]
    return out
