"""Forward returns and rank information coefficients.

Two things here are easy to get subtly wrong, so both are explicit.

**Forward returns are labelled by the date the position is taken.** ``fwd_5`` for date
``t`` is the return from ``t`` to ``t+5``. That value is *not* knowable at ``t`` -- it
is the label, not a feature. Nothing may feed it back into a factor, and the purged
cross-validation in :mod:`flowalpha.validation.purged_cv` exists precisely because
these labels overlap.

**IC t-statistics must be autocorrelation-adjusted.** Daily ICs from overlapping
h-day forward returns are strongly autocorrelated by construction: consecutive
5-day-forward ICs share four days of return. A naive ``mean / (sd / sqrt(n))``
t-statistic therefore overstates significance by roughly ``sqrt(h)``. Newey-West with
a Bartlett kernel is applied throughout, and the reported t-stats say so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl

FWD_PREFIX = "fwd_"


def forward_returns(
    prices: pl.DataFrame,
    horizons: Sequence[int],
    *,
    price_col: str = "adj_close",
) -> pl.DataFrame:
    """Forward returns over each horizon, labelled by the *entry* date.

    Returns ``(date, symbol, fwd_<h>, ...)``. Rows near the end of the sample have
    null labels: there is no future to measure yet, and inventing one by clipping to
    the last available price would fabricate a shorter-horizon return under a
    longer-horizon name.

    The shift is taken over a **dense session grid**, not over each symbol's own rows. A
    per-row shift crosses gaps: for a name that did not trade for two sessions, a
    ``shift(-1)`` spans three calendar sessions while still being labelled ``fwd_1``. That
    mislabels the horizon precisely for the names whose data is least trustworthy. On the
    dense grid a gap yields a NULL label instead, which is the honest answer.
    """
    schema = {"date": pl.Date, "symbol": pl.Utf8}
    schema.update({f"{FWD_PREFIX}{int(h)}": pl.Float64 for h in horizons})
    if prices.is_empty():
        return pl.DataFrame(schema=schema)
    if price_col not in prices.columns:
        raise ValueError(f"prices has no column {price_col!r}")

    for h in horizons:
        if int(h) <= 0:
            raise ValueError("forward-return horizons must be positive")

    observed = prices.select("date", "symbol", price_col)
    sessions = pl.DataFrame(
        {"date": sorted(set(observed["date"].to_list()))}, schema={"date": pl.Date}
    )
    names = pl.DataFrame(
        {"symbol": sorted(set(observed["symbol"].to_list()))}, schema={"symbol": pl.Utf8}
    )
    dense = (
        sessions.join(names, how="cross")
        .join(observed, on=["date", "symbol"], how="left")
        .sort(["symbol", "date"])
    )
    exprs = [
        (pl.col(price_col).shift(-int(h)).over("symbol") / pl.col(price_col) - 1.0)
        .alias(f"{FWD_PREFIX}{int(h)}")
        for h in horizons
    ]
    return (
        dense.with_columns(exprs)
        .filter(pl.col(price_col).is_not_null())   # a label needs a base price to start from
        .drop(price_col)
        .sort(["symbol", "date"])
    )


def rank_ic(
    panel: pl.DataFrame,
    fwd: pl.DataFrame,
    horizon: int,
    *,
    min_names: int = 10,
    value_col: str = "value",
) -> pl.DataFrame:
    """Per-date Spearman rank IC between factor value and forward return.

    Returns ``(date, ic, n)``. Dates with fewer than ``min_names`` joint observations
    are dropped: a rank correlation over six names is noise with a decimal point.

    Spearman is computed as the Pearson correlation of average ranks, which is the
    definition and handles ties correctly without a per-date Python loop.
    """
    col = f"{FWD_PREFIX}{int(horizon)}"
    if col not in fwd.columns:
        raise ValueError(f"forward-return frame has no {col!r}; got {fwd.columns}")
    schema = {"date": pl.Date, "ic": pl.Float64, "n": pl.UInt32}
    if panel.is_empty() or fwd.is_empty():
        return pl.DataFrame(schema=schema)

    joined = (
        panel.select("date", "symbol", value_col)
        .join(fwd.select("date", "symbol", col), on=["date", "symbol"], how="inner")
        .filter(pl.col(value_col).is_not_null() & pl.col(col).is_not_null())
    )
    if joined.is_empty():
        return pl.DataFrame(schema=schema)

    ranked = joined.with_columns(
        pl.col(value_col).rank(method="average").over("date").alias("_rx"),
        pl.col(col).rank(method="average").over("date").alias("_ry"),
    )
    out = (
        ranked.group_by("date")
        .agg(
            pl.corr("_rx", "_ry", method="pearson").alias("ic"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= min_names)
        .sort("date")
    )
    return out.select(pl.col("date"), pl.col("ic").cast(pl.Float64), pl.col("n").cast(pl.UInt32))


def newey_west_se(values: np.ndarray, lags: int | None = None) -> tuple[float, int]:
    """Newey-West standard error of the mean, Bartlett kernel.

    Returns ``(se, lags_used)``. When ``lags`` is None it uses the standard
    ``floor(4 * (n/100) ** (2/9))`` rule, which is the usual automatic choice and
    scales sensibly from a few hundred to a few thousand observations.

    A negative variance estimate (possible in small samples) falls back to the
    unadjusted standard error, which is reported rather than hidden.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 2:
        return float("nan"), 0
    if lags is None:
        lags = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(int(lags), n - 1))

    demeaned = arr - arr.mean()
    gamma0 = float(np.dot(demeaned, demeaned) / n)
    total = gamma0
    for k in range(1, lags + 1):
        gamma_k = float(np.dot(demeaned[k:], demeaned[:-k]) / n)
        weight = 1.0 - k / (lags + 1.0)
        total += 2.0 * weight * gamma_k
    if not (total > 0):
        return float(np.std(arr, ddof=1) / math.sqrt(n)), lags
    return float(math.sqrt(total / n)), lags


@dataclass(frozen=True)
class ICSummary:
    """Summary of an IC series.

    ``t_stat`` is Newey-West adjusted and **not** deflated for multiple testing. The
    trial-count deflation lives in :mod:`flowalpha.validation.deflated_sharpe` and
    applies to Sharpe ratios; an IC t-stat quoted from this object must be described
    as raw.
    """

    factor: str
    horizon: int
    n_dates: int
    mean: float
    std: float
    ic_ir: float
    t_stat: float
    nw_lags: int
    hit_rate: float
    mean_names: float

    def to_dict(self) -> dict:
        return {
            "factor": self.factor, "horizon": self.horizon, "n_dates": self.n_dates,
            "mean_ic": self.mean, "std_ic": self.std, "ic_ir": self.ic_ir,
            "t_stat_newey_west": self.t_stat, "nw_lags": self.nw_lags,
            "hit_rate": self.hit_rate, "mean_names": self.mean_names,
            "deflated": False,
        }


def summarize_ic(
    ic: pl.DataFrame,
    *,
    factor: str,
    horizon: int,
    nw_lags: int | None = None,
) -> ICSummary:
    """Aggregate an IC series into a reportable summary.

    ``nw_lags`` defaults to ``horizon - 1`` when the horizon exceeds one day, because
    that is exactly how many days of overlap consecutive labels share; beyond that the
    automatic rule takes over if it asks for more.
    """
    values = np.asarray(ic["ic"].to_list(), dtype=float) if not ic.is_empty() else np.array([])
    values = values[np.isfinite(values)]
    n = values.size
    if n == 0:
        return ICSummary(factor, int(horizon), 0, float("nan"), float("nan"),
                         float("nan"), float("nan"), 0, float("nan"), float("nan"))
    if nw_lags is None:
        auto = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        nw_lags = max(int(horizon) - 1, auto)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else float("nan")
    se, used = newey_west_se(values, nw_lags)
    counts = np.asarray(ic["n"].to_list(), dtype=float) if "n" in ic.columns else np.array([np.nan])
    return ICSummary(
        factor=factor,
        horizon=int(horizon),
        n_dates=n,
        mean=mean,
        std=std,
        ic_ir=(mean / std) if std and math.isfinite(std) and std > 0 else float("nan"),
        t_stat=(mean / se) if se and math.isfinite(se) and se > 0 else float("nan"),
        nw_lags=used,
        hit_rate=float((values > 0).mean()),
        mean_names=float(np.nanmean(counts)) if counts.size else float("nan"),
    )


def ic_table(
    panels: dict[str, pl.DataFrame],
    fwd: pl.DataFrame,
    horizons: Sequence[int],
    *,
    min_names: int = 10,
) -> pl.DataFrame:
    """IC summaries for every (factor, horizon) pair, as a tidy frame."""
    rows = []
    for name, panel in panels.items():
        for h in horizons:
            series = rank_ic(panel, fwd, int(h), min_names=min_names)
            rows.append(summarize_ic(series, factor=name, horizon=int(h)).to_dict())
    if not rows:
        return pl.DataFrame(
            schema={
                "factor": pl.Utf8, "horizon": pl.Int64, "n_dates": pl.Int64,
                "mean_ic": pl.Float64, "std_ic": pl.Float64, "ic_ir": pl.Float64,
                "t_stat_newey_west": pl.Float64, "nw_lags": pl.Int64,
                "hit_rate": pl.Float64, "mean_names": pl.Float64, "deflated": pl.Boolean,
            }
        )
    return pl.DataFrame(rows).sort(["factor", "horizon"])


def trailing_ic_series(
    panel: pl.DataFrame,
    fwd: pl.DataFrame,
    horizon: int,
    *,
    min_names: int = 10,
) -> pl.DataFrame:
    """IC series with the label-availability date attached.

    ``usable_from`` is the first date on which the IC for date ``d`` is knowable:
    ``d + horizon`` sessions, since the label needs that long to realise. Trailing-IC
    weighting schemes must filter on this column, not on ``date`` -- using ``date``
    would weight today's factors by an IC whose outcome has not happened yet, which is
    the most common way a "point-in-time" composite turns out not to be.
    """
    series = rank_ic(panel, fwd, horizon, min_names=min_names)
    if series.is_empty():
        return series.with_columns(pl.lit(None, pl.Date).alias("usable_from"))
    dates = sorted(set(fwd["date"].to_list()))
    index = {d: i for i, d in enumerate(dates)}
    h = int(horizon)

    def _usable(d):
        pos = index.get(d)
        if pos is None:
            return None
        target = pos + h
        # None, never a clamp to the last session. Clamping would mark an IC whose label
        # has NOT yet realised as usable on the final date, which is precisely the
        # look-ahead this column exists to prevent. Consumers treat None as "never usable".
        return dates[target] if target < len(dates) else None

    usable = [_usable(d) for d in series["date"].to_list()]
    return series.with_columns(pl.Series("usable_from", usable, dtype=pl.Date))
