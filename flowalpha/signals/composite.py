"""IC-weighted factor composite.

Weights come from each factor's **trailing** information coefficient over
``signals.composite.ic_window`` sessions. The point-in-time subtlety that decides
whether this is research or self-deception:

    An IC observation dated ``d`` is not knowable on ``d``. Its label is a
    ``ic_horizon``-day forward return, so the observation only becomes usable
    ``ic_horizon`` sessions later.

So the weight applied on session ``t`` is built from IC observations whose *label has
already realised* by ``t`` -- that is, ``usable_from <= t`` -- and which fall inside the
trailing window. Filtering on the IC's own ``date`` instead would weight today's factors
by an outcome that has not happened yet. That is the most common way a composite
described as point-in-time turns out not to be, and it is why
:func:`flowalpha.validation.ic.trailing_ic_series` carries a ``usable_from`` column at
all.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import polars as pl

from ..config import Config
from ..validation.ic import trailing_ic_series

SCHEMES = ("equal", "ic_mean", "ic_ir")


class CompositeError(Exception):
    """Raised for an unknown weighting scheme or an unusable factor set."""


@dataclass
class CompositeWeights:
    """Per-date factor weights, plus the diagnostics behind them."""

    #: (date, factor, weight)
    weights: pl.DataFrame
    scheme: str
    ic_window: int
    ic_horizon: int
    #: Factors dropped for non-positive trailing IC, per date count.
    n_dropped_negative: int
    factors: tuple[str, ...]

    def on(self, day: _dt.date) -> dict[str, float]:
        row = self.weights.filter(pl.col("date") == day)
        return dict(zip(row["factor"].to_list(), row["weight"].to_list()))

    def to_dict(self) -> dict:
        return {
            "scheme": self.scheme, "ic_window": self.ic_window,
            "ic_horizon": self.ic_horizon, "factors": list(self.factors),
            "n_dropped_negative_ic": self.n_dropped_negative,
            "n_dates": self.weights["date"].n_unique() if not self.weights.is_empty() else 0,
        }


def compute_weights(
    panels: Mapping[str, pl.DataFrame],
    fwd: pl.DataFrame,
    cfg: Config,
    *,
    sessions: Sequence[_dt.date] | None = None,
) -> CompositeWeights:
    """Trailing-IC weights per session, using only realised IC observations.

    Schemes:

    ``equal``
        Every available factor gets the same weight. The honest baseline: it cannot be
        overfit, so any IC-weighted scheme has to beat it to justify its extra freedom.
    ``ic_mean``
        Weight proportional to trailing mean IC.
    ``ic_ir``
        Weight proportional to trailing mean IC divided by its standard deviation --
        rewards consistency, not just magnitude.
    """
    ccfg = cfg["signals"]["composite"]
    scheme = str(ccfg["scheme"])
    if scheme not in SCHEMES:
        raise CompositeError(f"unknown composite scheme {scheme!r}; expected one of {SCHEMES}")
    window = int(ccfg["ic_window"])
    horizon = int(ccfg["ic_horizon"])
    drop_negative = bool(ccfg.get("drop_negative_ic", True))

    factors = tuple(sorted(panels))
    if not factors:
        return CompositeWeights(
            weights=pl.DataFrame(schema={"date": pl.Date, "factor": pl.Utf8, "weight": pl.Float64}),
            scheme=scheme, ic_window=window, ic_horizon=horizon,
            n_dropped_negative=0, factors=(),
        )

    if sessions is None:
        sessions = sorted(set(fwd["date"].to_list()))
    sessions = list(sessions)
    session_index = {d: i for i, d in enumerate(sessions)}

    # Precompute each factor's IC series with its label-availability date attached.
    ic_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, panel in panels.items():
        series = trailing_ic_series(panel, fwd, horizon)
        if series.is_empty():
            continue
        dates = np.array([session_index.get(d, -1) for d in series["date"].to_list()])
        usable = np.array([session_index.get(d, len(sessions)) for d in series["usable_from"].to_list()])
        values = np.asarray(series["ic"].to_list(), dtype=float)
        keep = dates >= 0
        ic_data[name] = (dates[keep], usable[keep], values[keep])

    rows: list[dict] = []
    n_dropped = 0
    for t, day in enumerate(sessions):
        raw: dict[str, float] = {}
        for name in factors:
            entry = ic_data.get(name)
            if entry is None:
                continue
            dates, usable, values = entry
            # Realised by t, and inside the trailing window measured on the IC's own date.
            mask = (usable <= t) & (dates > t - window) & (dates <= t)
            sample = values[mask]
            # Drop non-finite ICs. A gated factor is deliberately flat on its OFF dates,
            # so its cross-section has zero dispersion and its IC for those dates is
            # undefined. Letting one NaN through makes the weight NaN, which makes the
            # weighted sum NaN for EVERY stock on that date, which empties the whole
            # composite -- a failure that looks like "no signal" rather than "bad guard".
            sample = sample[np.isfinite(sample)]
            if sample.size < 2:
                continue
            mean = float(sample.mean())
            if scheme == "equal":
                score = 1.0
            elif scheme == "ic_mean":
                score = mean
            else:
                sd = float(sample.std(ddof=1))
                score = mean / sd if sd > 0 else 0.0
            if not np.isfinite(score):
                continue
            if drop_negative and scheme != "equal" and score <= 0:
                n_dropped += 1
                continue
            raw[name] = score

        if not raw:
            continue
        total = sum(abs(v) for v in raw.values())
        if not np.isfinite(total) or total <= 0:
            continue
        for name, score in raw.items():
            rows.append({"date": day, "factor": name, "weight": score / total})

    weights = (
        pl.DataFrame(rows, schema={"date": pl.Date, "factor": pl.Utf8, "weight": pl.Float64})
        if rows
        else pl.DataFrame(schema={"date": pl.Date, "factor": pl.Utf8, "weight": pl.Float64})
    )
    return CompositeWeights(
        weights=weights, scheme=scheme, ic_window=window, ic_horizon=horizon,
        n_dropped_negative=n_dropped, factors=factors,
    )


def build_composite(
    panels: Mapping[str, pl.DataFrame],
    weights: CompositeWeights,
    cfg: Config,
) -> pl.DataFrame:
    """Combine standardised factor panels into one signal.

    Requires at least ``signals.composite.min_coverage`` non-null factor values for a
    stock on a date. A stock scored from one factor is not comparable with one scored
    from seven, and letting both into the same cross-section makes the thin-coverage
    names systematically more extreme.

    Returns ``(date, symbol, value)`` where value is the weighted average of the
    available factors, re-standardised cross-sectionally.
    """
    from ..validation.neutralization import zscore

    min_coverage = int(cfg.get("signals.composite.min_coverage", 1))
    if weights.weights.is_empty() or not panels:
        return pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64})

    stacked = pl.concat(
        [
            panel.select("date", "symbol", "value").with_columns(pl.lit(name).alias("factor"))
            for name, panel in panels.items()
            if not panel.is_empty()
        ],
        how="vertical",
    )
    joined = (
        stacked.join(weights.weights, on=["date", "factor"], how="inner")
        .filter(
            pl.col("value").is_not_null()
            & pl.col("value").is_finite()
            & pl.col("weight").is_not_null()
            & pl.col("weight").is_finite()
        )
    )
    if joined.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64})

    agg = (
        joined.group_by(["date", "symbol"])
        .agg(
            (pl.col("value") * pl.col("weight")).sum().alias("_num"),
            pl.col("weight").abs().sum().alias("_den"),
            pl.len().alias("coverage"),
        )
        .filter((pl.col("coverage") >= min_coverage) & (pl.col("_den") > 0))
        .with_columns((pl.col("_num") / pl.col("_den")).alias("value"))
        .select("date", "symbol", "value")
    )
    return zscore(agg).sort(["symbol", "date"])


def weight_history(weights: CompositeWeights) -> pl.DataFrame:
    """Wide (date x factor) weight history, for plotting and audit."""
    if weights.weights.is_empty():
        return pl.DataFrame(schema={"date": pl.Date})
    return (
        weights.weights.pivot(on="factor", index="date", values="weight", aggregate_function="first")
        .sort("date")
        .fill_null(0.0)
    )
