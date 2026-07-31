"""Flow features and regime labels.

These are the conditioning variables the whole study turns on. Three properties are
load-bearing:

1. **One-session shift.** The row stamped ``d`` uses flow only through ``d-1``. NSE
   publishes the participant archive after the close, so ``d``'s own flow is not
   knowable on ``d``. Because the shift is baked in here, ``flow_features`` is
   registered in the store at **lag 0** -- see :mod:`flowalpha.data.store`. Adding a
   further lag would double-lag it.

2. **Gap-tolerant trailing sums.** A missing session nulls only the windows that
   actually span it. The obvious ``np.cumsum`` implementation propagates a single NaN
   to *every later value*: real NSE data has a handful of unpublished sessions, so
   with cumsum the first one silently kills every regime label from that date to the
   end of the sample. The symptom is regimes that stop dead years before the data
   does, which reads as a modelling problem rather than an arithmetic one. See
   :func:`trailing_sum` and its regression test.

3. **Expanding windows, never full-sample.** Every threshold -- terciles, the shock
   percentile -- is computed from history up to and including ``t``. A full-sample
   quantile is look-ahead: it tells you today which bucket today's value lands in
   relative to values you have not observed yet.
"""

from __future__ import annotations

import bisect
import datetime as _dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from ..config import Config
from ..data.calendar import TradingCalendar
from ..data.store import PointInTimeStore

FLOW_FEATURES_FILENAME = "flow_features.parquet"

#: The retail proxy. `Client` in NSE's taxonomy: includes HNIs and other
#: non-institutional accounts, so it proxies retail rather than measuring it.
RETAIL_CATEGORY = "Client"

TERCILE_LABELS_3 = ("low", "mid", "high")

FLOW_FEATURES_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "fii_flow_21d": pl.Float64,
    "dii_flow_21d": pl.Float64,
    "retail_flow_21d": pl.Float64,
    "fii_daily": pl.Float64,
    "retail_daily": pl.Float64,
    "fii_regime": pl.Utf8,
    "retail_regime": pl.Utf8,
    "retail_extreme": pl.Utf8,
    "flow_divergence": pl.Utf8,
    "flow_shock": pl.Boolean,
}

#: Regime columns, in the order reports display them.
REGIME_COLUMNS: tuple[str, ...] = (
    "fii_regime", "retail_regime", "retail_extreme", "flow_divergence", "flow_shock",
)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def trailing_sum(values: Sequence[float] | np.ndarray, window: int) -> np.ndarray:
    """Rolling sum over ``window`` observations, requiring the window to be complete.

    A window containing any NaN yields NaN; a window that does not is summed exactly.
    Crucially, the NaN does **not** propagate beyond the windows that contain it: the
    value at index ``i`` depends only on ``values[i-window+1 : i+1]``.

    This is written with an explicit sliding window rather than a cumulative sum
    because ``np.cumsum`` over a series containing NaN returns NaN for every
    subsequent element. With ~1900 sessions and a handful of unpublished ones, that
    difference is the difference between a study and a nullity.

    Examples
    --------
    >>> trailing_sum([1.0, 2.0, 3.0, 4.0], 2)
    array([nan,  3.,  5.,  7.])
    >>> trailing_sum([1.0, float("nan"), 3.0, 4.0], 2)
    array([nan, nan, nan,  7.])
    """
    arr = np.asarray(values, dtype=float)
    if window <= 0:
        raise ValueError("window must be positive")
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=float)
    if n < window:
        return out
    view = np.lib.stride_tricks.sliding_window_view(arr, window)
    complete = ~np.isnan(view).any(axis=1)
    sums = np.where(complete, np.nansum(view, axis=1), np.nan)
    out[window - 1 :] = sums
    return out


def shift_one(values: np.ndarray) -> np.ndarray:
    """Shift a session-indexed series forward by one session.

    ``out[i] = values[i-1]``, so the row for session ``d`` carries information as of
    the close of ``d-1`` -- which is when the underlying archive was published.
    """
    arr = np.asarray(values, dtype=float)
    out = np.full_like(arr, np.nan)
    if arr.shape[0] > 1:
        out[1:] = arr[:-1]
    return out


def expanding_bucket_labels(
    values: np.ndarray,
    *,
    n_buckets: int = 3,
    min_obs: int = 252,
    labels: Sequence[str] | None = None,
) -> list[str | None]:
    """Assign each observation to an expanding-window quantile bucket.

    The bucket for index ``i`` is decided from the empirical distribution of all
    *defined* values at indices ``0..i`` inclusive -- history only. A label is emitted
    only once at least ``min_obs`` defined values have accumulated, because a tercile
    computed from 30 observations is noise dressed as a regime.

    Ties are handled by mid-rank, so a series with many repeated values does not pile
    into one bucket by accident of comparison direction.

    Notes
    -----
    Deliberately not vectorised over the full sample: any implementation that sorts
    the whole series once and slices it is computing a full-sample quantile, which is
    the exact look-ahead this function exists to avoid.
    """
    if n_buckets < 2:
        raise ValueError("n_buckets must be at least 2")
    if labels is None:
        labels = TERCILE_LABELS_3 if n_buckets == 3 else tuple(f"q{i + 1}" for i in range(n_buckets))
    if len(labels) != n_buckets:
        raise ValueError(f"got {len(labels)} labels for {n_buckets} buckets")

    seen: list[float] = []
    out: list[str | None] = []
    for value in np.asarray(values, dtype=float):
        if math.isnan(value):
            out.append(None)
            continue
        bisect.insort(seen, float(value))
        count = len(seen)
        if count < min_obs:
            out.append(None)
            continue
        lo = bisect.bisect_left(seen, value)
        hi = bisect.bisect_right(seen, value)
        midrank = (lo + hi - 1) / 2.0
        frac = (midrank + 0.5) / count
        idx = min(int(frac * n_buckets), n_buckets - 1)
        out.append(labels[idx])
    return out


def expanding_percentile_flag(
    values: np.ndarray,
    *,
    percentile: float = 0.95,
    min_obs: int = 252,
) -> list[bool | None]:
    """Flag observations at or above their own expanding-window percentile.

    Same discipline as :func:`expanding_bucket_labels`: the threshold at ``i`` comes
    from values ``0..i`` only.
    """
    if not 0.0 < percentile < 1.0:
        raise ValueError("percentile must be in (0, 1)")
    seen: list[float] = []
    out: list[bool | None] = []
    for value in np.asarray(values, dtype=float):
        if math.isnan(value):
            out.append(None)
            continue
        bisect.insort(seen, float(value))
        count = len(seen)
        if count < min_obs:
            out.append(None)
            continue
        # Nearest-rank threshold on the observed sample; no interpolation, so the
        # flag is a statement about observed history rather than a fitted quantile.
        k = max(0, min(count - 1, int(math.ceil(percentile * count)) - 1))
        out.append(bool(value >= seen[k]))
    return out


def divergence_label(fii: float, dii: float) -> str | None:
    """Classify FII/DII directional agreement.

    ``aligned`` when both are pushing the same way, ``opposed`` when they disagree,
    ``neutral`` when either is flat. ``None`` when either is undefined -- a regime
    invented from a missing input is worse than an absent one.
    """
    if fii is None or dii is None or math.isnan(fii) or math.isnan(dii):
        return None
    sf, sd = np.sign(fii), np.sign(dii)
    if sf == 0 or sd == 0:
        return "neutral"
    return "aligned" if sf == sd else "opposed"


def collapse_extreme(label: str | None) -> str | None:
    """Derive ``retail_extreme`` from a tercile label: mid stays mid, tails collapse.

    The hypothesis is about *magnitude* of retail participation, not direction, so
    heavy buying and heavy selling belong in the same bucket.
    """
    if label is None:
        return None
    return "mid" if label == "mid" else "extreme"


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FlowFeatureInputs:
    """Session-indexed raw inputs, aligned to the calendar with NaN for gaps."""

    sessions: list[_dt.date]
    fii_daily: np.ndarray
    dii_daily: np.ndarray
    retail_daily: np.ndarray

    @property
    def n_missing(self) -> int:
        return int(np.isnan(self.fii_daily).sum())


def align_to_sessions(
    frame: pl.DataFrame,
    sessions: Sequence[_dt.date],
    *,
    participant: str,
    value_col: str = "net_futures",
) -> np.ndarray:
    """Project one participant's series onto the session grid, NaN where absent.

    Explicit NaN rather than zero: a session with no published archive is *unknown*
    flow, and zero is a specific, wrong claim about it.
    """
    index = {d: i for i, d in enumerate(sessions)}
    out = np.full(len(sessions), np.nan, dtype=float)
    if frame.is_empty():
        return out
    sub = frame.filter(pl.col("participant") == participant)
    for day, value in zip(sub["date"].to_list(), sub[value_col].to_list()):
        pos = index.get(day)
        if pos is not None and value is not None:
            out[pos] = float(value)
    return out


def load_flow_inputs(
    store: PointInTimeStore,
    sessions: Sequence[_dt.date],
    *,
    as_of_date: _dt.date,
) -> FlowFeatureInputs:
    """Read FII/DII and retail series through the store, aligned to ``sessions``."""
    daily = store.get("flows_daily", as_of_date=as_of_date)
    try:
        participants = store.get("participant_flows", as_of_date=as_of_date)
    except FileNotFoundError:
        participants = pl.DataFrame(schema={"date": pl.Date, "participant": pl.Utf8, "net_futures": pl.Float64})
    return FlowFeatureInputs(
        sessions=list(sessions),
        fii_daily=align_to_sessions(daily, sessions, participant="FII"),
        dii_daily=align_to_sessions(daily, sessions, participant="DII"),
        retail_daily=align_to_sessions(participants, sessions, participant=RETAIL_CATEGORY),
    )


def build_flow_features(
    store: PointInTimeStore,
    cfg: Config,
    *,
    as_of_date: _dt.date | None = None,
    calendar: TradingCalendar | None = None,
) -> pl.DataFrame:
    """Build the point-in-time flow-feature panel.

    Each row's features are functions of flow data strictly before that row's date,
    and of expanding statistics over that row's own history. The panel is therefore
    internally point-in-time: truncating it at any date gives exactly what would have
    been computed on that date, which is what the look-ahead injection test asserts.

    Parameters
    ----------
    as_of_date:
        Query date for reading the underlying flow datasets and the last session
        included. Defaults to the config end date.
    """
    calendar = calendar or store.calendar
    end = as_of_date or cfg.end_date
    sessions = calendar.sessions_between(cfg.start_date, end)
    if not sessions:
        return pl.DataFrame(schema=FLOW_FEATURES_SCHEMA)

    cond = cfg["conditioning"]
    window = int(cond["regime_window"])
    n_buckets = int(cond["n_buckets"])
    min_obs = int(cond["expanding_min_obs"])
    shock_pct = float(cond["shock_percentile"])

    raw = load_flow_inputs(store, sessions, as_of_date=end)

    # Shift first, then accumulate: the row for d must see flow only through d-1.
    fii_usable = shift_one(raw.fii_daily)
    dii_usable = shift_one(raw.dii_daily)
    retail_usable = shift_one(raw.retail_daily)

    fii_21 = trailing_sum(fii_usable, window)
    dii_21 = trailing_sum(dii_usable, window)
    retail_21 = trailing_sum(retail_usable, window)

    fii_regime = expanding_bucket_labels(fii_21, n_buckets=n_buckets, min_obs=min_obs)
    retail_regime = expanding_bucket_labels(retail_21, n_buckets=n_buckets, min_obs=min_obs)
    retail_extreme = [collapse_extreme(x) for x in retail_regime]
    divergence = [divergence_label(f, d) for f, d in zip(fii_21, dii_21)]
    shock = expanding_percentile_flag(np.abs(fii_usable), percentile=shock_pct, min_obs=min_obs)

    frame = pl.DataFrame(
        {
            "date": sessions,
            "fii_flow_21d": fii_21,
            "dii_flow_21d": dii_21,
            "retail_flow_21d": retail_21,
            "fii_daily": fii_usable,
            "retail_daily": retail_usable,
            "fii_regime": fii_regime,
            "retail_regime": retail_regime,
            "retail_extreme": retail_extreme,
            "flow_divergence": divergence,
            "flow_shock": shock,
        },
        schema=FLOW_FEATURES_SCHEMA,
    )
    # NaN is not null in polars. The numpy pipeline above produces NaN for "not
    # computable", but every consumer filters with is_null()/is_not_null(), and a
    # float NaN sails straight through such a filter and into an average. Normalise
    # once, here, at the boundary between the numpy and polars halves of the code.
    return normalize_nan(frame)


def normalize_nan(frame: pl.DataFrame) -> pl.DataFrame:
    """Convert float NaN to null in every Float64 column.

    Keeps the "missing" concept single-valued. Without this, ``is_not_null()`` filters
    silently admit NaN and downstream means become NaN for the whole series.
    """
    float_cols = [c for c, dt in frame.schema.items() if dt == pl.Float64]
    if not float_cols:
        return frame
    return frame.with_columns([pl.col(c).fill_nan(None) for c in float_cols])


def write_flow_features(features: pl.DataFrame, processed_dir: str | Path) -> Path:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / FLOW_FEATURES_FILENAME
    features.write_parquet(path)
    return path


def regime_coverage(features: pl.DataFrame) -> dict:
    """Summarise how much of the sample carries a usable regime label.

    Reports the **last** defined date per regime column. That number is the tell for
    the cumsum bug described in the module docstring: if regimes stop years before
    the price history ends, the trailing sum is propagating a gap.
    """
    if features.is_empty():
        return {"n_sessions": 0, "regimes": {}}
    out: dict = {
        "n_sessions": features.height,
        "first_session": min(features["date"].to_list()).isoformat(),
        "last_session": max(features["date"].to_list()).isoformat(),
        "regimes": {},
    }
    for col in REGIME_COLUMNS:
        defined = features.filter(pl.col(col).is_not_null())
        entry = {"n_defined": defined.height}
        if defined.height:
            dates = defined["date"].to_list()
            entry["first_defined"] = min(dates).isoformat()
            entry["last_defined"] = max(dates).isoformat()
            if col != "flow_shock":
                counts = (
                    defined.group_by(col).len().sort(col).rows()
                )
                entry["buckets"] = {str(k): int(v) for k, v in counts}
            else:
                entry["n_true"] = int(defined.filter(pl.col(col)).height)
        out["regimes"][col] = entry
    return out
