"""Data quality checks.

Each check returns a :class:`CheckResult` with a PASS/WARN/FAIL verdict and enough
detail to act on. Thresholds come from ``config.yaml`` where the parameter belongs
there (notably the universe cardinality band); the rest are stated as module
constants with the reasoning attached, because a threshold with no rationale is a
number someone will later "fix".

The point of a FAIL is to stop the pipeline. A WARN is a fact the report must carry.
Nothing here silently repairs data: a check that fixes what it finds is a check whose
findings you never see.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import polars as pl

from ..config import Config
from .calendar import TradingCalendar

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

#: A calendar session with no price rows at all means the panel and the persisted
#: calendar disagree. A couple of such days is a tolerable artefact of a partial
#: re-download; more than this means one of the two is wrong.
MAX_MISSING_SESSION_FRACTION = 0.02

#: Per-name gap tolerance inside a name's own active span. Names with a worse record
#: than this are unusable for rolling windows.
MAX_PER_NAME_GAP_FRACTION = 0.05
#: How many names may exceed the above before the whole panel is unusable.
MAX_BAD_NAME_FRACTION = 0.10

#: A one-day move beyond this is almost always an unadjusted corporate action rather
#: than a real return. Indian single names do move 20-30% on news, so the bar is set
#: well above that.
RETURN_OUTLIER_THRESHOLD = 0.40
MAX_RETURN_OUTLIER_FRACTION = 0.01

#: Flow archives are occasionally unpublished. Beyond this share, the rolling flow
#: windows have too many holes to support regime labelling.
MAX_FLOW_GAP_FRACTION = 0.05


@dataclass
class CheckResult:
    """Outcome of one check."""

    name: str
    status: str
    message: str
    details: dict = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status == FAIL

    def to_dict(self) -> dict:
        return {
            "name": self.name, "status": self.status,
            "message": self.message, "details": self.details,
        }


def _verdict(value: float, limit: float) -> str:
    """FAIL past the limit, WARN at half of it, else PASS.

    The WARN band exists so that drift is visible before it becomes a stoppage.
    """
    if value > limit:
        return FAIL
    if value > limit / 2.0:
        return WARN
    return PASS


# ---------------------------------------------------------------------------
# 1. Missing trading days
# ---------------------------------------------------------------------------

def check_missing_trading_days(
    prices: pl.DataFrame,
    calendar: TradingCalendar,
    *,
    start: _dt.date,
    end: _dt.date,
) -> CheckResult:
    """Calendar sessions carrying no price rows at all.

    A consistency check between the persisted calendar and the current panel. It is
    trivially zero when the calendar was just derived from this panel, and becomes
    informative the moment the two are built at different times -- which is exactly
    when a stale calendar would start silently mis-dating every availability lag.
    """
    sessions = calendar.sessions_between(start, end)
    if not sessions:
        return CheckResult("missing_trading_days", FAIL,
                           "calendar has no sessions in the configured window")
    present = set(prices["date"].to_list()) if not prices.is_empty() else set()
    missing = [d for d in sessions if d not in present]
    frac = len(missing) / len(sessions)
    status = _verdict(frac, MAX_MISSING_SESSION_FRACTION)
    return CheckResult(
        "missing_trading_days", status,
        f"{len(missing)}/{len(sessions)} calendar sessions ({frac:.2%}) have no price "
        f"rows; limit {MAX_MISSING_SESSION_FRACTION:.0%}",
        {
            "n_sessions": len(sessions), "n_missing": len(missing),
            "fraction": frac, "limit": MAX_MISSING_SESSION_FRACTION,
            "first_missing": [d.isoformat() for d in missing[:10]],
        },
    )


# ---------------------------------------------------------------------------
# 2. Per-stock coverage
# ---------------------------------------------------------------------------

def check_per_stock_coverage(
    prices: pl.DataFrame,
    calendar: TradingCalendar,
) -> CheckResult:
    """Gaps inside each name's own active span.

    Measured over first-observed to last-observed date per symbol, **not** over the
    whole window. Judging a 2024 listing against a 2019 start would flag every recent
    IPO as 80% missing, which says nothing about data quality and would bury the
    names that really are broken.
    """
    if prices.is_empty():
        return CheckResult("per_stock_coverage", FAIL, "price panel is empty")

    spans = prices.group_by("symbol").agg(
        pl.col("date").min().alias("first"),
        pl.col("date").max().alias("last"),
        pl.len().alias("n_rows"),
    )
    rows = []
    for row in spans.iter_rows(named=True):
        expected = calendar.n_sessions_between(row["first"], row["last"])
        gap = max(0, expected - int(row["n_rows"]))
        rows.append(
            {
                "symbol": row["symbol"],
                "span_sessions": expected,
                "rows": int(row["n_rows"]),
                "gap_fraction": (gap / expected) if expected else 0.0,
            }
        )
    frame = pl.DataFrame(rows)
    bad = frame.filter(pl.col("gap_fraction") > MAX_PER_NAME_GAP_FRACTION)
    frac_bad = bad.height / frame.height
    status = _verdict(frac_bad, MAX_BAD_NAME_FRACTION)
    worst = bad.sort("gap_fraction", descending=True).head(10)
    return CheckResult(
        "per_stock_coverage", status,
        f"{bad.height}/{frame.height} names ({frac_bad:.2%}) have more than "
        f"{MAX_PER_NAME_GAP_FRACTION:.0%} missing sessions inside their active span; "
        f"limit {MAX_BAD_NAME_FRACTION:.0%} of names",
        {
            "n_names": frame.height, "n_bad": bad.height, "fraction_bad": frac_bad,
            "limit": MAX_BAD_NAME_FRACTION,
            "worst": worst.to_dicts(),
            "median_gap_fraction": float(frame["gap_fraction"].median() or 0.0),
        },
    )


# ---------------------------------------------------------------------------
# 3. Return outliers
# ---------------------------------------------------------------------------

def check_return_outliers(prices: pl.DataFrame) -> CheckResult:
    """One-day moves beyond the outlier threshold.

    A cluster of these means corporate actions are not being adjusted -- the adjusted
    close should absorb splits and bonuses, so a 500% single-day "return" is a split
    that the adjustment missed.
    """
    if prices.is_empty():
        return CheckResult("return_outliers", FAIL, "price panel is empty")
    rets = (
        prices.sort(["symbol", "date"])
        .with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1.0).alias("ret")
        )
        .filter(pl.col("ret").is_not_null())
    )
    if rets.is_empty():
        return CheckResult("return_outliers", FAIL, "no computable returns in the panel")
    outliers = rets.filter(pl.col("ret").abs() > RETURN_OUTLIER_THRESHOLD)
    frac = outliers.height / rets.height
    status = _verdict(frac, MAX_RETURN_OUTLIER_FRACTION)
    worst = (
        outliers.with_columns(pl.col("ret").abs().alias("abs_ret"))
        .sort("abs_ret", descending=True)
        .head(10)
        .select("date", "symbol", "ret")
    )
    return CheckResult(
        "return_outliers", status,
        f"{outliers.height}/{rets.height} daily returns ({frac:.3%}) exceed "
        f"{RETURN_OUTLIER_THRESHOLD:.0%}; limit {MAX_RETURN_OUTLIER_FRACTION:.0%}",
        {
            "n_returns": rets.height, "n_outliers": outliers.height, "fraction": frac,
            "threshold": RETURN_OUTLIER_THRESHOLD, "limit": MAX_RETURN_OUTLIER_FRACTION,
            "worst": [
                {"date": r["date"].isoformat(), "symbol": r["symbol"], "ret": r["ret"]}
                for r in worst.iter_rows(named=True)
            ],
        },
    )


# ---------------------------------------------------------------------------
# 4. Flow gaps
# ---------------------------------------------------------------------------

def check_flow_gaps(
    flows: pl.DataFrame,
    calendar: TradingCalendar,
    *,
    start: _dt.date,
    end: _dt.date,
    label: str = "flows_daily",
) -> CheckResult:
    """Sessions with no published flow archive."""
    sessions = calendar.sessions_between(start, end)
    if not sessions:
        return CheckResult(f"{label}_gaps", FAIL, "calendar has no sessions in the window")
    covered = set(flows["date"].to_list()) if not flows.is_empty() else set()
    missing = [d for d in sessions if d not in covered]
    frac = len(missing) / len(sessions)
    status = _verdict(frac, MAX_FLOW_GAP_FRACTION)
    return CheckResult(
        f"{label}_gaps", status,
        f"{len(missing)}/{len(sessions)} sessions ({frac:.2%}) have no {label} data; "
        f"limit {MAX_FLOW_GAP_FRACTION:.0%}",
        {
            "n_sessions": len(sessions), "n_missing": len(missing), "fraction": frac,
            "limit": MAX_FLOW_GAP_FRACTION,
            "missing": [d.isoformat() for d in missing[:20]],
        },
    )


# ---------------------------------------------------------------------------
# 5. Universe cardinality
# ---------------------------------------------------------------------------

def rebalance_dates(
    calendar: TradingCalendar,
    start: _dt.date,
    end: _dt.date,
    freq: str = "quarterly",
) -> list[_dt.date]:
    """Last session of each period in the window."""
    sessions = calendar.sessions_between(start, end)
    if not sessions:
        return []
    if freq == "quarterly":
        key = lambda d: (d.year, (d.month - 1) // 3)          # noqa: E731
    elif freq == "monthly":
        key = lambda d: (d.year, d.month)                      # noqa: E731
    elif freq == "annual":
        key = lambda d: (d.year,)                              # noqa: E731
    else:
        raise ValueError(f"unsupported rebalance_freq {freq!r}")
    last: dict[tuple, _dt.date] = {}
    for day in sessions:
        last[key(day)] = day
    return sorted(last.values())


def universe_cardinality(
    prices: pl.DataFrame,
    calendar: TradingCalendar,
    cfg: Config,
) -> pl.DataFrame:
    """Per-rebalance membership count under the configured reconstruction rule.

    A name is eligible at a rebalance date when it has at least
    ``universe.reconstruction.min_history_days`` sessions of price history as of that date.
    Eligibility is evaluated **as of** the rebalance date, so the count reflects what a
    portfolio could actually have held then, and the count is then capped at
    ``universe.size``.

    ``universe.reconstruction.method`` is deliberately NOT applied here, and the reported
    ``method`` field says so. This function produces a *count*, and a count is invariant to
    how the eligible names are ranked -- capping at ``size`` gives the same number whatever
    the ordering. Applying `mktcap_rank` would also be meaningless on this data, since
    share counts are unavailable and market cap collapses to price. Selecting *which* names
    to hold is portfolio construction's job, not this check's.
    """
    recon = cfg["universe"].get("reconstruction", {})
    min_history = int(recon.get("min_history_days", 60))
    freq = str(recon.get("rebalance_freq", "quarterly"))
    size = int(cfg["universe"]["size"])
    dates = rebalance_dates(calendar, cfg.start_date, cfg.end_date, freq)
    if prices.is_empty() or not dates:
        return pl.DataFrame(schema={"rebalance_date": pl.Date, "n_members": pl.UInt32})

    counts = (
        prices.select("date", "symbol", "close", "volume")
        .sort(["symbol", "date"])
    )
    rows = []
    for day in dates:
        upto = counts.filter(pl.col("date") <= day)
        if upto.is_empty():
            rows.append({"rebalance_date": day, "n_members": 0})
            continue
        eligible = (
            upto.group_by("symbol")
            .agg(pl.len().alias("history"), pl.col("close").last().alias("last_close"))
            .filter((pl.col("history") >= min_history) & pl.col("last_close").is_not_null())
        )
        rows.append({"rebalance_date": day, "n_members": min(eligible.height, size)})
    return pl.DataFrame(rows).with_columns(
        pl.col("rebalance_date").cast(pl.Date), pl.col("n_members").cast(pl.UInt32)
    )


def _reconstruction_note(cfg: Config) -> str:
    recon = cfg["universe"].get("reconstruction", {}) or {}
    return (
        f"method={recon.get('method')!r} is recorded but NOT applied: this check reports a "
        "COUNT, which is invariant to how eligible names are ranked."
    )


def check_universe_cardinality(
    prices: pl.DataFrame,
    calendar: TradingCalendar,
    cfg: Config,
) -> CheckResult:
    """Median per-rebalance cardinality against ``universe.cardinality_band``.

    The band is read from config, never hardcoded. It is deliberately not centred on
    ``universe.size``: with a current-constituent list applied historically, the count
    *grows* across the sample, so the first->last span is reported alongside the median
    to make that growth visible rather than averaged away.
    """
    band = cfg["universe"]["cardinality_band"]
    lo, hi = int(band[0]), int(band[1])
    frame = universe_cardinality(prices, calendar, cfg)
    if frame.is_empty():
        return CheckResult("universe_cardinality", FAIL,
                           "could not compute cardinality (empty panel or no rebalance dates)")
    counts = frame["n_members"].to_list()
    median = float(np.median(counts))
    first, last = int(counts[0]), int(counts[-1])
    status = PASS if lo <= median <= hi else FAIL
    message = (
        f"median per-rebalance cardinality {median:.0f} against configured band "
        f"[{lo}, {hi}]; span {first} -> {last} across "
        f"{frame.height} rebalances"
    )
    if status == PASS and (first < lo or last > hi):
        status = WARN
        message += " (endpoints outside the band -- survivorship-driven growth)"
    return CheckResult(
        "universe_cardinality", status, message,
        {
            "band": [lo, hi], "median": median, "first": first, "last": last,
            "n_rebalances": frame.height,
            "reconstruction": _reconstruction_note(cfg),
            "series": [
                {"date": r["rebalance_date"].isoformat(), "n": int(r["n_members"])}
                for r in frame.iter_rows(named=True)
            ],
            "note": (
                "Growth across the sample is expected: NSE publishes only the current "
                "constituent list, and post-2019 listings have no early history. This "
                "is survivorship bias, recorded as a provenance caveat."
            ),
        },
    )


# ---------------------------------------------------------------------------
# 6. Adjustment sanity
# ---------------------------------------------------------------------------

def check_adjustment_sanity(prices: pl.DataFrame) -> CheckResult:
    """No non-positive adjustment factors, no null adjusted closes.

    A zero or negative factor propagates through every adjusted price for that name
    and turns its whole return series into nonsense, so this is a FAIL on any single
    occurrence rather than a fraction.
    """
    if prices.is_empty():
        return CheckResult("adjustment_sanity", FAIL, "price panel is empty")
    bad_factor = prices.filter(
        pl.col("adj_factor").is_null() | ~pl.col("adj_factor").is_finite() | (pl.col("adj_factor") <= 0)
    )
    null_adj = prices.filter(pl.col("adj_close").is_null())
    status = FAIL if (bad_factor.height or null_adj.height) else PASS
    return CheckResult(
        "adjustment_sanity", status,
        f"{bad_factor.height} rows with non-positive/non-finite adj_factor, "
        f"{null_adj.height} rows with null adj_close",
        {
            "n_bad_factor": bad_factor.height,
            "n_null_adj_close": null_adj.height,
            "examples": [
                {"date": r["date"].isoformat(), "symbol": r["symbol"],
                 "adj_factor": r["adj_factor"]}
                for r in bad_factor.head(10).iter_rows(named=True)
            ],
        },
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_checks(
    cfg: Config,
    prices: pl.DataFrame,
    calendar: TradingCalendar,
    *,
    flows: pl.DataFrame | None = None,
    extra: Iterable[CheckResult] = (),
) -> list[CheckResult]:
    """Run every check and return the results in report order."""
    results = [
        check_missing_trading_days(prices, calendar, start=cfg.start_date, end=cfg.end_date),
        check_per_stock_coverage(prices, calendar),
        check_return_outliers(prices),
    ]
    if flows is not None:
        results.append(
            check_flow_gaps(flows, calendar, start=cfg.start_date, end=cfg.end_date)
        )
    results.append(check_universe_cardinality(prices, calendar, cfg))
    results.append(check_adjustment_sanity(prices))
    results.extend(extra)
    return results


def overall_status(results: Sequence[CheckResult]) -> str:
    if any(r.status == FAIL for r in results):
        return FAIL
    if any(r.status == WARN for r in results):
        return WARN
    return PASS
