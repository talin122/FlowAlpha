#!/usr/bin/env python
"""Download the price panel from Yahoo Finance and build ``prices.parquet``.

Incremental: each symbol's raw bars are cached at
``data/raw/yahoo/<SYMBOL>.parquet`` and skipped on re-runs unless ``--force``.
Rebuilding the processed panel never requires the network.

What Yahoo does NOT provide, and what that costs
-----------------------------------------------
* **delivery percentage** -- absent. The delivery factor is therefore inert (it
  computes to an empty panel and is skipped with a printed warning). Recorded as
  ``unavailable``.
* **shares outstanding** -- absent. Written as 1.0 for every name, which degrades the
  size factor to a *log-price* proxy. Recorded as ``unavailable``; the factor is
  named ``size_logprice_proxy`` so the degradation is visible at every call site.

Both are recorded rather than papered over. The nulls are fine; silence about them
would not be.

``--provenance-only`` rewrites the provenance record from what is already on disk
without contacting Yahoo.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

import polars as pl

from _cli import REPO_ROOT, bootstrap, say
from flowalpha.data.calendar import TradingCalendar, ensure_holidays_file
from flowalpha.data.prices import PRICES_SCHEMA, process_prices, write_prices
from flowalpha.data.provenance import (
    CAVEAT_DIVIDEND_ADJUSTMENT,
    REAL,
    UNAVAILABLE,
    Provenance,
)

YAHOO_SUFFIX = ".NS"
RAW_SUBDIR = "yahoo"
SOURCE = "Yahoo Finance via yfinance (auto_adjust=False, group_by=ticker)"


def read_universe(reference_dir: Path) -> list[str]:
    """Read ``universe_tickers.csv`` as plain lines.

    Line-based on purpose: the file is written single-column so this reader stays
    trivial. A ``symbol`` header line is tolerated and skipped.
    """
    path = reference_dir / "universe_tickers.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/fetch_universe_tickers.py first. "
            "This script has no built-in fallback universe by design."
        )
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    out = [ln for ln in lines if ln and ln.lower() != "symbol"]
    if not out:
        raise ValueError(f"{path} contains no symbols")
    return out


def _to_polars(pdf, symbol: str) -> pl.DataFrame:
    """Convert one ticker's pandas frame to a typed long-format polars frame."""
    import pandas as pd  # local import: pandas is a downloader-only dependency

    if pdf is None or len(pdf) == 0:
        return pl.DataFrame(
            schema={
                "date": pl.Date, "symbol": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
                "low": pl.Float64, "close": pl.Float64, "adj_close": pl.Float64,
                "volume": pl.Float64,
            }
        )
    frame = pdf.copy()
    frame = frame.reset_index()
    date_col = "Date" if "Date" in frame.columns else frame.columns[0]
    frame = frame.rename(
        columns={
            date_col: "date", "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
        }
    )
    keep = [c for c in ("date", "open", "high", "low", "close", "adj_close", "volume") if c in frame.columns]
    frame = frame[keep]
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    out = pl.from_pandas(frame)
    return out.with_columns(pl.lit(symbol).alias("symbol")).select(
        pl.col("date").cast(pl.Date),
        pl.col("symbol").cast(pl.Utf8),
        *[
            pl.col(c).cast(pl.Float64)
            for c in ("open", "high", "low", "close", "adj_close", "volume")
            if c in out.columns
        ],
    )


def download_batch(symbols: list[str], start: str, end: str) -> dict[str, pl.DataFrame]:
    """Download one batch of symbols, returning ``{symbol: long-format frame}``."""
    import yfinance as yf

    tickers = [s + YAHOO_SUFFIX for s in symbols]
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
        actions=False,
    )
    out: dict[str, pl.DataFrame] = {}
    for sym, ticker in zip(symbols, tickers):
        try:
            sub = raw[ticker] if len(tickers) > 1 else raw
        except (KeyError, TypeError):
            continue
        sub = sub.dropna(how="all")
        frame = _to_polars(sub, sym)
        if not frame.is_empty():
            out[sym] = frame
    return out


def build_panel(raw_dir: Path, symbols: list[str]) -> tuple[pl.DataFrame, list[str]]:
    """Assemble the cached per-symbol files into one processed panel."""
    frames: list[pl.DataFrame] = []
    absent: list[str] = []
    for sym in symbols:
        path = raw_dir / f"{sym}.parquet"
        if not path.exists():
            absent.append(sym)
            continue
        frame = pl.read_parquet(path)
        if frame.is_empty():
            absent.append(sym)
            continue
        frames.append(frame)
    if not frames:
        return pl.DataFrame(schema=PRICES_SCHEMA), absent
    combined = pl.concat(frames, how="vertical_relaxed")
    return process_prices(combined), absent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None, help="ISO start date (default: config dates.start)")
    parser.add_argument("--end", default=None, help="ISO end date (default: config dates.end)")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--force", action="store_true", help="re-download symbols already cached")
    parser.add_argument("--limit", type=int, default=None, help="only the first N symbols (smoke runs)")
    parser.add_argument(
        "--provenance-only",
        action="store_true",
        help="rebuild the panel and provenance from cached files without downloading",
    )
    args = parser.parse_args(argv)

    cfg, _ = bootstrap("download_yahoo")
    start = args.start or cfg["dates"]["start"]
    # yfinance treats `end` as exclusive; extend by a day so the declared end
    # session is actually fetched.
    end_date = _dt.date.fromisoformat(args.end or cfg["dates"]["end"])
    end_exclusive = (end_date + _dt.timedelta(days=1)).isoformat()

    raw_dir = cfg.path("raw") / RAW_SUBDIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    symbols = read_universe(cfg.path("reference"))
    if args.limit:
        symbols = symbols[: args.limit]
    say(f"universe: {len(symbols)} symbols; window {start} .. {end_date}")

    if not args.provenance_only:
        pending = [s for s in symbols if args.force or not (raw_dir / f"{s}.parquet").exists()]
        say(f"cached: {len(symbols) - len(pending)}; to download: {len(pending)}")
        for i in range(0, len(pending), args.batch_size):
            batch = pending[i : i + args.batch_size]
            got = download_batch(batch, start, end_exclusive)
            for sym, frame in got.items():
                frame.write_parquet(raw_dir / f"{sym}.parquet")
            say(
                f"  batch {i // args.batch_size + 1}: "
                f"{len(got)}/{len(batch)} symbols returned data"
            )

    prices, absent = build_panel(raw_dir, symbols)
    if prices.is_empty():
        say("FATAL: no price data on disk. Nothing was downloaded successfully.")
        return 1

    n_sym = prices["symbol"].n_unique()
    sessions = sorted(set(prices["date"].to_list()))
    say(f"panel: {prices.height} rows, {n_sym}/{len(symbols)} symbols, {len(sessions)} sessions")
    say(f"        {sessions[0]} .. {sessions[-1]}")
    if absent:
        say(f"WARNING: {len(absent)} symbols returned no data: {', '.join(absent[:12])}"
            + (" ..." if len(absent) > 12 else ""))

    path = write_prices(prices, cfg.path("processed"))
    say(f"wrote {path.relative_to(REPO_ROOT)}")

    calendar = TradingCalendar.from_prices(prices)
    sessions_path, holidays_path = ensure_holidays_file(calendar, cfg.path("reference"))
    say(
        f"calendar: {len(calendar)} sessions, {len(calendar.holidays())} implied holidays "
        f"-> {sessions_path.name}, {holidays_path.name}"
    )

    # Shares outstanding is not available from Yahoo. Write 1.0 so the size factor
    # still computes, and name the degradation in provenance.
    shares_path = cfg.path("reference") / "shares_outstanding.csv"
    pl.DataFrame(
        {"symbol": sorted(set(prices["symbol"].to_list())), "shares": 1.0}
    ).write_csv(shares_path)
    say(f"wrote {shares_path.relative_to(REPO_ROOT)} (all 1.0 -- see provenance)")

    prov = Provenance.load(REPO_ROOT / "data")
    prov.mark(
        "prices",
        REAL,
        SOURCE,
        note=(
            f"{n_sym} symbols, {len(sessions)} sessions, {sessions[0]}..{sessions[-1]}. "
            "adj_factor = adj_close/close applied to open/high/low; volume left raw so "
            "turnover = close*volume is split-invariant."
        ),
    )
    prov.mark(
        "delivery_pct",
        UNAVAILABLE,
        SOURCE,
        note=(
            "Yahoo publishes no delivery percentage. The delivery factor computes to an "
            "empty panel and is skipped with a warning rather than counted as a trial."
        ),
    )
    prov.mark(
        "shares_outstanding",
        UNAVAILABLE,
        SOURCE,
        note=(
            "Yahoo publishes no share count. Written as 1.0, which degrades the size "
            "factor to a log-PRICE proxy (named size_logprice_proxy), not log market cap."
        ),
    )
    prov.mark(
        "trading_calendar",
        REAL,
        "derived from observed price dates",
        note=(
            f"{len(calendar)} sessions derived from the price panel; weekdays in range "
            f"with no trading ({len(calendar.holidays())}) are treated as holidays."
        ),
    )
    prov.add_caveat(CAVEAT_DIVIDEND_ADJUSTMENT)
    prov.save(REPO_ROOT / "data")
    say(f"provenance: {prov.label()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
