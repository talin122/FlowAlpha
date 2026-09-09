#!/usr/bin/env python
"""Fetch the NIFTY 500 constituent list from NSE.

Writes three files under ``data/reference``:

``universe_tickers.csv``
    One bare symbol per line under a ``symbol`` header. The price downloader reads
    this **line by line**, so it must stay single-column. Adding a second column here
    breaks that reader.
``nse_industry.csv``
    ``symbol,industry`` from the same file, kept separate for exactly that reason.
``sectors.csv``
    ``symbol,sector`` -- the industry column under the name the factor code expects.

There is deliberately **no hardcoded fallback universe**. If the list cannot be
fetched this script exits non-zero. A silent fallback to a short built-in list would
produce a ~50-name universe while ``config.yaml`` declares 500, and every
cross-sectional rank in the project would then be computed over a tenth of the
intended breadth -- a failure that looks like a working pipeline.
"""

from __future__ import annotations

import argparse
import io
import sys

import polars as pl

from _cli import REPO_ROOT, bootstrap, say
from flowalpha.data.fetch import FetchError, NSESession
from flowalpha.data.provenance import (
    CAVEAT_SURVIVORSHIP,
    REAL,
    Provenance,
)

NIFTY500_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"

EXPECTED_COLUMNS = ("Company Name", "Industry", "Symbol", "Series", "ISIN Code")

#: NSE issues temporary placeholder tickers during corporate actions -- a demerger or
#: scheme of arrangement produces e.g. ``DUMMYHEG`` while the new entity is being set up.
#: They appear in the published constituent list but are not tradable instruments and no
#: price vendor carries them.
#:
#: This is not hypothetical: ``DUMMYHEG`` entered this universe on 2026-09-09, replacing
#: HEG. Nothing downstream broke, because Yahoo returned no data and the panel simply
#: omitted it -- but the symbol still reached universe_tickers.csv and sectors.csv, and a
#: name that is silently dropped three stages later is a name nobody ever decided to drop.
#: Excluded here, at the point of ingestion, and reported by count.
PLACEHOLDER_PREFIXES = ("DUMMY",)


def is_placeholder(symbol: str) -> bool:
    """True for an NSE corporate-action placeholder rather than a tradable symbol."""
    return symbol.upper().startswith(PLACEHOLDER_PREFIXES)


def parse_constituents(content: bytes | str) -> tuple[pl.DataFrame, list[str]]:
    """Parse the published constituent CSV, keeping only tradable ``EQ`` rows.

    Returns ``(members, dropped_placeholders)``. The placeholders are returned rather
    than discarded so the caller can report them: a universe that quietly shrinks is the
    kind of change that makes two runs incomparable without anyone noticing.
    """
    text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
    head = text.lstrip()[:200].lower()
    if head.startswith("<!doctype html") or head.startswith("<html"):
        raise ValueError("constituent list response is an HTML error page, not CSV")
    frame = pl.read_csv(io.StringIO(text), infer_schema_length=0)
    frame = frame.rename({c: c.strip() for c in frame.columns})
    missing = [c for c in ("Symbol", "Industry", "Series") if c not in frame.columns]
    if missing:
        raise ValueError(f"constituent list missing columns {missing}; got {frame.columns}")
    out = (
        frame.with_columns(
            pl.col("Symbol").str.strip_chars().alias("symbol"),
            pl.col("Industry").str.strip_chars().alias("industry"),
            pl.col("Series").str.strip_chars().alias("series"),
        )
        .filter((pl.col("series") == "EQ") & (pl.col("symbol") != ""))
        .select("symbol", "industry")
        .unique(subset=["symbol"], keep="first")
        .sort("symbol")
    )
    dropped = [s for s in out["symbol"].to_list() if is_placeholder(s)]
    if dropped:
        out = out.filter(~pl.col("symbol").map_elements(is_placeholder, return_dtype=pl.Boolean))
    if out.is_empty():
        raise ValueError("constituent list parsed to zero tradable EQ symbols")
    return out, dropped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default=NIFTY500_URL, help="override the constituent list URL"
    )
    args = parser.parse_args(argv)

    cfg, _ = bootstrap("fetch_universe_tickers")
    ref = cfg.path("reference")
    ref.mkdir(parents=True, exist_ok=True)

    say(f"fetching {cfg['universe']['name']} constituents from {args.url}")
    try:
        with NSESession(cfg) as session:
            result = session.get(args.url, allow_missing=False)
        members, placeholders = parse_constituents(result.content)
    except (FetchError, ValueError) as exc:
        say(f"FATAL: could not obtain the constituent list: {exc}")
        say(
            "Refusing to fall back to a hardcoded universe. A short built-in list would "
            "silently shrink every cross-sectional rank in the project. Fix the network "
            "or pass --url and re-run."
        )
        return 1

    declared = int(cfg["universe"]["size"])
    n = members.height
    if placeholders:
        say(
            f"excluded {len(placeholders)} corporate-action placeholder(s): "
            f"{', '.join(placeholders)}"
        )
        say(
            "  These are NSE scaffolding symbols, not tradable instruments. Dropped here "
            "rather than left to vanish silently when a price vendor returns nothing."
        )
    say(f"parsed {n} tradable EQ symbols (config declares universe.size = {declared})")
    if n != declared:
        say(
            f"WARNING: constituent count {n} differs from declared size {declared}. "
            "The published list is authoritative; consider updating config.yaml."
        )

    tickers_path = ref / "universe_tickers.csv"
    members.select("symbol").write_csv(tickers_path)
    industry_path = ref / "nse_industry.csv"
    members.rename({"industry": "industry"}).write_csv(industry_path)
    sectors_path = ref / "sectors.csv"
    members.rename({"industry": "sector"}).write_csv(sectors_path)

    say(f"wrote {tickers_path.relative_to(REPO_ROOT)} ({n} symbols, single column)")
    say(f"wrote {industry_path.relative_to(REPO_ROOT)}")
    say(f"wrote {sectors_path.relative_to(REPO_ROOT)}")

    prov = Provenance.load(REPO_ROOT / "data")
    prov.mark(
        "universe",
        REAL,
        args.url,
        note=(
            f"NSE published {cfg['universe']['name']} constituent list, {n} EQ symbols, "
            "CURRENT snapshot only"
        ),
    )
    prov.mark(
        "sectors",
        REAL,
        args.url,
        note=(
            "Industry classification from the same constituent file. This is a CURRENT "
            "snapshot, not point-in-time: a name reclassified since listing carries its "
            "present industry for its whole history."
        ),
    )
    prov.add_caveat(CAVEAT_SURVIVORSHIP)
    prov.add_caveat(
        "Sector labels are a current snapshot, so sector-neutralisation and sector caps "
        "inherit a mild classification look-ahead. factors.sector_neutralize is false by "
        "default for this reason."
    )
    prov.save(REPO_ROOT / "data")
    say(f"provenance: {prov.label()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
