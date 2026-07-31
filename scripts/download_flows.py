#!/usr/bin/env python
"""Accumulate the FII/DII **cash-market** series from NSE's current-day API.

``api/fiidiiTradeReact`` returns **today only**. There is no free historical
endpoint, so this series can only grow one session at a time, by running this script
daily. Expect it to span days or weeks at first, not years -- that is the expected
state, not a failure.

It writes ``fii_dii_cash.parquet`` and **never** ``flows_daily.parquet``. The latter
spans the full 2019->present window from the F&O archive; overwriting it with a few
months of cash data would null nearly every regime label downstream, which looks
like a modelling problem rather than a file-handling one. The distinct filename is
the guard, and this docstring is the reason.

Units differ from the rest of the flow stack: Rs crores here, net futures contracts
in ``flows_daily``. Both are recorded separately in provenance.
"""

from __future__ import annotations

import argparse
import sys

import polars as pl

from _cli import REPO_ROOT, bootstrap, say
from flowalpha.data.fetch import FetchError, NSESession
from flowalpha.data.nse_flows import (
    FII_DII_CASH_FILENAME,
    FII_DII_CASH_URL,
    FlowParseError,
    parse_fii_dii_cash,
    write_fii_dii_cash,
)
from flowalpha.data.provenance import REAL, UNAVAILABLE, Provenance

CASH_NOTE = (
    "NSE api/fiidiiTradeReact is CURRENT-DAY ONLY. This series is accumulated by "
    "running scripts/download_flows.py daily, so its span reflects how long the "
    "script has been run, not data availability. Units: Rs crores (cash market), "
    "unlike flows_daily which is in net futures contracts."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance-only", action="store_true",
                        help="re-derive provenance from the parquet already on disk")
    args = parser.parse_args(argv)

    cfg, _ = bootstrap("download_flows")
    processed = cfg.path("processed")
    raw_dir = cfg.path("raw") / "flows" / "cash"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = processed / FII_DII_CASH_FILENAME

    if not args.provenance_only:
        say(f"requesting {FII_DII_CASH_URL} (current day only)")
        try:
            with NSESession(cfg) as session:
                result = session.get_json_api(FII_DII_CASH_URL)
            if not result.ok:
                raise FlowParseError(f"HTTP {result.status}")
            rows = parse_fii_dii_cash(result.content)
        except (FetchError, FlowParseError) as exc:
            say(f"WARNING: cash FII/DII unavailable this run: {exc}")
            say("Recording the dataset as unavailable rather than substituting a proxy.")
            prov = Provenance.load(REPO_ROOT / "data")
            if prov.status_of("fii_dii_cash") is None:
                prov.mark("fii_dii_cash", UNAVAILABLE, FII_DII_CASH_URL,
                          note=CASH_NOTE + " Not reachable on the most recent attempt.")
                prov.save(REPO_ROOT / "data")
            return 0

        # Cache the raw payload so the parse can be re-audited without re-fetching.
        session_dates = sorted(set(rows["date"].to_list()))
        stamp = session_dates[-1].isoformat() if session_dates else "unknown"
        (raw_dir / f"fiidii_{stamp}.json").write_bytes(result.content)

        say(f"parsed {rows.height} rows for {stamp}")
        for row in rows.iter_rows(named=True):
            say(f"  {row['participant']}: net {row['net_crore']} crore")
        write_fii_dii_cash(rows, processed)

    if not path.exists():
        say("no cash series on disk yet")
        return 0

    series = pl.read_parquet(path)
    dates = sorted(set(series["date"].to_list()))
    say(f"wrote {path.relative_to(REPO_ROOT)}: {series.height} rows, "
        f"{len(dates)} sessions, {dates[0]} .. {dates[-1]}")
    say("NOTE: this file is separate from flows_daily.parquet on purpose. "
        "flows_daily spans the full window from the F&O archive.")

    prov = Provenance.load(REPO_ROOT / "data")
    prov.mark("fii_dii_cash", REAL, FII_DII_CASH_URL,
              note=f"{len(dates)} sessions accumulated ({dates[0]}..{dates[-1]}). " + CASH_NOTE)
    prov.add_caveat(
        "The cash FII/DII series (fii_dii_cash) is short because the source API is "
        "current-day only. Regime construction therefore uses the F&O participant "
        "series, not the cash series."
    )
    prov.save(REPO_ROOT / "data")
    say(f"provenance: {prov.label()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
