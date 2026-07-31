#!/usr/bin/env python
"""Build the flow panels and the derived flow-feature panel from the raw cache.

Writes ``participant_flows.parquet``, ``flows_daily.parquet`` and
``flow_features.parquet``.

Purely local: it reads the cached CSVs written by
``scripts/download_participant_flows.py`` and never touches the network, so the panel
can always be rebuilt after a parser change without re-fetching ~1900 files.

Sessions with no published archive are **reported, not filled**. Forward-filling or
interpolating a flow value would manufacture exactly the signal this study tests.

The regime-coverage block at the end is the GATE 5 check: the **last defined regime
date** must reach the end of the price history. If it stops years earlier, the
trailing sum is propagating a gap.
"""

from __future__ import annotations

import argparse
import json
import sys

import polars as pl

from _cli import REPO_ROOT, bootstrap, say
from flowalpha.data.calendar import TradingCalendar
from flowalpha.data.nse_flows import ingest_flows
from flowalpha.data.provenance import CAVEAT_FLOW_UNITS, REAL, Provenance
from flowalpha.data.store import PointInTimeStore
from flowalpha.factors.flow_features import (
    build_flow_features,
    regime_coverage,
    write_flow_features,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance-only", action="store_true",
                        help="re-derive provenance from the parquet already on disk")
    args = parser.parse_args(argv)

    cfg, _ = bootstrap("build_flows")
    processed = cfg.path("processed")

    expected = None
    sessions_path = cfg.path("reference") / "trading_sessions.csv"
    if sessions_path.exists():
        calendar = TradingCalendar.from_file(sessions_path)
        expected = calendar.sessions_between(cfg.start_date, cfg.end_date)
        say(f"expecting flows for {len(expected)} trading sessions in the config window")
    else:
        say("WARNING: no trading calendar on disk; missing-session reporting disabled")

    if not args.provenance_only:
        summary = ingest_flows(cfg.path("raw"), processed, expected_sessions=expected)
    else:
        summary = {}

    if not args.provenance_only:
        say(f"participant_flows: {summary['participant_rows']} rows, "
            f"{summary['sessions_covered']} sessions "
            f"({summary['first_session']} .. {summary['last_session']})")
        say(f"flows_daily:       {summary['flows_daily_rows']} rows (FII + DII)")
        if summary["parse_failures"]:
            say(f"WARNING: {len(summary['parse_failures'])} cached files failed to parse:")
            for day, reason in summary["parse_failures"][:10]:
                say(f"    {day}: {reason}")
        n_missing = summary.get("n_missing_sessions", 0)
        if n_missing:
            pct = 100.0 * n_missing / max(1, len(expected or []))
            say(f"{n_missing} of {len(expected or [])} sessions ({pct:.2f}%) have no flow data.")
            say("  Reported, not filled: the rolling-window builder nulls only the windows "
                "that actually span a gap.")
            say("  first few: " + ", ".join(summary["missing_sessions"][:8]))
        (cfg.path("results")).mkdir(parents=True, exist_ok=True)
        (cfg.path("results") / "flow_ingest_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    daily_path = processed / "flows_daily.parquet"
    part_path = processed / "participant_flows.parquet"
    if not daily_path.exists():
        say("FATAL: flows_daily.parquet was not produced. Is the raw cache empty?")
        return 1

    daily = pl.read_parquet(daily_path)
    part = pl.read_parquet(part_path)
    dates = sorted(set(daily["date"].to_list()))
    say(f"on disk: flows_daily {daily.height} rows over {len(dates)} sessions "
        f"({dates[0]} .. {dates[-1]})")
    say(f"         participant_flows categories: "
        f"{sorted(set(part['participant'].to_list()))}")

    # --- derived flow features (the conditioning variables) -----------------
    store = PointInTimeStore.from_config(cfg)
    features = build_flow_features(store, cfg)
    features_path = write_flow_features(features, processed)
    say(f"wrote {features_path.relative_to(REPO_ROOT)}: {features.height} sessions")

    coverage = regime_coverage(features)
    (cfg.path("results")).mkdir(parents=True, exist_ok=True)
    (cfg.path("results") / "regime_coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="utf-8"
    )
    say("regime coverage (GATE 5 -- last_defined must reach the end of price history):")
    for col, entry in coverage.get("regimes", {}).items():
        detail = entry.get("buckets") or {"n_true": entry.get("n_true")}
        say(f"  {col:<18} defined={entry['n_defined']:<6} "
            f"first={entry.get('first_defined')} last={entry.get('last_defined')}  {detail}")
    price_end = coverage.get("last_session")
    lasts = [e.get("last_defined") for e in coverage.get("regimes", {}).values() if e.get("last_defined")]
    if lasts and price_end and min(lasts) != price_end:
        say(f"WARNING: earliest last-defined regime date {min(lasts)} != last session "
            f"{price_end}. Check the trailing-sum implementation for gap propagation.")
    else:
        say(f"all regimes defined through {price_end}")

    prov = Provenance.load(REPO_ROOT / "data")
    prov.mark(
        "flow_features",
        REAL,
        "derived from flows_daily + participant_flows",
        note=(
            "DERIVED, read at store lag 0 because construction already shifts the "
            "underlying flow by one session. Expanding-window terciles and percentiles "
            "only; no full-sample quantiles."
        ),
    )
    note = (
        f"{len(dates)} sessions ({dates[0]}..{dates[-1]}), derived from the cached "
        "fao_participant_vol archive. Units: NET FUTURES CONTRACTS, not cash rupees."
    )
    prov.mark("flows_daily", REAL,
              "https://nsearchives.nseindia.com/content/nsccl/fao_participant_vol_DDMMYYYY.csv",
              note=note)
    prov.mark("participant_flows", REAL,
              "https://nsearchives.nseindia.com/content/nsccl/fao_participant_vol_DDMMYYYY.csv",
              note=note + " Categories: Client (retail proxy), DII, FII, Pro.")
    prov.add_caveat(CAVEAT_FLOW_UNITS)
    prov.save(REPO_ROOT / "data")
    say(f"provenance: {prov.label()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
