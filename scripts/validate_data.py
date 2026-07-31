#!/usr/bin/env python
"""Data quality gate. Writes ``results/data_quality_report.json`` and exits non-zero on FAIL.

Warnings are acceptable and are carried through into the HTML report's limitations
section. A FAIL stops the pipeline: every downstream number would be computed on data
we have just established is unusable.

Nothing here repairs what it finds. A check that silently fixes its own findings is a
check whose findings you never see.
"""

from __future__ import annotations

import argparse
import json
import sys

import polars as pl

from _cli import REPO_ROOT, bootstrap, say
from flowalpha.data.calendar import TradingCalendar
from flowalpha.data.prices import read_prices
from flowalpha.data.quality import FAIL, PASS, WARN, overall_status, run_all_checks

SYMBOL = {PASS: "[ PASS ]", WARN: "[ WARN ]", FAIL: "[ FAIL ]"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-fail", action="store_true",
                        help="report but exit 0 (for inspecting a broken tree)")
    args = parser.parse_args(argv)

    cfg, prov = bootstrap("validate_data")
    results_dir = cfg.path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        prices = read_prices(cfg.path("processed"))
    except FileNotFoundError as exc:
        say(f"FATAL: {exc}")
        return 1

    sessions_path = cfg.path("reference") / "trading_sessions.csv"
    if sessions_path.exists():
        calendar = TradingCalendar.from_file(sessions_path)
        cal_source = str(sessions_path.relative_to(REPO_ROOT))
    else:
        calendar = TradingCalendar.from_prices(prices)
        cal_source = "derived from the price panel (no persisted sessions file)"
    say(f"calendar: {len(calendar)} sessions from {cal_source}")

    flows_path = cfg.path("processed") / "flows_daily.parquet"
    flows = pl.read_parquet(flows_path) if flows_path.exists() else None
    if flows is None:
        say("WARNING: flows_daily.parquet absent; flow-gap check skipped")

    results = run_all_checks(cfg, prices, calendar, flows=flows)
    status = overall_status(results)

    say("")
    for r in results:
        say(f"{SYMBOL[r.status]} {r.name}")
        say(f"         {r.message}")
    say("")
    say(f"overall: {status}")

    report = {
        "overall": status,
        "provenance_label": prov.label(),
        "config_window": {"start": cfg["dates"]["start"], "end": cfg["dates"]["end"]},
        "calendar_source": cal_source,
        "n_sessions": len(calendar),
        "n_price_rows": prices.height,
        "n_symbols": prices["symbol"].n_unique(),
        "checks": [r.to_dict() for r in results],
        "caveats": list(prov.caveats),
    }
    out_path = results_dir / "data_quality_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    say(f"wrote {out_path.relative_to(REPO_ROOT)}")

    if status == FAIL and not args.allow_fail:
        say("exiting non-zero: downstream results computed on this tree would not be usable")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
