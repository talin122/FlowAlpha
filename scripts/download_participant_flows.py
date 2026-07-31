#!/usr/bin/env python
"""Download NSE's participant-wise F&O volume archive, one file per session.

This is the **only** free source that spans 2019 -> present for institutional order
flow, which is why the project's flow series is built from it. See
:mod:`flowalpha.data.nse_flows` for the unit caveat: these are net futures
*contracts*, not cash rupees.

Roughly 1900 requests. Three things keep that considerate and cheap:

* Every raw CSV is cached under ``data/raw/flows/participant/`` and skipped on
  re-runs, so an interrupted download resumes for free.
* A handful of worker threads share a single politeness delay (``download.flow_delay_sec``),
  so the aggregate request rate is bounded regardless of worker count.
* A 404 is not retried. Sessions with no published archive are recorded in
  ``data/raw/flows/participant/_missing.json`` and skipped next time, instead of
  costing five requests each, every run.

Which dates are attempted comes from the trading calendar derived from the price
panel, so flows are only sought for sessions the market actually traded.

``--provenance-only`` re-derives the provenance record from the cache without
downloading.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from _cli import REPO_ROOT, bootstrap, say
from flowalpha.data.calendar import TradingCalendar
from flowalpha.data.fetch import FetchError, NSESession
from flowalpha.data.nse_flows import (
    FlowParseError,
    parse_participant_vol,
    participant_cache_path,
    participant_url,
)
from flowalpha.data.provenance import CAVEAT_RETAIL_PROXY, REAL, Provenance

MISSING_FILENAME = "_missing.json"


def load_calendar(cfg) -> TradingCalendar:
    path = cfg.path("reference") / "trading_sessions.csv"
    if path.exists():
        return TradingCalendar.from_file(path)
    prices_path = cfg.path("processed") / "prices.parquet"
    if prices_path.exists():
        import polars as pl

        return TradingCalendar.from_prices(pl.read_parquet(prices_path, columns=["date"]))
    raise FileNotFoundError(
        "no trading calendar available. Run scripts/download_yahoo.py first so the "
        "calendar can be derived from observed sessions."
    )


def read_missing(cache_dir: Path) -> set[_dt.date]:
    path = cache_dir / MISSING_FILENAME
    if not path.exists():
        return set()
    try:
        return {_dt.date.fromisoformat(d) for d in json.loads(path.read_text(encoding="utf-8"))}
    except (json.JSONDecodeError, ValueError, OSError):
        return set()


def write_missing(cache_dir: Path, missing: set[_dt.date]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / MISSING_FILENAME).write_text(
        json.dumps(sorted(d.isoformat() for d in missing), indent=2) + "\n", encoding="utf-8"
    )


def fetch_one(session: NSESession, raw_root: Path, day: _dt.date) -> tuple[_dt.date, str]:
    """Fetch and validate one session's archive. Returns ``(date, status)``.

    Status is ``ok``, ``missing`` (no published file) or ``bad:<reason>``. The payload
    is parsed *before* being cached, so an HTML error page never lands on disk
    pretending to be data.
    """
    url = participant_url(day)
    try:
        result = session.get(url, allow_missing=True)
    except FetchError as exc:
        return day, f"bad:{exc}"
    if not result.ok:
        return day, "missing"
    try:
        parse_participant_vol(result.content, session_date=day)
    except FlowParseError as exc:
        return day, f"bad:{exc}"
    path = participant_cache_path(raw_root, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(result.content)
    return day, "ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None, help="ISO start (default: config dates.start)")
    parser.add_argument("--end", default=None, help="ISO end (default: config dates.end)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--delay", type=float, default=None, help="per-request politeness delay")
    parser.add_argument("--retry-missing", action="store_true",
                        help="re-attempt sessions previously recorded as having no archive")
    parser.add_argument("--limit", type=int, default=None, help="only the first N pending sessions")
    parser.add_argument("--provenance-only", action="store_true")
    args = parser.parse_args(argv)

    cfg, _ = bootstrap("download_participant_flows")
    raw_root = cfg.path("raw")
    cache_dir = raw_root / "flows" / "participant"
    cache_dir.mkdir(parents=True, exist_ok=True)

    calendar = load_calendar(cfg)
    start = _dt.date.fromisoformat(args.start) if args.start else cfg.start_date
    end = _dt.date.fromisoformat(args.end) if args.end else cfg.end_date
    wanted = calendar.sessions_between(start, end)
    say(f"sessions in window: {len(wanted)} ({start} .. {end})")

    known_missing = set() if args.retry_missing else read_missing(cache_dir)
    if not args.provenance_only:
        pending = [
            d for d in wanted
            if not participant_cache_path(raw_root, d).exists() and d not in known_missing
        ]
        if args.limit:
            pending = pending[: args.limit]
        say(
            f"cached: {sum(1 for d in wanted if participant_cache_path(raw_root, d).exists())}; "
            f"known-missing: {len(known_missing & set(wanted))}; to fetch: {len(pending)}"
        )

        workers = args.workers or int(cfg.get("download.flow_workers", 6))
        delay = args.delay if args.delay is not None else float(cfg.get("download.flow_delay_sec", 0.35))
        counts = {"ok": 0, "missing": 0, "bad": 0}
        newly_missing: set[_dt.date] = set()
        bad: list[tuple[_dt.date, str]] = []

        if pending:
            with NSESession(cfg, delay_sec=delay) as session:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(fetch_one, session, raw_root, d): d for d in pending}
                    done = 0
                    for fut in as_completed(futures):
                        day, status = fut.result()
                        done += 1
                        if status == "ok":
                            counts["ok"] += 1
                        elif status == "missing":
                            counts["missing"] += 1
                            newly_missing.add(day)
                        else:
                            counts["bad"] += 1
                            bad.append((day, status[4:]))
                        if done % 200 == 0:
                            say(f"  {done}/{len(pending)} fetched "
                                f"(ok={counts['ok']} missing={counts['missing']} bad={counts['bad']})")
        say(f"fetch complete: ok={counts['ok']} missing={counts['missing']} bad={counts['bad']}")
        if bad:
            say("  rejected payloads (not cached):")
            for day, reason in bad[:10]:
                say(f"    {day}: {reason}")
        write_missing(cache_dir, known_missing | newly_missing)

    cached = sorted(d for d in wanted if participant_cache_path(raw_root, d).exists())
    missing_now = sorted(set(wanted) - set(cached))
    say(f"cache covers {len(cached)}/{len(wanted)} sessions in window")
    if missing_now:
        say(f"  {len(missing_now)} sessions have no archive; first few: "
            + ", ".join(str(d) for d in missing_now[:8]))
        say("  These are reported, not filled. Interpolating a flow value would "
            "manufacture the signal under test.")

    if cached:
        prov = Provenance.load(REPO_ROOT / "data")
        prov.mark(
            "participant_flows",
            REAL,
            "https://nsearchives.nseindia.com/content/nsccl/fao_participant_vol_DDMMYYYY.csv",
            note=(
                f"{len(cached)} sessions cached ({cached[0]}..{cached[-1]}). "
                "Units: NET FUTURES CONTRACTS (index+stock futures, long minus short; "
                "option legs excluded), NOT cash rupees."
            ),
        )
        prov.add_caveat(CAVEAT_RETAIL_PROXY)
        prov.save(REPO_ROOT / "data")
        say(f"provenance: {prov.label()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
