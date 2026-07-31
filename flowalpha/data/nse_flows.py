"""NSE order-flow ingestion.

What is actually obtainable
--------------------------
This project's thesis is about institutional flow, so the flow source determines
what the research measures. Endpoint availability, verified:

===========================================================  ==================  =====================================
source                                                       span                content
===========================================================  ==================  =====================================
``nsccl/fao_participant_vol_DDMMYYYY.csv``                    2019 -> present     Client/DII/FII/Pro F&O long & short
                                                                                  positions, in **contracts**
``api/fiidiiTradeReact``                                      current day only    FII/DII **cash**, Rs crores
``content/fo/fii_stats_DD-MMM-YYYY.xls``                      2019 -> present     FII F&O values; real legacy ``.xls``
``cm_participant_vol_*.csv``                                  --                  **404, does not exist**
===========================================================  ==================  =====================================

Consequence: **daily cash-market FII/DII history is not freely obtainable.** Only the
participant-wise F&O file spans the full window. So ``flows_daily`` and
``participant_flows`` are built from that file and are denominated in *net futures
contracts*, not cash rupees. The cash series lives in a **separate** dataset
(``fii_dii_cash``) accumulated one session at a time, and will be short.

That unit difference is stated in provenance, in ``DATA_SOURCE.md``, in the README
and in the report. Regime features are quantile- and sign-based, so bucketing is
unaffected -- but "FII net" in this codebase means net futures positioning, and
anyone reading a result without knowing that will misinterpret it.

Parsing quirks of ``fao_participant_vol`` (all real, all load-bearing)
---------------------------------------------------------------------
* Line 0 is a **quoted title** carrying the date. Older files use one level of
  quoting (``"...as on January 01, 2019"``), newer files two (``""...as on Jul 29,
  2026""``), and the month format differs between them. The header is line 1.
* Some column names carry **trailing spaces** (``"Future Stock Short       "``).
* There is a **``TOTAL`` summary row**; including it in the participant categories
  would double the panel and make every regime meaningless.
* The net is taken over **futures legs only** (index + stock, long minus short).
  Option legs are excluded deliberately: a long call and a short put are not
  comparable exposures, so summing option contracts does not measure direction.
* A missing archive returns an **HTML error page**. A parser that accepts it writes
  garbage into the panel.
"""

from __future__ import annotations

import datetime as _dt
import io
import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import polars as pl

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

#: The four participant categories NSE publishes. `TOTAL` is NOT one of them.
PARTICIPANT_CATEGORIES: tuple[str, ...] = ("Client", "DII", "FII", "Pro")

#: Category used as the retail proxy. Recorded as a proxy, not a measurement:
#: `Client` includes HNIs and other non-institutional accounts.
RETAIL_CATEGORY = "Client"

_FUT_LONG = ("fut_index_long", "fut_stock_long")
_FUT_SHORT = ("fut_index_short", "fut_stock_short")

#: Canonical name for each published column, after stripping and casefolding.
_COLUMN_MAP: dict[str, str] = {
    "client type": "participant",
    "future index long": "fut_index_long",
    "future index short": "fut_index_short",
    "future stock long": "fut_stock_long",
    "future stock short": "fut_stock_short",
    "option index call long": "opt_index_call_long",
    "option index put long": "opt_index_put_long",
    "option index call short": "opt_index_call_short",
    "option index put short": "opt_index_put_short",
    "option stock call long": "opt_stock_call_long",
    "option stock put long": "opt_stock_put_long",
    "option stock call short": "opt_stock_call_short",
    "option stock put short": "opt_stock_put_short",
    "total long contracts": "total_long",
    "total short contracts": "total_short",
}

_NUMERIC_COLUMNS: tuple[str, ...] = tuple(v for v in _COLUMN_MAP.values() if v != "participant")

PARTICIPANT_FLOWS_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "participant": pl.Utf8,
    **{c: pl.Float64 for c in _NUMERIC_COLUMNS},
    "net_futures": pl.Float64,
}

FLOWS_DAILY_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "participant": pl.Utf8,
    "net_futures": pl.Float64,
    "gross_futures_long": pl.Float64,
    "gross_futures_short": pl.Float64,
}

FII_DII_CASH_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "participant": pl.Utf8,
    "buy_crore": pl.Float64,
    "sell_crore": pl.Float64,
    "net_crore": pl.Float64,
}

DEALS_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "symbol": pl.Utf8,
    "client_name": pl.Utf8,
    "buy_sell": pl.Utf8,
    "quantity": pl.Float64,
    "price": pl.Float64,
}

PARTICIPANT_FLOWS_FILENAME = "participant_flows.parquet"
#: Deliberately distinct from the cash file. Overwriting the full-window flow panel
#: with the (few-month) cash series nulls almost every regime downstream.
FLOWS_DAILY_FILENAME = "flows_daily.parquet"
FII_DII_CASH_FILENAME = "fii_dii_cash.parquet"
BULK_DEALS_FILENAME = "bulk_deals.parquet"
BLOCK_DEALS_FILENAME = "block_deals.parquet"

PARTICIPANT_URL = "https://nsearchives.nseindia.com/content/nsccl/fao_participant_vol_{ddmmyyyy}.csv"
FII_DII_CASH_URL = "https://www.nseindia.com/api/fiidiiTradeReact"


class FlowParseError(Exception):
    """Raised when a flow payload is missing, malformed, or an HTML error page."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode(content: bytes | str) -> str:
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return content


def _looks_like_html(text: str) -> bool:
    head = text.lstrip()[:400].lower()
    return head.startswith("<!doctype html") or head.startswith("<html") or "<head>" in head


_TITLE_DATE_RE = re.compile(r"as on\s+([A-Za-z]+)\s+(\d{1,2})\s*,\s*(\d{4})")

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def participant_title_date(line: str) -> _dt.date | None:
    """Extract the session date from the archive's title line, if present.

    Handles both published month formats (``January 01, 2019`` and ``Jul 29, 2026``)
    and both quoting levels. Returns ``None`` when the line carries no date, in
    which case the caller must supply ``session_date`` -- the date appears nowhere
    else in the file.
    """
    m = _TITLE_DATE_RE.search(line)
    if not m:
        return None
    month = _MONTHS.get(m.group(1)[:3].lower())
    if month is None:
        return None
    try:
        return _dt.date(int(m.group(3)), month, int(m.group(2)))
    except ValueError:
        return None


def empty_participant_flows() -> pl.DataFrame:
    return pl.DataFrame(schema=PARTICIPANT_FLOWS_SCHEMA)


def empty_flows_daily() -> pl.DataFrame:
    return pl.DataFrame(schema=FLOWS_DAILY_SCHEMA)


def empty_fii_dii_cash() -> pl.DataFrame:
    return pl.DataFrame(schema=FII_DII_CASH_SCHEMA)


# ---------------------------------------------------------------------------
# Participant-wise F&O volume
# ---------------------------------------------------------------------------

def parse_participant_vol(
    content: bytes | str,
    *,
    session_date: _dt.date | None = None,
    require_date: bool = True,
) -> pl.DataFrame:
    """Parse one ``fao_participant_vol_DDMMYYYY.csv`` into :data:`PARTICIPANT_FLOWS_SCHEMA`.

    Parameters
    ----------
    content:
        Raw bytes or text of the archive file. Both are accepted because the
        downloader caches bytes while tests pass strings.
    session_date:
        The session this file describes. Preferred over the title line, since the
        filename is the authoritative date and the title has changed format at least
        once. When omitted, the title line is parsed instead.
    require_date:
        When True (default) a file with neither an explicit ``session_date`` nor a
        parseable title date is an error rather than a dateless frame.

    Returns
    -------
    One row per participant category, ``TOTAL`` excluded, with ``net_futures`` =
    (index long + stock long) - (index short + stock short).
    """
    text = _decode(content)
    if _looks_like_html(text):
        raise FlowParseError(
            "participant flow payload is an HTML error page (archive missing for this date), not CSV"
        )
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise FlowParseError("participant flow payload is empty")

    title_date = participant_title_date(lines[0])
    # The first line is a title only if it is not itself the header.
    header_is_first = "client type" in lines[0].split(",")[0].strip().strip('"').lower()
    body_lines = lines if header_is_first else lines[1:]
    if not body_lines:
        raise FlowParseError("participant flow payload has a title but no data rows")

    resolved = session_date or title_date
    if resolved is None and require_date:
        raise FlowParseError(
            "participant flow payload carries no parseable session date; pass session_date "
            "explicitly (the date appears nowhere but the title line)"
        )

    frame = pl.read_csv(
        io.StringIO("\n".join(body_lines)),
        infer_schema_length=0,
        truncate_ragged_lines=True,
    )
    # Trailing (and leading) spaces on published column names are real.
    renames: dict[str, str] = {}
    for col in frame.columns:
        key = col.strip().strip('"').strip().lower()
        if key in _COLUMN_MAP:
            renames[col] = _COLUMN_MAP[key]
    frame = frame.rename(renames)

    if "participant" not in frame.columns:
        raise FlowParseError(
            f"participant flow payload has no 'Client Type' column; got {frame.columns}"
        )
    for col in _FUT_LONG + _FUT_SHORT:
        if col not in frame.columns:
            raise FlowParseError(f"participant flow payload missing futures column '{col}'")

    frame = frame.with_columns(pl.col("participant").cast(pl.Utf8).str.strip_chars())

    # Drop the TOTAL summary row (and any other non-category row) before anything
    # downstream can treat it as a participant.
    frame = frame.filter(pl.col("participant").is_in(list(PARTICIPANT_CATEGORIES)))
    if frame.is_empty():
        raise FlowParseError(
            f"participant flow payload contains none of the expected categories "
            f"{PARTICIPANT_CATEGORIES}"
        )

    for col in _NUMERIC_COLUMNS:
        if col in frame.columns:
            frame = frame.with_columns(
                pl.col(col).cast(pl.Utf8).str.strip_chars().str.replace_all(",", "")
                .replace({"-": None, "": None})
                .cast(pl.Float64, strict=False)
                .alias(col)
            )
        else:
            frame = frame.with_columns(pl.lit(None, pl.Float64).alias(col))

    frame = frame.with_columns(
        pl.lit(resolved).cast(pl.Date).alias("date"),
        (
            sum(pl.col(c) for c in _FUT_LONG) - sum(pl.col(c) for c in _FUT_SHORT)
        ).alias("net_futures"),
    )
    return frame.select(
        [pl.col(n).cast(dt).alias(n) for n, dt in PARTICIPANT_FLOWS_SCHEMA.items()]
    ).sort("participant")


def participant_url(session_date: _dt.date) -> str:
    return PARTICIPANT_URL.format(ddmmyyyy=session_date.strftime("%d%m%Y"))


def participant_cache_path(raw_dir: str | Path, session_date: _dt.date) -> Path:
    """Where one session's raw archive is cached, so re-runs are incremental."""
    return (
        Path(raw_dir) / "flows" / "participant" / f"fao_participant_vol_{session_date:%d%m%Y}.csv"
    )


def parse_participant_cache_dir(
    cache_dir: str | Path,
    *,
    on_error: str = "collect",
) -> tuple[pl.DataFrame, list[tuple[_dt.date, str]]]:
    """Parse every cached participant archive in a directory.

    The session date comes from the *filename*, which is authoritative, not the
    title line. Returns ``(panel, failures)`` where failures is a list of
    ``(date, reason)`` -- surfaced rather than swallowed, because a silently dropped
    session becomes a gap in a rolling window later.
    """
    cache_dir = Path(cache_dir)
    frames: list[pl.DataFrame] = []
    failures: list[tuple[_dt.date, str]] = []
    if not cache_dir.exists():
        return empty_participant_flows(), failures

    for path in sorted(cache_dir.glob("fao_participant_vol_*.csv")):
        stem = path.stem.replace("fao_participant_vol_", "")
        try:
            session = _dt.datetime.strptime(stem, "%d%m%Y").date()
        except ValueError:
            failures.append((_dt.date(1970, 1, 1), f"{path.name}: unparseable date in filename"))
            continue
        try:
            frames.append(parse_participant_vol(path.read_bytes(), session_date=session))
        except (FlowParseError, OSError) as exc:
            if on_error == "raise":
                raise
            failures.append((session, str(exc)))
    if not frames:
        return empty_participant_flows(), failures
    return pl.concat(frames, how="vertical").sort(["date", "participant"]), failures


def flows_daily_from_participants(participants: pl.DataFrame) -> pl.DataFrame:
    """Project the participant panel down to the institutional (FII/DII) view.

    This -- not the cash API -- is what writes ``flows_daily.parquet``, and the
    values are net futures contracts. See the module docstring.
    """
    if participants.is_empty():
        return empty_flows_daily()
    out = (
        participants.filter(pl.col("participant").is_in(["FII", "DII"]))
        .with_columns(
            sum(pl.col(c) for c in _FUT_LONG).alias("gross_futures_long"),
            sum(pl.col(c) for c in _FUT_SHORT).alias("gross_futures_short"),
        )
        .select([pl.col(n).cast(dt).alias(n) for n, dt in FLOWS_DAILY_SCHEMA.items()])
        .sort(["date", "participant"])
    )
    return out


# ---------------------------------------------------------------------------
# FII/DII cash (current-day API only)
# ---------------------------------------------------------------------------

def parse_fii_dii_cash(content: bytes | str) -> pl.DataFrame:
    """Parse the ``fiidiiTradeReact`` JSON payload into :data:`FII_DII_CASH_SCHEMA`.

    The API returns **today only**. There is no free historical endpoint, so a
    long cash series can only be accumulated by running the downloader daily. The
    values are Rs crores, unlike every other flow series in this project.
    """
    text = _decode(content)
    if _looks_like_html(text):
        raise FlowParseError("fii/dii cash payload is an HTML page, not JSON")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FlowParseError(f"fii/dii cash payload is not valid JSON: {exc}") from exc
    if isinstance(payload, Mapping):
        payload = payload.get("data", payload)
    if not isinstance(payload, Sequence):
        raise FlowParseError(f"fii/dii cash payload has unexpected shape: {type(payload).__name__}")

    rows: list[dict] = []
    for entry in payload:
        if not isinstance(entry, Mapping):
            continue
        category = str(entry.get("category", "")).strip()
        participant = "FII" if category.upper().startswith(("FII", "FPI")) else (
            "DII" if category.upper().startswith("DII") else None
        )
        if participant is None:
            continue
        session = _parse_cash_date(str(entry.get("date", "")))
        if session is None:
            raise FlowParseError(f"fii/dii cash payload has unparseable date {entry.get('date')!r}")
        rows.append(
            {
                "date": session,
                "participant": participant,
                "buy_crore": _to_float(entry.get("buyValue")),
                "sell_crore": _to_float(entry.get("sellValue")),
                "net_crore": _to_float(entry.get("netValue")),
            }
        )
    if not rows:
        raise FlowParseError("fii/dii cash payload contained no FII or DII rows")
    return (
        pl.DataFrame(rows)
        .select([pl.col(n).cast(dt).alias(n) for n, dt in FII_DII_CASH_SCHEMA.items()])
        .sort(["date", "participant"])
    )


_CASH_DATE_FORMATS = ("%d-%b-%Y", "%d-%B-%Y", "%d %b %Y", "%Y-%m-%d", "%d-%m-%Y")


def _parse_cash_date(text: str) -> _dt.date | None:
    text = text.strip()
    if not text:
        return None
    for fmt in _CASH_DATE_FORMATS:
        try:
            return _dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _to_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "NA", "na"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Bulk / block deals
# ---------------------------------------------------------------------------

_DEAL_COLUMN_ALIASES = {
    "date": "date", "deal date": "date", "symbol": "symbol",
    "security name": "security_name", "client name": "client_name",
    "buy/sell": "buy_sell", "buy / sell": "buy_sell",
    "quantity traded": "quantity", "quantity": "quantity",
    "trade price / wght. avg. price": "price", "trade price": "price",
    "price": "price", "trade price / wght.avg.price": "price",
}


def parse_deals(content: bytes | str, *, kind: str = "bulk") -> pl.DataFrame:
    """Parse an NSE bulk- or block-deals CSV into :data:`DEALS_SCHEMA`.

    These are per-name, per-counterparty prints. They are ingested as their own
    datasets (with their own availability lag) rather than folded into the daily
    flow series, because their coverage is sparse and mixing a sparse per-name
    series into an aggregate would misstate both.
    """
    text = _decode(content)
    if _looks_like_html(text):
        raise FlowParseError(f"{kind} deals payload is an HTML error page, not CSV")
    if not text.strip():
        raise FlowParseError(f"{kind} deals payload is empty")
    frame = pl.read_csv(io.StringIO(text), infer_schema_length=0, truncate_ragged_lines=True)
    renames = {}
    for col in frame.columns:
        key = col.strip().strip('"').lower()
        if key in _DEAL_COLUMN_ALIASES:
            renames[col] = _DEAL_COLUMN_ALIASES[key]
    frame = frame.rename(renames)
    for required in ("date", "symbol"):
        if required not in frame.columns:
            raise FlowParseError(f"{kind} deals payload missing '{required}'; got {frame.columns}")

    frame = frame.with_columns(
        pl.coalesce(
            pl.col("date").str.strip_chars().str.to_date("%d-%b-%Y", strict=False),
            pl.col("date").str.strip_chars().str.to_date("%d-%m-%Y", strict=False),
            pl.col("date").str.strip_chars().str.to_date("%Y-%m-%d", strict=False),
        ).alias("date"),
        pl.col("symbol").str.strip_chars().alias("symbol"),
    )
    for col, dtype in DEALS_SCHEMA.items():
        if col not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype).alias(col))
    frame = frame.with_columns(
        pl.col("quantity").cast(pl.Utf8).str.replace_all(",", "").cast(pl.Float64, strict=False),
        pl.col("price").cast(pl.Utf8).str.replace_all(",", "").cast(pl.Float64, strict=False),
        pl.col("buy_sell").cast(pl.Utf8).str.strip_chars(),
        pl.col("client_name").cast(pl.Utf8).str.strip_chars(),
    )
    return frame.select([pl.col(n).cast(dt).alias(n) for n, dt in DEALS_SCHEMA.items()]).sort(
        ["date", "symbol"]
    )


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest_flows(
    raw_dir: str | Path,
    processed_dir: str | Path,
    *,
    expected_sessions: Iterable[_dt.date] | None = None,
) -> dict:
    """Build ``participant_flows.parquet`` and ``flows_daily.parquet`` from the cache.

    Reads only what is already on disk -- downloading is a separate concern
    (``scripts/download_participant_flows.py``) so that rebuilding the panel never
    depends on the network being up.

    Returns a summary dict with coverage counts, parse failures and, when
    ``expected_sessions`` is supplied, the sessions for which no archive exists.
    Missing sessions are reported, never filled: interpolating a flow value would
    manufacture the very signal the study is testing.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(raw_dir) / "flows" / "participant"

    participants, failures = parse_participant_cache_dir(cache_dir)
    daily = flows_daily_from_participants(participants)

    participants.write_parquet(processed_dir / PARTICIPANT_FLOWS_FILENAME)
    daily.write_parquet(processed_dir / FLOWS_DAILY_FILENAME)

    covered = set(participants["date"].to_list()) if not participants.is_empty() else set()
    summary = {
        "participant_rows": participants.height,
        "flows_daily_rows": daily.height,
        "sessions_covered": len(covered),
        "first_session": min(covered).isoformat() if covered else None,
        "last_session": max(covered).isoformat() if covered else None,
        "parse_failures": [(d.isoformat(), reason) for d, reason in failures],
        "missing_sessions": [],
    }
    if expected_sessions is not None:
        missing = sorted(set(expected_sessions) - covered)
        summary["missing_sessions"] = [d.isoformat() for d in missing]
        summary["n_missing_sessions"] = len(missing)
    return summary


def write_fii_dii_cash(new_rows: pl.DataFrame, processed_dir: str | Path) -> Path:
    """Append to the cash series, keeping it in its OWN file.

    Never writes ``flows_daily.parquet``: that file spans the full window from the
    F&O archive, and clobbering it with a few months of cash data would null nearly
    every regime label downstream. The separate filename is the guard.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / FII_DII_CASH_FILENAME
    combined = new_rows
    if path.exists():
        existing = pl.read_parquet(path)
        combined = pl.concat([existing, new_rows], how="vertical")
    combined = (
        combined.select([pl.col(n).cast(dt).alias(n) for n, dt in FII_DII_CASH_SCHEMA.items()])
        .unique(subset=["date", "participant"], keep="last")
        .sort(["date", "participant"])
    )
    combined.write_parquet(path)
    return path
