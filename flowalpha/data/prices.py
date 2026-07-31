"""Price panel construction.

Two ingestion paths exist:

``parse_bhavcopy``
    Parses NSE's ``sec_bhavdata_full`` daily archive. This is the only free source
    that carries **delivery percentage**, which is why it is supported at all -- the
    delivery factor is inert without it.

``process_prices``
    Normalises a raw OHLCV panel (whatever the source) into the canonical schema:
    derives ``adj_factor = adj_close / close`` and applies it to open/high/low.

Adjustment policy
-----------------
``volume`` is deliberately left unadjusted and ``turnover`` is computed as
``close * volume`` from the *raw* series. That product is split-invariant, so it is
the actual rupee value traded -- which is what Amihud illiquidity and ADV
participation caps need. Adjusting both price and volume would double-count.

The adjustment itself is proportional (a single factor applied to the whole bar),
not a reconstruction of each corporate action. That approximation is recorded as a
standing provenance caveat.
"""

from __future__ import annotations

import datetime as _dt
import io
from pathlib import Path

import polars as pl

#: Canonical price-panel schema. Every consumer may rely on these names and dtypes.
PRICES_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "symbol": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "adj_close": pl.Float64,
    "volume": pl.Float64,
    "adj_factor": pl.Float64,
    "adj_open": pl.Float64,
    "adj_high": pl.Float64,
    "adj_low": pl.Float64,
    "turnover": pl.Float64,
    "delivery_pct": pl.Float64,
}

PRICES_FILENAME = "prices.parquet"


class PriceParseError(Exception):
    """Raised when a price payload is not the format it claims to be."""


def empty_prices() -> pl.DataFrame:
    """An empty price panel with the full explicit schema.

    Built from the schema rather than from empty Python lists: an untyped empty
    frame gets ``Null``-typed columns, and joining a ``Null`` date column against a
    real ``Date`` key raises ``SchemaError`` deep inside an unrelated module.
    """
    return pl.DataFrame(schema=PRICES_SCHEMA)


def _decode(content: bytes | str) -> str:
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return content


def _reject_html(text: str, what: str) -> None:
    """NSE serves a styled HTML error page for missing archives, with HTTP 200-ish
    handling in some clients. Treating that page as data silently poisons a panel."""
    head = text.lstrip()[:400].lower()
    if head.startswith("<!doctype html") or head.startswith("<html") or "<head>" in head:
        raise PriceParseError(f"{what}: response is an HTML error page, not data")


def parse_bhavcopy(
    content: bytes | str,
    *,
    session_date: _dt.date | None = None,
    series: tuple[str, ...] = ("EQ",),
) -> pl.DataFrame:
    """Parse an NSE ``sec_bhavdata_full`` CSV into the canonical schema.

    Parameters
    ----------
    content:
        Raw file bytes or text.
    session_date:
        Override for the session date. The file carries ``DATE1`` per row, so this is
        only needed when that column is absent or unparseable.
    series:
        Which NSE series to keep. ``EQ`` only, by default: ``BE``/``BZ`` names sit in
        trade-to-trade segments with different microstructure.

    Notes
    -----
    Column names in this file carry **leading spaces** (`` SERIES``, `` DELIV_PER``),
    and ``DELIV_QTY``/``DELIV_PER`` are literal ``-`` for names with no delivery
    reporting on that day. Both are handled here rather than at every call site.
    """
    text = _decode(content)
    _reject_html(text, "bhavcopy")
    if not text.strip():
        raise PriceParseError("bhavcopy: empty payload")

    frame = pl.read_csv(
        io.StringIO(text),
        infer_schema_length=0,  # read everything as text; we coerce explicitly below
        truncate_ragged_lines=True,
    )
    frame = frame.rename({c: c.strip().upper() for c in frame.columns})

    required = {"SYMBOL", "SERIES", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE", "TTL_TRD_QNTY"}
    missing = required - set(frame.columns)
    if missing:
        raise PriceParseError(f"bhavcopy: missing columns {sorted(missing)}")

    frame = frame.with_columns(
        pl.col("SYMBOL").str.strip_chars().alias("symbol"),
        pl.col("SERIES").str.strip_chars().alias("series"),
    ).filter(pl.col("series").is_in(list(series)))

    if session_date is not None:
        frame = frame.with_columns(pl.lit(session_date).cast(pl.Date).alias("date"))
    elif "DATE1" in frame.columns:
        frame = frame.with_columns(
            pl.col("DATE1").str.strip_chars().str.to_date("%d-%b-%Y", strict=False).alias("date")
        )
    else:
        raise PriceParseError("bhavcopy: no DATE1 column and no session_date supplied")

    if frame["date"].null_count() == frame.height and frame.height:
        raise PriceParseError("bhavcopy: DATE1 could not be parsed as %d-%b-%Y")

    def num(col: str) -> pl.Expr:
        return (
            pl.col(col).str.strip_chars().replace({"-": None, "": None}).cast(pl.Float64, strict=False)
        )

    out = frame.with_columns(
        num("OPEN_PRICE").alias("open"),
        num("HIGH_PRICE").alias("high"),
        num("LOW_PRICE").alias("low"),
        num("CLOSE_PRICE").alias("close"),
        num("TTL_TRD_QNTY").alias("volume"),
        (num("DELIV_PER") if "DELIV_PER" in frame.columns else pl.lit(None, pl.Float64)).alias("delivery_pct"),
        # TURNOVER_LACS is in lakhs of rupees; canonical turnover is plain rupees.
        (
            (num("TURNOVER_LACS") * 1e5) if "TURNOVER_LACS" in frame.columns else pl.lit(None, pl.Float64)
        ).alias("turnover_reported"),
    )

    out = out.with_columns(
        # Bhavcopy prices are unadjusted; adj_close is filled by process_prices when
        # an adjusted series is available, and equals close otherwise.
        pl.col("close").alias("adj_close"),
        pl.coalesce(pl.col("turnover_reported"), pl.col("close") * pl.col("volume")).alias("turnover"),
    )
    return _finalise(out)


def process_prices(
    raw: pl.DataFrame,
    *,
    require_adj_close: bool = True,
) -> pl.DataFrame:
    """Normalise a raw OHLCV panel into :data:`PRICES_SCHEMA`.

    Expects at least ``date, symbol, open, high, low, close, volume`` and, unless
    ``require_adj_close`` is False, ``adj_close``.

    Rows with a non-positive or null ``close`` are dropped: they cannot produce a
    usable adjustment factor, and keeping them would let a divide-by-zero propagate
    an ``inf`` adjustment through the whole symbol's history.
    """
    needed = ["date", "symbol", "open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise PriceParseError(f"process_prices: missing columns {missing}")
    if require_adj_close and "adj_close" not in raw.columns:
        raise PriceParseError("process_prices: missing adj_close (pass require_adj_close=False to skip)")

    frame = raw
    if "adj_close" not in frame.columns:
        frame = frame.with_columns(pl.col("close").alias("adj_close"))
    if "delivery_pct" not in frame.columns:
        # Not an error: Yahoo has no delivery data. Recorded as `unavailable` in
        # provenance so the inert delivery factor is visible rather than mysterious.
        frame = frame.with_columns(pl.lit(None, pl.Float64).alias("delivery_pct"))

    frame = frame.with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("symbol").cast(pl.Utf8).str.strip_chars(),
        *[pl.col(c).cast(pl.Float64) for c in ("open", "high", "low", "close", "adj_close", "volume", "delivery_pct")],
    )

    frame = frame.filter(
        pl.col("date").is_not_null()
        & pl.col("symbol").is_not_null()
        & (pl.col("symbol") != "")
        & pl.col("close").is_not_null()
        & (pl.col("close") > 0)
        & pl.col("adj_close").is_not_null()
    )

    frame = frame.with_columns((pl.col("adj_close") / pl.col("close")).alias("adj_factor"))
    frame = frame.with_columns(
        (pl.col("open") * pl.col("adj_factor")).alias("adj_open"),
        (pl.col("high") * pl.col("adj_factor")).alias("adj_high"),
        (pl.col("low") * pl.col("adj_factor")).alias("adj_low"),
        # Raw close x raw volume: split-invariant rupee value actually traded.
        (pl.col("close") * pl.col("volume")).alias("turnover"),
    )
    return _finalise(frame)


def _finalise(frame: pl.DataFrame) -> pl.DataFrame:
    """Fill any absent canonical column, order columns, dedupe and sort."""
    for name, dtype in PRICES_SCHEMA.items():
        if name not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype).alias(name))
    if "adj_factor" in frame.columns:
        frame = frame.with_columns(
            pl.when(pl.col("adj_factor").is_null())
            .then(pl.lit(1.0))
            .otherwise(pl.col("adj_factor"))
            .alias("adj_factor")
        )
    for name in ("adj_open", "adj_high", "adj_low"):
        base = name.removeprefix("adj_")
        frame = frame.with_columns(
            pl.when(pl.col(name).is_null())
            .then(pl.col(base) * pl.col("adj_factor"))
            .otherwise(pl.col(name))
            .alias(name)
        )
    frame = frame.select(
        [pl.col(name).cast(dtype).alias(name) for name, dtype in PRICES_SCHEMA.items()]
    )
    return frame.unique(subset=["date", "symbol"], keep="last").sort(["symbol", "date"])


def write_prices(prices: pl.DataFrame, processed_dir: str | Path) -> Path:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / PRICES_FILENAME
    prices.write_parquet(path)
    return path


def read_prices(processed_dir: str | Path) -> pl.DataFrame:
    path = Path(processed_dir) / PRICES_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/download_yahoo.py (or ingest bhavcopies) first."
        )
    return pl.read_parquet(path)


def daily_returns(prices: pl.DataFrame, price_col: str = "adj_close") -> pl.DataFrame:
    """Per-symbol simple daily returns from the adjusted close.

    Returns ``(date, symbol, ret)``. Uses the adjusted series so dividends and
    splits do not show up as one-day return outliers.
    """
    if prices.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "ret": pl.Float64})
    return (
        prices.sort(["symbol", "date"])
        .with_columns(
            (pl.col(price_col) / pl.col(price_col).shift(1).over("symbol") - 1.0).alias("ret")
        )
        .select("date", "symbol", "ret")
    )
