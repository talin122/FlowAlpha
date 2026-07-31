"""Persisted portfolio state.

Yesterday's holdings are an *input* to today's trade list. Without them, every daily
run would compute trades as if starting from cash, reporting a full-book turnover every
day and a cost figure that bears no relation to what would actually be paid.

State is stored as JSON next to the daily outputs, keyed by date, so a day's run can be
re-derived and audited from the files alone.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

HOLDINGS_FILENAME = "holdings.json"

HOLDINGS_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.Utf8,
    "weight": pl.Float64,
}


@dataclass
class Holdings:
    """A dated set of target weights."""

    as_of: _dt.date | None
    weights: dict[str, float] = field(default_factory=dict)
    #: Free-form provenance of this state, e.g. the run label that produced it.
    source: str = ""

    @property
    def gross(self) -> float:
        return sum(abs(w) for w in self.weights.values())

    @property
    def net(self) -> float:
        return sum(self.weights.values())

    def to_frame(self) -> pl.DataFrame:
        if not self.weights:
            return pl.DataFrame(schema=HOLDINGS_SCHEMA)
        items = sorted(self.weights.items())
        return pl.DataFrame(
            {"symbol": [k for k, _ in items], "weight": [v for _, v in items]},
            schema=HOLDINGS_SCHEMA,
        )

    @classmethod
    def from_frame(
        cls, frame: pl.DataFrame, as_of: _dt.date | None = None, source: str = ""
    ) -> "Holdings":
        if frame.is_empty():
            return cls(as_of=as_of, weights={}, source=source)
        return cls(
            as_of=as_of,
            weights={
                str(s): float(w)
                for s, w in zip(frame["symbol"].to_list(), frame["weight"].to_list())
                if w is not None
            },
            source=source,
        )

    @classmethod
    def empty(cls) -> "Holdings":
        """The starting state: flat.

        Returned when no state file exists, so a first run reports a full initial build
        -- which is what genuinely happens -- rather than pretending to be flat-to-flat.
        """
        return cls(as_of=None, weights={}, source="no prior state (first run)")

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "source": self.source,
            "gross": self.gross,
            "net": self.net,
            "weights": dict(sorted(self.weights.items())),
        }


def state_path(results_dir: str | Path) -> Path:
    return Path(results_dir) / HOLDINGS_FILENAME


def load_holdings(results_dir: str | Path) -> Holdings:
    """Read persisted holdings, or the flat state when none exist."""
    path = state_path(results_dir)
    if not path.exists():
        return Holdings.empty()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return Holdings.empty()
    as_of = raw.get("as_of")
    return Holdings(
        as_of=_dt.date.fromisoformat(as_of) if as_of else None,
        weights={str(k): float(v) for k, v in (raw.get("weights") or {}).items()},
        source=str(raw.get("source", "")),
    )


def save_holdings(holdings: Holdings, results_dir: str | Path) -> Path:
    path = state_path(results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(holdings.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def archive_holdings(holdings: Holdings, daily_dir: str | Path) -> Path:
    """Write a dated copy alongside the day's digest, so history is auditable."""
    path = Path(daily_dir) / HOLDINGS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(holdings.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path
