"""Regime label access.

:class:`RegimeSet` is a thin, typed view over the ``flow_features`` panel. It exists so
that conditioning code never re-derives a regime: the labels were built once, with
expanding windows and the flow lag baked in, and any second implementation would
eventually disagree with the first.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
import polars as pl

from ..config import Config
from ..data.store import PointInTimeStore
from ..factors.flow_features import REGIME_COLUMNS


class RegimeError(Exception):
    """Raised for an unknown regime column or an empty regime panel."""


@dataclass
class RegimeSet:
    """Date-indexed regime labels.

    Parameters
    ----------
    features:
        The ``flow_features`` panel: one row per session, one column per regime.
    """

    features: pl.DataFrame

    @classmethod
    def from_store(
        cls,
        store: PointInTimeStore,
        cfg: Config,
        *,
        as_of_date: _dt.date | None = None,
    ) -> "RegimeSet":
        """Read ``flow_features`` through the store at lag 0.

        Lag 0 is correct and deliberate: the row stamped ``d`` was built from flow
        through ``d-1``, so the snapshot for ``d`` is readable on ``d``. See
        :mod:`flowalpha.data.store`.
        """
        features = store.get("flow_features", as_of_date=as_of_date or cfg.end_date)
        return cls(features=features)

    # -- introspection ------------------------------------------------------
    @property
    def columns(self) -> tuple[str, ...]:
        """Regime columns actually present in the panel."""
        return tuple(c for c in REGIME_COLUMNS if c in self.features.columns)

    def _require(self, column: str) -> None:
        if column not in self.features.columns:
            raise RegimeError(
                f"unknown regime column {column!r}; available: {list(self.columns)}"
            )

    def buckets(self, column: str) -> list[str]:
        """Distinct non-null labels, sorted for stable report ordering."""
        self._require(column)
        values = (
            self.features.filter(pl.col(column).is_not_null())[column].unique().to_list()
        )
        return sorted(str(v) for v in values)

    def labels(self, column: str) -> pl.DataFrame:
        """``(date, regime)`` for dates where the label is defined."""
        self._require(column)
        return (
            self.features.filter(pl.col(column).is_not_null())
            .select(pl.col("date"), pl.col(column).cast(pl.Utf8).alias("regime"))
            .sort("date")
        )

    def dates_in(self, column: str, bucket: str | bool) -> list[_dt.date]:
        """Sessions whose label equals ``bucket``."""
        self._require(column)
        return (
            self.features.filter(pl.col(column).cast(pl.Utf8) == str(bucket))["date"]
            .sort()
            .to_list()
        )

    def label_on(self, column: str, day: _dt.date) -> str | None:
        """The label for one session, or None when undefined."""
        self._require(column)
        row = self.features.filter(pl.col("date") == day)
        if row.is_empty() or row[column][0] is None:
            return None
        return str(row[column][0])

    def coverage(self, column: str) -> dict:
        """Per-bucket session counts plus the first and last defined dates."""
        self._require(column)
        defined = self.features.filter(pl.col(column).is_not_null())
        if defined.is_empty():
            return {"n_defined": 0, "buckets": {}}
        counts = defined.group_by(pl.col(column).cast(pl.Utf8)).len().sort(column).rows()
        dates = defined["date"].to_list()
        return {
            "n_defined": defined.height,
            "first_defined": min(dates).isoformat(),
            "last_defined": max(dates).isoformat(),
            "buckets": {str(k): int(v) for k, v in counts},
        }

    def attach(
        self, panel: pl.DataFrame, column: str, *, out_col: str = "regime"
    ) -> pl.DataFrame:
        """Inner-join a regime label onto any date-indexed frame.

        An inner join on purpose: rows on dates with no defined regime are dropped
        rather than pooled into an "unknown" bucket. Pooling them would mix the early
        sample -- before the expanding window had enough observations -- into whichever
        bucket happened to absorb it.
        """
        self._require(column)
        labels = self.labels(column).rename({"regime": out_col})
        return panel.join(labels, on="date", how="inner")

    def summary(self) -> dict[str, dict]:
        return {col: self.coverage(col) for col in self.columns}
