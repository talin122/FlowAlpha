"""Per-factor summary cards.

A card is everything a reader needs to judge one factor at once: what it bets on, its
IC across horizons with autocorrelation-adjusted t-statistics, its backtest Sharpe both
gross and net, the trial count its Deflated Sharpe was deflated against, and its
conditional IC by regime.

The cards deliberately keep the raw and deflated numbers adjacent and labelled. Quoting
a Sharpe without its trial count is the failure this whole module exists to prevent, so
the data structure makes them inseparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import polars as pl

from ..conditioning.analysis import ConditionalIC
from ..validation.deflated_sharpe import DeflatedSharpeResult
from ..validation.ic import ICSummary


@dataclass
class FactorCard:
    """Everything reportable about one factor."""

    name: str
    hypothesis: str
    ic: list[ICSummary] = field(default_factory=list)
    conditional_ic: list[ConditionalIC] = field(default_factory=list)
    backtest: dict | None = None
    dsr: DeflatedSharpeResult | None = None
    n_trials: int = 0
    notes: list[str] = field(default_factory=list)

    # -- accessors ----------------------------------------------------------
    def ic_at(self, horizon: int) -> ICSummary | None:
        for entry in self.ic:
            if entry.horizon == horizon:
                return entry
        return None

    @property
    def headline_ic(self) -> ICSummary | None:
        """The shortest available horizon, used for one-line summaries."""
        return min(self.ic, key=lambda s: s.horizon) if self.ic else None

    def to_dict(self) -> dict:
        return {
            "factor": self.name,
            "hypothesis": self.hypothesis,
            "ic": [s.to_dict() for s in self.ic],
            "conditional_ic": [c.to_dict() for c in self.conditional_ic],
            "backtest": dict(self.backtest) if self.backtest else None,
            "deflated_sharpe": self.dsr.to_dict() if self.dsr else None,
            "n_trials_at_evaluation": self.n_trials,
            "notes": list(self.notes),
        }

    # -- presentation -------------------------------------------------------
    def summary_lines(self) -> list[str]:
        """ASCII lines for console output."""
        out = [f"  {self.name}", f"    hypothesis: {self.hypothesis}"]
        for entry in sorted(self.ic, key=lambda s: s.horizon):
            out.append(
                f"    IC h={entry.horizon:<3} mean={_fmt(entry.mean, 4)} "
                f"IR={_fmt(entry.ic_ir, 3)} t(NW)={_fmt(entry.t_stat, 2)} "
                f"hit={_fmt(entry.hit_rate, 3)} n={entry.n_dates} [RAW, not deflated]"
            )
        if self.backtest:
            out.append(
                f"    backtest  gross SR={_fmt(self.backtest.get('gross_sharpe'), 2)} "
                f"net SR={_fmt(self.backtest.get('net_sharpe'), 2)} "
                f"turnover={_fmt(self.backtest.get('annual_turnover'), 1)}x/yr "
                f"cost={_fmt(self.backtest.get('total_cost_bps'), 0)}bps/yr"
            )
        if self.dsr:
            out.append(
                f"    DSR={_fmt(self.dsr.dsr, 3)} against {self.dsr.n_trials} cumulative "
                f"trials (raw annual SR {_fmt(self.dsr.sharpe_annual, 2)})"
            )
        for entry in self.conditional_ic:
            out.append(
                f"    cond IC {entry.regime_column}={entry.bucket:<10} "
                f"mean={_fmt(entry.mean_ic, 4)} t(NW)={_fmt(entry.t_stat, 2)} "
                f"n={entry.n_dates}"
            )
        for note in self.notes:
            out.append(f"    note: {note}")
        return out


def _fmt(value, digits: int) -> str:
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(f):
        return "n/a"
    return f"{f:.{digits}f}"


def cards_to_frame(cards: Sequence[FactorCard], horizon: int) -> pl.DataFrame:
    """Flatten cards into one tidy table at a chosen horizon.

    Every Sharpe column is accompanied by ``n_trials`` and ``dsr`` in the same row, so a
    reader cannot pick up the Sharpe without the correction beside it.
    """
    rows = []
    for card in cards:
        entry = card.ic_at(horizon)
        bt = card.backtest or {}
        rows.append(
            {
                "factor": card.name,
                "horizon": horizon,
                "mean_ic": entry.mean if entry else float("nan"),
                "ic_ir": entry.ic_ir if entry else float("nan"),
                "t_stat_newey_west_raw": entry.t_stat if entry else float("nan"),
                "hit_rate": entry.hit_rate if entry else float("nan"),
                "n_dates": entry.n_dates if entry else 0,
                "gross_sharpe": float(bt.get("gross_sharpe", float("nan"))),
                "net_sharpe": float(bt.get("net_sharpe", float("nan"))),
                "annual_turnover": float(bt.get("annual_turnover", float("nan"))),
                "cost_bps_per_year": float(bt.get("total_cost_bps", float("nan"))),
                "n_trials": card.n_trials,
                "deflated_sharpe_prob": card.dsr.dsr if card.dsr else float("nan"),
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={
                "factor": pl.Utf8, "horizon": pl.Int64, "mean_ic": pl.Float64,
                "ic_ir": pl.Float64, "t_stat_newey_west_raw": pl.Float64,
                "hit_rate": pl.Float64, "n_dates": pl.Int64, "gross_sharpe": pl.Float64,
                "net_sharpe": pl.Float64, "annual_turnover": pl.Float64,
                "cost_bps_per_year": pl.Float64, "n_trials": pl.Int64,
                "deflated_sharpe_prob": pl.Float64,
            }
        )
    return pl.DataFrame(rows).sort("factor")
