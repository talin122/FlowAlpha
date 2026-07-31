"""Regime overlay: gate factors on and off by conditional IC.

A factor is switched **ON** for session ``t`` when its trailing *conditional* IC
t-statistic, measured within the regime bucket that ``t`` is currently in, exceeds
``signals.regime_overlay.gate_t_threshold``.

The map from factor to regime lives in ``config.yaml`` and is a **declared hypothesis,
not a search result**. Nobody scanned factor-regime pairs looking for the best pair;
momentum is mapped to FII flow and reversal to retail extremes because those are the
two effects the project set out to test. This matters because a gate chosen by search
would need its own multiple-testing correction on top of everything else, and the
correction would swamp the effect.

Point-in-time, as everywhere: the gate on ``t`` uses only IC observations whose labels
had realised by ``t`` (``usable_from <= t``), within the trailing window.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import polars as pl

from ..config import Config
from ..conditioning.regimes import RegimeSet
from ..validation.ic import newey_west_se, trailing_ic_series


@dataclass
class GateDecision:
    """One factor's gate state on one session."""

    date: _dt.date
    factor: str
    regime_column: str
    bucket: str | None
    t_stat: float
    n_obs: int
    on: bool


@dataclass
class OverlayResult:
    """Gate decisions and the gated panels."""

    #: (date, factor, regime_column, bucket, t_stat, n_obs, on)
    decisions: pl.DataFrame
    panels: dict[str, pl.DataFrame]
    gate_t_threshold: float
    factor_regime_map: dict[str, str]
    #: Fraction of dates each mapped factor was ON. Reported because a gate that is
    #: almost never on produces a strategy that is almost never invested.
    on_fraction: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "gate_t_threshold": self.gate_t_threshold,
            "factor_regime_map": dict(self.factor_regime_map),
            "on_fraction": dict(self.on_fraction),
            "n_decisions": self.decisions.height,
            "declared_not_mined": True,
        }


def resolve_map(cfg: Config, factor_names: Sequence[str]) -> dict[str, str]:
    """Expand the declared prefix map onto concrete factor names.

    ``reversal -> retail_extreme`` covers ``reversal_5d`` and ``reversal_21d`` without
    the config repeating the windows it already declares under ``factors.reversal``.
    """
    declared = cfg.get("signals.regime_overlay.factor_regime_map", {}) or {}
    out: dict[str, str] = {}
    for prefix, regime_col in declared.items():
        for name in factor_names:
            if name == prefix or name.startswith(str(prefix)):
                out[name] = str(regime_col)
    return out


def compute_gates(
    panels: Mapping[str, pl.DataFrame],
    fwd: pl.DataFrame,
    cfg: Config,
    regimes: RegimeSet,
    *,
    sessions: Sequence[_dt.date] | None = None,
    min_obs: int = 20,
) -> OverlayResult:
    """Decide, per session and per mapped factor, whether the factor is ON.

    Unmapped factors are passed through untouched: the overlay is a hypothesis about
    two specific effects, not a general-purpose filter.
    """
    threshold = float(cfg.get("signals.regime_overlay.gate_t_threshold", 1.5))
    window = int(cfg.get("signals.composite.ic_window", 252))
    horizon = int(cfg.get("signals.composite.ic_horizon", 5))
    mapping = resolve_map(cfg, sorted(panels))

    if sessions is None:
        sessions = sorted(set(fwd["date"].to_list()))
    sessions = list(sessions)
    index = {d: i for i, d in enumerate(sessions)}

    decisions: list[dict] = []
    gated: dict[str, pl.DataFrame] = {}

    for name, panel in panels.items():
        regime_col = mapping.get(name)
        if regime_col is None or regime_col not in regimes.columns:
            gated[name] = panel
            continue

        series = trailing_ic_series(panel, fwd, horizon)
        if series.is_empty():
            gated[name] = panel
            continue
        labelled = regimes.attach(series, regime_col)
        if labelled.is_empty():
            gated[name] = panel
            continue

        ic_pos = np.array([index.get(d, -1) for d in labelled["date"].to_list()])
        usable = np.array([index.get(d, len(sessions)) for d in labelled["usable_from"].to_list()])
        values = np.asarray(labelled["ic"].to_list(), dtype=float)
        buckets = np.array(labelled["regime"].to_list(), dtype=object)
        keep = ic_pos >= 0
        ic_pos, usable, values, buckets = ic_pos[keep], usable[keep], values[keep], buckets[keep]

        today_bucket = {
            d: b for d, b in zip(
                regimes.labels(regime_col)["date"].to_list(),
                regimes.labels(regime_col)["regime"].to_list(),
            )
        }

        on_dates: list[_dt.date] = []
        for t, day in enumerate(sessions):
            bucket = today_bucket.get(day)
            if bucket is None:
                decisions.append({
                    "date": day, "factor": name, "regime_column": regime_col,
                    "bucket": None, "t_stat": float("nan"), "n_obs": 0, "on": False,
                })
                continue
            mask = (
                (usable <= t) & (ic_pos > t - window) & (ic_pos <= t) & (buckets == bucket)
            )
            sample = values[mask]
            sample = sample[np.isfinite(sample)]
            if sample.size < min_obs:
                decisions.append({
                    "date": day, "factor": name, "regime_column": regime_col,
                    "bucket": str(bucket), "t_stat": float("nan"),
                    "n_obs": int(sample.size), "on": False,
                })
                continue
            se, _ = newey_west_se(sample, max(horizon - 1, 1))
            t_stat = float(sample.mean() / se) if se and np.isfinite(se) and se > 0 else float("nan")
            is_on = bool(np.isfinite(t_stat) and t_stat >= threshold)
            if is_on:
                on_dates.append(day)
            decisions.append({
                "date": day, "factor": name, "regime_column": regime_col,
                "bucket": str(bucket), "t_stat": t_stat,
                "n_obs": int(sample.size), "on": is_on,
            })

        on_set = pl.Series("date", on_dates, dtype=pl.Date)
        gated[name] = panel.with_columns(
            pl.when(pl.col("date").is_in(on_set)).then(pl.col("value")).otherwise(0.0).alias("value")
        )

    frame = (
        pl.DataFrame(
            decisions,
            schema={
                "date": pl.Date, "factor": pl.Utf8, "regime_column": pl.Utf8,
                "bucket": pl.Utf8, "t_stat": pl.Float64, "n_obs": pl.Int64, "on": pl.Boolean,
            },
        )
        if decisions
        else pl.DataFrame(
            schema={
                "date": pl.Date, "factor": pl.Utf8, "regime_column": pl.Utf8,
                "bucket": pl.Utf8, "t_stat": pl.Float64, "n_obs": pl.Int64, "on": pl.Boolean,
            }
        )
    )
    on_fraction = {}
    if not frame.is_empty():
        for name in frame["factor"].unique().to_list():
            sub = frame.filter(pl.col("factor") == name)
            on_fraction[name] = sub.filter(pl.col("on")).height / sub.height
    return OverlayResult(
        decisions=frame, panels=gated, gate_t_threshold=threshold,
        factor_regime_map=mapping, on_fraction=on_fraction,
    )


def gate_summary(result: OverlayResult) -> list[str]:
    """ASCII lines describing the gates, for console and report."""
    lines = [
        f"  gate threshold: trailing conditional IC t-stat >= {result.gate_t_threshold}",
        "  factor -> regime map is DECLARED in config.yaml, not selected by search:",
    ]
    for factor, regime in sorted(result.factor_regime_map.items()):
        frac = result.on_fraction.get(factor)
        share = f"{frac:.1%}" if frac is not None else "n/a"
        lines.append(f"    {factor:<20} gated on {regime:<16} ON {share} of sessions")
    if not result.factor_regime_map:
        lines.append("    (no factors mapped -- overlay is inert)")
    return lines
