"""Shared pipeline assembly for the run_* scripts.

Loads one as-of snapshot through the point-in-time store, computes factor panels and
forward returns, and evaluates factors into :class:`FactorCard` objects with their
Deflated Sharpe deflated against the **cumulative** trial registry.

Every stage that evaluates a strategy registers a trial before computing its DSR, so the
deflation always reflects the number of looks taken including the current one.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from _cli import REPO_ROOT, say
from flowalpha.backtest.costs import CostModel, break_even_aum, rolling_adv
from flowalpha.backtest.engine import BacktestResult, run_backtest
from flowalpha.config import Config
from flowalpha.data.prices import read_prices
from flowalpha.data.store import PointInTimeStore
from flowalpha.factors.base import FactorContext, _load_shares
from flowalpha.factors.library import build_panels, default_library
from flowalpha.signals.factor_card import FactorCard, cards_to_frame
from flowalpha.validation.deflated_sharpe import (
    TrialRegistry,
    deflated_sharpe_ratio,
    pbo_cscv,
)
from flowalpha.validation.ic import forward_returns, rank_ic, summarize_ic


@dataclass
class Pipeline:
    """One loaded snapshot, ready to evaluate."""

    cfg: Config
    store: PointInTimeStore
    ctx: FactorContext
    prices: pl.DataFrame
    fwd: pl.DataFrame
    panels: dict[str, pl.DataFrame]
    skipped: list[str]
    registry: TrialRegistry
    shares_available: bool

    @property
    def horizons(self) -> list[int]:
        return [int(h) for h in self.cfg["forward_returns"]["horizons"]]

    @property
    def sessions(self) -> list[_dt.date]:
        return list(self.ctx.sessions)


def load_pipeline(cfg: Config, *, as_of_date: _dt.date | None = None, verbose: bool = True) -> Pipeline:
    """Build the standard snapshot: store -> context -> panels -> forward returns."""
    store = PointInTimeStore.from_config(cfg)
    as_of = as_of_date or cfg.end_date
    ctx = FactorContext.from_store(store, cfg, as_of_date=as_of)
    if ctx.n_sessions == 0:
        raise SystemExit("no price data available through the store; run the download stages first")

    _, shares_available = _load_shares(cfg.path("reference"))
    factors = default_library(cfg, shares_available=shares_available)
    panels, skipped = build_panels(factors, ctx, verbose=verbose)

    prices = read_prices(cfg.path("processed")).filter(
        (pl.col("date") >= cfg.start_date) & (pl.col("date") <= as_of)
    )
    fwd = forward_returns(prices, [int(h) for h in cfg["forward_returns"]["horizons"]])

    if verbose:
        say(f"snapshot as of {as_of}: {ctx.n_sessions} sessions, {ctx.n_symbols} symbols")
        say(f"factors computed: {len(panels)} ({', '.join(sorted(panels))})")
        if skipped:
            say(f"factors skipped:  {len(skipped)}")
            for reason in skipped:
                say(f"  - {reason}")
        say(f"shares_outstanding usable: {shares_available} "
            f"(size factor is {'log market cap' if shares_available else 'a LOG-PRICE PROXY'})")

    return Pipeline(
        cfg=cfg, store=store, ctx=ctx, prices=prices, fwd=fwd,
        panels=panels, skipped=skipped,
        registry=TrialRegistry.load(cfg.path("results")),
        shares_available=shares_available,
    )


def evaluate_panel(
    pipe: Pipeline,
    name: str,
    panel: pl.DataFrame,
    *,
    hypothesis: str = "",
    kind: str = "factor",
    notional: float = 1_000_000_000.0,
    register: bool = True,
) -> tuple[FactorCard, BacktestResult]:
    """IC across horizons, backtest, and DSR against the cumulative trial count."""
    cfg = pipe.cfg
    ic_entries = []
    for h in pipe.horizons:
        series = rank_ic(panel, pipe.fwd, h)
        ic_entries.append(summarize_ic(series, factor=name, horizon=h))

    result = run_backtest(panel, pipe.prices, cfg, notional=notional)

    if register:
        pipe.registry.register(name, kind=kind, detail={"notional": notional})
    n_trials = max(1, pipe.registry.n_trials)
    dsr = deflated_sharpe_ratio(
        result.net_returns,
        n_trials,
        benchmark_sr_annual=float(cfg.get("validation.deflated_sharpe.benchmark_sr", 0.0)),
    )
    card = FactorCard(
        name=name, hypothesis=hypothesis, ic=ic_entries,
        backtest=result.to_dict(), dsr=dsr, n_trials=n_trials,
    )
    return card, result


def evaluate_library(
    pipe: Pipeline,
    *,
    notional: float = 1_000_000_000.0,
    verbose: bool = True,
    register: bool = True,
) -> tuple[list[FactorCard], dict[str, BacktestResult]]:
    """Evaluate every computed factor. Skipped factors are NOT registered as trials."""
    hypotheses = {
        f.name: f.hypothesis
        for f in default_library(pipe.cfg, shares_available=pipe.shares_available)
    }
    cards: list[FactorCard] = []
    results: dict[str, BacktestResult] = {}
    for name in sorted(pipe.panels):
        card, result = evaluate_panel(
            pipe, name, pipe.panels[name],
            hypothesis=hypotheses.get(name, ""), notional=notional, register=register,
        )
        cards.append(card)
        results[name] = result
        if verbose:
            for line in card.summary_lines():
                say(line)
    return cards, results


#: Notional at which square-root impact is numerically negligible, so the resulting
#: net return reflects the FIXED costs only (STT, stamp duty, brokerage). Fixed costs
#: are proportional to notional and therefore capacity-independent; impact is not. A
#: strategy that fails even here fails for reasons no amount of size discipline fixes.
FIXED_COST_ONLY_NOTIONAL = 1.0


def evaluate_library_two_scales(
    pipe: Pipeline, notional: float, *, verbose: bool = True
) -> tuple[list[FactorCard], dict[str, BacktestResult], list[FactorCard], dict[str, BacktestResult]]:
    """Evaluate at the stated notional and again with fixed costs only.

    Reporting both is what makes the cost story legible: the first answers "is this
    tradeable at this size", the second answers "does the signal survive taxes and
    brokerage at all". Only the first pass registers trials -- the second is the same
    strategy measured a different way, not another look at the data.
    """
    cards, results = evaluate_library(pipe, notional=notional, verbose=verbose)
    fixed_cards, fixed_results = evaluate_library(
        pipe, notional=FIXED_COST_ONLY_NOTIONAL, verbose=False, register=False
    )
    return cards, results, fixed_cards, fixed_results


def pbo_over_factors(results: Mapping[str, BacktestResult], cfg: Config) -> dict | None:
    """PBO across the factor set, treating each factor as one configuration.

    Series of differing length are right-aligned to the shortest, since CSCV needs a
    rectangular matrix. The trim is reported so a shortened window is visible.
    """
    import numpy as np

    usable = {k: v for k, v in results.items() if v.n_days > 0}
    if len(usable) < 2:
        return None
    names = sorted(usable)
    common = min(usable[n].n_days for n in names)
    matrix = np.column_stack([usable[n].net_returns[-common:] for n in names])
    try:
        out = pbo_cscv(matrix, n_partitions=int(cfg.get("validation.pbo.n_partitions", 16)))
    except ValueError as exc:
        return {"error": str(exc), "configs": names}
    d = out.to_dict()
    d["configs"] = names
    d["n_periods_used"] = common
    d["n_periods_trimmed"] = max(usable[n].n_days for n in names) - common
    return d


def capacity_note(
    result: BacktestResult, prices: pl.DataFrame, cfg: Config
) -> dict:
    """Break-even AUM for a backtest, using median trailing ADV as the liquidity scale."""
    model = CostModel.from_config(cfg)
    adv = rolling_adv(prices, model.adv_window)
    median_adv = float(adv["adv"].median() or 0.0)
    aum = break_even_aum(
        result.gross_annual_return if result.n_days else 0.0,
        result.annual_turnover if result.n_days else 0.0,
        model,
        mean_adv=median_adv,
        gross_leverage=float(cfg["backtest"]["gross_leverage"]),
    )
    return {
        "median_trailing_adv_rupees": median_adv,
        "break_even_aum_rupees": aum,
        "break_even_aum_crore": (aum / 1e7) if aum not in (float("inf"),) else float("inf"),
    }


def write_json(payload, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def cards_table(cards: Sequence[FactorCard], horizon: int) -> pl.DataFrame:
    return cards_to_frame(cards, horizon)


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:  # pragma: no cover - defensive
        return str(path)
