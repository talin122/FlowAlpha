"""Daily long/short backtest engine.

Mechanics, stated precisely because the details determine whether the result means
anything:

* On session ``t`` the factor value known at ``t`` determines target weights.
* Those weights earn the return from ``t`` to ``t+1``. **No same-day return is
  earned**: a signal computed from ``t``'s close cannot capture ``t``'s move.
* With ``holding_days = h``, the book is a set of ``h`` overlapping sleeves, each
  rebalanced every ``h`` sessions and each carrying ``1/h`` of the gross. Daily P&L is
  the average across sleeves. This is how a 5-day holding period is run daily without
  either overlapping the same capital ``h`` times or trading the whole book every day.
* Costs are charged on the actual weight *changes*, on the day they occur.

The engine reports gross and net separately, plus turnover, so a strategy that only
works before costs is visibly that.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import polars as pl

from ..validation.deflated_sharpe import TRADING_DAYS, sharpe_ratio
from .costs import CostModel, rolling_adv


@dataclass
class BacktestResult:
    """Daily P&L series and summary statistics."""

    #: (date, gross_return, net_return, cost, turnover, n_long, n_short)
    daily: pl.DataFrame
    gross_sharpe: float
    net_sharpe: float
    gross_annual_return: float
    net_annual_return: float
    annual_turnover: float
    total_cost_bps: float
    n_days: int
    holding_days: int
    cost_detail: dict = field(default_factory=dict)

    @property
    def net_returns(self) -> np.ndarray:
        return np.asarray(self.daily["net_return"].to_list(), dtype=float)

    @property
    def gross_returns(self) -> np.ndarray:
        return np.asarray(self.daily["gross_return"].to_list(), dtype=float)

    def to_dict(self) -> dict:
        return {
            "gross_sharpe": self.gross_sharpe,
            "net_sharpe": self.net_sharpe,
            "gross_annual_return": self.gross_annual_return,
            "net_annual_return": self.net_annual_return,
            "annual_turnover": self.annual_turnover,
            "total_cost_bps": self.total_cost_bps,
            "n_days": self.n_days,
            "holding_days": self.holding_days,
            **self.cost_detail,
        }


def quintile_weights(
    values: np.ndarray,
    *,
    long_only: bool = False,
    gross_leverage: float = 1.0,
    n_quantiles: int = 5,
) -> np.ndarray:
    """Equal-weight long the top quintile, short the bottom, scaled to gross leverage.

    ``NaN`` factor values get zero weight. The long and short books are each scaled to
    half the gross, so the portfolio is dollar-neutral by construction rather than by
    luck of the cross-section.
    """
    out = np.zeros_like(values, dtype=float)
    finite = np.isfinite(values)
    n = int(finite.sum())
    if n < n_quantiles:
        return out
    idx = np.where(finite)[0]
    order = idx[np.argsort(values[idx], kind="stable")]
    k = max(1, n // n_quantiles)
    short_idx, long_idx = order[:k], order[-k:]

    if long_only:
        out[long_idx] = gross_leverage / len(long_idx)
        return out
    out[long_idx] = 0.5 * gross_leverage / len(long_idx)
    out[short_idx] = -0.5 * gross_leverage / len(short_idx)
    return out


def _panel_to_matrix(
    panel: pl.DataFrame, sessions: Sequence[_dt.date], symbols: Sequence[str], value_col: str
) -> np.ndarray:
    out = np.full((len(sessions), len(symbols)), np.nan)
    date_idx = {d: i for i, d in enumerate(sessions)}
    sym_idx = {s: j for j, s in enumerate(symbols)}
    for d, s, v in zip(panel["date"].to_list(), panel["symbol"].to_list(), panel[value_col].to_list()):
        i, j = date_idx.get(d), sym_idx.get(s)
        if i is not None and j is not None and v is not None:
            out[i, j] = float(v)
    return out


def run_backtest(
    panel: pl.DataFrame,
    prices: pl.DataFrame,
    cfg,
    *,
    cost_model: CostModel | None = None,
    holding_days: int | None = None,
    long_only: bool | None = None,
    gross_leverage: float | None = None,
    notional: float = 1.0,
    value_col: str = "value",
) -> BacktestResult:
    """Run the daily long/short backtest of one factor panel.

    Parameters
    ----------
    panel:
        ``(date, symbol, value)`` signal, already point-in-time.
    prices:
        Canonical price panel; ``adj_close`` drives returns and ``turnover`` drives ADV.
    notional:
        Book size in rupees, used only to scale impact costs. Weights are fractions of
        this, so with the default of 1.0 impact is negligible and the run is effectively
        a capacity-free upper bound -- pass a realistic AUM to see impact bite.
    """
    bcfg = cfg["backtest"]
    holding = int(holding_days if holding_days is not None else bcfg["holding_days"])
    leverage = float(gross_leverage if gross_leverage is not None else bcfg["gross_leverage"])
    long_only_flag = bool(
        long_only if long_only is not None else cfg.get("signals.construction.long_only", False)
    )
    model = cost_model or CostModel.from_config(cfg)

    if panel.is_empty() or prices.is_empty():
        return _empty_result(holding)

    sessions = sorted(set(prices["date"].to_list()))
    symbols = sorted(set(prices["symbol"].to_list()))
    n, m = len(sessions), len(symbols)
    # Need at least one realisable step (two sessions), and enough sessions for each
    # sleeve to be established once.
    if n < max(2, holding + 1):
        return _empty_result(holding)

    signal = _panel_to_matrix(panel, sessions, symbols, value_col)
    close = _panel_to_matrix(
        prices.select("date", "symbol", "adj_close"), sessions, symbols, "adj_close"
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        fwd = np.full((n, m), np.nan)
        fwd[:-1] = close[1:] / close[:-1] - 1.0

    adv_frame = rolling_adv(prices, model.adv_window)
    adv = _panel_to_matrix(adv_frame, sessions, symbols, "adv")

    # Target weights per session, before the sleeve structure.
    target = np.zeros((n, m))
    for i in range(n):
        target[i] = quintile_weights(
            signal[i], long_only=long_only_flag, gross_leverage=leverage, n_quantiles=5
        )

    # `holding` overlapping sleeves, each holding 1/holding of the gross and
    # rebalancing every `holding` sessions on a staggered schedule.
    held = np.zeros((holding, m))
    sleeve_scale = 1.0 / holding

    dates_out, gross_out, net_out, cost_out, turn_out = [], [], [], [], []
    n_long_out, n_short_out = [], []
    total_fixed = total_impact = total_traded = 0.0
    missing_adv_rows = 0
    unpriced_days = 0
    unpriced_gross = 0.0

    for i in range(n - 1):
        sleeve = i % holding
        prev = held[sleeve].copy()
        new = target[i] * sleeve_scale
        delta = new - prev
        held[sleeve] = new

        traded_value = float(np.abs(delta).sum()) * notional
        buys = float(delta[delta > 0].sum()) * notional
        sells = float(-delta[delta < 0].sum()) * notional
        fixed = model.fixed_cost(buys, sells)

        impact = 0.0
        for j in np.where(np.abs(delta) > 0)[0]:
            a = adv[i, j]
            if not np.isfinite(a) or a <= 0:
                missing_adv_rows += 1
                continue
            impact += model.impact_cost(float(abs(delta[j])) * notional, float(a))

        book = held.sum(axis=0)
        step = fwd[i]
        # A held name with no next-session price contributes zero. That is an assumption,
        # not a measurement: it books the delisting or halt as a costless exit at the last
        # observed price. Counted and reported so the assumption is visible rather than
        # buried in an aggregate.
        unpriced = (np.abs(book) > 1e-12) & ~np.isfinite(step)
        if unpriced.any():
            unpriced_days += int(unpriced.sum())
            unpriced_gross += float(np.abs(book[unpriced]).sum())
        contrib = np.where(np.isfinite(step), book * np.nan_to_num(step, nan=0.0), 0.0)
        gross = float(contrib.sum())
        cost_frac = (fixed + impact) / notional if notional > 0 else 0.0

        dates_out.append(sessions[i + 1])
        gross_out.append(gross)
        cost_out.append(cost_frac)
        net_out.append(gross - cost_frac)
        turn_out.append(traded_value / notional if notional > 0 else 0.0)
        n_long_out.append(int((book > 1e-12).sum()))
        n_short_out.append(int((book < -1e-12).sum()))

        total_fixed += fixed
        total_impact += impact
        total_traded += traded_value

    daily = pl.DataFrame(
        {
            "date": dates_out,
            "gross_return": gross_out,
            "net_return": net_out,
            "cost": cost_out,
            "turnover": turn_out,
            "n_long": n_long_out,
            "n_short": n_short_out,
        },
        schema={
            "date": pl.Date, "gross_return": pl.Float64, "net_return": pl.Float64,
            "cost": pl.Float64, "turnover": pl.Float64,
            "n_long": pl.UInt32, "n_short": pl.UInt32,
        },
    )
    days = daily.height
    gross_arr = np.asarray(gross_out, dtype=float)
    net_arr = np.asarray(net_out, dtype=float)
    years = days / TRADING_DAYS if days else float("nan")
    return BacktestResult(
        daily=daily,
        gross_sharpe=sharpe_ratio(gross_arr),
        net_sharpe=sharpe_ratio(net_arr),
        gross_annual_return=float(gross_arr.mean() * TRADING_DAYS) if days else float("nan"),
        net_annual_return=float(net_arr.mean() * TRADING_DAYS) if days else float("nan"),
        annual_turnover=float(np.sum(turn_out) / years) if years else float("nan"),
        total_cost_bps=float(1e4 * (total_fixed + total_impact) / notional / years)
        if years and notional > 0 else float("nan"),
        n_days=days,
        holding_days=holding,
        cost_detail={
            "total_fixed_cost": total_fixed,
            "total_impact_cost": total_impact,
            "total_traded_value": total_traded,
            "trades_without_adv": missing_adv_rows,
            "notional": notional,
            # Position-days where a held name had no next-session price and was therefore
            # marked flat. Non-zero means some of the P&L rests on that assumption.
            "unpriced_position_days": unpriced_days,
            "unpriced_gross_exposure": unpriced_gross,
        },
    )


def _empty_result(holding: int) -> BacktestResult:
    return BacktestResult(
        daily=pl.DataFrame(
            schema={
                "date": pl.Date, "gross_return": pl.Float64, "net_return": pl.Float64,
                "cost": pl.Float64, "turnover": pl.Float64,
                "n_long": pl.UInt32, "n_short": pl.UInt32,
            }
        ),
        gross_sharpe=float("nan"), net_sharpe=float("nan"),
        gross_annual_return=float("nan"), net_annual_return=float("nan"),
        annual_turnover=float("nan"), total_cost_bps=float("nan"),
        n_days=0, holding_days=holding,
    )


def equity_curve(result: BacktestResult, *, column: str = "net_return") -> pl.DataFrame:
    """Cumulative compounded equity curve from a daily return column."""
    if result.daily.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "equity": pl.Float64})
    rets = np.asarray(result.daily[column].to_list(), dtype=float)
    return pl.DataFrame(
        {"date": result.daily["date"].to_list(), "equity": np.cumprod(1.0 + rets)},
        schema={"date": pl.Date, "equity": pl.Float64},
    )


def max_drawdown(result: BacktestResult, *, column: str = "net_return") -> float:
    """Worst peak-to-trough decline of the compounded curve."""
    curve = equity_curve(result, column=column)
    if curve.is_empty():
        return float("nan")
    equity = np.asarray(curve["equity"].to_list(), dtype=float)
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))
