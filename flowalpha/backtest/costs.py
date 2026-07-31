"""Indian equity transaction costs.

Components, all rates from ``config.yaml``:

===============================  =========  ==========================================
component                        side       note
===============================  =========  ==========================================
STT (delivery)                   sell       0.10% of turnover. Delivery-based equity.
STT (intraday)                   sell       0.025%. Applies when the position is not
                                            held to settlement.
Stamp duty                       buy        0.015%, buy side only.
Brokerage + exchange + SEBI+GST  both       ~0.03% blended.
Market impact                    both       ``impact_coef * sqrt(participation)``.
===============================  =========  ==========================================

Why square-root impact
----------------------
Empirically, price impact scales roughly with the square root of the fraction of daily
volume traded, not linearly. Linear impact understates the cost of small trades and
wildly overstates large ones; the square root is the standard reduced-form fit and is
what makes a break-even AUM calculation meaningful.

Participation is measured against ``adv_window``-day average rupee turnover. A name
with no ADV estimate gets **no impact charge and is flagged**, rather than a guessed
one: a fabricated ADV would let the backtest trade unlimited size in an illiquid name
for free, which is the single most flattering error a cost model can make.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
import polars as pl


@dataclass(frozen=True)
class CostModel:
    """Rates and impact parameters. Construct via :meth:`from_config`."""

    stt_delivery: float = 0.001
    stt_intraday_sell: float = 0.00025
    stamp_duty_buy: float = 0.00015
    brokerage_exchange: float = 0.0003
    impact_coef: float = 0.1
    adv_window: int = 21
    #: Whether positions are assumed held to settlement (delivery) or squared off.
    delivery: bool = True

    @classmethod
    def from_config(cls, cfg, *, delivery: bool = True) -> "CostModel":
        c = cfg["backtest"]["costs"]
        return cls(
            stt_delivery=float(c["stt_delivery"]),
            stt_intraday_sell=float(c["stt_intraday_sell"]),
            stamp_duty_buy=float(c["stamp_duty_buy"]),
            brokerage_exchange=float(c["brokerage_exchange"]),
            impact_coef=float(c["impact_coef"]),
            adv_window=int(c["adv_window"]),
            delivery=delivery,
        )

    # -- per-rupee rates ----------------------------------------------------
    @property
    def buy_rate(self) -> float:
        """Fixed (non-impact) cost per rupee bought.

        No STT on the buy side of a delivery trade; stamp duty applies to buys only.
        """
        return self.stamp_duty_buy + self.brokerage_exchange

    @property
    def sell_rate(self) -> float:
        """Fixed cost per rupee sold, including STT."""
        stt = self.stt_delivery if self.delivery else self.stt_intraday_sell
        return stt + self.brokerage_exchange

    def fixed_cost(self, buy_value: float, sell_value: float) -> float:
        """Rupee cost of buying ``buy_value`` and selling ``sell_value``."""
        return self.buy_rate * abs(buy_value) + self.sell_rate * abs(sell_value)

    def impact_bps(self, trade_value: float, adv: float | None) -> float:
        """Square-root impact in basis points of the traded value.

        Returns 0.0 when ADV is unknown; the caller is expected to surface that, and
        :func:`trade_costs` counts such rows in ``n_missing_adv``.
        """
        if adv is None or not math.isfinite(adv) or adv <= 0 or abs(trade_value) <= 0:
            return 0.0
        participation = abs(trade_value) / adv
        return 1e4 * self.impact_coef * math.sqrt(participation)

    def impact_cost(self, trade_value: float, adv: float | None) -> float:
        return abs(trade_value) * self.impact_bps(trade_value, adv) / 1e4

    def total_cost(
        self, buy_value: float, sell_value: float, adv: float | None = None
    ) -> float:
        """Fixed plus impact cost for a single name's trade."""
        traded = abs(buy_value) + abs(sell_value)
        return self.fixed_cost(buy_value, sell_value) + self.impact_cost(traded, adv)


def rolling_adv(prices: pl.DataFrame, window: int) -> pl.DataFrame:
    """Trailing average rupee turnover per name.

    ``(date, symbol, adv)``. Shifted by one session: today's own turnover is not known
    when today's trade is sized, and including it would let a large trade justify its
    own impact estimate.
    """
    if prices.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "adv": pl.Float64})
    if "turnover" not in prices.columns:
        raise ValueError("prices must carry a 'turnover' column to estimate ADV")
    return (
        prices.sort(["symbol", "date"])
        .with_columns(
            pl.col("turnover")
            .shift(1)
            .rolling_mean(window_size=window, min_samples=max(2, window // 2))
            .over("symbol")
            .alias("adv")
        )
        .select("date", "symbol", "adv")
    )


@dataclass(frozen=True)
class CostBreakdown:
    """Aggregated costs for one rebalance."""

    fixed: float
    impact: float
    buy_value: float
    sell_value: float
    n_missing_adv: int

    @property
    def total(self) -> float:
        return self.fixed + self.impact

    def bps_of(self, notional: float) -> float:
        return 1e4 * self.total / notional if notional > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "fixed": self.fixed, "impact": self.impact, "total": self.total,
            "buy_value": self.buy_value, "sell_value": self.sell_value,
            "n_missing_adv": self.n_missing_adv,
        }


def trade_costs(
    trades: pl.DataFrame,
    model: CostModel,
    *,
    value_col: str = "trade_value",
    adv_col: str = "adv",
) -> CostBreakdown:
    """Aggregate costs over a set of trades.

    ``trades`` needs a signed ``trade_value`` (positive = buy) and optionally ``adv``.
    Names with a missing ADV are counted and reported rather than charged a guessed
    impact.
    """
    if trades.is_empty():
        return CostBreakdown(0.0, 0.0, 0.0, 0.0, 0)
    values = np.asarray(trades[value_col].to_list(), dtype=float)
    values = np.nan_to_num(values, nan=0.0)
    buys = float(values[values > 0].sum())
    sells = float(-values[values < 0].sum())
    fixed = model.fixed_cost(buys, sells)

    if adv_col in trades.columns:
        advs = np.asarray(
            [np.nan if v is None else float(v) for v in trades[adv_col].to_list()], dtype=float
        )
    else:
        advs = np.full(values.shape, np.nan)
    missing = int(np.sum(~np.isfinite(advs) & (np.abs(values) > 0)))
    impact = 0.0
    for value, adv in zip(values, advs):
        impact += model.impact_cost(value, None if not math.isfinite(adv) else adv)
    return CostBreakdown(
        fixed=fixed, impact=impact, buy_value=buys, sell_value=sells, n_missing_adv=missing
    )


def break_even_aum(
    gross_annual_return: float,
    turnover_per_year: float,
    model: CostModel,
    *,
    mean_adv: float,
    gross_leverage: float = 1.0,
) -> float:
    """Largest AUM at which the strategy still breaks even, in rupees.

    Fixed costs are proportional to notional, so they cannot be outgrown -- they simply
    reduce the gross return. Impact grows with ``sqrt(AUM)``, so there is a finite AUM
    at which impact consumes whatever gross return survives the fixed costs:

        gross - fixed_rate * turnover - impact_coef * turnover * sqrt(AUM * L / ADV) = 0

    Returns ``0.0`` when fixed costs alone already exceed the gross return, and
    ``inf`` when there is no impact charge to bind (``impact_coef`` or turnover zero).
    A finite answer here is the difference between a strategy and a curiosity.
    """
    if turnover_per_year <= 0 or mean_adv <= 0:
        return float("inf")
    fixed_rate = (model.buy_rate + model.sell_rate) / 2.0
    net_of_fixed = gross_annual_return - fixed_rate * turnover_per_year
    if net_of_fixed <= 0:
        return 0.0
    if model.impact_coef <= 0:
        return float("inf")
    # net_of_fixed = impact_coef * turnover * sqrt(AUM * L / ADV)
    root = net_of_fixed / (model.impact_coef * turnover_per_year)
    return float(root ** 2 * mean_adv / max(gross_leverage, 1e-12))


def describe(model: CostModel) -> list[str]:
    """ASCII lines summarising the model, for report tables."""
    return [
        f"  STT ({'delivery' if model.delivery else 'intraday'}, sell side): "
        f"{(model.stt_delivery if model.delivery else model.stt_intraday_sell) * 1e4:.1f} bps",
        f"  stamp duty (buy side):                {model.stamp_duty_buy * 1e4:.1f} bps",
        f"  brokerage + exchange + SEBI + GST:    {model.brokerage_exchange * 1e4:.1f} bps",
        f"  round-trip fixed cost:                {(model.buy_rate + model.sell_rate) * 1e4:.1f} bps",
        f"  impact:  {model.impact_coef} * sqrt(participation) vs {model.adv_window}d ADV",
    ]
