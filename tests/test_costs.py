"""Indian transaction costs, verified against hand-computed rupee amounts."""

from __future__ import annotations

import datetime as _dt
import math

import polars as pl
import pytest

from flowalpha.backtest.costs import (
    CostModel,
    break_even_aum,
    describe,
    rolling_adv,
    trade_costs,
)

from conftest import make_config

DAYS = [_dt.date(2022, 1, 3) + _dt.timedelta(days=i) for i in range(40)
        if (_dt.date(2022, 1, 3) + _dt.timedelta(days=i)).weekday() < 5]


@pytest.fixture
def model() -> CostModel:
    return CostModel()  # config defaults, restated in the dataclass


def test_from_config_reads_every_rate(tmp_path):
    cfg = make_config(tmp_path, DAYS)
    m = CostModel.from_config(cfg)
    assert m.stt_delivery == 0.001
    assert m.stt_intraday_sell == 0.00025
    assert m.stamp_duty_buy == 0.00015
    assert m.brokerage_exchange == 0.0003
    assert m.impact_coef == 0.1
    assert m.adv_window == 10


def test_buy_rate_has_stamp_duty_but_no_stt(model):
    """STT is a sell-side tax on delivery equity; stamp duty is buy-side."""
    assert model.buy_rate == pytest.approx(0.00015 + 0.0003)
    assert model.buy_rate == pytest.approx(0.00045)  # 4.5 bps


def test_sell_rate_includes_delivery_stt(model):
    assert model.sell_rate == pytest.approx(0.001 + 0.0003)
    assert model.sell_rate == pytest.approx(0.0013)  # 13 bps


def test_intraday_sell_rate_is_much_lower():
    m = CostModel(delivery=False)
    assert m.sell_rate == pytest.approx(0.00025 + 0.0003)
    assert m.sell_rate < CostModel(delivery=True).sell_rate


def test_round_trip_fixed_cost_hand_computed(model):
    """Buy Rs 1,000,000 and sell Rs 1,000,000.

    buy:  1e6 * 0.00045 = Rs 450
    sell: 1e6 * 0.00130 = Rs 1300
    total Rs 1750, i.e. 17.5 bps of the one-way notional.
    """
    cost = model.fixed_cost(1_000_000.0, 1_000_000.0)
    assert cost == pytest.approx(1750.0)
    assert 1e4 * cost / 1_000_000.0 == pytest.approx(17.5)


def test_fixed_cost_uses_absolute_values(model):
    assert model.fixed_cost(-1000.0, -1000.0) == pytest.approx(model.fixed_cost(1000.0, 1000.0))


def test_impact_is_square_root_of_participation(model):
    """10% participation -> 0.1 * sqrt(0.1) = 3.162% ... in bps: 316.2."""
    adv = 1_000_000.0
    assert model.impact_bps(100_000.0, adv) == pytest.approx(1e4 * 0.1 * math.sqrt(0.1))
    assert model.impact_bps(100_000.0, adv) == pytest.approx(316.2278, rel=1e-4)


def test_impact_quadruples_the_trade_for_double_the_cost(model):
    """Square-root scaling: 4x the size costs 2x the bps."""
    adv = 1_000_000.0
    small = model.impact_bps(10_000.0, adv)
    big = model.impact_bps(40_000.0, adv)
    assert big == pytest.approx(2.0 * small)


def test_impact_cost_is_bps_times_value(model):
    adv = 1_000_000.0
    value = 100_000.0
    assert model.impact_cost(value, adv) == pytest.approx(value * model.impact_bps(value, adv) / 1e4)


def test_missing_adv_charges_nothing_and_is_reported(model):
    """A guessed ADV would let the backtest trade unlimited size for free -- the most
    flattering error a cost model can make. So it is reported, not invented."""
    assert model.impact_bps(1e9, None) == 0.0
    assert model.impact_bps(1e9, 0.0) == 0.0
    assert model.impact_bps(1e9, float("nan")) == 0.0

    trades = pl.DataFrame(
        {"symbol": ["A", "B"], "trade_value": [1e6, -2e6], "adv": [1e8, None]},
        schema={"symbol": pl.Utf8, "trade_value": pl.Float64, "adv": pl.Float64},
    )
    breakdown = trade_costs(trades, model)
    assert breakdown.n_missing_adv == 1
    assert breakdown.impact > 0  # from A only


def test_total_cost_combines_fixed_and_impact(model):
    fixed = model.fixed_cost(500_000.0, 500_000.0)
    impact = model.impact_cost(1_000_000.0, 1e8)
    assert model.total_cost(500_000.0, 500_000.0, 1e8) == pytest.approx(fixed + impact)


def test_trade_costs_hand_computed(model):
    """Buys Rs 3m, sells Rs 1m, ADV Rs 1bn for both.

    fixed  = 3e6 * 0.00045 + 1e6 * 0.0013 = 1350 + 1300 = Rs 2650
    impact = 0.1 * sqrt(3e6/1e9) * 3e6 + 0.1 * sqrt(1e6/1e9) * 1e6
    """
    trades = pl.DataFrame(
        {"symbol": ["A", "B"], "trade_value": [3e6, -1e6], "adv": [1e9, 1e9]},
        schema={"symbol": pl.Utf8, "trade_value": pl.Float64, "adv": pl.Float64},
    )
    got = trade_costs(trades, model)
    assert got.buy_value == pytest.approx(3e6)
    assert got.sell_value == pytest.approx(1e6)
    assert got.fixed == pytest.approx(2650.0)
    expected_impact = (
        0.1 * math.sqrt(3e6 / 1e9) * 3e6 + 0.1 * math.sqrt(1e6 / 1e9) * 1e6
    )
    assert got.impact == pytest.approx(expected_impact)
    assert got.total == pytest.approx(2650.0 + expected_impact)


def test_trade_costs_bps_of_notional(model):
    trades = pl.DataFrame(
        {"trade_value": [1e6], "adv": [1e12]},
        schema={"trade_value": pl.Float64, "adv": pl.Float64},
    )
    got = trade_costs(trades, model)
    assert got.bps_of(1e6) == pytest.approx(1e4 * got.total / 1e6)
    assert got.bps_of(0.0) == 0.0


def test_trade_costs_on_empty_input(model):
    empty = pl.DataFrame(schema={"trade_value": pl.Float64, "adv": pl.Float64})
    got = trade_costs(empty, model)
    assert got.total == 0.0 and got.n_missing_adv == 0


def test_trade_costs_without_an_adv_column(model):
    trades = pl.DataFrame({"trade_value": [1e6]}, schema={"trade_value": pl.Float64})
    got = trade_costs(trades, model)
    assert got.impact == 0.0
    assert got.n_missing_adv == 1


# --- ADV ------------------------------------------------------------------

def _prices(turnovers: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": DAYS[: len(turnovers)],
            "symbol": ["A"] * len(turnovers),
            "turnover": turnovers,
        },
        schema={"date": pl.Date, "symbol": pl.Utf8, "turnover": pl.Float64},
    )


def test_rolling_adv_excludes_today():
    """Today's own turnover is unknown when today's trade is sized, and including it
    would let a large trade justify its own impact estimate."""
    prices = _prices([100.0, 200.0, 300.0, 400.0, 500.0])
    adv = rolling_adv(prices, 2).sort("date")
    values = adv["adv"].to_list()
    assert values[0] is None
    # At date index 2 the window is turnover[0:2] = (100, 200) -> 150.
    assert values[2] == pytest.approx(150.0)
    assert values[4] == pytest.approx(350.0)  # (300, 400)


def test_rolling_adv_requires_turnover():
    with pytest.raises(ValueError, match="turnover"):
        rolling_adv(pl.DataFrame({"date": DAYS[:2], "symbol": ["A", "A"]}), 2)


def test_rolling_adv_on_empty_panel_keeps_schema():
    got = rolling_adv(pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8,
                                           "turnover": pl.Float64}), 5)
    assert got.is_empty() and got.schema["adv"] == pl.Float64


# --- break-even AUM -------------------------------------------------------

def test_break_even_aum_is_finite_and_scales_with_liquidity(model):
    small = break_even_aum(0.10, 4.0, model, mean_adv=1e8)
    large = break_even_aum(0.10, 4.0, model, mean_adv=1e10)
    assert math.isfinite(small) and small > 0
    assert large == pytest.approx(100.0 * small)  # linear in ADV


def test_break_even_aum_is_zero_when_fixed_costs_already_exceed_gross(model):
    """No amount of size discipline saves a strategy that loses to fixed costs."""
    assert break_even_aum(0.001, 20.0, model, mean_adv=1e10) == 0.0


def test_break_even_aum_falls_as_turnover_rises(model):
    low = break_even_aum(0.15, 2.0, model, mean_adv=1e9)
    high = break_even_aum(0.15, 8.0, model, mean_adv=1e9)
    assert high < low


def test_break_even_aum_is_infinite_without_impact():
    free = CostModel(impact_coef=0.0)
    assert break_even_aum(0.10, 4.0, free, mean_adv=1e9) == float("inf")
    assert break_even_aum(0.10, 0.0, CostModel(), mean_adv=1e9) == float("inf")


def test_break_even_aum_hand_computed():
    """With only impact (no fixed costs) and gross 10%, turnover 1x, ADV 1e9:

    0.10 = 0.1 * 1 * sqrt(AUM / 1e9)  ->  sqrt(AUM/1e9) = 1  ->  AUM = 1e9
    """
    m = CostModel(stt_delivery=0.0, stamp_duty_buy=0.0, brokerage_exchange=0.0,
                  impact_coef=0.1)
    assert break_even_aum(0.10, 1.0, m, mean_adv=1e9) == pytest.approx(1e9)


def test_break_even_aum_accounts_for_leverage():
    m = CostModel(stt_delivery=0.0, stamp_duty_buy=0.0, brokerage_exchange=0.0)
    single = break_even_aum(0.10, 1.0, m, mean_adv=1e9, gross_leverage=1.0)
    doubled = break_even_aum(0.10, 1.0, m, mean_adv=1e9, gross_leverage=2.0)
    assert doubled == pytest.approx(single / 2.0)


def test_describe_is_ascii_and_mentions_every_component(model):
    lines = "\n".join(describe(model))
    lines.encode("ascii")  # must not raise
    for token in ("STT", "stamp duty", "brokerage", "impact", "round-trip"):
        assert token in lines
