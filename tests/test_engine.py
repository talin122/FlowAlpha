"""Backtest engine, verified against a hand-computed two-day two-stock example."""

from __future__ import annotations

import datetime as _dt
import math

import numpy as np
import polars as pl
import pytest

from flowalpha.backtest.costs import CostModel
from flowalpha.backtest.engine import (
    equity_curve,
    max_drawdown,
    quintile_weights,
    run_backtest,
)

from conftest import make_config, make_tiny_prices

D = _dt.date


# --- weighting -------------------------------------------------------------

def test_quintile_weights_are_dollar_neutral():
    values = np.arange(10.0)
    w = quintile_weights(values, gross_leverage=1.0)
    assert w.sum() == pytest.approx(0.0)
    assert np.abs(w).sum() == pytest.approx(1.0)


def test_quintile_weights_long_the_top_short_the_bottom():
    values = np.arange(10.0)
    w = quintile_weights(values)
    assert w[-1] > 0 and w[-2] > 0
    assert w[0] < 0 and w[1] < 0
    assert w[5] == 0.0


def test_quintile_weights_hand_computed_for_ten_names():
    """10 names, quintiles -> 2 long, 2 short, gross 1.0.

    Each long gets +0.5/2 = +0.25, each short -0.25.
    """
    w = quintile_weights(np.arange(10.0), gross_leverage=1.0)
    assert sorted(w.tolist()) == pytest.approx([-0.25, -0.25, 0, 0, 0, 0, 0, 0, 0.25, 0.25])


def test_quintile_weights_long_only():
    w = quintile_weights(np.arange(10.0), long_only=True, gross_leverage=1.0)
    assert (w >= 0).all()
    assert w.sum() == pytest.approx(1.0)


def test_quintile_weights_ignores_nan():
    values = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0])
    w = quintile_weights(values)
    assert w[0] == 0.0 and w[6] == 0.0
    assert np.abs(w).sum() == pytest.approx(1.0)


def test_quintile_weights_too_few_names_is_all_zero():
    assert (quintile_weights(np.array([1.0, 2.0])) == 0).all()


def test_quintile_weights_scale_with_leverage():
    w = quintile_weights(np.arange(10.0), gross_leverage=2.0)
    assert np.abs(w).sum() == pytest.approx(2.0)


# --- the hand-computed engine example -------------------------------------

def test_engine_reproduces_a_hand_computed_two_stock_example(tmp_path):
    """Two stocks, three sessions, holding_days=1, ten names padded for quintiles.

    Setup: ten symbols. The signal ranks S0..S9 ascending on every date, so with
    quintiles the book is long {S8, S9} at +0.25 each and short {S0, S1} at -0.25 each.
    Only S0 and S9 move; everything else is flat.

        session 0 -> 1: S9 +10%, S0 -10%
        session 1 -> 2: S9  -5%, S0  +5%

    Gross P&L, session 0->1:  0.25*0.10 + (-0.25)*(-0.10) = 0.05
    Gross P&L, session 1->2:  0.25*(-0.05) + (-0.25)*(0.05) = -0.025

    With holding_days=1 the whole book is rebalanced daily, but the weights never
    change after day 0, so cost is charged only on day 0's initial build:
        traded = 4 names x 0.25 = 1.0 of notional
        buys  = 0.25 + 0.25 = 0.5   (the two longs)
        sells = 0.25 + 0.25 = 0.5   (the two shorts)
        fixed = 0.5*0.00045 + 0.5*0.0013 = 0.000875
    Impact is zero here because the price panel carries no turnover, so ADV is unknown
    and the engine refuses to invent one.
    """
    sessions = [D(2022, 1, 3), D(2022, 1, 4), D(2022, 1, 5)]
    symbols = [f"S{i}" for i in range(10)]

    levels = {s: [100.0, 100.0, 100.0] for s in symbols}
    levels["S9"] = [100.0, 110.0, 104.5]   # +10%, then -5%
    levels["S0"] = [100.0, 90.0, 94.5]     # -10%, then +5%

    rows = []
    for sym in symbols:
        for day, px in zip(sessions, levels[sym]):
            rows.append({"date": day, "symbol": sym, "adj_close": px, "turnover": None})
    prices = pl.DataFrame(
        rows,
        schema={"date": pl.Date, "symbol": pl.Utf8, "adj_close": pl.Float64,
                "turnover": pl.Float64},
    )
    panel = pl.DataFrame(
        [{"date": day, "symbol": f"S{i}", "value": float(i)}
         for day in sessions for i in range(10)],
        schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64},
    )

    cfg = make_config(tmp_path, sessions)
    result = run_backtest(panel, prices, cfg, holding_days=1, notional=1.0)

    assert result.n_days == 2
    gross = result.daily["gross_return"].to_list()
    assert gross[0] == pytest.approx(0.05)
    assert gross[1] == pytest.approx(-0.025)

    costs = result.daily["cost"].to_list()
    assert costs[0] == pytest.approx(0.5 * 0.00045 + 0.5 * 0.0013)
    assert costs[1] == pytest.approx(0.0)  # weights unchanged, so nothing traded

    net = result.daily["net_return"].to_list()
    assert net[0] == pytest.approx(0.05 - costs[0])
    assert net[1] == pytest.approx(-0.025)

    turnover = result.daily["turnover"].to_list()
    assert turnover[0] == pytest.approx(1.0)
    assert turnover[1] == pytest.approx(0.0)

    assert result.daily["n_long"].to_list() == [2, 2]
    assert result.daily["n_short"].to_list() == [2, 2]
    # P&L is dated to the session it was EARNED, i.e. one after the signal date.
    assert result.daily["date"].to_list() == sessions[1:]


def test_engine_earns_no_same_day_return(tmp_path):
    """A signal from t's close cannot capture t's move.

    S9 rises only on the FIRST step, so a same-day-return engine would show a gain on
    a date the correct engine cannot.
    """
    sessions = [D(2022, 1, 3), D(2022, 1, 4)]
    symbols = [f"S{i}" for i in range(10)]
    rows = []
    for i, sym in enumerate(symbols):
        for k, day in enumerate(sessions):
            px = 100.0 * (1.5 if (sym == "S9" and k == 0) else 1.0)
            rows.append({"date": day, "symbol": sym, "adj_close": px, "turnover": None})
    prices = pl.DataFrame(rows, schema={"date": pl.Date, "symbol": pl.Utf8,
                                        "adj_close": pl.Float64, "turnover": pl.Float64})
    panel = pl.DataFrame(
        [{"date": day, "symbol": f"S{i}", "value": float(i)}
         for day in sessions for i in range(10)],
        schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64},
    )
    cfg = make_config(tmp_path, sessions)
    result = run_backtest(panel, prices, cfg, holding_days=1, notional=1.0)
    # The only realisable step is S9 falling back from 150 to 100 -> a LOSS on the long.
    assert result.daily["gross_return"].to_list()[0] < 0


def test_engine_charges_impact_when_adv_is_known(tmp_path):
    sessions = [D(2022, 1, 3) + _dt.timedelta(days=i) for i in range(40)
                if (D(2022, 1, 3) + _dt.timedelta(days=i)).weekday() < 5]
    symbols = tuple(f"S{i}" for i in range(10))
    prices = make_tiny_prices(sessions, symbols=symbols)
    # A CHANGING signal, so trades happen after the ADV window has filled. With a
    # constant signal the only trade is the initial build, when ADV is still unknown
    # by construction -- and then the engine correctly charges no impact at all.
    rng = np.random.default_rng(7)
    panel = pl.DataFrame(
        [{"date": day, "symbol": s, "value": float(rng.normal())}
         for day in sessions for s in symbols],
        schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64},
    )
    cfg = make_config(tmp_path, sessions)
    small = run_backtest(panel, prices, cfg, holding_days=1, notional=1e6)
    large = run_backtest(panel, prices, cfg, holding_days=1, notional=1e11)
    assert large.cost_detail["total_impact_cost"] > 0
    # Impact grows sub-linearly in notional, so cost as a FRACTION still rises.
    assert large.total_cost_bps > small.total_cost_bps


def test_engine_reports_trades_without_adv(tmp_path):
    sessions = [D(2022, 1, 3), D(2022, 1, 4), D(2022, 1, 5)]
    rows = [
        {"date": day, "symbol": f"S{i}", "adj_close": 100.0 + i, "turnover": None}
        for day in sessions for i in range(10)
    ]
    prices = pl.DataFrame(rows, schema={"date": pl.Date, "symbol": pl.Utf8,
                                        "adj_close": pl.Float64, "turnover": pl.Float64})
    panel = pl.DataFrame(
        [{"date": day, "symbol": f"S{i}", "value": float(i)} for day in sessions for i in range(10)],
        schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64},
    )
    cfg = make_config(tmp_path, sessions)
    result = run_backtest(panel, prices, cfg, holding_days=1, notional=1e9)
    assert result.cost_detail["trades_without_adv"] > 0
    assert result.cost_detail["total_impact_cost"] == 0.0


# --- sleeve structure -----------------------------------------------------

def test_longer_holding_period_reduces_turnover(tmp_path):
    sessions = [D(2022, 1, 3) + _dt.timedelta(days=i) for i in range(120)
                if (D(2022, 1, 3) + _dt.timedelta(days=i)).weekday() < 5]
    symbols = tuple(f"S{i}" for i in range(10))
    prices = make_tiny_prices(sessions, symbols=symbols)
    rng = np.random.default_rng(0)
    panel = pl.DataFrame(
        [
            {"date": day, "symbol": sym, "value": float(rng.normal())}
            for day in sessions for sym in symbols
        ],
        schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64},
    )
    cfg = make_config(tmp_path, sessions)
    fast = run_backtest(panel, prices, cfg, holding_days=1, notional=1.0)
    slow = run_backtest(panel, prices, cfg, holding_days=5, notional=1.0)
    assert slow.annual_turnover < fast.annual_turnover
    assert slow.holding_days == 5


def test_sleeves_keep_gross_leverage_at_target(tmp_path):
    """With h sleeves each at 1/h of gross, total gross must still be ~1.0."""
    sessions = [D(2022, 1, 3) + _dt.timedelta(days=i) for i in range(60)
                if (D(2022, 1, 3) + _dt.timedelta(days=i)).weekday() < 5]
    symbols = tuple(f"S{i}" for i in range(10))
    prices = make_tiny_prices(sessions, symbols=symbols)
    panel = pl.DataFrame(
        [{"date": day, "symbol": f"S{i}", "value": float(i)} for day in sessions for i in range(10)],
        schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64},
    )
    cfg = make_config(tmp_path, sessions)
    result = run_backtest(panel, prices, cfg, holding_days=5, notional=1.0)
    # A constant signal means every sleeve holds the same book, so after the ramp-up
    # the long and short counts settle at the single-sleeve values.
    tail = result.daily.tail(10)
    assert tail["n_long"].to_list() == [2] * 10
    assert tail["n_short"].to_list() == [2] * 10


# --- summary statistics ---------------------------------------------------

def test_empty_inputs_give_an_empty_result(tmp_path):
    cfg = make_config(tmp_path, [D(2022, 1, 3), D(2022, 1, 4)])
    empty_panel = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64})
    empty_prices = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8,
                                        "adj_close": pl.Float64, "turnover": pl.Float64})
    result = run_backtest(empty_panel, empty_prices, cfg)
    assert result.n_days == 0
    assert math.isnan(result.net_sharpe)
    assert result.daily.schema["date"] == pl.Date


def test_too_short_history_gives_an_empty_result(tmp_path):
    sessions = [D(2022, 1, 3), D(2022, 1, 4)]
    prices = make_tiny_prices(sessions, symbols=tuple(f"S{i}" for i in range(10)))
    panel = pl.DataFrame(
        [{"date": day, "symbol": f"S{i}", "value": float(i)} for day in sessions for i in range(10)],
        schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64},
    )
    cfg = make_config(tmp_path, sessions)
    assert run_backtest(panel, prices, cfg, holding_days=5).n_days == 0


def test_gross_and_net_are_reported_separately(tmp_path):
    """A strategy that only works before costs must be visibly that."""
    sessions = [D(2022, 1, 3) + _dt.timedelta(days=i) for i in range(80)
                if (D(2022, 1, 3) + _dt.timedelta(days=i)).weekday() < 5]
    symbols = tuple(f"S{i}" for i in range(10))
    prices = make_tiny_prices(sessions, symbols=symbols)
    rng = np.random.default_rng(1)
    panel = pl.DataFrame(
        [{"date": day, "symbol": s, "value": float(rng.normal())}
         for day in sessions for s in symbols],
        schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64},
    )
    cfg = make_config(tmp_path, sessions)
    result = run_backtest(panel, prices, cfg, holding_days=1, notional=1.0)
    d = result.to_dict()
    assert d["gross_sharpe"] != d["net_sharpe"]
    assert result.net_annual_return < result.gross_annual_return
    assert d["total_cost_bps"] > 0


def test_equity_curve_and_drawdown(tmp_path):
    sessions = [D(2022, 1, 3) + _dt.timedelta(days=i) for i in range(60)
                if (D(2022, 1, 3) + _dt.timedelta(days=i)).weekday() < 5]
    symbols = tuple(f"S{i}" for i in range(10))
    prices = make_tiny_prices(sessions, symbols=symbols)
    panel = pl.DataFrame(
        [{"date": day, "symbol": f"S{i}", "value": float(i)} for day in sessions for i in range(10)],
        schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64},
    )
    cfg = make_config(tmp_path, sessions)
    result = run_backtest(panel, prices, cfg, holding_days=1, notional=1.0)
    curve = equity_curve(result)
    assert curve.height == result.n_days
    assert curve["equity"][0] == pytest.approx(1.0 + result.daily["net_return"][0])
    dd = max_drawdown(result)
    assert dd <= 0.0


def test_equity_curve_of_empty_result(tmp_path):
    cfg = make_config(tmp_path, [D(2022, 1, 3), D(2022, 1, 4)])
    empty = run_backtest(
        pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64}),
        pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "adj_close": pl.Float64,
                             "turnover": pl.Float64}),
        cfg,
    )
    assert equity_curve(empty).is_empty()
    assert math.isnan(max_drawdown(empty))


def test_custom_cost_model_is_honoured(tmp_path):
    sessions = [D(2022, 1, 3) + _dt.timedelta(days=i) for i in range(60)
                if (D(2022, 1, 3) + _dt.timedelta(days=i)).weekday() < 5]
    symbols = tuple(f"S{i}" for i in range(10))
    prices = make_tiny_prices(sessions, symbols=symbols)
    rng = np.random.default_rng(2)
    panel = pl.DataFrame(
        [{"date": day, "symbol": s, "value": float(rng.normal())}
         for day in sessions for s in symbols],
        schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64},
    )
    cfg = make_config(tmp_path, sessions)
    free = run_backtest(panel, prices, cfg, holding_days=1, notional=1.0,
                        cost_model=CostModel(stt_delivery=0.0, stamp_duty_buy=0.0,
                                             brokerage_exchange=0.0, impact_coef=0.0))
    assert free.cost_detail["total_fixed_cost"] == pytest.approx(0.0)
    assert free.gross_sharpe == pytest.approx(free.net_sharpe)
