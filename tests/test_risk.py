"""Portfolio-level risk overlays: volatility targeting and drawdown control.

Three properties matter more than the arithmetic, and each has a test named for the
failure it prevents:

* a scale must never depend on a return the book has not yet earned;
* rescaling must cost turnover, or the overlay manufactures Sharpe from nothing;
* with the overlays off, results must be bit-identical to a run with no overlay at all,
  because the whole repository's existing numbers were produced that way.
"""

from __future__ import annotations

import datetime as _dt
import math

import numpy as np
import polars as pl
import pytest

from flowalpha.backtest.engine import max_drawdown, run_backtest
from flowalpha.backtest.risk import (
    NEUTRAL_SCALE,
    DrawdownConfig,
    DrawdownController,
    RiskOverlay,
    VolatilityTargeter,
    VolTargetConfig,
    max_drawdown_from_returns,
    time_under_water,
)
from flowalpha.validation.deflated_sharpe import TRADING_DAYS

from conftest import make_config

D = _dt.date


# ---------------------------------------------------------------------------
# Volatility targeting
# ---------------------------------------------------------------------------

def _targeter(**kw) -> VolatilityTargeter:
    base = dict(enabled=True, target_annual_vol=0.10, window=20, min_obs=20,
                max_leverage_multiple=4.0, min_leverage_multiple=0.1)
    base.update(kw)
    return VolatilityTargeter(VolTargetConfig(**base))


def test_scale_is_neutral_until_min_obs_is_reached():
    """An unmeasurable volatility must not become a leverage decision. Below min_obs the
    honest answer is 'leave the book alone', not a guess from three observations."""
    t = _targeter(min_obs=20)
    for _ in range(19):
        t.update(0.01)
        assert t.scale() == NEUTRAL_SCALE
    t.update(0.01)
    # 20 identical returns have zero dispersion, so still no usable estimate.
    assert t.scale() == NEUTRAL_SCALE


def test_scale_hits_the_target_when_realised_vol_is_known():
    """Feed a series whose sample sd is exactly known and check the ratio."""
    t = _targeter(window=100, min_obs=100, target_annual_vol=0.10)
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.02, 100)
    for r in rets:
        t.update(float(r))
    expected = (0.10 / math.sqrt(TRADING_DAYS)) / float(rets.std(ddof=1))
    assert t.scale() == pytest.approx(np.clip(expected, 0.1, 4.0))


def test_higher_realised_vol_produces_lower_leverage():
    """The defining property, stated as a direction rather than a constant."""
    calm, wild = _targeter(), _targeter()
    rng = np.random.default_rng(1)
    for r in rng.normal(0, 0.005, 40):
        calm.update(float(r))
    for r in rng.normal(0, 0.05, 40):
        wild.update(float(r))
    assert calm.scale() > wild.scale()


def test_leverage_multiple_is_capped_in_both_directions():
    """A near-zero denominator would otherwise lever the book arbitrarily just before the
    quiet patch ends."""
    quiet = _targeter(max_leverage_multiple=2.0)
    rng = np.random.default_rng(2)
    for r in rng.normal(0, 1e-5, 40):
        quiet.update(float(r))
    assert quiet.scale() == pytest.approx(2.0)

    loud = _targeter(min_leverage_multiple=0.25)
    for r in rng.normal(0, 5.0, 40):
        loud.update(float(r))
    assert loud.scale() == pytest.approx(0.25)


def test_zero_volatility_returns_neutral_not_the_cap():
    """A degenerate measurement is a measurement failure, not a signal to lever up. The
    cap is a risk decision and must not be reached by accident."""
    t = _targeter(max_leverage_multiple=3.0)
    for _ in range(40):
        t.update(0.01)  # constant series -> sd exactly 0
    assert t.scale() == NEUTRAL_SCALE


def test_non_finite_returns_are_skipped_not_zero_filled():
    """Booking a missing session as 0.0 would understate volatility and lever up in
    response to an absence of data."""
    skipped, zeroed = _targeter(window=30, min_obs=10), _targeter(window=30, min_obs=10)
    rng = np.random.default_rng(3)
    for r in rng.normal(0, 0.02, 30):
        skipped.update(float(r))
        zeroed.update(float(r))
    for _ in range(10):
        skipped.update(float("nan"))
        zeroed.update(0.0)
    assert skipped.scale() < zeroed.scale()


def test_a_disabled_targeter_is_inert():
    t = _targeter(enabled=False)
    for r in (0.5, -0.4, 0.3):
        t.update(r)
    assert t.scale() == NEUTRAL_SCALE


@pytest.mark.parametrize("kw", [
    {"target_annual_vol": 0.0},
    {"window": 1},
    {"min_obs": 1},
    {"min_leverage_multiple": 0.0},
    {"min_leverage_multiple": 3.0, "max_leverage_multiple": 2.0},
])
def test_invalid_vol_config_is_rejected_at_construction(kw):
    base = dict(enabled=True, target_annual_vol=0.1, window=20, min_obs=20,
                max_leverage_multiple=2.0, min_leverage_multiple=0.25)
    base.update(kw)
    with pytest.raises(ValueError):
        VolatilityTargeter(VolTargetConfig(**base))


# ---------------------------------------------------------------------------
# Drawdown control
# ---------------------------------------------------------------------------

def _controller(**kw) -> DrawdownController:
    base = dict(enabled=True, threshold=0.10, derisk_to=0.5, restore_threshold=0.05)
    base.update(kw)
    return DrawdownController(DrawdownConfig(**base))


def test_no_derisk_until_the_threshold_is_breached():
    c = _controller(threshold=0.10)
    for _ in range(9):
        c.update(-0.01)  # ~-8.6% compounded
    assert not c.derisked
    assert c.scale() == NEUTRAL_SCALE


def test_derisk_triggers_past_the_threshold():
    c = _controller(threshold=0.10, derisk_to=0.4)
    for _ in range(12):
        c.update(-0.01)  # ~-11.4% compounded
    assert c.derisked
    assert c.scale() == pytest.approx(0.4)


def test_hysteresis_prevents_thrashing_at_the_boundary():
    """Without a restore band the control flips on and off on alternating sessions,
    paying turnover each time for no change in risk. Sit just past the trigger and
    oscillate: the state must not flip back."""
    c = _controller(threshold=0.10, restore_threshold=0.05)
    for _ in range(11):
        c.update(-0.01)
    assert c.derisked
    flips = 0
    was = c.derisked
    for _ in range(20):
        c.update(0.001)
        c.update(-0.001)
        if c.derisked != was:
            flips += 1
            was = c.derisked
    assert flips == 0


def test_restore_requires_recovering_past_the_shallower_band():
    c = _controller(threshold=0.10, restore_threshold=0.05)
    for _ in range(12):
        c.update(-0.01)
    assert c.derisked
    c.update(0.04)  # still deeper than -5%
    assert c.derisked
    c.update(0.05)  # now shallower than -5%
    assert not c.derisked
    assert c.scale() == NEUTRAL_SCALE


def test_a_new_equity_peak_resets_the_drawdown():
    c = _controller()
    for _ in range(12):
        c.update(-0.01)
    for _ in range(30):
        c.update(0.02)
    assert not c.derisked
    assert c.last_drawdown == pytest.approx(0.0)


@pytest.mark.parametrize("kw", [
    {"threshold": 0.0},
    {"threshold": 1.0},
    {"derisk_to": -0.1},
    {"derisk_to": 1.5},
    {"restore_threshold": 0.10},   # equal to threshold -> no hysteresis band
    {"restore_threshold": 0.20},   # wider than threshold
])
def test_invalid_drawdown_config_is_rejected_at_construction(kw):
    base = dict(enabled=True, threshold=0.10, derisk_to=0.5, restore_threshold=0.05)
    base.update(kw)
    with pytest.raises(ValueError):
        DrawdownController(DrawdownConfig(**base))


# ---------------------------------------------------------------------------
# Drawdown measurement
# ---------------------------------------------------------------------------

def test_max_drawdown_on_a_hand_computed_path():
    # 1.0 -> 1.2 -> 0.6 : worst decline is 0.6/1.2 - 1 = -0.5
    assert max_drawdown_from_returns(np.array([0.2, -0.5])) == pytest.approx(-0.5)


def test_max_drawdown_of_a_monotonic_gain_is_zero():
    assert max_drawdown_from_returns(np.array([0.01] * 10)) == pytest.approx(0.0)


def test_max_drawdown_of_an_empty_series_is_nan():
    assert math.isnan(max_drawdown_from_returns(np.array([])))


def test_time_under_water_counts_the_longest_run_below_a_peak():
    # up, then 3 sessions below the peak, recover above, then 1 below
    r = np.array([0.10, -0.01, -0.01, -0.01, 0.10, -0.01])
    assert time_under_water(r) == 3


def test_time_under_water_of_a_monotonic_gain_is_zero():
    assert time_under_water(np.array([0.01] * 5)) == 0


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

def _sessions(n: int) -> list[_dt.date]:
    out, day = [], D(2021, 1, 4)
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += _dt.timedelta(days=1)
    return out


def _panel_and_prices(n_days: int = 260, seed: int = 7, n_symbols: int = 10):
    """A book with a volatility regime shift partway through, so a vol-targeting overlay
    has something to react to.

    Ten names, not two: ``quintile_weights`` splits into fifths, so a two-name universe
    yields empty long and short legs, an all-zero book and a NaN Sharpe -- which looks
    like a broken overlay when it is really a broken fixture. The signal rotates slowly
    so there is baseline turnover to compare against.
    """
    rng = np.random.default_rng(seed)
    sessions = _sessions(n_days)
    symbols = [f"S{i:02d}" for i in range(n_symbols)]

    prow, srow = [], []
    level = {s: 100.0 for s in symbols}
    for i, d in enumerate(sessions):
        sd = 0.004 if i < n_days // 2 else 0.030
        for k, s in enumerate(symbols):
            level[s] *= 1.0 + float(rng.normal(0.0002, sd))
            prow.append({"date": d, "symbol": s, "adj_close": level[s],
                         "turnover": 5e8})
            srow.append({
                "date": d, "symbol": s,
                "value": math.sin(2 * math.pi * (k / n_symbols + i / 40.0)),
            })
    prices = pl.DataFrame(prow, schema={"date": pl.Date, "symbol": pl.Utf8,
                                        "adj_close": pl.Float64, "turnover": pl.Float64})
    panel = pl.DataFrame(srow, schema={"date": pl.Date, "symbol": pl.Utf8,
                                       "value": pl.Float64})
    return panel, prices, sessions


def _cfg(tmp_path, sessions, **risk):
    """Config with an optional risk block.

    The whole ``backtest`` node is restated because ``make_config`` merges overrides
    shallowly -- passing only ``{"risk": ...}`` would silently drop the cost model and
    the backtest would run for free.
    """
    backtest = {
        "rebalance": "daily", "holding_days": 5, "gross_leverage": 1.0,
        "costs": {
            "stt_delivery": 0.001, "stt_intraday_sell": 0.00025,
            "stamp_duty_buy": 0.00015, "brokerage_exchange": 0.0003,
            "impact_coef": 0.1, "adv_window": 10,
        },
    }
    if risk:
        backtest["risk"] = risk
    return make_config(tmp_path, sessions, backtest=backtest)


def test_overlays_default_off_leave_results_identical(tmp_path):
    """The regression lock. Every backtested number in this repository was produced with
    no overlay; if the default run diverges by so much as a float, those numbers silently
    stop meaning what the report says they mean."""
    panel, prices, sessions = _panel_and_prices()
    cfg = _cfg(tmp_path, sessions)
    a = run_backtest(panel, prices, cfg)
    b = run_backtest(panel, prices, cfg, risk=RiskOverlay.from_config(cfg))
    assert a.net_sharpe == b.net_sharpe
    assert a.annual_turnover == b.annual_turnover
    assert np.array_equal(a.net_returns, b.net_returns)
    assert a.risk_detail == {}
    assert "risk_scale" not in a.daily.columns


def test_drawdown_is_measured_even_with_control_disabled(tmp_path):
    """Wiring the previously-dead max_drawdown in. A risk that is never quantified cannot
    be reasoned about, whether or not it is being controlled."""
    panel, prices, sessions = _panel_and_prices()
    res = run_backtest(panel, prices, _cfg(tmp_path, sessions))
    assert np.isfinite(res.max_drawdown)
    assert res.max_drawdown <= 0.0
    assert res.time_under_water_days >= 0
    assert res.max_drawdown == pytest.approx(max_drawdown(res))
    assert "max_drawdown" in res.to_dict()


def test_a_future_shock_cannot_change_any_earlier_scale(tmp_path):
    """The look-ahead proof. Corrupt prices from session k onward; every risk scale at or
    before k must be untouched. If the overlay ever indexes a forward return this fails
    at exactly the injection point."""
    panel, prices, sessions = _panel_and_prices()
    cfg = _cfg(tmp_path, sessions, volatility_target={"enabled": True, "window": 30, "min_obs": 30},
               drawdown_control={"enabled": True})
    base = run_backtest(panel, prices, cfg)

    sessions = sorted(set(prices["date"].to_list()))
    k = len(sessions) // 2
    cutoff = sessions[k]
    shocked = prices.with_columns(
        pl.when(pl.col("date") >= cutoff)
        .then(pl.col("adj_close") * 5.0)
        .otherwise(pl.col("adj_close"))
        .alias("adj_close")
    )
    after = run_backtest(panel, shocked, cfg)

    a = base.daily.filter(pl.col("date") <= cutoff)["risk_scale"].to_list()
    b = after.daily.filter(pl.col("date") <= cutoff)["risk_scale"].to_list()
    assert a == b
    # And the shock must actually have mattered later, or the test proves nothing.
    assert base.daily["risk_scale"].to_list() != after.daily["risk_scale"].to_list()


def test_volatility_targeting_increases_turnover(tmp_path):
    """No free lunch: changing exposure is trading, and trading is charged. An overlay
    that moved the book without paying for it would manufacture Sharpe."""
    panel, prices, sessions = _panel_and_prices()
    plain = run_backtest(panel, prices, _cfg(tmp_path, sessions))
    scaled = run_backtest(
        panel, prices,
        _cfg(tmp_path, sessions, volatility_target={"enabled": True, "window": 30, "min_obs": 30}),
    )
    assert scaled.annual_turnover > plain.annual_turnover
    assert scaled.cost_detail["total_fixed_cost"] > plain.cost_detail["total_fixed_cost"]


def test_volatility_targeting_reduces_exposure_in_the_volatile_regime(tmp_path):
    """The fixture's volatility rises 7.5x halfway through; the applied scale must fall."""
    panel, prices, sessions = _panel_and_prices()
    res = run_backtest(
        panel, prices,
        _cfg(tmp_path, sessions, volatility_target={"enabled": True, "window": 30, "min_obs": 30}),
    )
    scales = res.daily["risk_scale"].to_list()
    half = len(scales) // 2
    early = np.nanmean(scales[half - 40:half])
    late = np.nanmean(scales[-40:])
    assert late < early


def test_risk_scale_is_recorded_for_every_session(tmp_path):
    """Diagnostics must line up row-for-row with the P&L, or a reader cannot attribute a
    return to the exposure that produced it."""
    panel, prices, sessions = _panel_and_prices()
    res = run_backtest(
        panel, prices,
        _cfg(tmp_path, sessions, volatility_target={"enabled": True, "window": 30, "min_obs": 30},
             drawdown_control={"enabled": True}),
    )
    for col in ("risk_scale", "realized_vol", "drawdown", "derisked"):
        assert col in res.daily.columns
        assert res.daily[col].len() == res.n_days
    assert res.risk_detail["enabled"] is True
    assert res.risk_detail["n_sessions"] == res.n_days


def test_drawdown_control_reduces_exposure_after_losses(tmp_path):
    """A book that only loses must end up de-risked, and hold less than the uncontrolled
    version over the run."""
    sessions = _sessions(200)
    n_symbols = 10
    symbols = [f"S{i:02d}" for i in range(n_symbols)]
    prow, srow = [], []
    level = {s: 100.0 for s in symbols}
    for d in sessions:
        for k, s in enumerate(symbols):
            # Names the signal ranks highest fall; the ones it ranks lowest rise. A
            # steadily losing long/short book, by construction.
            level[s] *= 0.997 if k < n_symbols // 2 else 1.003
            prow.append({"date": d, "symbol": s, "adj_close": level[s], "turnover": 5e8})
            srow.append({"date": d, "symbol": s, "value": float(n_symbols - k)})
    prices = pl.DataFrame(prow, schema={"date": pl.Date, "symbol": pl.Utf8,
                                        "adj_close": pl.Float64, "turnover": pl.Float64})
    panel = pl.DataFrame(srow, schema={"date": pl.Date, "symbol": pl.Utf8,
                                       "value": pl.Float64})
    res = run_backtest(
        panel, prices,
        _cfg(tmp_path, sessions, drawdown_control={"enabled": True, "threshold": 0.05,
                                         "derisk_to": 0.5, "restore_threshold": 0.02}),
    )
    assert res.risk_detail["sessions_derisked"] > 0
    assert res.daily["derisked"].to_list()[-1] is True
    # De-risking a losing book must lose less than not de-risking it.
    plain = run_backtest(panel, prices, _cfg(tmp_path, sessions))
    assert res.max_drawdown > plain.max_drawdown


def test_overlay_summary_reports_what_actually_happened(tmp_path):
    panel, prices, sessions = _panel_and_prices()
    res = run_backtest(
        panel, prices,
        _cfg(tmp_path, sessions, volatility_target={"enabled": True, "window": 30, "min_obs": 30}),
    )
    s = res.risk_detail
    assert s["volatility_target_enabled"] is True
    assert s["drawdown_control_enabled"] is False
    assert s["min_risk_scale"] <= s["mean_risk_scale"] <= s["max_risk_scale"]
    assert s["target_annual_vol"] == pytest.approx(0.10)


def test_calmar_is_nan_rather_than_infinite_without_a_drawdown():
    """Dividing by a zero denominator would report a spectacular ratio for a series too
    short to have lost money yet."""
    from flowalpha.backtest.engine import BacktestResult

    r = BacktestResult(
        daily=pl.DataFrame(schema={"date": pl.Date, "net_return": pl.Float64}),
        gross_sharpe=1.0, net_sharpe=1.0, gross_annual_return=0.1,
        net_annual_return=0.1, annual_turnover=1.0, total_cost_bps=1.0,
        n_days=0, holding_days=5,
    )
    assert math.isnan(r.calmar)
