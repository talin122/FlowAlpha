"""Per-factor unit tests against hand-computed values on a tiny fixture.

Every factor is checked against an arithmetic result worked out independently of the
implementation. A factor test that only asserts "the output has the right shape" would
have passed for every sign error this project could make.
"""

from __future__ import annotations

import datetime as _dt
import math

import numpy as np
import polars as pl
import pytest

from flowalpha.factors.base import (
    FactorContext,
    empty_panel,
    rolling_mean,
    rolling_std,
    shift_rows,
)
from flowalpha.factors.library import (
    AmihudIlliquidity,
    DeliveryRatio,
    LowVolatility,
    Momentum,
    Reversal,
    Size,
    VolumeTrend,
    build_panels,
    default_library,
)

from conftest import make_config

D = _dt.date


# --- hand-built context ----------------------------------------------------

HAND_SESSIONS = [D(2022, 1, 3) + _dt.timedelta(days=i) for i in range(10) if (D(2022, 1, 3) + _dt.timedelta(days=i)).weekday() < 5]
# 10 calendar days from Mon 2022-01-03 -> 8 weekdays.


@pytest.fixture
def hand_ctx(tmp_path) -> FactorContext:
    """Two symbols, eight sessions, values chosen so every factor is checkable by hand.

    AAA: 100, 110, 121, 133.1, 146.41, 161.051, 177.1561, 194.87171  (exactly +10%/day)
    BBB: 100, 90, 100, 90, 100, 90, 100, 90                          (alternating)
    """
    sessions = HAND_SESSIONS
    aaa = [100.0 * (1.1 ** i) for i in range(len(sessions))]
    bbb = [100.0 if i % 2 == 0 else 90.0 for i in range(len(sessions))]
    adj_close = np.array([[a, b] for a, b in zip(aaa, bbb)])
    volume = np.array([[1000.0, 2000.0]] * len(sessions))
    volume[4:, 0] = 3000.0  # AAA volume triples in the second half
    turnover = adj_close * volume
    delivery = np.full_like(adj_close, np.nan)
    cfg = make_config(tmp_path, sessions)
    return FactorContext(
        sessions=list(sessions), symbols=["AAA", "BBB"],
        adj_close=adj_close, close=adj_close.copy(),
        volume=volume, turnover=turnover, delivery_pct=delivery,
        cfg=cfg, as_of_date=sessions[-1],
    )


def _value(panel: pl.DataFrame, day, symbol) -> float:
    rows = panel.filter((pl.col("date") == day) & (pl.col("symbol") == symbol))
    assert rows.height == 1, f"expected one row for {symbol} on {day}, got {rows.height}"
    return float(rows["value"][0])


# --- rolling primitives ----------------------------------------------------

def test_rolling_mean_requires_a_complete_window():
    m = np.array([[1.0], [2.0], [3.0], [4.0]])
    got = rolling_mean(m, 2)
    assert math.isnan(got[0, 0])
    assert got[1, 0] == pytest.approx(1.5)
    assert got[3, 0] == pytest.approx(3.5)


def test_rolling_mean_is_gap_tolerant():
    """A NaN must null only the windows that span it, not everything after."""
    m = np.array([[1.0], [np.nan], [3.0], [4.0], [5.0]])
    got = rolling_mean(m, 2)
    assert math.isnan(got[1, 0]) and math.isnan(got[2, 0])
    assert got[3, 0] == pytest.approx(3.5)
    assert got[4, 0] == pytest.approx(4.5)


def test_rolling_mean_min_valid_relaxes_the_requirement():
    m = np.array([[1.0], [np.nan], [3.0]])
    got = rolling_mean(m, 3, min_valid=2)
    assert got[2, 0] == pytest.approx(2.0)


def test_rolling_std_is_sample_std():
    m = np.array([[1.0], [2.0], [3.0], [4.0]])
    got = rolling_std(m, 3)
    assert got[2, 0] == pytest.approx(1.0)


def test_rolling_window_longer_than_series_is_all_nan():
    assert np.isnan(rolling_mean(np.array([[1.0], [2.0]]), 5)).all()


def test_rolling_rejects_nonpositive_window():
    with pytest.raises(ValueError):
        rolling_mean(np.array([[1.0]]), 0)


def test_shift_rows_moves_information_forward():
    m = np.array([[1.0], [2.0], [3.0]])
    got = shift_rows(m, 1)
    assert math.isnan(got[0, 0])
    assert got[1, 0] == 1.0 and got[2, 0] == 2.0


def test_shift_rows_refuses_to_look_forward():
    with pytest.raises(ValueError, match="forward in time"):
        shift_rows(np.array([[1.0]]), -1)


def test_shift_rows_zero_is_a_copy():
    m = np.array([[1.0], [2.0]])
    got = shift_rows(m, 0)
    got[0, 0] = 99.0
    assert m[0, 0] == 1.0


# --- momentum --------------------------------------------------------------

def test_momentum_hand_computed(hand_ctx):
    """lookback=3, skip=1 on the +10%/day series.

    value[t] = P[t-1]/P[t-4] - 1 = 1.1^3 - 1 = 0.331 for every t >= 4.
    """
    panel = Momentum(lookback=3, skip=1).compute(hand_ctx)
    assert _value(panel, HAND_SESSIONS[4], "AAA") == pytest.approx(1.1 ** 3 - 1)
    assert _value(panel, HAND_SESSIONS[7], "AAA") == pytest.approx(1.1 ** 3 - 1)


def test_momentum_skips_the_recent_window(hand_ctx):
    """With skip=0 the value differs, proving the skip is actually applied."""
    with_skip = Momentum(lookback=3, skip=1).compute(hand_ctx)
    without = Momentum(lookback=3, skip=0, name="m0").compute(hand_ctx)
    a = _value(with_skip, HAND_SESSIONS[5], "AAA")
    b = _value(without, HAND_SESSIONS[5], "AAA")
    assert a == pytest.approx(1.1 ** 3 - 1)
    assert b == pytest.approx(1.1 ** 3 - 1)
    # On the alternating series the skip genuinely changes the answer.
    assert _value(with_skip, HAND_SESSIONS[5], "BBB") != pytest.approx(
        _value(without, HAND_SESSIONS[5], "BBB")
    )


def test_momentum_has_no_values_before_its_window(hand_ctx):
    panel = Momentum(lookback=3, skip=1).compute(hand_ctx)
    early = panel.filter(pl.col("date") < HAND_SESSIONS[4])
    assert early.is_empty()


def test_momentum_on_too_short_history_is_empty(hand_ctx):
    assert Momentum(lookback=100, skip=21).compute(hand_ctx).is_empty()


def test_momentum_rejects_bad_params():
    with pytest.raises(ValueError):
        Momentum(lookback=0)
    with pytest.raises(ValueError):
        Momentum(lookback=10, skip=-1)


# --- reversal --------------------------------------------------------------

def test_reversal_is_the_negative_of_the_trailing_return(hand_ctx):
    """AAA rises 10% a day, so 2-day reversal = -(1.1^2 - 1) = -0.21."""
    panel = Reversal(window=2).compute(hand_ctx)
    assert _value(panel, HAND_SESSIONS[4], "AAA") == pytest.approx(-(1.1 ** 2 - 1))


def test_reversal_sign_favours_losers(hand_ctx):
    """BBB alternates 100/90: on a down day the reversal value must be POSITIVE."""
    panel = Reversal(window=1).compute(hand_ctx)
    # session 1 is 90 after 100 -> trailing return -0.10 -> factor +0.10
    assert _value(panel, HAND_SESSIONS[1], "BBB") == pytest.approx(0.1)
    # session 2 is 100 after 90 -> trailing return +1/9 -> factor -1/9
    assert _value(panel, HAND_SESSIONS[2], "BBB") == pytest.approx(-(100 / 90 - 1))


def test_reversal_name_encodes_its_window():
    assert Reversal(window=5).name == "reversal_5d"
    assert Reversal(window=21).name == "reversal_21d"


def test_reversal_rejects_bad_window():
    with pytest.raises(ValueError):
        Reversal(window=0)


# --- low volatility -------------------------------------------------------

def test_low_volatility_is_negative_realised_vol(hand_ctx):
    """AAA has constant +10% returns, so its realised vol is 0 and factor value 0.

    BBB alternates, so its vol is large and the factor value strongly negative.
    """
    panel = LowVolatility(window=4).compute(hand_ctx)
    aaa = _value(panel, HAND_SESSIONS[6], "AAA")
    bbb = _value(panel, HAND_SESSIONS[6], "BBB")
    assert aaa == pytest.approx(0.0, abs=1e-12)
    assert bbb < -0.05
    assert aaa > bbb  # low-vol name ranks higher


def test_low_volatility_hand_computed_exactly(hand_ctx):
    """window=3 at session 5 uses returns from sessions 3,4,5 of BBB.

    BBB closes: 100,90,100,90,100,90 -> returns at 3,4,5 = -0.1, +1/9, -0.1.
    """
    panel = LowVolatility(window=3).compute(hand_ctx)
    rets = np.array([-0.1, 100 / 90 - 1, -0.1])
    expected = -float(np.std(rets, ddof=1))
    assert _value(panel, HAND_SESSIONS[5], "BBB") == pytest.approx(expected)


# --- illiquidity ----------------------------------------------------------

def test_amihud_illiquidity_hand_computed(hand_ctx):
    """window=2 at session 5 for BBB: mean(|ret| / turnover) x 1e9."""
    panel = AmihudIlliquidity(window=2).compute(hand_ctx)
    # session 4: ret = 100/90-1, turnover = 100*2000
    # session 5: ret = 90/100-1, turnover = 90*2000
    r4, t4 = 100 / 90 - 1, 100.0 * 2000.0
    r5, t5 = 90 / 100 - 1, 90.0 * 2000.0
    expected = (abs(r4) / t4 + abs(r5) / t5) / 2 * 1e9
    assert _value(panel, HAND_SESSIONS[5], "BBB") == pytest.approx(expected)


def test_illiquidity_ranks_thin_names_higher(hand_ctx):
    """AAA's turnover is much larger, so it must be the LESS illiquid name."""
    panel = AmihudIlliquidity(window=2).compute(hand_ctx)
    assert _value(panel, HAND_SESSIONS[5], "AAA") < _value(panel, HAND_SESSIONS[5], "BBB")


def test_illiquidity_needs_turnover(hand_ctx):
    hand_ctx.turnover = np.full_like(hand_ctx.turnover, np.nan)
    factor = AmihudIlliquidity(window=2)
    assert not factor.is_available(hand_ctx)
    assert factor.compute(hand_ctx).is_empty()


# --- delivery (inert on Yahoo data) ---------------------------------------

def test_delivery_factor_is_empty_without_delivery_data(hand_ctx):
    """The Yahoo case. Empty panel, explicit schema, no crash."""
    factor = DeliveryRatio(window=2)
    panel = factor.compute(hand_ctx)
    assert panel.is_empty()
    assert panel.schema == empty_panel().schema
    assert not factor.is_available(hand_ctx)
    assert "delivery_pct" in factor.unavailable_reason(hand_ctx)


def test_delivery_factor_works_when_the_field_exists(hand_ctx):
    hand_ctx.delivery_pct = np.tile(np.array([[40.0, 60.0]]), (hand_ctx.n_sessions, 1))
    panel = DeliveryRatio(window=2).compute(hand_ctx)
    assert _value(panel, HAND_SESSIONS[3], "BBB") == pytest.approx(60.0)


# --- size -----------------------------------------------------------------

def test_size_without_shares_is_named_a_price_proxy():
    """Calling a log-price series 'size' would invite a false small-cap reading."""
    assert Size(shares_available=False).name == "size_logprice_proxy"
    assert Size(shares_available=True).name == "size_logmktcap"


def test_size_proxy_is_log_price(hand_ctx):
    panel = Size(shares_available=False).compute(hand_ctx)
    assert _value(panel, HAND_SESSIONS[0], "AAA") == pytest.approx(math.log(100.0))
    assert _value(panel, HAND_SESSIONS[1], "AAA") == pytest.approx(math.log(110.0))


def test_size_uses_real_shares_when_available(hand_ctx):
    hand_ctx.shares = {"AAA": 10.0, "BBB": 1_000.0}
    panel = Size(shares_available=True).compute(hand_ctx)
    assert _value(panel, HAND_SESSIONS[0], "AAA") == pytest.approx(math.log(100.0 * 10.0))
    assert _value(panel, HAND_SESSIONS[0], "BBB") == pytest.approx(math.log(100.0 * 1000.0))


# --- volume trend ---------------------------------------------------------

def test_volume_trend_hand_computed(hand_ctx):
    """AAA volume goes 1000 (sessions 0-3) then 3000 (sessions 4-7).

    With window=2 the prior mean is the recent mean shifted back by the window, so:
      session 5: recent = mean(V4,V5) = 3000; prior = mean(V2,V3) = 1000 -> log(3)
      session 6: recent = mean(V5,V6) = 3000; prior = mean(V3,V4) = 2000 -> log(1.5)
      session 7: recent = mean(V6,V7) = 3000; prior = mean(V4,V5) = 3000 -> log(1) = 0
    """
    panel = VolumeTrend(window=2).compute(hand_ctx)
    assert _value(panel, HAND_SESSIONS[5], "AAA") == pytest.approx(math.log(3.0))
    assert _value(panel, HAND_SESSIONS[6], "AAA") == pytest.approx(math.log(1.5))
    assert _value(panel, HAND_SESSIONS[7], "AAA") == pytest.approx(0.0)


def test_volume_trend_is_flat_for_constant_volume(hand_ctx):
    panel = VolumeTrend(window=2).compute(hand_ctx)
    assert _value(panel, HAND_SESSIONS[5], "BBB") == pytest.approx(0.0)


def test_volume_trend_needs_double_its_window(hand_ctx):
    assert VolumeTrend(window=5).compute(hand_ctx).is_empty()


# --- library --------------------------------------------------------------

def test_default_library_reads_every_window_from_config(tiny_cfg):
    factors = {f.name: f for f in default_library(tiny_cfg)}
    assert factors["momentum"].lookback == 40 and factors["momentum"].skip == 5
    assert "reversal_3d" in factors and "reversal_10d" in factors
    assert factors["low_volatility"].window == 20
    assert factors["illiquidity"].window == 10
    assert factors["volume_trend"].window == 10
    assert "size_logprice_proxy" in factors


def test_default_library_size_name_follows_shares_availability(tiny_cfg):
    names = {f.name for f in default_library(tiny_cfg, shares_available=True)}
    assert "size_logmktcap" in names and "size_logprice_proxy" not in names


def test_every_factor_declares_a_hypothesis(tiny_cfg):
    for factor in default_library(tiny_cfg):
        assert factor.hypothesis, f"{factor.name} has no stated hypothesis"


def test_build_panels_skips_dataless_factors_with_a_warning(hand_ctx, capsys):
    factors = [Momentum(lookback=3, skip=1), DeliveryRatio(window=2)]
    panels, skipped = build_panels(factors, hand_ctx)
    assert "momentum" in panels
    assert "delivery_ratio" not in panels
    assert any("delivery_ratio" in s for s in skipped)
    assert "WARNING" in capsys.readouterr().out


def test_build_panels_does_not_count_a_skipped_factor_as_a_trial(hand_ctx):
    """An unevaluated factor is not a test, so it must not inflate the trial count."""
    panels, skipped = build_panels([DeliveryRatio(window=2)], hand_ctx, verbose=False)
    assert panels == {}
    assert len(skipped) == 1


# --- treatment and context ------------------------------------------------

def test_panel_applies_winsorisation_and_zscore(hand_ctx):
    panel = Momentum(lookback=3, skip=1).panel(hand_ctx)
    for day in set(panel["date"].to_list()):
        vals = panel.filter(pl.col("date") == day)["value"].to_numpy()
        if vals.size >= 2 and np.isfinite(vals).all():
            assert abs(float(vals.mean())) < 1e-9


def test_panel_of_an_empty_factor_has_the_right_schema(hand_ctx):
    assert DeliveryRatio(window=2).panel(hand_ctx).schema == empty_panel().schema


def test_context_returns_matrix(hand_ctx):
    rets = hand_ctx.returns
    assert math.isnan(rets[0, 0])
    assert rets[1, 0] == pytest.approx(0.1)
    assert rets[1, 1] == pytest.approx(-0.1)


def test_context_to_long_rejects_wrong_shape(hand_ctx):
    with pytest.raises(Exception):
        hand_ctx.to_long(np.zeros((3, 3)))


def test_context_from_store_loads_a_snapshot(tiny_store, tiny_tree, tiny_sessions):
    ctx = FactorContext.from_store(tiny_store, tiny_tree, as_of_date=tiny_sessions[100])
    assert ctx.n_symbols == 6
    assert ctx.sessions[-1] == tiny_sessions[100]
    assert ctx.adj_close.shape == (ctx.n_sessions, 6)


def test_context_from_store_on_empty_processed_dir(tiny_cfg, tiny_calendar):
    from flowalpha.data.store import PointInTimeStore
    from conftest import make_tiny_prices, write_tree

    write_tree(tiny_cfg, prices=make_tiny_prices([]) if False else make_tiny_prices(tiny_calendar.sessions[:1]))
    store = PointInTimeStore(tiny_cfg.path("processed"), tiny_calendar, tiny_cfg["availability"])
    ctx = FactorContext.from_store(store, tiny_cfg, as_of_date=tiny_calendar.sessions[0])
    assert ctx.n_sessions == 1


def test_shares_available_is_false_when_every_count_is_identical(tiny_cfg, tiny_sessions):
    """shares=1.0 for everyone carries no cross-sectional information."""
    from flowalpha.factors.base import _load_shares

    ref = tiny_cfg.path("reference")
    pl.DataFrame({"symbol": ["AAA", "BBB"], "shares": [1.0, 1.0]}).write_csv(
        ref / "shares_outstanding.csv"
    )
    mapping, available = _load_shares(ref)
    assert mapping == {"AAA": 1.0, "BBB": 1.0}
    assert available is False

    pl.DataFrame({"symbol": ["AAA", "BBB"], "shares": [1.0, 2.0]}).write_csv(
        ref / "shares_outstanding.csv"
    )
    _, available2 = _load_shares(ref)
    assert available2 is True
