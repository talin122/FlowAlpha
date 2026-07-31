"""Portfolio construction, constraints, trade list, and persisted state."""

from __future__ import annotations

import datetime as _dt
import json
import math

import numpy as np
import polars as pl
import pytest

from flowalpha.backtest.costs import CostModel
from flowalpha.signals.portfolio import (
    TRADE_SCHEMA,
    _cap_and_redistribute,
    build_portfolio,
)
from flowalpha.signals.state import (
    Holdings,
    archive_holdings,
    load_holdings,
    save_holdings,
)

from conftest import make_config, make_tiny_prices

D = _dt.date
AS_OF = D(2022, 3, 1)


def _sessions(n: int = 40) -> list[_dt.date]:
    out, day = [], D(2022, 1, 3)
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += _dt.timedelta(days=1)
    return out


SESSIONS = _sessions()
SYMBOLS = [f"S{i:02d}" for i in range(40)]


def _signal(as_of=None) -> pl.DataFrame:
    as_of = as_of or SESSIONS[-1]
    return pl.DataFrame(
        [{"date": as_of, "symbol": s, "value": float(i)} for i, s in enumerate(SYMBOLS)],
        schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64},
    )


@pytest.fixture
def prices():
    return make_tiny_prices(SESSIONS, symbols=tuple(SYMBOLS))


@pytest.fixture
def cfg(tmp_path):
    base = make_config(tmp_path, SESSIONS)
    return base.with_overrides(
        signals={
            **base["signals"],
            "construction": {
                "method": "quintile_ls", "long_only": False, "top_n": 10,
                "max_weight": 0.10, "sector_cap": 0.8, "gross_leverage": 1.0,
                "adv_participation_cap": 1.0, "rebalance_band": 0.0,
            },
        }
    )


# --- cap and redistribute --------------------------------------------------

def test_cap_redistributes_rather_than_shrinking_gross():
    """A plain clip would take gross from 1.0 to 0.75 without saying so.

    The cap must be feasible for the redistribution to preserve gross: four names per
    side at a 0.3 cap can hold 0.5 of gross, so the excess from the 0.4 name has
    somewhere to go.
    """
    weights = np.array([0.4, 0.1, -0.4, -0.1])
    out = _cap_and_redistribute(weights, 0.3)
    assert np.abs(out).max() <= 0.3 + 1e-12
    assert np.abs(out).sum() == pytest.approx(np.abs(weights).sum())
    assert out[out > 0].sum() == pytest.approx(0.5)


def test_cap_preserves_the_long_short_split():
    weights = np.array([0.3, 0.2, -0.2, -0.3])
    out = _cap_and_redistribute(weights, 0.26)
    assert out[out > 0].sum() == pytest.approx(0.5)
    assert out[out < 0].sum() == pytest.approx(-0.5)


def test_cap_gives_up_gracefully_when_there_is_no_room():
    """Two names cannot hold gross 1.0 under a 0.1 cap; gross falls and the caller is
    told (via the portfolio warning), rather than the cap being silently ignored."""
    weights = np.array([0.5, -0.5])
    out = _cap_and_redistribute(weights, 0.1)
    assert np.abs(out).max() <= 0.1 + 1e-12
    assert np.abs(out).sum() < np.abs(weights).sum()


def test_cap_of_zero_is_a_noop():
    weights = np.array([0.5, -0.5])
    assert _cap_and_redistribute(weights, 0.0) == pytest.approx(weights)


# --- construction ---------------------------------------------------------

def test_quintile_long_short_is_dollar_neutral(cfg, prices):
    p = build_portfolio(_signal(), prices, cfg, as_of_date=SESSIONS[-1], notional=1e9)
    assert p.n_long == 10 and p.n_short == 10
    assert p.net == pytest.approx(0.0, abs=1e-12)
    assert p.gross == pytest.approx(1.0)


def test_long_only_holds_only_longs(cfg, prices):
    long_cfg = cfg.with_overrides(
        signals={**cfg["signals"],
                 "construction": {**cfg["signals"]["construction"], "long_only": True}}
    )
    p = build_portfolio(_signal(), prices, long_cfg, as_of_date=SESSIONS[-1], notional=1e9)
    assert p.n_short == 0
    assert p.net == pytest.approx(1.0)


def test_max_weight_is_respected(cfg, prices):
    tight = cfg.with_overrides(
        signals={**cfg["signals"],
                 "construction": {**cfg["signals"]["construction"], "max_weight": 0.06}}
    )
    p = build_portfolio(_signal(), prices, tight, as_of_date=SESSIONS[-1], notional=1e9)
    assert p.holdings["weight"].abs().max() <= 0.06 + 1e-9


def test_impossible_max_weight_is_reported(cfg, prices):
    """Two names per leg cannot hold 0.5 of gross under a 1% per-name cap."""
    tiny = cfg.with_overrides(
        signals={**cfg["signals"],
                 "construction": {**cfg["signals"]["construction"],
                                  "top_n": 2, "max_weight": 0.01}}
    )
    p = build_portfolio(_signal(), prices, tiny, as_of_date=SESSIONS[-1], notional=1e9)
    assert p.constraints_applied["projection"]["feasible"] is False
    assert any("jointly infeasible" in w for w in p.warnings)
    # The per-name cap itself is still honoured.
    assert float(p.holdings["weight"].abs().max()) <= 0.01 + 1e-9


def test_gross_leverage_is_applied(cfg, prices):
    levered = cfg.with_overrides(
        signals={**cfg["signals"],
                 "construction": {**cfg["signals"]["construction"], "gross_leverage": 2.0,
                                  "max_weight": 0.5}}
    )
    p = build_portfolio(_signal(), prices, levered, as_of_date=SESSIONS[-1], notional=1e9)
    assert p.gross == pytest.approx(2.0)


def _sector_map(n_sectors: int = 4) -> dict[str, str]:
    """Assign sectors so BOTH legs span several of them, with one dominant.

    Both properties matter. A per-leg share cap of ``c`` needs at least ``1/c`` sectors
    *on that leg* to be satisfiable at full gross -- with a single sector per leg the cap
    is unachievable while staying dollar-neutral, whatever the code does. And one sector
    has to dominate or the cap never binds and the test proves nothing.

    Shorts are indices 0-9, longs 30-39 (top_n=10 of 40 names).
    """
    names = ["IT", "Bank", "Pharma", "Energy"][:n_sectors]
    out = {}
    for i, s in enumerate(SYMBOLS):
        if i < 10:                      # short leg: 5 IT, 2 Bank, 2 Pharma, 1 Energy
            out[s] = names[0] if i < 5 else names[min(1 + (i - 5) // 2, n_sectors - 1)]
        elif i >= 30:                   # long leg: same shape
            j = i - 30
            out[s] = names[0] if j < 5 else names[min(1 + (j - 5) // 2, n_sectors - 1)]
        else:
            out[s] = names[i % n_sectors]
    return out


def test_sector_cap_binds_and_holds_on_the_final_book(cfg, prices):
    capped = cfg.with_overrides(
        signals={**cfg["signals"],
                 "construction": {**cfg["signals"]["construction"], "sector_cap": 0.4}}
    )
    p = build_portfolio(_signal(), prices, capped, as_of_date=SESSIONS[-1],
                        sectors=_sector_map(), notional=1e9)
    shares = (
        p.holdings.with_columns(pl.col("weight").abs().alias("g"))
        .group_by("sector").agg(pl.col("g").sum())
    )
    for row in shares.iter_rows(named=True):
        assert float(row["g"]) / p.gross <= 0.4 + 1e-6, row
    assert p.constraints_applied["satisfied"]["sector_cap"] is True
    assert p.constraints_applied["satisfied"]["gross_leverage"] is True


def test_sector_cap_does_not_break_dollar_neutrality(cfg, prices):
    """The bug this guards.

    Capping a sector across BOTH legs at once removes more exposure from whichever leg the
    sector is concentrated in, so a nominally market-neutral book acquires a directional
    bet. Measured at -13% net before each leg was projected separately.
    """
    capped = cfg.with_overrides(
        signals={**cfg["signals"],
                 "construction": {**cfg["signals"]["construction"], "sector_cap": 0.4}}
    )
    # Deliberately lopsided: the dominant sector sits mostly on the long leg.
    sectors = {}
    for i, s in enumerate(SYMBOLS):
        if i >= 30:
            sectors[s] = "IT" if i < 38 else ("Bank" if i == 38 else "Pharma")
        elif i < 10:
            sectors[s] = ["IT", "Bank", "Pharma", "Energy"][i % 4]
        else:
            sectors[s] = "Energy"
    p = build_portfolio(_signal(), prices, capped, as_of_date=SESSIONS[-1],
                        sectors=sectors, notional=1e9)
    assert p.net == pytest.approx(0.0, abs=1e-9)
    assert p.constraints_applied["satisfied"]["dollar_neutral"] is True
    long_gross = float(p.holdings.filter(pl.col("weight") > 0)["weight"].sum())
    short_gross = float(-p.holdings.filter(pl.col("weight") < 0)["weight"].sum())
    assert long_gross == pytest.approx(short_gross, abs=1e-9)


def test_single_sector_makes_the_cap_inert_not_infeasible(cfg, prices):
    """A share cap needs >= 2 buckets. With one, its share IS 1.0.

    Treating that as binding capped the leg at `sector_cap * target` and ran the whole
    book at a fraction of its intended size -- 0.005 gross against a target of 1.0 with the
    shipped sector_cap of 0.25.
    """
    capped = cfg.with_overrides(
        signals={**cfg["signals"],
                 "construction": {**cfg["signals"]["construction"], "sector_cap": 0.25}}
    )
    p = build_portfolio(_signal(), prices, capped, as_of_date=SESSIONS[-1], notional=1e9)
    assert p.gross == pytest.approx(1.0, abs=1e-9)
    assert p.constraints_applied["satisfied"]["gross_leverage"] is True
    assert any("inert" in w for w in p.warnings)


def test_infeasible_constraints_are_reported_not_hidden(cfg, prices):
    """Shorts all in one sector, longs all in another: the 40% cap cannot hold while
    dollar-neutral, because each leg is then exactly 50% of gross. The code must SAY so."""
    sectors = {s: ("IT" if i < 30 else "Bank") for i, s in enumerate(SYMBOLS)}
    capped = cfg.with_overrides(
        signals={**cfg["signals"],
                 "construction": {**cfg["signals"]["construction"], "sector_cap": 0.4}}
    )
    p = build_portfolio(_signal(), prices, capped, as_of_date=SESSIONS[-1],
                        sectors=sectors, notional=1e9)
    assert p.constraints_applied["satisfied"]["sector_cap"] is False
    assert any("sector_cap is NOT satisfied" in w for w in p.warnings)
    # Dollar-neutrality is never sacrificed to a concentration cap.
    assert p.net == pytest.approx(0.0, abs=1e-9)


def test_max_weight_holds_on_the_final_book(cfg, prices):
    tight = cfg.with_overrides(
        signals={**cfg["signals"],
                 "construction": {**cfg["signals"]["construction"], "max_weight": 0.06}}
    )
    p = build_portfolio(_signal(), prices, tight, as_of_date=SESSIONS[-1],
                        sectors=_sector_map(), notional=1e9)
    assert float(p.holdings["weight"].abs().max()) <= 0.06 + 1e-9
    assert p.constraints_applied["satisfied"]["max_weight"] is True


def test_adv_participation_cap_limits_position_size(cfg, prices):
    from flowalpha.backtest.costs import rolling_adv

    adv = rolling_adv(prices, 10)
    capped = cfg.with_overrides(
        signals={**cfg["signals"],
                 "construction": {**cfg["signals"]["construction"],
                                  "adv_participation_cap": 0.0001}}
    )
    p = build_portfolio(_signal(), prices, capped, as_of_date=SESSIONS[-1],
                        adv=adv, notional=1e12)
    assert p.constraints_applied["adv_participation_cap"]["n_binding"] > 0
    # An ADV limit this tight cannot absorb the target gross, and that is reported rather
    # than papered over by scaling the book back up.
    assert p.constraints_applied["projection"]["feasible"] is False
    assert any("jointly infeasible" in w for w in p.warnings)


def test_missing_adv_is_reported_not_invented(cfg, prices):
    capped = cfg.with_overrides(
        signals={**cfg["signals"],
                 "construction": {**cfg["signals"]["construction"],
                                  "adv_participation_cap": 0.05}}
    )
    p = build_portfolio(_signal(), prices, capped, as_of_date=SESSIONS[-1],
                        adv=None, notional=1e9)
    assert p.constraints_applied["adv_participation_cap"]["n_without_adv"] > 0
    assert any("no ADV estimate" in w for w in p.warnings)


def test_rebalance_band_leaves_small_drifts_alone(cfg, prices):
    """A book must not churn on noise."""
    banded = cfg.with_overrides(
        signals={**cfg["signals"],
                 "construction": {**cfg["signals"]["construction"], "rebalance_band": 0.5}}
    )
    prior = Holdings(as_of=SESSIONS[-2], weights={s: 0.0 for s in SYMBOLS})
    p = build_portfolio(_signal(), prices, banded, as_of_date=SESSIONS[-1],
                        prior=prior, notional=1e9)
    assert p.constraints_applied["rebalance_band"]["n_left_alone"] > 0


def test_no_signal_for_the_date_gives_an_empty_portfolio(cfg, prices):
    p = build_portfolio(_signal(SESSIONS[0]), prices, cfg, as_of_date=SESSIONS[-1],
                        notional=1e9)
    assert p.holdings.is_empty()
    assert p.trades.is_empty()
    assert any("no signal" in w for w in p.warnings)


def test_too_few_names_gives_an_empty_portfolio(cfg, prices):
    thin = pl.DataFrame(
        [{"date": SESSIONS[-1], "symbol": "S00", "value": 1.0}],
        schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64},
    )
    p = build_portfolio(thin, prices, cfg, as_of_date=SESSIONS[-1], notional=1e9)
    assert p.holdings.is_empty()


# --- trades ---------------------------------------------------------------

def test_first_run_trades_the_whole_book(cfg, prices):
    p = build_portfolio(_signal(), prices, cfg, as_of_date=SESSIONS[-1], notional=1e9)
    assert p.turnover == pytest.approx(1.0)
    assert p.trades.height == 20


def test_no_change_means_no_trades(cfg, prices):
    first = build_portfolio(_signal(), prices, cfg, as_of_date=SESSIONS[-1], notional=1e9)
    second = build_portfolio(
        _signal(), prices, cfg, as_of_date=SESSIONS[-1],
        prior=first.to_holdings(), notional=1e9,
    )
    assert second.trades.is_empty()
    assert second.turnover == pytest.approx(0.0)


def test_exits_of_dropped_names_appear_in_the_trade_list(cfg, prices):
    """Names in yesterday's book but absent from today's candidates must be sold.

    Omitting them would understate turnover and leave the state file describing a book
    that was never held.
    """
    prior = Holdings(as_of=SESSIONS[-2], weights={"ZZZ_NOT_IN_UNIVERSE": 0.2})
    p = build_portfolio(_signal(), prices, cfg, as_of_date=SESSIONS[-1],
                        prior=prior, notional=1e9)
    exits = p.trades.filter(pl.col("symbol") == "ZZZ_NOT_IN_UNIVERSE")
    assert exits.height == 1
    assert exits["side"][0] == "SELL"
    assert float(exits["target_weight"][0]) == pytest.approx(0.0)


def test_trade_list_schema_and_cost_bps(cfg, prices):
    from flowalpha.backtest.costs import rolling_adv

    adv = rolling_adv(prices, 10)
    p = build_portfolio(_signal(), prices, cfg, as_of_date=SESSIONS[-1],
                        adv=adv, notional=1e9)
    assert p.trades.schema == TRADE_SCHEMA
    assert (p.trades["cost_bps"] > 0).all()
    buys = p.trades.filter(pl.col("side") == "BUY")
    sells = p.trades.filter(pl.col("side") == "SELL")
    # Sell-side carries STT, so it must cost more in fixed terms.
    assert float(sells["cost_bps"].mean()) > float(buys["cost_bps"].mean())


def test_trade_cost_reflects_the_cost_model(cfg, prices):
    free = CostModel(stt_delivery=0.0, stamp_duty_buy=0.0, brokerage_exchange=0.0,
                     impact_coef=0.0)
    p = build_portfolio(_signal(), prices, cfg, as_of_date=SESSIONS[-1],
                        notional=1e9, cost_model=free)
    assert p.trades["cost_bps"].abs().max() == pytest.approx(0.0)
    assert p.total_cost_bps == pytest.approx(0.0)


def test_participation_is_blank_without_adv(cfg, prices):
    p = build_portfolio(_signal(), prices, cfg, as_of_date=SESSIONS[-1], notional=1e9)
    assert all(math.isnan(v) for v in p.trades["participation"].to_list())


def test_portfolio_to_dict_records_constraints(cfg, prices):
    p = build_portfolio(_signal(), prices, cfg, as_of_date=SESSIONS[-1], notional=1e9)
    d = p.to_dict()
    assert d["n_long"] == 10 and d["n_short"] == 10
    assert "max_weight" in d["constraints"]
    assert d["as_of"] == SESSIONS[-1].isoformat()


# --- state ---------------------------------------------------------------

def test_missing_state_is_flat_and_says_so(tmp_path):
    """A first run must report a genuine full build, not a fake flat-to-flat."""
    h = load_holdings(tmp_path)
    assert h.weights == {}
    assert h.as_of is None
    assert "first run" in h.source


def test_state_roundtrip(tmp_path):
    h = Holdings(as_of=D(2026, 7, 30), weights={"A": 0.5, "B": -0.5}, source="test")
    save_holdings(h, tmp_path)
    again = load_holdings(tmp_path)
    assert again.weights == h.weights
    assert again.as_of == h.as_of
    assert again.source == "test"


def test_state_gross_and_net(tmp_path):
    h = Holdings(as_of=None, weights={"A": 0.3, "B": -0.2})
    assert h.gross == pytest.approx(0.5)
    assert h.net == pytest.approx(0.1)


def test_corrupt_state_falls_back_to_flat(tmp_path):
    (tmp_path / "holdings.json").write_text("{not json", encoding="utf-8")
    assert load_holdings(tmp_path).weights == {}


def test_state_frame_roundtrip():
    h = Holdings(as_of=D(2026, 1, 1), weights={"A": 0.5, "B": -0.5})
    again = Holdings.from_frame(h.to_frame(), as_of=h.as_of)
    assert again.weights == h.weights


def test_empty_holdings_frame_has_schema():
    assert Holdings.empty().to_frame().schema["weight"] == pl.Float64


def test_archive_writes_a_dated_copy(tmp_path):
    h = Holdings(as_of=D(2026, 7, 30), weights={"A": 1.0})
    path = archive_holdings(h, tmp_path / "daily" / "2026-07-30")
    assert path.exists()
    assert json.loads(path.read_text())["weights"] == {"A": 1.0}


def test_portfolio_to_holdings_carries_the_date(cfg, prices):
    p = build_portfolio(_signal(), prices, cfg, as_of_date=SESSIONS[-1], notional=1e9)
    h = p.to_holdings(source="run")
    assert h.as_of == SESSIONS[-1]
    assert len(h.weights) == 20
    assert h.source == "run"


def test_second_day_trades_are_real_deltas(cfg, prices):
    """Yesterday's holdings are an input; without them every day reports a full build."""
    day1 = build_portfolio(_signal(SESSIONS[-2]), prices, cfg, as_of_date=SESSIONS[-2],
                           notional=1e9)
    reversed_signal = pl.DataFrame(
        [{"date": SESSIONS[-1], "symbol": s, "value": float(-i)}
         for i, s in enumerate(SYMBOLS)],
        schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64},
    )
    day2 = build_portfolio(reversed_signal, prices, cfg, as_of_date=SESSIONS[-1],
                           prior=day1.to_holdings(), notional=1e9)
    # A fully reversed signal must trade roughly twice the book.
    assert day2.turnover > 1.5
