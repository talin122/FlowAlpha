"""Regression tests for defects found in the post-build audit.

Each test corresponds to a bug that was live in the codebase and is named for the
*symptom a user would have seen*, not for the line that was wrong. They are collected
here rather than scattered so the audit's findings stay legible as a set.
"""

from __future__ import annotations

import datetime as _dt
import math

import numpy as np
import polars as pl
import pytest

from flowalpha.config import load_config
from flowalpha.data.quality import check_universe_cardinality
from flowalpha.data.store import LookAheadError, PointInTimeStore
from flowalpha.mining.grammar import build_vocabulary, is_complete
from flowalpha.mining.miner import UNSCORABLE_REWARD, formula_reward, mine
from flowalpha.signals.portfolio import _project_side, build_portfolio
from flowalpha.validation.ic import forward_returns, trailing_ic_series

from conftest import make_config, make_tiny_flows, make_tiny_prices, write_tree

D = _dt.date


def _sessions(n: int, start=D(2022, 1, 3)) -> list[_dt.date]:
    out, day = [], start
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += _dt.timedelta(days=1)
    return out


SESSIONS = _sessions(60)
SYMBOLS = tuple(f"S{i:02d}" for i in range(40))


def _signal(as_of=None) -> pl.DataFrame:
    as_of = as_of or SESSIONS[-1]
    return pl.DataFrame(
        [{"date": as_of, "symbol": s, "value": float(i)} for i, s in enumerate(SYMBOLS)],
        schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64},
    )


@pytest.fixture
def prices():
    return make_tiny_prices(SESSIONS, symbols=SYMBOLS)


def _construction(base, **over):
    c = {"method": "quintile_ls", "long_only": False, "top_n": 10, "max_weight": 0.2,
         "sector_cap": 0.25, "gross_leverage": 1.0, "adv_participation_cap": 1.0,
         "rebalance_band": 0.0}
    c.update(over)
    return base.with_overrides(signals={**base["signals"], "construction": c})


# ---------------------------------------------------------------------------
# A. Config immutability was only skin-deep
# ---------------------------------------------------------------------------

def test_nested_config_mutation_cannot_reach_the_live_config():
    """`cfg["universe"]["size"] = 999` used to succeed.

    Blocking only top-level assignment left the nested dicts writable, so a run could
    mutate its own parameters mid-flight and its run-directory config snapshot would then
    describe something other than what actually ran.
    """
    cfg = load_config()
    original = cfg["universe"]["size"]
    handed_out = cfg["universe"]
    handed_out["size"] = 999
    assert cfg["universe"]["size"] == original
    assert cfg.get("universe.size") == original


def test_dotted_get_also_hands_out_a_detached_copy():
    cfg = load_config()
    original = cfg.get("factors.momentum.lookback")
    node = cfg.get("factors.momentum")
    node["lookback"] = -1
    assert cfg.get("factors.momentum.lookback") == original


def test_list_values_are_detached_too():
    cfg = load_config()
    band = cfg["universe"]["cardinality_band"]
    original = list(band)
    band[0] = -1
    assert cfg["universe"]["cardinality_band"] == original


def test_top_level_assignment_is_still_blocked():
    cfg = load_config()
    with pytest.raises(TypeError):
        cfg["seed"] = 1


# ---------------------------------------------------------------------------
# B. A single-sector book ran at a fraction of its target gross
# ---------------------------------------------------------------------------

def test_single_sector_book_still_reaches_target_gross(prices, tmp_path):
    """With no sector labels and the shipped sector_cap of 0.25, gross collapsed to 0.005
    against a target of 1.0 -- a 190x under-investment -- while simultaneously reporting
    the cap as inert. A share cap needs at least two buckets to mean anything."""
    cfg = _construction(make_config(tmp_path, SESSIONS), sector_cap=0.25)
    p = build_portfolio(_signal(), prices, cfg, as_of_date=SESSIONS[-1], notional=1e9)
    assert p.gross == pytest.approx(1.0, abs=1e-9)
    assert p.constraints_applied["satisfied"]["gross_leverage"] is True
    assert any("inert" in w for w in p.warnings)


def test_a_uniform_rescale_cannot_change_a_share_based_cap():
    """The guard that caused the collapse tested a scale-invariant quantity."""
    from flowalpha.signals.portfolio import _sector_shares

    w = np.array([0.3, -0.2, 0.5, -0.1])
    sec = np.array(["A", "A", "B", "B"], dtype=object)
    assert _sector_shares(w, sec) == pytest.approx(_sector_shares(w * 0.37, sec))


# ---------------------------------------------------------------------------
# K. The sector cap gave a market-neutral book a directional bet
# ---------------------------------------------------------------------------

def test_sector_cap_cannot_introduce_net_exposure(prices, tmp_path):
    """Capping a sector across BOTH legs at once removed more exposure from whichever leg
    it was concentrated in. A nominally dollar-neutral book acquired a -13% net position.
    """
    sectors = {}
    for i, s in enumerate(SYMBOLS):
        if i >= 30:                                    # long leg: IT-heavy
            sectors[s] = "IT" if i < 38 else ("Bank" if i == 38 else "Pharma")
        elif i < 10:                                   # short leg: spread
            sectors[s] = ["IT", "Bank", "Pharma", "Energy"][i % 4]
        else:
            sectors[s] = "Energy"
    cfg = _construction(make_config(tmp_path, SESSIONS), sector_cap=0.4)
    p = build_portfolio(_signal(), prices, cfg, as_of_date=SESSIONS[-1],
                        sectors=sectors, notional=1e9)
    assert p.net == pytest.approx(0.0, abs=1e-9)
    long_gross = float(p.holdings.filter(pl.col("weight") > 0)["weight"].sum())
    short_gross = float(-p.holdings.filter(pl.col("weight") < 0)["weight"].sum())
    assert long_gross == pytest.approx(short_gross, abs=1e-9)


def test_an_infeasible_leg_never_leaves_the_book_lopsided(prices, tmp_path):
    """If one leg cannot reach its target, trimming the other keeps neutrality.

    Long 0.40 against short 0.50 is a 10% net market bet nobody asked for.
    """
    sectors = {}
    for i, s in enumerate(SYMBOLS):
        if i >= 30:
            sectors[s] = "IT" if i < 38 else ("Bank" if i == 38 else "Pharma")
        elif i < 10:
            sectors[s] = ["IT", "Bank", "Pharma", "Energy"][i % 4]
        else:
            sectors[s] = "Energy"
    cfg = _construction(make_config(tmp_path, SESSIONS), sector_cap=0.4, max_weight=0.10)
    p = build_portfolio(_signal(), prices, cfg, as_of_date=SESSIONS[-1],
                        sectors=sectors, notional=1e9)
    assert p.net == pytest.approx(0.0, abs=1e-9)
    if not p.constraints_applied["projection"]["legs_balanced"]:
        assert any("dollar-neutral" in w for w in p.warnings)


def test_every_constraint_is_verified_on_the_final_book(prices, tmp_path):
    """Constraints interact, so the only trustworthy claim is one made after the fact."""
    cfg = _construction(make_config(tmp_path, SESSIONS), max_weight=0.07, sector_cap=0.5)
    p = build_portfolio(_signal(), prices, cfg, as_of_date=SESSIONS[-1], notional=1e9)
    satisfied = p.constraints_applied["satisfied"]
    assert set(satisfied) == {"max_weight", "sector_cap", "gross_leverage", "dollar_neutral"}
    if satisfied["max_weight"]:
        assert float(p.holdings["weight"].abs().max()) <= 0.07 + 1e-9
    for name, ok in satisfied.items():
        if not ok:
            assert any(name in w or "infeasible" in w or "different gross" in w
                       for w in p.warnings), f"{name} violated with no warning"


def test_water_fill_lands_exactly_on_the_caps():
    """The previous iterative rescale converged slowly and could stall just over a cap."""
    mag = np.array([0.5, 0.3, 0.2])
    caps = np.array([0.15, 0.15, 0.15])
    sec = np.array(["A", "B", "C"], dtype=object)
    out, feasible, capacity = _project_side(
        mag, sec, caps, sector_cap=0.5, target_gross=0.45
    )
    assert feasible
    assert np.abs(out).sum() == pytest.approx(0.45)
    assert np.abs(out).max() <= 0.15 + 1e-12


def test_projection_reports_infeasibility_rather_than_guessing():
    mag = np.array([0.5, 0.5])
    caps = np.array([0.05, 0.05])
    sec = np.array(["A", "B"], dtype=object)
    out, feasible, capacity = _project_side(
        mag, sec, caps, sector_cap=0.9, target_gross=1.0
    )
    assert feasible is False
    assert capacity == pytest.approx(0.10)
    assert np.abs(out).sum() == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# C. The miner kept formulas it had never scored
# ---------------------------------------------------------------------------

def test_an_unscorable_formula_is_penalised_not_scored_zero():
    """Reward 0.0 made an unmeasured formula beat every genuinely negative-IC one, so the
    top-k filled with candidates that had never been evaluated."""
    assert UNSCORABLE_REWARD < 0.0


def test_a_short_ic_series_yields_nan_not_zero():
    days = _sessions(20)
    syms = [f"S{i:02d}" for i in range(20)]
    panel = pl.DataFrame(
        [{"date": d, "symbol": s, "value": float(i)} for d in days for i, s in enumerate(syms)],
        schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64},
    )
    prices = pl.DataFrame(
        [{"date": d, "symbol": s, "adj_close": 100.0 + i} for i, d in enumerate(days) for s in syms],
        schema={"date": pl.Date, "symbol": pl.Utf8, "adj_close": pl.Float64},
    )
    reward, ic, _, n_dates, _ = formula_reward(
        panel, forward_returns(prices, [5]), 5, kind="pooled", min_dates=60
    )
    assert math.isnan(reward) and math.isnan(ic)
    assert 0 < n_dates < 60


def test_mine_never_keeps_a_candidate_with_a_nan_ic(tmp_path):
    from flowalpha.factors.base import FactorContext

    sess = _sessions(120)
    rng = np.random.default_rng(3)
    n, m = len(sess), 15
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.015, size=(n, m)), axis=0)
    vol = 1e5 * (1.0 + rng.random((n, m)))
    cfg = make_config(tmp_path, sess)
    ctx = FactorContext(
        sessions=sess, symbols=[f"S{i:02d}" for i in range(m)],
        adj_close=close, close=close.copy(), volume=vol, turnover=close * vol,
        delivery_pct=np.full((n, m), np.nan), cfg=cfg, as_of_date=sess[-1],
    )
    prices = pl.DataFrame(
        {
            "date": np.repeat(np.array(sess, dtype="object"), m).tolist(),
            "symbol": np.tile(np.array(ctx.symbols, dtype="object"), n).tolist(),
            "adj_close": close.reshape(-1),
        },
        schema={"date": pl.Date, "symbol": pl.Utf8, "adj_close": pl.Float64},
    )
    mcfg = cfg.with_overrides(
        mining={**cfg["mining"], "vocab": "price_only", "max_formula_len": 8,
                "reward": "pooled"}
    )
    result = mine(ctx, forward_returns(prices, [5]), mcfg, episodes=40, seeds=[0],
                  policy_kind="random", top_k=8)
    for c in result.candidates:
        assert np.isfinite(c.ic), f"kept an unscored formula: {c.formula}"
        assert c.n_dates > 0
    assert "n_unscorable_signals" in result.to_dict()


# ---------------------------------------------------------------------------
# D. usable_from was clamped to the last session
# ---------------------------------------------------------------------------

def test_usable_from_is_null_rather_than_clamped_to_the_last_session():
    """Clamping would mark an IC whose label has not realised as usable on the final
    date, which is the exact look-ahead the column exists to prevent."""
    days = _sessions(40)
    syms = [f"S{i:02d}" for i in range(15)]
    prices = pl.DataFrame(
        [{"date": d, "symbol": s, "adj_close": 100.0 + i + j}
         for i, d in enumerate(days) for j, s in enumerate(syms)],
        schema={"date": pl.Date, "symbol": pl.Utf8, "adj_close": pl.Float64},
    )
    panel = prices.select("date", "symbol").with_columns(
        pl.col("symbol").str.slice(1).cast(pl.Float64).alias("value")
    )
    fwd = forward_returns(prices, [5])
    series = trailing_ic_series(panel, fwd, 5, min_names=5)
    index = {d: i for i, d in enumerate(days)}
    for row in series.iter_rows(named=True):
        if row["usable_from"] is None:
            continue
        # Whenever it IS set, it must be exactly the horizon away -- never nearer.
        assert index[row["usable_from"]] - index[row["date"]] == 5


# ---------------------------------------------------------------------------
# F. forward returns crossed per-name gaps
# ---------------------------------------------------------------------------

def test_forward_return_is_null_across_a_per_name_gap():
    """A per-row shift spanned the gap and labelled a 3-session return as fwd_1."""
    days = [D(2022, 1, 3), D(2022, 1, 4), D(2022, 1, 5), D(2022, 1, 6), D(2022, 1, 7)]
    rows = [{"date": d, "symbol": "A", "adj_close": 100.0} for d in days]
    for d in days:
        if d not in (D(2022, 1, 5), D(2022, 1, 6)):     # B is halted for two sessions
            rows.append({"date": d, "symbol": "B",
                         "adj_close": 100.0 if d < D(2022, 1, 7) else 200.0})
    px = pl.DataFrame(rows, schema={"date": pl.Date, "symbol": pl.Utf8,
                                    "adj_close": pl.Float64})
    fwd = forward_returns(px, [1])
    b = fwd.filter(pl.col("symbol") == "B").sort("date")
    got = dict(zip(b["date"].to_list(), b["fwd_1"].to_list()))
    # 01-04 -> 01-05 is unobservable for B, so the label must be NULL, not +100%.
    assert got[D(2022, 1, 4)] is None
    assert got[D(2022, 1, 3)] == pytest.approx(0.0)


def test_forward_returns_unchanged_for_a_gapless_panel():
    """The fix must not disturb the ordinary case."""
    days = _sessions(30)
    syms = [f"S{i}" for i in range(5)]
    px = pl.DataFrame(
        [{"date": d, "symbol": s, "adj_close": float(100 + 10 * j + i)}
         for i, d in enumerate(days) for j, s in enumerate(syms)],
        schema={"date": pl.Date, "symbol": pl.Utf8, "adj_close": pl.Float64},
    )
    fwd = forward_returns(px, [1, 5])
    one = fwd.filter((pl.col("symbol") == "S0") & (pl.col("date") == days[0]))
    assert float(one["fwd_1"][0]) == pytest.approx(1.0 / 100.0)
    assert float(one["fwd_5"][0]) == pytest.approx(5.0 / 100.0)
    assert fwd.filter(pl.col("date") == days[-1])["fwd_1"].null_count() == len(syms)


# ---------------------------------------------------------------------------
# G. A held name with no next price was booked as a flat exit
# ---------------------------------------------------------------------------

def test_engine_reports_positions_it_could_not_mark(tmp_path):
    """Contributing 0 is an assumption -- a costless exit at the last observed price --
    not a measurement, so it has to be counted and surfaced."""
    from flowalpha.backtest.engine import run_backtest

    sess = _sessions(10)
    rows = []
    for i, d in enumerate(sess):
        for j in range(10):
            if j == 9 and i > 2:        # the top-ranked (long) name stops trading
                continue
            rows.append({"date": d, "symbol": f"S{j}", "adj_close": 100.0, "turnover": None})
    px = pl.DataFrame(rows, schema={"date": pl.Date, "symbol": pl.Utf8,
                                    "adj_close": pl.Float64, "turnover": pl.Float64})
    panel = pl.DataFrame(
        [{"date": d, "symbol": f"S{j}", "value": float(j)} for d in sess for j in range(10)],
        schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64},
    )
    cfg = make_config(tmp_path, sess)
    r = run_backtest(panel, px, cfg, holding_days=1, notional=1.0)
    assert r.cost_detail["unpriced_position_days"] > 0
    assert r.cost_detail["unpriced_gross_exposure"] > 0


def test_engine_reports_zero_unpriced_days_on_a_clean_panel(tmp_path):
    from flowalpha.backtest.engine import run_backtest

    sess = _sessions(30)
    syms = tuple(f"S{i}" for i in range(10))
    px = make_tiny_prices(sess, symbols=syms)
    panel = pl.DataFrame(
        [{"date": d, "symbol": f"S{j}", "value": float(j)} for d in sess for j in range(10)],
        schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64},
    )
    cfg = make_config(tmp_path, sess)
    r = run_backtest(panel, px, cfg, holding_days=1, notional=1.0)
    assert r.cost_detail["unpriced_position_days"] == 0


# ---------------------------------------------------------------------------
# E / H / J. Claim-vs-behaviour mismatches
# ---------------------------------------------------------------------------

def test_cardinality_check_states_that_the_method_is_not_applied(prices, tmp_path):
    """The docstring claimed a market-cap ranking the code never performed."""
    from flowalpha.data.calendar import TradingCalendar

    cfg = make_config(tmp_path, SESSIONS).with_overrides(
        universe={"name": "T", "size": len(SYMBOLS), "cardinality_band": [10, 60],
                  "reconstruction": {"method": "mktcap_rank", "rebalance_freq": "quarterly",
                                     "min_history_days": 20}}
    )
    r = check_universe_cardinality(prices, TradingCalendar(SESSIONS), cfg)
    assert "NOT applied" in r.details["reconstruction"]
    assert "mktcap_rank" in r.details["reconstruction"]


def test_is_complete_rejects_a_midsequence_underflow_by_position():
    """The guard excused an underflow whenever the offending token happened to equal the
    final token by value."""
    vocab = build_vocabulary("price_only")
    assert not is_complete(["add", "close", "add"], vocab)
    assert is_complete(["close", "volume", "add"], vocab)


def test_symbol_filter_on_an_aggregate_dataset_is_an_error(tiny_cfg, tiny_calendar,
                                                           tiny_sessions):
    """It used to be silently ignored, so a caller could believe it had restricted the
    universe when it had not."""
    daily, part = make_tiny_flows(tiny_sessions)
    write_tree(tiny_cfg, prices=make_tiny_prices(tiny_sessions),
               flows_daily=daily, participant_flows=part)
    store = PointInTimeStore(tiny_cfg.path("processed"), tiny_calendar,
                             tiny_cfg["availability"])
    with pytest.raises(LookAheadError, match="aggregate series"):
        store.get("flows_daily", as_of_date=tiny_sessions[30], symbols=["FII"])
    # The per-name datasets still accept it.
    assert not store.get("prices", as_of_date=tiny_sessions[30], symbols=["AAA"]).is_empty()
