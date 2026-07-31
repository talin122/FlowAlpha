"""Shared fixtures.

Two flavours, on purpose:

**Tiny, hand-built** (``tiny_*``) -- a small calendar with a deliberate mid-series
holiday and deterministic values, used where a test asserts an exact expected number.

**Synthetic tree** (``synthetic_tree``, session-scoped) -- a full schema-exact
pipeline input with a *known* data-generating process, used for integration-scale
checks that the validation machinery recovers effects that are genuinely there.

The synthetic fixture **pins its own date window** rather than inheriting ``dates``
from ``config.yaml``. Those tests assert statistical properties of one specific
random draw; inheriting the live window would mean that extending the study period
silently flips DGP-recovery assertions, and the resulting "regression" has nothing to
do with the code.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from flowalpha.config import Config
from flowalpha.data.calendar import TradingCalendar
from flowalpha.data.prices import process_prices
from flowalpha.data.store import PointInTimeStore

# --- tiny fixtures ---------------------------------------------------------

TINY_START = _dt.date(2021, 1, 4)
TINY_N_SESSIONS = 160
#: A weekday in the middle of the tiny window on which "the market did not trade".
#: Its presence keeps every calendar-arithmetic test honest about long gaps.
TINY_HOLIDAY = _dt.date(2021, 4, 2)
TINY_SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")


def _weekdays(start: _dt.date, count: int, skip: set[_dt.date]) -> list[_dt.date]:
    out: list[_dt.date] = []
    day = start
    while len(out) < count:
        if day.weekday() < 5 and day not in skip:
            out.append(day)
        day += _dt.timedelta(days=1)
    return out


@pytest.fixture
def tiny_sessions() -> list[_dt.date]:
    return _weekdays(TINY_START, TINY_N_SESSIONS, {TINY_HOLIDAY})


@pytest.fixture
def tiny_calendar(tiny_sessions) -> TradingCalendar:
    return TradingCalendar(tiny_sessions)


def make_config(root: Path, sessions: list[_dt.date], **overrides) -> Config:
    """A Config rooted at ``root`` with the window pinned to ``sessions``.

    Pinned, not inherited: see the module docstring.
    """
    data = {
        "seed": 20240101,
        "paths": {
            "raw": "raw", "processed": "processed",
            "reference": "reference", "results": "results",
        },
        "universe": {
            "name": "TINY", "size": len(TINY_SYMBOLS),
            "cardinality_band": [2, 20],
            "reconstruction": {"method": "mktcap_rank", "rebalance_freq": "quarterly",
                               "min_history_days": 5},
        },
        "dates": {
            "start": sessions[0].isoformat(),
            "end": sessions[-1].isoformat(),
            "timezone": "Asia/Kolkata",
        },
        "availability": {
            "prices": {"lag_days": 0},
            "flows_daily": {"lag_days": 1},
            "participant_flows": {"lag_days": 1},
            "fii_dii_cash": {"lag_days": 1},
            "bulk_deals": {"lag_days": 1},
            "block_deals": {"lag_days": 1},
            "shareholding": {"lag_calendar_days": 21},
        },
        "download": {
            "request_delay_sec": 0.0, "max_retries": 2, "backoff_base_sec": 1.0,
            "timeout_sec": 5, "user_agent": "test", "flow_workers": 2,
            "flow_delay_sec": 0.0,
        },
        "factors": {
            "winsorize": [0.01, 0.99],
            "sector_neutralize": False,
            "momentum": {"lookback": 40, "skip": 5},
            "reversal": {"windows": [3, 10]},
            "volatility": {"window": 20},
            "liquidity": {"window": 10},
            "delivery": {"window": 10},
            "volume_trend": {"window": 10},
        },
        "forward_returns": {"horizons": [1, 5]},
        "validation": {
            "purged_kfold": {"n_splits": 4, "embargo_days": 2},
            "deflated_sharpe": {"benchmark_sr": 0.0},
            "pbo": {"n_partitions": 6},
        },
        "conditioning": {
            "regime_window": 5, "n_buckets": 3,
            "expanding_min_obs": 20, "shock_percentile": 0.9,
        },
        "backtest": {
            "rebalance": "daily", "holding_days": 5, "gross_leverage": 1.0,
            "costs": {
                "stt_delivery": 0.001, "stt_intraday_sell": 0.00025,
                "stamp_duty_buy": 0.00015, "brokerage_exchange": 0.0003,
                "impact_coef": 0.1, "adv_window": 10,
            },
        },
        "signals": {
            "composite": {
                "scheme": "ic_ir", "ic_window": 60, "ic_horizon": 5,
                "min_coverage": 2, "drop_negative_ic": True,
            },
            "regime_overlay": {
                "enabled": True, "gate_t_threshold": 1.5,
                "factor_regime_map": {"momentum": "fii_regime", "reversal_3d": "retail_extreme"},
            },
            "construction": {
                "method": "quintile_ls", "long_only": False, "top_n": 2,
                "max_weight": 0.5, "sector_cap": 0.8, "gross_leverage": 1.0,
                "adv_participation_cap": 0.10, "rebalance_band": 0.0025,
            },
        },
        "mining": {
            "max_formula_len": 8, "vocab": "flow_augmented", "reward": "pooled",
            "novelty_lambda": 0.2, "episodes": 20, "smoke_episodes": 5,
            "seeds": [0], "policy": "gru", "learning_rate": 0.01,
            "baseline_momentum": 0.9, "reward_clip": 1.0, "entropy_coef": 0.01,
        },
        "synthetic": {"enabled": True, "n_symbols": len(TINY_SYMBOLS), "seed": 7},
    }
    data.update(overrides)
    for key in ("raw", "processed", "reference", "results"):
        (root / data["paths"][key]).mkdir(parents=True, exist_ok=True)
    return Config(_data=data, root=root, source_file=None)


@pytest.fixture
def tiny_cfg(tmp_path, tiny_sessions) -> Config:
    return make_config(tmp_path, tiny_sessions)


def make_tiny_prices(sessions: list[_dt.date], symbols=TINY_SYMBOLS, seed: int = 11) -> pl.DataFrame:
    """Deterministic price panel: geometric random walk, distinct per symbol."""
    rng = np.random.default_rng(seed)
    rows = []
    for s_i, sym in enumerate(symbols):
        level = 100.0 * (1.0 + s_i)
        for day in sessions:
            step = float(rng.normal(0.0005, 0.015))
            level *= 1.0 + step
            close = round(level, 4)
            rows.append(
                {
                    "date": day, "symbol": sym,
                    "open": close * 0.995, "high": close * 1.01, "low": close * 0.99,
                    "close": close, "adj_close": close,
                    "volume": 10_000.0 * (1 + s_i) + 100.0 * (day.toordinal() % 17),
                }
            )
    return process_prices(pl.DataFrame(rows))


def make_tiny_flows(
    sessions: list[_dt.date],
    seed: int = 12,
    *,
    drop_sessions: set[_dt.date] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Deterministic ``(flows_daily, participant_flows)`` frames.

    ``drop_sessions`` omits sessions entirely, so tests can reproduce the real
    condition of an unpublished NSE archive.
    """
    rng = np.random.default_rng(seed)
    drop = drop_sessions or set()
    daily_rows, part_rows = [], []
    for day in sessions:
        if day in drop:
            continue
        values = {
            "FII": float(rng.normal(500, 3000)),
            "DII": float(rng.normal(-200, 2000)),
            "Client": float(rng.normal(0, 5000)),
            "Pro": float(rng.normal(0, 1500)),
        }
        for participant, net in values.items():
            long_leg = max(net, 0.0) + 1000.0
            short_leg = long_leg - net
            part_rows.append(
                {
                    "date": day, "participant": participant,
                    "fut_index_long": long_leg * 0.4, "fut_index_short": short_leg * 0.4,
                    "fut_stock_long": long_leg * 0.6, "fut_stock_short": short_leg * 0.6,
                    "opt_index_call_long": 0.0, "opt_index_put_long": 0.0,
                    "opt_index_call_short": 0.0, "opt_index_put_short": 0.0,
                    "opt_stock_call_long": 0.0, "opt_stock_put_long": 0.0,
                    "opt_stock_call_short": 0.0, "opt_stock_put_short": 0.0,
                    "total_long": long_leg, "total_short": short_leg,
                    "net_futures": net,
                }
            )
            if participant in ("FII", "DII"):
                daily_rows.append(
                    {
                        "date": day, "participant": participant, "net_futures": net,
                        "gross_futures_long": long_leg, "gross_futures_short": short_leg,
                    }
                )
    return pl.DataFrame(daily_rows), pl.DataFrame(part_rows)


def write_tree(
    cfg: Config,
    *,
    prices: pl.DataFrame | None = None,
    flows_daily: pl.DataFrame | None = None,
    participant_flows: pl.DataFrame | None = None,
    shareholding: pl.DataFrame | None = None,
) -> Path:
    processed = cfg.path("processed")
    processed.mkdir(parents=True, exist_ok=True)
    if prices is not None:
        prices.write_parquet(processed / "prices.parquet")
    if flows_daily is not None:
        flows_daily.write_parquet(processed / "flows_daily.parquet")
    if participant_flows is not None:
        participant_flows.write_parquet(processed / "participant_flows.parquet")
    if shareholding is not None:
        shareholding.write_parquet(processed / "shareholding.parquet")
    return processed


@pytest.fixture
def tiny_tree(tiny_cfg, tiny_sessions) -> Config:
    prices = make_tiny_prices(tiny_sessions)
    daily, part = make_tiny_flows(tiny_sessions)
    write_tree(tiny_cfg, prices=prices, flows_daily=daily, participant_flows=part)
    return tiny_cfg


@pytest.fixture
def tiny_store(tiny_tree, tiny_calendar) -> PointInTimeStore:
    return PointInTimeStore(
        tiny_tree.path("processed"), tiny_calendar, tiny_tree["availability"]
    )


# --- synthetic tree (session-scoped) ---------------------------------------

#: The synthetic fixture's window is PINNED here, not inherited from config.yaml.
#: The DGP-recovery tests assert statistical properties of one specific random draw;
#: inheriting the live window would mean that extending the study period silently flips
#: those assertions, and the resulting "regression" would have nothing to do with the code.
SYNTH_START = _dt.date(2015, 1, 5)
SYNTH_N_SESSIONS = 900
SYNTH_N_SYMBOLS = 90
SYNTH_SEED = 7


@pytest.fixture(scope="session")
def synthetic_data():
    """Generated fixture with a KNOWN data-generating process."""
    from flowalpha.data.synthetic import SyntheticSpec, generate

    return generate(
        SyntheticSpec(
            n_symbols=SYNTH_N_SYMBOLS,
            n_sessions=SYNTH_N_SESSIONS,
            start_date=SYNTH_START,
            seed=SYNTH_SEED,
        )
    )


@pytest.fixture(scope="session")
def synthetic_tree(tmp_path_factory, synthetic_data):
    """Write the synthetic fixture as a full pipeline input.

    Returns ``(cfg, store, data)``. Config windows are set from the generated sessions and
    the factor/conditioning windows are shortened to fit 900 sessions.
    """
    from flowalpha.data.store import PointInTimeStore
    from flowalpha.data.synthetic import write_tree as write_synth_tree

    root = tmp_path_factory.mktemp("synthetic")
    write_synth_tree(synthetic_data, root)
    sessions = list(synthetic_data.calendar.sessions)
    cfg = make_config(
        root,
        sessions,
        factors={
            "winsorize": [0.01, 0.99],
            "sector_neutralize": False,
            "momentum": {"lookback": 126, "skip": 21},
            "reversal": {"windows": [5, 21]},
            "volatility": {"window": 60},
            "liquidity": {"window": 21},
            "delivery": {"window": 21},
            "volume_trend": {"window": 21},
        },
        conditioning={
            "regime_window": 21, "n_buckets": 3,
            "expanding_min_obs": 126, "shock_percentile": 0.95,
        },
        forward_returns={"horizons": [1, 5, 21]},
        universe={
            "name": "SYNTHETIC", "size": SYNTH_N_SYMBOLS,
            "cardinality_band": [int(0.5 * SYNTH_N_SYMBOLS), int(1.5 * SYNTH_N_SYMBOLS)],
            "reconstruction": {"method": "mktcap_rank", "rebalance_freq": "quarterly",
                               "min_history_days": 60},
        },
    )
    store = PointInTimeStore(
        cfg.path("processed"), synthetic_data.calendar, cfg["availability"]
    )
    return cfg, store, synthetic_data


@pytest.fixture(scope="session")
def synthetic_features(synthetic_tree):
    """Flow features for the synthetic tree, written into its processed dir."""
    from flowalpha.factors.flow_features import build_flow_features, write_flow_features

    cfg, store, data = synthetic_tree
    features = build_flow_features(store, cfg, calendar=data.calendar)
    write_flow_features(features, cfg.path("processed"))
    store.invalidate("flow_features")
    return features
