"""Regime overlay gating."""

from __future__ import annotations

import datetime as _dt

import numpy as np
import polars as pl
import pytest

from flowalpha.conditioning.regimes import RegimeSet
from flowalpha.factors.flow_features import FLOW_FEATURES_SCHEMA
from flowalpha.signals.regime_overlay import (
    compute_gates,
    gate_summary,
    resolve_map,
)
from flowalpha.validation.ic import forward_returns

from conftest import make_config

D = _dt.date


def _sessions(n: int = 400) -> list[_dt.date]:
    out, day = [], D(2021, 1, 4)
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += _dt.timedelta(days=1)
    return out


SESSIONS = _sessions()
SYMBOLS = [f"S{i:02d}" for i in range(30)]


def _features(**labels) -> pl.DataFrame:
    data = {"date": SESSIONS}
    for col, dtype in FLOW_FEATURES_SCHEMA.items():
        if col == "date":
            continue
        data[col] = labels.get(col, [None] * len(SESSIONS))
    return pl.DataFrame(data, schema=FLOW_FEATURES_SCHEMA)


@pytest.fixture(scope="module")
def overlay_fixture():
    """momentum predictive only when fii_regime == 'high'; reversal_5d pure noise."""
    rng = np.random.default_rng(21)
    regime = ["high" if (i // 25) % 2 == 0 else "low" for i in range(len(SESSIONS))]
    levels = {s: 100.0 for s in SYMBOLS}
    mom_rows, rev_rows, price_rows = [], [], []
    pending, pending_on = None, False
    for i, day in enumerate(SESSIONS):
        for s in SYMBOLS:
            if pending is not None:
                strength = 1.5 if pending_on else 0.0
                levels[s] *= 1.0 + strength * 0.01 * pending[s] + 0.01 * float(rng.normal())
            price_rows.append({"date": day, "symbol": s, "adj_close": levels[s]})
        signal = {s: float(rng.normal()) for s in SYMBOLS}
        for s in SYMBOLS:
            mom_rows.append({"date": day, "symbol": s, "value": signal[s]})
            rev_rows.append({"date": day, "symbol": s, "value": float(rng.normal())})
        pending, pending_on = signal, regime[i] == "high"

    schema = {"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64}
    panels = {
        "momentum": pl.DataFrame(mom_rows, schema=schema),
        "reversal_5d": pl.DataFrame(rev_rows, schema=schema),
    }
    prices = pl.DataFrame(
        price_rows, schema={"date": pl.Date, "symbol": pl.Utf8, "adj_close": pl.Float64}
    )
    regimes = RegimeSet(features=_features(
        fii_regime=regime,
        retail_extreme=["extreme" if r == "high" else "mid" for r in regime],
    ))
    return panels, prices, regimes


@pytest.fixture
def cfg(tmp_path):
    base = make_config(tmp_path, SESSIONS)
    return base.with_overrides(
        signals={
            **base["signals"],
            "composite": {"scheme": "ic_ir", "ic_window": 120, "ic_horizon": 1,
                          "min_coverage": 1, "drop_negative_ic": True},
            "regime_overlay": {
                "enabled": True, "gate_t_threshold": 1.5,
                "factor_regime_map": {"momentum": "fii_regime",
                                      "reversal": "retail_extreme"},
            },
        }
    )


# --- map resolution --------------------------------------------------------

def test_resolve_map_expands_prefixes(cfg):
    got = resolve_map(cfg, ["momentum", "reversal_5d", "reversal_21d", "illiquidity"])
    assert got == {
        "momentum": "fii_regime",
        "reversal_5d": "retail_extreme",
        "reversal_21d": "retail_extreme",
    }


def test_resolve_map_leaves_unmapped_factors_out(cfg):
    got = resolve_map(cfg, ["illiquidity"])
    assert got == {}


# --- gating ----------------------------------------------------------------

def test_unmapped_factors_pass_through_untouched(cfg, overlay_fixture):
    """The overlay is a hypothesis about two specific effects, not a general filter."""
    panels, prices, regimes = overlay_fixture
    panels = {**panels, "illiquidity": panels["momentum"]}
    out = compute_gates(panels, forward_returns(prices, [1]), cfg, regimes, sessions=SESSIONS)
    assert out.panels["illiquidity"].equals(panels["illiquidity"])
    assert "illiquidity" not in out.factor_regime_map


def test_gate_turns_on_in_the_favourable_regime(cfg, overlay_fixture):
    panels, prices, regimes = overlay_fixture
    out = compute_gates(panels, forward_returns(prices, [1]), cfg, regimes, sessions=SESSIONS)
    mom = out.decisions.filter(pl.col("factor") == "momentum")
    on_high = mom.filter((pl.col("bucket") == "high") & pl.col("on")).height
    on_low = mom.filter((pl.col("bucket") == "low") & pl.col("on")).height
    assert on_high > on_low


def test_gated_panel_is_flat_when_off(cfg, overlay_fixture):
    panels, prices, regimes = overlay_fixture
    out = compute_gates(panels, forward_returns(prices, [1]), cfg, regimes, sessions=SESSIONS)
    decisions = out.decisions.filter(pl.col("factor") == "momentum")
    off_dates = decisions.filter(~pl.col("on"))["date"]
    gated = out.panels["momentum"].filter(pl.col("date").is_in(off_dates))
    assert gated["value"].abs().max() == pytest.approx(0.0)


def test_gated_panel_keeps_values_when_on(cfg, overlay_fixture):
    panels, prices, regimes = overlay_fixture
    out = compute_gates(panels, forward_returns(prices, [1]), cfg, regimes, sessions=SESSIONS)
    on_dates = out.decisions.filter((pl.col("factor") == "momentum") & pl.col("on"))["date"]
    if on_dates.len() == 0:
        pytest.skip("gate never fired in this fixture")
    gated = out.panels["momentum"].filter(pl.col("date").is_in(on_dates))
    original = panels["momentum"].filter(pl.col("date").is_in(on_dates))
    assert gated.sort(["symbol", "date"]).equals(original.sort(["symbol", "date"]))


def test_gate_is_off_before_enough_observations(cfg, overlay_fixture):
    panels, prices, regimes = overlay_fixture
    out = compute_gates(panels, forward_returns(prices, [1]), cfg, regimes,
                        sessions=SESSIONS, min_obs=50)
    early = out.decisions.filter(pl.col("date") <= SESSIONS[20])
    assert not early["on"].any()


def test_gate_is_off_when_the_regime_is_undefined(cfg, overlay_fixture):
    panels, prices, _ = overlay_fixture
    regimes = RegimeSet(features=_features(
        fii_regime=[None] * 200 + ["high"] * (len(SESSIONS) - 200)
    ))
    out = compute_gates(panels, forward_returns(prices, [1]), cfg, regimes, sessions=SESSIONS)
    mom = out.decisions.filter(pl.col("factor") == "momentum")
    undefined = mom.filter(pl.col("bucket").is_null())
    assert undefined.height == 200
    assert not undefined["on"].any()


def test_gate_is_point_in_time(cfg, overlay_fixture):
    """Decisions up to a cutoff must not move when later prices change."""
    panels, prices, regimes = overlay_fixture
    cutoff = SESSIONS[250]
    a = compute_gates(panels, forward_returns(prices, [1]), cfg, regimes, sessions=SESSIONS)
    poisoned = prices.with_columns(
        pl.when(pl.col("date") > cutoff).then(pl.lit(1e6)).otherwise(pl.col("adj_close"))
        .alias("adj_close")
    )
    b = compute_gates(panels, forward_returns(poisoned, [1]), cfg, regimes, sessions=SESSIONS)
    upto_a = a.decisions.filter(pl.col("date") <= cutoff).sort(["factor", "date"])
    upto_b = b.decisions.filter(pl.col("date") <= cutoff).sort(["factor", "date"])
    assert upto_a.equals(upto_b)


def test_higher_threshold_gates_less_often(cfg, overlay_fixture):
    panels, prices, regimes = overlay_fixture
    fwd = forward_returns(prices, [1])
    loose = compute_gates(panels, fwd, cfg, regimes, sessions=SESSIONS)
    strict_cfg = cfg.with_overrides(
        signals={**cfg["signals"],
                 "regime_overlay": {**cfg["signals"]["regime_overlay"],
                                    "gate_t_threshold": 5.0}}
    )
    strict = compute_gates(panels, fwd, strict_cfg, regimes, sessions=SESSIONS)
    assert strict.decisions["on"].sum() < loose.decisions["on"].sum()


def test_on_fraction_is_reported(cfg, overlay_fixture):
    """A gate that is almost never on produces a strategy almost never invested."""
    panels, prices, regimes = overlay_fixture
    out = compute_gates(panels, forward_returns(prices, [1]), cfg, regimes, sessions=SESSIONS)
    for factor in out.factor_regime_map:
        assert 0.0 <= out.on_fraction[factor] <= 1.0


def test_to_dict_records_that_the_map_is_declared(cfg, overlay_fixture):
    panels, prices, regimes = overlay_fixture
    out = compute_gates(panels, forward_returns(prices, [1]), cfg, regimes, sessions=SESSIONS)
    d = out.to_dict()
    assert d["declared_not_mined"] is True
    assert d["gate_t_threshold"] == 1.5
    assert "momentum" in d["factor_regime_map"]


def test_gate_summary_is_ascii_and_names_the_map(cfg, overlay_fixture):
    panels, prices, regimes = overlay_fixture
    out = compute_gates(panels, forward_returns(prices, [1]), cfg, regimes, sessions=SESSIONS)
    text = "\n".join(gate_summary(out))
    text.encode("ascii")
    assert "DECLARED" in text
    assert "momentum" in text


def test_gate_summary_with_no_mapped_factors(cfg, overlay_fixture):
    panels, prices, regimes = overlay_fixture
    out = compute_gates({"illiquidity": panels["momentum"]}, forward_returns(prices, [1]),
                        cfg, regimes, sessions=SESSIONS)
    assert any("inert" in line for line in gate_summary(out))


def test_empty_panel_passes_through(cfg, overlay_fixture):
    _, prices, regimes = overlay_fixture
    empty = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64})
    out = compute_gates({"momentum": empty}, forward_returns(prices, [1]), cfg, regimes,
                        sessions=SESSIONS)
    assert out.panels["momentum"].is_empty()


def test_missing_regime_column_passes_the_factor_through(cfg, overlay_fixture):
    panels, prices, _ = overlay_fixture
    regimes = RegimeSet(features=_features())  # everything null
    out = compute_gates(panels, forward_returns(prices, [1]), cfg, regimes, sessions=SESSIONS)
    assert out.panels["momentum"].equals(panels["momentum"])
