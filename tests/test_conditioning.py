"""Regime access, conditional IC, and declared experiments."""

from __future__ import annotations

import datetime as _dt
import math

import numpy as np
import polars as pl
import pytest

from flowalpha.conditioning.analysis import (
    ConditioningExperiment,
    conditional_ic,
    conditional_ic_table,
    conditional_panel,
    declared_experiments,
    expand_experiments,
    regime_exposure,
    run_experiment,
)
from flowalpha.conditioning.regimes import RegimeError, RegimeSet
from flowalpha.factors.flow_features import FLOW_FEATURES_SCHEMA
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


def _features(labels: dict[str, list]) -> pl.DataFrame:
    data = {"date": SESSIONS}
    for col, dtype in FLOW_FEATURES_SCHEMA.items():
        if col == "date":
            continue
        if col in labels:
            data[col] = labels[col]
        elif dtype == pl.Boolean:
            data[col] = [None] * len(SESSIONS)
        elif dtype == pl.Utf8:
            data[col] = [None] * len(SESSIONS)
        else:
            data[col] = [None] * len(SESSIONS)
    return pl.DataFrame(data, schema=FLOW_FEATURES_SCHEMA)


@pytest.fixture(scope="module")
def conditional_fixture():
    """A factor whose forward predictiveness is switched on in the 'high' regime only.

    The regime alternates in long blocks, so the effect is recoverable but the buckets are
    not trivially separated by time.
    """
    rng = np.random.default_rng(5)
    regime = ["high" if (i // 20) % 2 == 0 else "low" for i in range(len(SESSIONS))]
    levels = {s: 100.0 for s in SYMBOLS}
    panel_rows, price_rows = [], []
    pending = None
    pending_active = False
    for i, day in enumerate(SESSIONS):
        for s in SYMBOLS:
            if pending is not None:
                strength = 1.2 if pending_active else 0.0
                step = strength * 0.01 * pending[s] + 0.01 * float(rng.normal())
                levels[s] *= 1.0 + step
            price_rows.append({"date": day, "symbol": s, "adj_close": levels[s]})
        signal = {s: float(rng.normal()) for s in SYMBOLS}
        for s in SYMBOLS:
            panel_rows.append({"date": day, "symbol": s, "value": signal[s]})
        pending = signal
        pending_active = regime[i] == "high"

    panel = pl.DataFrame(
        panel_rows, schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64}
    )
    prices = pl.DataFrame(
        price_rows, schema={"date": pl.Date, "symbol": pl.Utf8, "adj_close": pl.Float64}
    )
    features = _features(
        {
            "fii_regime": regime,
            "retail_extreme": ["extreme" if r == "high" else "mid" for r in regime],
        }
    )
    return panel, prices, RegimeSet(features=features)


# --- RegimeSet -------------------------------------------------------------

def test_columns_lists_only_present_regimes(conditional_fixture):
    _, _, regimes = conditional_fixture
    assert "fii_regime" in regimes.columns
    assert "retail_extreme" in regimes.columns


def test_unknown_column_raises(conditional_fixture):
    _, _, regimes = conditional_fixture
    with pytest.raises(RegimeError, match="unknown regime column"):
        regimes.buckets("not_a_regime")


def test_buckets_are_sorted_strings(conditional_fixture):
    _, _, regimes = conditional_fixture
    assert regimes.buckets("fii_regime") == ["high", "low"]


def test_labels_drops_undefined_dates():
    features = _features({"fii_regime": ["high"] * 10 + [None] * (len(SESSIONS) - 10)})
    regimes = RegimeSet(features=features)
    assert regimes.labels("fii_regime").height == 10


def test_dates_in_and_label_on(conditional_fixture):
    _, _, regimes = conditional_fixture
    highs = regimes.dates_in("fii_regime", "high")
    assert SESSIONS[0] in highs
    assert regimes.label_on("fii_regime", SESSIONS[0]) == "high"
    assert regimes.label_on("fii_regime", D(1990, 1, 1)) is None


def test_coverage_reports_buckets_and_bounds(conditional_fixture):
    _, _, regimes = conditional_fixture
    cov = regimes.coverage("fii_regime")
    assert cov["n_defined"] == len(SESSIONS)
    assert set(cov["buckets"]) == {"high", "low"}
    assert cov["last_defined"] == SESSIONS[-1].isoformat()


def test_coverage_of_an_all_null_column():
    regimes = RegimeSet(features=_features({}))
    assert regimes.coverage("fii_regime")["n_defined"] == 0


def test_attach_is_an_inner_join(conditional_fixture):
    """Rows on undefined-regime dates are dropped, not pooled into 'unknown'.

    Pooling would put the early sample -- before the expanding window had enough
    observations -- into whichever bucket absorbed it.
    """
    features = _features({"fii_regime": [None] * 100 + ["high"] * (len(SESSIONS) - 100)})
    regimes = RegimeSet(features=features)
    frame = pl.DataFrame({"date": SESSIONS}, schema={"date": pl.Date})
    got = regimes.attach(frame, "fii_regime")
    assert got.height == len(SESSIONS) - 100
    assert set(got["regime"].to_list()) == {"high"}


def test_summary_covers_every_column(conditional_fixture):
    _, _, regimes = conditional_fixture
    summary = regimes.summary()
    assert set(summary) == set(regimes.columns)


# --- conditional IC --------------------------------------------------------

def test_conditional_ic_recovers_the_embedded_regime_effect(conditional_fixture):
    """The fixture switches predictiveness on in 'high' only; the IC must show it."""
    panel, prices, regimes = conditional_fixture
    fwd = forward_returns(prices, [1])
    entries = {e.bucket: e for e in conditional_ic(panel, fwd, 1, regimes, "fii_regime",
                                                   factor="f")}
    assert entries["high"].mean_ic > entries["low"].mean_ic
    assert entries["high"].t_stat > 2.0
    assert abs(entries["low"].mean_ic) < abs(entries["high"].mean_ic)


def test_conditional_ic_marks_itself_not_deflated(conditional_fixture):
    panel, prices, regimes = conditional_fixture
    fwd = forward_returns(prices, [1])
    for entry in conditional_ic(panel, fwd, 1, regimes, "fii_regime", factor="f"):
        assert entry.to_dict()["deflated"] is False


def test_conditional_ic_drops_thin_buckets(conditional_fixture):
    panel, prices, regimes = conditional_fixture
    fwd = forward_returns(prices, [1])
    entries = conditional_ic(panel, fwd, 1, regimes, "fii_regime", factor="f",
                             min_dates=10_000)
    assert entries == []


def test_conditional_ic_on_empty_panel(conditional_fixture):
    _, prices, regimes = conditional_fixture
    empty = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64})
    assert conditional_ic(empty, forward_returns(prices, [1]), 1, regimes, "fii_regime") == []


def test_conditional_ic_table_covers_every_combination(conditional_fixture):
    panel, prices, regimes = conditional_fixture
    fwd = forward_returns(prices, [1])
    table = conditional_ic_table(
        {"a": panel, "b": panel}, fwd, 1, regimes, ["fii_regime", "retail_extreme"]
    )
    assert set(table["factor"].to_list()) == {"a", "b"}
    assert set(table["regime_column"].to_list()) == {"fii_regime", "retail_extreme"}


def test_conditional_ic_table_skips_absent_regime_columns(conditional_fixture):
    panel, prices, regimes = conditional_fixture
    table = conditional_ic_table(
        {"a": panel}, forward_returns(prices, [1]), 1, regimes, ["not_a_regime"]
    )
    assert table.is_empty()


def test_conditional_ic_table_empty_keeps_schema(conditional_fixture):
    _, prices, regimes = conditional_fixture
    table = conditional_ic_table({}, forward_returns(prices, [1]), 1, regimes, ["fii_regime"])
    assert table.is_empty() and "mean_ic" in table.columns


# --- experiments -----------------------------------------------------------

def test_declared_experiment_in_the_right_direction_is_supported(conditional_fixture):
    panel, prices, regimes = conditional_fixture
    exp = ConditioningExperiment(
        factor="f", regime_column="fii_regime", favourable="high", unfavourable="low"
    )
    result = run_experiment(exp, panel, forward_returns(prices, [1]), 1, regimes)
    assert result.difference > 0
    assert result.supported
    assert result.verdict == "SUPPORTED"


def test_a_reversed_hypothesis_is_refuted_not_rebranded(conditional_fixture):
    """Directions are declared in advance, so backwards is a refutation."""
    panel, prices, regimes = conditional_fixture
    exp = ConditioningExperiment(
        factor="f", regime_column="fii_regime", favourable="low", unfavourable="high"
    )
    result = run_experiment(exp, panel, forward_returns(prices, [1]), 1, regimes)
    assert result.difference < 0
    assert not result.supported
    assert result.verdict.startswith("REFUTED")


def _noise_experiment(seed: int):
    rng = np.random.default_rng(seed)
    rows_p, rows_f = [], []
    levels = {s: 100.0 for s in SYMBOLS}
    for day in SESSIONS:
        for s in SYMBOLS:
            levels[s] *= 1.0 + 0.01 * float(rng.normal())
            rows_f.append({"date": day, "symbol": s, "adj_close": levels[s]})
            rows_p.append({"date": day, "symbol": s, "value": float(rng.normal())})
    panel = pl.DataFrame(rows_p, schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64})
    prices = pl.DataFrame(rows_f, schema={"date": pl.Date, "symbol": pl.Utf8,
                                          "adj_close": pl.Float64})
    regimes = RegimeSet(features=_features(
        {"fii_regime": ["high" if (i // 20) % 2 == 0 else "low" for i in range(len(SESSIONS))]}
    ))
    exp = ConditioningExperiment(factor="f", regime_column="fii_regime",
                                 favourable="high", unfavourable="low")
    return run_experiment(exp, panel, forward_returns(prices, [1]), 1, regimes)


def test_no_embedded_effect_is_rarely_supported():
    """With no effect present, the gate must almost always decline to support.

    Tested as a RATE across seeds rather than on one draw, because a |t| >= 1.5 gate is
    genuinely crossed by noise a meaningful fraction of the time -- roughly 13% per side
    for a two-sided normal. A single-draw assertion would be a coin flip dressed as a
    test, and the rate is the property that actually matters: the gate is not free, which
    is exactly why the Deflated Sharpe machinery exists alongside it.
    """
    results = [_noise_experiment(seed) for seed in range(8)]
    n_supported = sum(1 for r in results if r.supported)
    assert n_supported <= 2, [r.to_dict() for r in results if r.supported]
    for r in results:
        assert abs(r.difference) < 0.05  # no large effect is ever found
        assert r.verdict.startswith(("NOT SUPPORTED", "REFUTED", "SUPPORTED", "INCONCLUSIVE"))


def test_a_noise_draw_that_crosses_the_gate_is_still_reported_honestly():
    """When noise does cross, the verdict names the direction rather than hiding it."""
    results = [_noise_experiment(seed) for seed in range(8)]
    for r in results:
        if r.difference < 0 and abs(r.t_stat) >= r.gate_t_threshold:
            assert r.verdict.startswith("REFUTED")
            assert not r.supported


def test_experiment_on_empty_panel_is_inconclusive(conditional_fixture):
    _, prices, regimes = conditional_fixture
    empty = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64})
    exp = ConditioningExperiment(factor="f", regime_column="fii_regime",
                                 favourable="high", unfavourable="low")
    result = run_experiment(exp, empty, forward_returns(prices, [1]), 1, regimes)
    assert math.isnan(result.difference)
    assert "INCONCLUSIVE" in result.verdict


def test_experiment_to_dict_records_the_declared_direction(conditional_fixture):
    panel, prices, regimes = conditional_fixture
    exp = ConditioningExperiment(factor="f", regime_column="fii_regime",
                                 favourable="high", unfavourable="low",
                                 rationale="because")
    d = run_experiment(exp, panel, forward_returns(prices, [1]), 1, regimes).to_dict()
    assert d["favourable_bucket"] == "high"
    assert d["unfavourable_bucket"] == "low"
    assert d["rationale"] == "because"
    assert "verdict" in d


def test_experiment_name_is_stable(conditional_fixture):
    exp = ConditioningExperiment(factor="momentum", regime_column="fii_regime",
                                 favourable="high", unfavourable="low")
    assert exp.name == "momentum|fii_regime|high-vs-low"


def test_declared_experiments_come_from_config(tmp_path, conditional_fixture):
    """The map is a stated hypothesis, read from config -- not a search result."""
    _, _, regimes = conditional_fixture
    cfg = make_config(tmp_path, SESSIONS)
    exps = declared_experiments(cfg, regimes)
    by_col = {e.regime_column: e for e in exps}
    assert by_col["fii_regime"].favourable == "high"
    assert by_col["fii_regime"].unfavourable == "low"
    assert by_col["retail_extreme"].favourable == "extreme"
    assert by_col["retail_extreme"].unfavourable == "mid"
    for e in exps:
        assert e.rationale


def test_declared_experiments_skip_absent_regime_columns(tmp_path):
    regimes = RegimeSet(features=_features({}))
    cfg = make_config(tmp_path, SESSIONS)
    assert declared_experiments(cfg, regimes) == []


def test_expand_experiments_matches_by_prefix():
    exp = ConditioningExperiment(factor="reversal", regime_column="retail_extreme",
                                 favourable="extreme", unfavourable="mid")
    pairs = expand_experiments([exp], ["reversal_5d", "reversal_21d", "momentum"])
    assert [name for _, name in pairs] == ["reversal_5d", "reversal_21d"]


def test_expand_experiments_unmatched_prefix_yields_nothing():
    exp = ConditioningExperiment(factor="nonexistent", regime_column="fii_regime",
                                 favourable="high", unfavourable="low")
    assert expand_experiments([exp], ["momentum"]) == []


# --- conditional strategy --------------------------------------------------

def test_conditional_panel_zeroes_rather_than_dropping(conditional_fixture):
    """Keeping the dates flat means the conditional and unconditional strategies are
    measured over the SAME window. Dropping them would shorten the sample in a way
    correlated with the signal, flattering the result for that reason alone."""
    panel, _, regimes = conditional_fixture
    out = conditional_panel(panel, regimes, "fii_regime", ["high"])
    assert out.height == panel.height
    low_dates = regimes.dates_in("fii_regime", "low")
    off = out.filter(pl.col("date").is_in(pl.Series("d", low_dates, dtype=pl.Date)))
    assert off["value"].abs().max() == pytest.approx(0.0)
    on = out.filter(~pl.col("date").is_in(pl.Series("d", low_dates, dtype=pl.Date)))
    assert on["value"].abs().max() > 0


def test_conditional_panel_can_drop_when_asked(conditional_fixture):
    panel, _, regimes = conditional_fixture
    out = conditional_panel(panel, regimes, "fii_regime", ["high"], flat_outside=False)
    assert out.height < panel.height


def test_conditional_panel_of_empty_input(conditional_fixture):
    _, _, regimes = conditional_fixture
    empty = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64})
    assert conditional_panel(empty, regimes, "fii_regime", ["high"]).is_empty()


def test_conditional_panel_handles_undefined_regime_dates():
    features = _features({"fii_regime": [None] * 200 + ["high"] * (len(SESSIONS) - 200)})
    regimes = RegimeSet(features=features)
    panel = pl.DataFrame(
        [{"date": d, "symbol": "A", "value": 1.0} for d in SESSIONS],
        schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64},
    )
    out = conditional_panel(panel, regimes, "fii_regime", ["high"])
    assert out.height == len(SESSIONS)
    early = out.filter(pl.col("date") <= SESSIONS[199])
    assert early["value"].abs().max() == pytest.approx(0.0)


def test_regime_exposure_is_the_invested_fraction(conditional_fixture):
    """Reported alongside every conditional result: a strategy in the market a third of
    the time has a mechanically lower total return."""
    _, _, regimes = conditional_fixture
    frac = regime_exposure(regimes, "fii_regime", ["high"])
    assert 0.4 < frac < 0.6
    assert regime_exposure(regimes, "fii_regime", ["high", "low"]) == pytest.approx(1.0)
    assert regime_exposure(regimes, "fii_regime", []) == pytest.approx(0.0)


def test_regime_exposure_of_an_all_null_column():
    regimes = RegimeSet(features=_features({}))
    assert math.isnan(regime_exposure(regimes, "fii_regime", ["high"]))
