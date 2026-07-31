"""Integration: does the pipeline recover effects that are known to be present?

Run against the synthetic fixture, whose data-generating process is documented in
:data:`flowalpha.data.synthetic.KNOWN_EFFECTS`. This is what makes a *negative* result on
real data meaningful: if the machinery can find effects that are there, then not finding
them is informative rather than merely unexplained.

The fixture's date window is pinned in ``conftest.py``, not inherited from
``config.yaml`` -- see the note there.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from flowalpha.conditioning.analysis import (
    ConditioningExperiment,
    conditional_ic,
    conditional_panel,
    regime_exposure,
    run_experiment,
)
from flowalpha.conditioning.regimes import RegimeSet
from flowalpha.data.quality import overall_status, run_all_checks
from flowalpha.data.synthetic import KNOWN_EFFECTS
from flowalpha.factors.base import FactorContext
from flowalpha.factors.library import build_panels, default_library
from flowalpha.signals.composite import build_composite, compute_weights
from flowalpha.validation.deflated_sharpe import TrialRegistry, deflated_sharpe_ratio
from flowalpha.validation.ic import forward_returns, ic_table, rank_ic, summarize_ic

HORIZON = 5


@pytest.fixture(scope="module")
def built(synthetic_tree, synthetic_features):
    """Factor panels and forward returns for the synthetic tree."""
    cfg, store, data = synthetic_tree
    ctx = FactorContext.from_store(store, cfg)
    factors = default_library(cfg, shares_available=True)
    panels, skipped = build_panels(factors, ctx, verbose=False)
    fwd = forward_returns(data.prices, [1, HORIZON, 21])
    regimes = RegimeSet(features=synthetic_features)
    return cfg, store, data, ctx, panels, skipped, fwd, regimes


def _ic(panels, fwd, name, horizon=HORIZON):
    return summarize_ic(rank_ic(panels[name], fwd, horizon), factor=name, horizon=horizon)


# --- the tree itself -------------------------------------------------------

def test_synthetic_tree_is_schema_exact(built):
    from flowalpha.data.prices import PRICES_SCHEMA
    from flowalpha.data.nse_flows import FLOWS_DAILY_SCHEMA, PARTICIPANT_FLOWS_SCHEMA

    _, _, data, *_ = built
    assert data.prices.schema == PRICES_SCHEMA
    assert data.flows_daily.schema == FLOWS_DAILY_SCHEMA
    assert data.participant_flows.schema == PARTICIPANT_FLOWS_SCHEMA


def test_synthetic_tree_passes_the_quality_gate(built):
    cfg, _, data, *_ = built
    results = run_all_checks(cfg, data.prices, data.calendar, flows=data.flows_daily)
    assert overall_status(results) in ("PASS", "WARN"), [
        r.to_dict() for r in results if r.status == "FAIL"
    ]


def test_synthetic_tree_marks_every_dataset_synthetic(synthetic_tree):
    from flowalpha.data.provenance import Provenance

    cfg, _, _ = synthetic_tree
    prov = Provenance.load(cfg.root)
    assert prov.all_synthetic
    assert prov.label() == "SYNTHETIC DATA"
    assert "SYNTHETIC" in prov.banner()


def test_synthetic_provenance_lists_the_known_effects(synthetic_tree):
    from flowalpha.data.provenance import Provenance

    cfg, _, _ = synthetic_tree
    caveats = " ".join(Provenance.load(cfg.root).caveats)
    for name in KNOWN_EFFECTS:
        assert name in caveats


def test_every_factor_computes_on_synthetic_data(built):
    """Including delivery, which is the only fixture where it is not inert."""
    *_, panels, skipped, _, _ = built
    assert "delivery_ratio" in panels, skipped
    assert "size_logmktcap" in panels
    assert len(panels) == 8


def test_synthetic_calendar_has_real_holidays(built):
    _, _, data, *_ = built
    assert len(data.calendar.holidays()) > 0


def test_synthetic_flows_have_real_gaps(built):
    """The gap handling in flow_features must be exercised by this fixture too."""
    _, _, data, *_ = built
    covered = set(data.flows_daily["date"].to_list())
    sessions = set(data.calendar.sessions)
    assert 0 < len(sessions - covered) <= data.spec.n_flow_gaps


# --- regime construction --------------------------------------------------

def test_regimes_are_defined_through_the_end_of_the_sample(built):
    """The cumsum-bug check, on the synthetic fixture."""
    *_, regimes = built
    _, _, data, *_ = built
    for col in ("fii_regime", "retail_regime", "retail_extreme"):
        cov = regimes.coverage(col)
        assert cov["n_defined"] > 0
        assert cov["last_defined"] == data.calendar.end.isoformat()


def test_regime_buckets_are_all_populated(built):
    *_, regimes = built
    assert set(regimes.buckets("fii_regime")) == {"low", "mid", "high"}
    assert set(regimes.buckets("retail_extreme")) == {"mid", "extreme"}


# --- DGP recovery: unconditional effects ----------------------------------

def test_momentum_recovers_the_persistent_driver(built):
    """KNOWN_EFFECTS['momentum']."""
    *_, panels, _, fwd, _ = built
    s = _ic(panels, fwd, "momentum")
    assert s.mean > 0, KNOWN_EFFECTS["momentum"]
    assert s.t_stat > 2.0


def test_reversal_recovers_the_transient_shock(built):
    """KNOWN_EFFECTS['reversal']."""
    *_, panels, _, fwd, _ = built
    s = _ic(panels, fwd, "reversal_5d", horizon=1)
    assert s.mean > 0, KNOWN_EFFECTS["reversal"]
    assert s.t_stat > 2.0


def test_low_volatility_recovers_the_premium(built):
    """KNOWN_EFFECTS['low_volatility']."""
    *_, panels, _, fwd, _ = built
    s = _ic(panels, fwd, "low_volatility", horizon=21)
    assert s.mean > 0, KNOWN_EFFECTS["low_volatility"]


def test_delivery_recovers_its_loading_on_the_driver(built):
    """KNOWN_EFFECTS['delivery_ratio'] -- the one fixture where delivery is not inert."""
    *_, panels, _, fwd, _ = built
    s = _ic(panels, fwd, "delivery_ratio", horizon=21)
    assert s.mean > 0, KNOWN_EFFECTS["delivery_ratio"]
    assert s.t_stat > 1.5


def test_a_shuffled_signal_recovers_nothing(built):
    """Control: destroying the cross-sectional alignment must destroy the IC."""
    *_, panels, _, fwd, _ = built
    rng = np.random.default_rng(0)
    panel = panels["momentum"]
    shuffled = panel.with_columns(
        pl.Series("value", rng.permutation(panel["value"].to_numpy()))
    )
    s = summarize_ic(rank_ic(shuffled, fwd, HORIZON), factor="shuffled", horizon=HORIZON)
    assert abs(s.t_stat) < 3.0
    assert abs(s.mean) < abs(_ic(panels, fwd, "momentum").mean)


def test_ic_table_covers_every_factor_and_horizon(built):
    cfg, _, _, _, panels, _, fwd, _ = built
    table = ic_table(panels, fwd, [1, HORIZON, 21])
    assert table.height == len(panels) * 3
    assert table["deflated"].to_list() == [False] * table.height


# --- DGP recovery: the conditional effects (the study's hypotheses) -------

def test_momentum_is_weaker_when_fii_flow_is_negative(built):
    """KNOWN_EFFECTS['momentum_conditional'].

    The DGP damps the persistent driver's payoff when trailing FII flow is negative, keyed
    to flow through t-1 -- exactly what the point-in-time features can see. So the
    conditional machinery must find it.
    """
    *_, panels, _, fwd, regimes = built
    entries = {
        e.bucket: e
        for e in conditional_ic(panels["momentum"], fwd, HORIZON, regimes, "fii_regime",
                                factor="momentum")
    }
    assert entries["high"].mean_ic > entries["low"].mean_ic, KNOWN_EFFECTS["momentum_conditional"]


def test_the_declared_momentum_hypothesis_is_supported_on_synthetic_data(built):
    """The hypothesis is stated in config and is TRUE in this fixture by construction."""
    *_, panels, _, fwd, regimes = built
    exp = ConditioningExperiment(
        factor="momentum", regime_column="fii_regime",
        favourable="high", unfavourable="low",
        rationale="embedded in the synthetic DGP",
    )
    result = run_experiment(exp, panels["momentum"], fwd, HORIZON, regimes)
    assert result.difference > 0
    assert result.supported, result.to_dict()
    assert result.verdict == "SUPPORTED"


def test_reversal_is_stronger_when_retail_participation_is_extreme(built):
    """KNOWN_EFFECTS['reversal_conditional']."""
    *_, panels, _, fwd, regimes = built
    entries = {
        e.bucket: e
        for e in conditional_ic(panels["reversal_5d"], fwd, 1, regimes, "retail_extreme",
                                factor="reversal_5d")
    }
    assert entries["extreme"].mean_ic > entries["mid"].mean_ic, (
        KNOWN_EFFECTS["reversal_conditional"]
    )


def test_a_conditional_strategy_is_flat_outside_its_regime(built):
    *_, panels, _, fwd, regimes = built
    cond = conditional_panel(panels["momentum"], regimes, "fii_regime", ["high"])
    low_dates = pl.Series("d", regimes.dates_in("fii_regime", "low"), dtype=pl.Date)
    assert cond.filter(pl.col("date").is_in(low_dates))["value"].abs().max() == pytest.approx(0.0)
    assert cond.height == panels["momentum"].height


def test_conditional_exposure_is_reported(built):
    *_, regimes = built
    frac = regime_exposure(regimes, "fii_regime", ["high"])
    assert 0.2 < frac < 0.5


def test_conditioning_on_an_irrelevant_regime_finds_nothing(built):
    """Control: momentum's payoff does not depend on flow divergence in the DGP."""
    *_, panels, _, fwd, regimes = built
    if "flow_divergence" not in regimes.columns:
        pytest.skip("flow_divergence undefined in this fixture")
    exp = ConditioningExperiment(
        factor="momentum", regime_column="flow_divergence",
        favourable="aligned", unfavourable="opposed",
    )
    result = run_experiment(exp, panels["momentum"], fwd, HORIZON, regimes)
    assert abs(result.difference) < abs(
        run_experiment(
            ConditioningExperiment("momentum", "fii_regime", "high", "low"),
            panels["momentum"], fwd, HORIZON, regimes,
        ).difference
    )


# --- composite and validation stack ---------------------------------------

def test_composite_recovers_a_positive_ic(built):
    cfg, _, _, ctx, panels, _, fwd, _ = built
    weights = compute_weights(panels, fwd, cfg, sessions=ctx.sessions)
    signal = build_composite(panels, weights, cfg)
    assert not signal.is_empty()
    s = summarize_ic(rank_ic(signal, fwd, HORIZON), factor="composite", horizon=HORIZON)
    assert s.mean > 0
    assert s.t_stat > 2.0


def test_composite_weights_favour_the_stronger_factors(built):
    """Averaged over the whole history, not read off one date.

    A trailing IC/IR at a single date is noisy enough that any one date can rank a
    no-effect factor above a real one; the mean over ~600 dates is the claim that
    actually means the weighting works.
    """
    cfg, _, _, ctx, panels, _, fwd, _ = built
    weights = compute_weights(panels, fwd, cfg, sessions=ctx.sessions)
    mean_weight = dict(
        weights.weights.group_by("factor").agg(pl.col("weight").mean()).rows()
    )
    # The DGP embeds no volume-trend effect, but does embed momentum.
    assert mean_weight["momentum"] > mean_weight["volume_trend"], mean_weight


def test_backtest_of_a_recovered_factor_is_positive_before_costs(built):
    from flowalpha.backtest.engine import run_backtest

    cfg, _, data, _, panels, _, _, _ = built
    result = run_backtest(panels["momentum"], data.prices, cfg,
                          notional=1.0, holding_days=5)
    assert result.n_days > 0
    assert result.gross_annual_return > 0


def test_deflation_reduces_confidence_as_trials_accumulate(built, tmp_path):
    from flowalpha.backtest.engine import run_backtest

    cfg, _, data, _, panels, _, _, _ = built
    result = run_backtest(panels["momentum"], data.prices, cfg, notional=1.0)
    registry = TrialRegistry.load(tmp_path)
    one = deflated_sharpe_ratio(result.net_returns, registry.register("momentum")).dsr
    for i in range(50):
        registry.register(f"probe_{i}")
    many = deflated_sharpe_ratio(result.net_returns, registry.n_trials).dsr
    assert many <= one


def test_purged_cv_over_the_synthetic_window(built):
    from flowalpha.validation.purged_cv import PurgedKFold, leakage_check

    cfg, _, data, *_ = built
    dates = list(data.calendar.sessions)
    cv = PurgedKFold.from_config(cfg, label_horizon=HORIZON)
    splits = cv.splits(dates)
    assert len(splits) == cfg["validation"]["purged_kfold"]["n_splits"]
    for split in splits:
        assert leakage_check(split, HORIZON, dates)


def test_pbo_over_the_synthetic_factor_set(built):
    from flowalpha.backtest.engine import run_backtest
    from flowalpha.validation.deflated_sharpe import pbo_cscv

    cfg, _, data, _, panels, _, _, _ = built
    series = []
    for name in sorted(panels):
        result = run_backtest(panels[name], data.prices, cfg, notional=1.0)
        if result.n_days:
            series.append(result.net_returns)
    common = min(len(s) for s in series)
    matrix = np.column_stack([s[-common:] for s in series])
    out = pbo_cscv(matrix, n_partitions=8)
    assert 0.0 <= out.pbo <= 1.0
    assert out.n_configs == len(series)


# --- look-ahead, end to end -----------------------------------------------

def test_no_factor_leaks_future_data_on_the_synthetic_tree(built):
    """Every factor recomputed on a poisoned-future panel must be unchanged in the past."""
    cfg, _, data, ctx, panels, _, _, _ = built
    cutoff = ctx.sessions[len(ctx.sessions) // 2]

    poisoned_ctx = FactorContext(
        sessions=list(ctx.sessions), symbols=list(ctx.symbols),
        adj_close=ctx.adj_close.copy(), close=ctx.close.copy(),
        volume=ctx.volume.copy(), turnover=ctx.turnover.copy(),
        delivery_pct=ctx.delivery_pct.copy(),
        cfg=cfg, as_of_date=ctx.as_of_date,
        sectors=dict(ctx.sectors), shares=dict(ctx.shares), shares_available=True,
    )
    idx = ctx.sessions.index(cutoff) + 1
    for matrix in (poisoned_ctx.adj_close, poisoned_ctx.close,
                   poisoned_ctx.volume, poisoned_ctx.turnover, poisoned_ctx.delivery_pct):
        matrix[idx:, :] = 1e9

    poisoned_panels, _ = build_panels(
        default_library(cfg, shares_available=True), poisoned_ctx, verbose=False
    )
    for name, panel in panels.items():
        a = panel.filter(pl.col("date") <= cutoff).sort(["symbol", "date"])
        b = poisoned_panels[name].filter(pl.col("date") <= cutoff).sort(["symbol", "date"])
        assert a.equals(b), f"{name} leaked future data"


def test_store_still_enforces_lags_on_the_synthetic_tree(synthetic_tree):
    cfg, store, data = synthetic_tree
    day = data.calendar.sessions[100]
    flows = store.get("flows_daily", as_of_date=day)
    assert max(flows["date"].to_list()) < day
    prices = store.get("prices", as_of_date=day)
    assert max(prices["date"].to_list()) == day
