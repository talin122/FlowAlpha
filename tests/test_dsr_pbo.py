"""Deflated Sharpe, PBO, and the cumulative trial registry."""

from __future__ import annotations

import datetime as _dt
import json
import math

import numpy as np
import pytest

from flowalpha.validation.deflated_sharpe import (
    TRADING_DAYS,
    TrialRegistry,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    pbo_cscv,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


# --- Sharpe ----------------------------------------------------------------

def test_sharpe_hand_computed():
    """mean 0.001, sd 0.01 over 252 periods -> annualised 0.1 * sqrt(252)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=10_000)
    x = 0.001 + 0.01 * (x - x.mean()) / x.std(ddof=1)
    assert sharpe_ratio(x, annualize=False) == pytest.approx(0.1, rel=1e-6)
    assert sharpe_ratio(x) == pytest.approx(0.1 * math.sqrt(TRADING_DAYS), rel=1e-6)


def test_sharpe_subtracts_the_risk_free_rate():
    x = np.full(500, 0.001)
    x[0] = 0.002  # give it some variance
    a = sharpe_ratio(x, risk_free=0.0)
    b = sharpe_ratio(x, risk_free=0.001)
    assert a > b


def test_sharpe_of_a_constant_series_is_nan():
    assert math.isnan(sharpe_ratio(np.full(100, 0.001)))


def test_sharpe_of_too_short_series_is_nan():
    assert math.isnan(sharpe_ratio([0.01]))
    assert math.isnan(sharpe_ratio([]))


def test_sharpe_ignores_non_finite_values():
    a = sharpe_ratio([0.01, 0.02, -0.01, 0.03])
    b = sharpe_ratio([0.01, np.nan, 0.02, -0.01, np.inf, 0.03])
    assert a == pytest.approx(b)


# --- expected maximum under the null --------------------------------------

def test_expected_max_sharpe_is_zero_for_one_trial():
    assert expected_max_sharpe(1) == 0.0


def test_expected_max_sharpe_grows_with_trials():
    values = [expected_max_sharpe(n) for n in (2, 10, 100, 1000)]
    assert values == sorted(values)
    assert values[0] > 0


def test_expected_max_sharpe_scales_with_variance():
    a = expected_max_sharpe(100, variance=1.0)
    b = expected_max_sharpe(100, variance=4.0)
    assert b == pytest.approx(2.0 * a)


# --- probabilistic Sharpe -------------------------------------------------

def test_psr_rises_with_the_observed_sharpe():
    a = probabilistic_sharpe_ratio(0.05, 0.0, 1000, 0.0, 0.0)
    b = probabilistic_sharpe_ratio(0.15, 0.0, 1000, 0.0, 0.0)
    assert 0.5 < a < b <= 1.0


def test_psr_penalises_negative_skew_and_fat_tails():
    """A Sharpe earned by selling tails is less trustworthy, and must score lower."""
    clean = probabilistic_sharpe_ratio(0.1, 0.0, 1000, 0.0, 0.0)
    skewed = probabilistic_sharpe_ratio(0.1, 0.0, 1000, -2.0, 0.0)
    fat = probabilistic_sharpe_ratio(0.1, 0.0, 1000, 0.0, 8.0)
    assert skewed < clean
    assert fat < clean


def test_psr_rises_with_sample_length():
    short = probabilistic_sharpe_ratio(0.1, 0.0, 100, 0.0, 0.0)
    long = probabilistic_sharpe_ratio(0.1, 0.0, 5000, 0.0, 0.0)
    assert long > short


def test_psr_on_degenerate_input_is_nan():
    assert math.isnan(probabilistic_sharpe_ratio(0.1, 0.0, 1, 0.0, 0.0))
    assert math.isnan(probabilistic_sharpe_ratio(float("nan"), 0.0, 100, 0.0, 0.0))


# --- deflated Sharpe ------------------------------------------------------

def _series(sr_annual: float, n: int = 1500, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    x = (x - x.mean()) / x.std(ddof=1)
    per_period = sr_annual / math.sqrt(TRADING_DAYS)
    return 0.01 * (x + per_period)


def test_dsr_falls_as_the_trial_count_rises():
    """The core claim: the same Sharpe means less when you looked more times."""
    returns = _series(1.2)
    values = [deflated_sharpe_ratio(returns, n).dsr for n in (1, 10, 100, 1000)]
    assert values == sorted(values, reverse=True)
    assert values[0] > values[-1]


def test_dsr_records_the_trial_count_it_used():
    """A deflated Sharpe is uninterpretable without its denominator."""
    d = deflated_sharpe_ratio(_series(1.0), 57).to_dict()
    assert d["n_trials"] == 57
    assert "expected_max_sharpe_per_period" in d


def test_dsr_of_a_zero_edge_strategy_is_unimpressive_after_many_trials():
    returns = _series(0.0, seed=99)
    assert deflated_sharpe_ratio(returns, 200).dsr < 0.9


def test_dsr_annualises_consistently():
    d = deflated_sharpe_ratio(_series(1.5), 1)
    assert d.sharpe_annual == pytest.approx(d.sharpe_per_period * math.sqrt(TRADING_DAYS))
    assert d.sharpe_annual == pytest.approx(1.5, abs=0.05)


def test_dsr_uses_the_configured_benchmark():
    returns = _series(1.0)
    a = deflated_sharpe_ratio(returns, 10, benchmark_sr_annual=0.0).dsr
    b = deflated_sharpe_ratio(returns, 10, benchmark_sr_annual=1.5).dsr
    assert b < a


def test_dsr_on_short_series_is_nan_but_does_not_crash():
    d = deflated_sharpe_ratio([0.01, 0.02], 5)
    assert math.isnan(d.dsr)
    assert d.n_obs == 2


# --- trial registry -------------------------------------------------------

def test_registry_starts_empty(tmp_path):
    reg = TrialRegistry.load(tmp_path)
    assert reg.n_trials == 0 and reg.n_distinct == 0


def test_registry_counts_repeat_evaluations(tmp_path):
    """A strategy evaluated 40 times has been tuned 40 times; the count must show it."""
    reg = TrialRegistry.load(tmp_path)
    reg.register("momentum")
    reg.register("momentum")
    reg.register("reversal_5d")
    assert reg.n_trials == 3
    assert reg.n_distinct == 2


def test_registry_persists_across_loads(tmp_path):
    reg = TrialRegistry.load(tmp_path)
    reg.register_many(["a", "b", "c"])
    reg.save()
    again = TrialRegistry.load(tmp_path)
    assert again.n_trials == 3
    again.register("d")
    assert again.n_trials == 4


def test_registry_is_cumulative_across_runs(tmp_path):
    """The honest denominator spans the project, not the script invocation."""
    for _ in range(3):
        reg = TrialRegistry.load(tmp_path)
        reg.register_many(["momentum", "reversal_5d"])
        reg.save()
    assert TrialRegistry.load(tmp_path).n_trials == 6


def test_registry_records_timestamps_and_detail(tmp_path):
    reg = TrialRegistry.load(tmp_path)
    ts = _dt.datetime(2026, 7, 30, tzinfo=_dt.timezone.utc)
    reg.register("x", kind="composite", detail={"scheme": "ic_ir"}, timestamp=ts)
    reg.save()
    raw = json.loads((tmp_path / "trial_registry.json").read_text())
    entry = raw["trials"]["x"]
    assert entry["kind"] == "composite"
    assert entry["first_seen"] == ts.isoformat()
    assert entry["detail"]["scheme"] == "ic_ir"


def test_registry_survives_a_corrupt_file(tmp_path):
    (tmp_path / "trial_registry.json").write_text("{not json", encoding="utf-8")
    assert TrialRegistry.load(tmp_path).n_trials == 0


def test_registry_summary_mentions_both_counts(tmp_path):
    reg = TrialRegistry.load(tmp_path)
    reg.register_many(["a", "a", "b"])
    text = reg.summary()
    assert "3 cumulative" in text and "2 distinct" in text


# --- PBO ------------------------------------------------------------------

def test_pbo_is_low_for_a_genuinely_dominant_configuration():
    """One config has real edge, so in-sample selection generalises."""
    rng = np.random.default_rng(1)
    n, k = 800, 6
    matrix = rng.normal(0.0, 0.01, size=(n, k))
    matrix[:, 0] += 0.0025  # a real, persistent edge
    result = pbo_cscv(matrix, n_partitions=8)
    assert result.pbo < 0.25
    assert result.n_configs == k
    assert result.n_combinations > 0


def test_pbo_is_high_when_every_configuration_is_noise():
    """Selection on noise does not generalise, so the winner lands below median."""
    rng = np.random.default_rng(2)
    matrix = rng.normal(0.0, 0.01, size=(600, 12))
    assert pbo_cscv(matrix, n_partitions=8).pbo > 0.35


def test_pbo_requires_more_than_one_configuration():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="at least two configurations"):
        pbo_cscv(rng.normal(size=(200, 1)))


def test_pbo_requires_an_even_partition_count():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="even"):
        pbo_cscv(rng.normal(size=(200, 3)), n_partitions=7)


def test_pbo_reduces_partitions_for_a_short_series():
    rng = np.random.default_rng(4)
    result = pbo_cscv(rng.normal(size=(20, 3)), n_partitions=16)
    assert result.n_partitions < 16


def test_pbo_rejects_a_hopelessly_short_series():
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="too short"):
        pbo_cscv(rng.normal(size=(3, 3)), n_partitions=16)


def test_pbo_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2-D"):
        pbo_cscv(np.zeros(10))


def test_pbo_reports_a_truncated_sweep_honestly():
    """A capped combination count must not be presented as exhaustive."""
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(400, 4))
    full = pbo_cscv(matrix, n_partitions=10, max_combinations=None)
    capped = pbo_cscv(matrix, n_partitions=10, max_combinations=20)
    assert capped.n_combinations == 20
    assert full.n_combinations > 20


def test_pbo_is_deterministic():
    rng = np.random.default_rng(6)
    matrix = rng.normal(size=(400, 5))
    assert pbo_cscv(matrix, n_partitions=8).pbo == pbo_cscv(matrix, n_partitions=8).pbo
