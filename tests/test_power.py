"""Power and minimum detectable effect for the declared conditioning experiments.

The point of these tests is not that the arithmetic matches scipy -- it is that a null
result cannot be reported as a finding unless the design could have detected the effect
it was declared to look for. Two failure modes are pinned specifically:

* adequacy judged against the *observed* effect, which is the t-statistic rewritten and
  would make an adequately-powered null unreachable by construction;
* an unquantifiable power silently defaulting to "adequate".
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from flowalpha.conditioning.analysis import (
    ConditioningExperiment,
    ExperimentResult,
    PowerAnalysis,
    power_analysis,
)

GATE = 1.5
SE = 0.01
# An unconditional IC of 0.02 with a 50% reference fraction puts the declared effect at
# 0.01 -- exactly one SE, which is deliberately below the gate so the default fixture is
# an underpowered design.
UNCOND = 0.02


def _pa(**kw) -> PowerAnalysis:
    base = dict(
        difference=0.005, se_difference=SE, gate_t_threshold=GATE,
        unconditional_ic=UNCOND, power_target=0.80, reference_effect_fraction=0.5,
    )
    base.update(kw)
    return power_analysis(**base)


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------

def test_coin_flip_threshold_is_the_gate_times_the_standard_error():
    """At delta = gate*se the statistic is centred exactly on the gate, so support is a
    coin flip. This is the boundary below which a miss is more likely than a find."""
    p = _pa()
    assert p.detectable_50 == pytest.approx(GATE * SE)
    assert _pa(difference=p.detectable_50).power_at_observed == pytest.approx(0.5)


def test_mde_is_the_effect_detected_at_exactly_the_target_power():
    """Inverse consistency: feeding the MDE back in as the true effect must return the
    target power. This catches a sign error or a swapped ppf/cdf, which a hand-computed
    constant would not."""
    p = _pa()
    assert p.mde == pytest.approx((GATE + norm.ppf(0.80)) * SE)
    assert _pa(difference=p.mde).power_at_observed == pytest.approx(0.80)


def test_mde_scales_with_the_standard_error_not_with_the_observed_effect():
    """The MDE is a property of the design. Doubling the noise doubles it; changing what
    was observed leaves it alone."""
    assert _pa(se_difference=2 * SE).mde == pytest.approx(2 * _pa().mde)
    assert _pa(difference=0.9).mde == pytest.approx(_pa(difference=0.0).mde)


def test_mde_is_expressed_as_a_multiple_of_the_unconditional_ic():
    p = _pa()
    assert p.mde_over_unconditional_ic == pytest.approx(p.mde / abs(UNCOND))
    # Sign of the unconditional IC must not flip the ratio: a negative-IC factor is
    # conditioned the same way.
    assert _pa(unconditional_ic=-UNCOND).mde_over_unconditional_ic == pytest.approx(
        p.mde_over_unconditional_ic
    )


def test_reference_effect_is_the_declared_fraction_of_the_unconditional_ic():
    assert _pa(reference_effect_fraction=0.5).reference_effect == pytest.approx(0.01)
    assert _pa(reference_effect_fraction=1.0).reference_effect == pytest.approx(0.02)


def test_a_bigger_sample_makes_a_previously_hopeless_design_adequate():
    """se shrinks with sqrt(n). The same declared effect that is invisible at this sample
    size becomes detectable once the noise falls far enough -- which is the whole reason
    the MDE is worth reporting."""
    assert not _pa().adequate
    assert _pa(se_difference=SE / 10).adequate


# ---------------------------------------------------------------------------
# Adequacy must not be the t-statistic in disguise
# ---------------------------------------------------------------------------

def test_adequacy_ignores_the_observed_difference_entirely():
    """The regression this file exists for. An earlier version set `adequate` from power
    at the observed effect; because that is monotone in t, adequacy became a restatement
    of significance and 'adequately powered null' could never occur."""
    tiny = _pa(difference=0.0)
    huge = _pa(difference=10.0)
    assert tiny.adequate == huge.adequate
    assert tiny.power_at_reference == pytest.approx(huge.power_at_reference)


def test_an_adequately_powered_null_is_reachable():
    """Construct a design with enough precision to see the declared effect but an
    observed difference far below the gate. If this combination cannot exist, the verdict
    taxonomy has a dead branch and every null would read as inconclusive."""
    p = _pa(se_difference=SE / 10, difference=0.0002)
    assert p.adequate
    t = 0.0002 / (SE / 10)
    assert t < GATE  # would not be declared supported


def test_power_at_observed_is_monotone_in_the_observed_effect():
    """Kept as a descriptive figure, so its behaviour is still pinned."""
    powers = [_pa(difference=d).power_at_observed for d in (0.0, 0.01, 0.02, 0.05)]
    assert powers == sorted(powers)


def test_power_uses_the_magnitude_so_a_wrong_signed_difference_is_still_quantified():
    """A refuted-direction difference is still a difference of some size, and the
    question 'could this design resolve an effect that big?' remains meaningful."""
    assert _pa(difference=-0.03).power_at_observed == pytest.approx(
        _pa(difference=+0.03).power_at_observed
    )


# ---------------------------------------------------------------------------
# Unknown power is never adequate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("se", [0.0, -1.0, float("nan"), float("inf")])
def test_an_unusable_standard_error_yields_nan_power_and_is_not_adequate(se):
    p = _pa(se_difference=se)
    assert not p.adequate
    assert math.isnan(p.power_at_reference)
    assert math.isnan(p.mde)


def test_a_zero_unconditional_ic_gives_an_infinite_ratio_not_a_crash():
    """No yardstick means the ratio is undefined-large, not zero. Reporting 0.0x would
    read as 'the MDE is negligible', the exact opposite of the truth."""
    p = _pa(unconditional_ic=0.0)
    assert p.mde_over_unconditional_ic == float("inf")
    assert not p.adequate  # reference effect collapses to 0, which nothing can detect


def test_a_nan_unconditional_ic_is_not_adequate():
    p = _pa(unconditional_ic=float("nan"))
    assert not p.adequate


# ---------------------------------------------------------------------------
# Verdict taxonomy
# ---------------------------------------------------------------------------

def _result(*, difference: float, t_stat: float, power: PowerAnalysis | None) -> ExperimentResult:
    exp = ConditioningExperiment(
        factor="momentum", regime_column="fii_regime",
        favourable="high", unfavourable="low",
    )
    return ExperimentResult(
        experiment=exp, horizon=5, ic_favourable=0.02, ic_unfavourable=0.01,
        n_favourable=600, n_unfavourable=600, difference=difference, t_stat=t_stat,
        supported=bool(difference > 0 and t_stat >= GATE), gate_t_threshold=GATE,
        power=power,
    )


def test_an_underpowered_null_reports_inconclusive_and_quotes_the_mde():
    r = _result(difference=0.005, t_stat=0.5, power=_pa(difference=0.005))
    assert r.verdict.startswith("INCONCLUSIVE (underpowered")
    assert f"{r.power.mde:.5f}" in r.verdict


def test_an_adequately_powered_null_reports_not_supported():
    """The distinction that licenses the stronger claim: this one IS evidence against a
    conditioning effect, where the underpowered case is not."""
    r = _result(difference=0.0002, t_stat=0.2, power=_pa(se_difference=SE / 10, difference=0.0002))
    assert r.verdict.startswith("NOT SUPPORTED")


def test_a_supported_result_is_unaffected_by_power():
    """Power bounds what a null may claim. It never downgrades a positive finding: an
    effect that cleared the gate was, by definition, detected."""
    r = _result(difference=0.05, t_stat=3.0, power=_pa(difference=0.05))
    assert not r.power.adequate
    assert r.verdict == "SUPPORTED"


def test_a_refutation_outranks_the_power_caveat():
    """A significant difference in the wrong direction is a refutation, not a power
    problem -- the design plainly resolved something."""
    r = _result(difference=-0.05, t_stat=-3.0, power=_pa(difference=-0.05))
    assert r.verdict.startswith("REFUTED")


def test_unquantifiable_power_says_so_rather_than_claiming_a_null():
    r = _result(difference=0.005, t_stat=0.5, power=_pa(se_difference=float("nan")))
    assert r.verdict == "INCONCLUSIVE (test power could not be quantified)"


def test_insufficient_data_outranks_everything():
    r = _result(difference=float("nan"), t_stat=float("nan"), power=None)
    assert r.verdict == "INCONCLUSIVE (insufficient data)"


def test_power_is_serialised_with_the_experiment():
    r = _result(difference=0.005, t_stat=0.5, power=_pa(difference=0.005))
    d = r.to_dict()
    assert d["power"]["adequately_powered"] is False
    assert d["power"]["mde_at_target_power"] == pytest.approx(r.power.mde)
    # The descriptive figure is named so a reader cannot mistake it for the deciding one.
    assert "descriptive_only" in "".join(d["power"])


def test_a_result_without_power_still_serialises():
    assert _result(difference=0.005, t_stat=0.5, power=None).to_dict()["power"] is None


# ---------------------------------------------------------------------------
# Integration: the real experiment path attaches power
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def experiment_inputs():
    """A factor that predicts the next session only while ``fii_regime`` is ``high``.

    Self-contained rather than shared with test_conditioning: this file needs to assert
    on the standard error the t-stat was actually formed with, so the fixture has to be
    close enough to read.
    """
    import datetime as _dt

    import polars as pl

    from flowalpha.conditioning.regimes import RegimeSet
    from flowalpha.factors.flow_features import FLOW_FEATURES_SCHEMA
    from flowalpha.validation.ic import forward_returns

    sessions, day = [], _dt.date(2021, 1, 4)
    while len(sessions) < 400:
        if day.weekday() < 5:
            sessions.append(day)
        day += _dt.timedelta(days=1)
    symbols = [f"S{i:02d}" for i in range(30)]

    rng = np.random.default_rng(11)
    regime = ["high" if (i // 20) % 2 == 0 else "low" for i in range(len(sessions))]
    level = {s: 100.0 for s in symbols}
    panel_rows, price_rows, pending, active = [], [], None, False
    for i, d in enumerate(sessions):
        for s in symbols:
            if pending is not None:
                level[s] *= 1.0 + (1.2 if active else 0.0) * 0.01 * pending[s] + 0.01 * float(
                    rng.normal()
                )
            price_rows.append({"date": d, "symbol": s, "adj_close": level[s]})
        signal = {s: float(rng.normal()) for s in symbols}
        panel_rows.extend({"date": d, "symbol": s, "value": signal[s]} for s in symbols)
        pending, active = signal, regime[i] == "high"

    panel = pl.DataFrame(
        panel_rows, schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64}
    )
    prices = pl.DataFrame(
        price_rows, schema={"date": pl.Date, "symbol": pl.Utf8, "adj_close": pl.Float64}
    )
    data: dict[str, list] = {"date": sessions}
    for col in FLOW_FEATURES_SCHEMA:
        if col == "date":
            continue
        if col == "fii_regime":
            data[col] = regime
        elif col == "retail_extreme":
            data[col] = ["extreme" if r == "high" else "mid" for r in regime]
        else:
            data[col] = [None] * len(sessions)
    features = pl.DataFrame(data, schema=FLOW_FEATURES_SCHEMA)
    fwd = forward_returns(prices, [5])
    return panel, fwd, RegimeSet(features=features)


def test_run_experiment_attaches_power_consistent_with_its_own_t_stat(experiment_inputs):
    """se(diff) must be the exact denominator the t-stat was formed with. If the MDE were
    computed from a separately-derived standard error it would describe a different test
    than the one that produced the verdict."""
    from flowalpha.conditioning.analysis import run_experiment

    panel, fwd, regimes = experiment_inputs
    exp = ConditioningExperiment(
        factor="factor", regime_column="fii_regime", favourable="high", unfavourable="low",
    )
    r = run_experiment(exp, panel, fwd, 5, regimes, gate_t_threshold=GATE)
    assert r.power is not None
    assert np.isfinite(r.power.se_difference) and r.power.se_difference > 0
    assert r.difference / r.power.se_difference == pytest.approx(r.t_stat)


def test_run_experiment_measures_the_unconditional_ic_over_all_dates(experiment_inputs):
    """The MDE yardstick comes from the same series the buckets partition, so it cannot
    disagree with the conditional numbers reported beside it."""
    from flowalpha.conditioning.analysis import run_experiment
    from flowalpha.validation.ic import rank_ic

    panel, fwd, regimes = experiment_inputs
    exp = ConditioningExperiment(
        factor="factor", regime_column="fii_regime", favourable="high", unfavourable="low",
    )
    r = run_experiment(exp, panel, fwd, 5, regimes, gate_t_threshold=GATE)
    series = rank_ic(panel, fwd, 5)
    expected = float(np.nanmean(np.asarray(series["ic"].to_list(), dtype=float)))
    assert r.power.unconditional_ic == pytest.approx(expected)
