"""Conditional IC, declared experiments, and the conditional strategy.

This is the study. The question is whether a factor's information coefficient differs
across order-flow regimes, and -- if it does -- whether trading the factor only in its
favourable regime beats trading it unconditionally **after costs**.

Two disciplines make the answer trustworthy:

**Declared hypotheses.** The factor-to-regime map lives in ``config.yaml`` under
``signals.regime_overlay.factor_regime_map`` and is a *stated hypothesis*, not the
output of a search over pairs. :class:`ConditioningExperiment` records the direction
expected in advance, so a result in the wrong direction is a refutation rather than a
finding with the sign flipped.

**Every evaluation is a trial.** Each conditional experiment registers in the
cumulative trial registry, so the Deflated Sharpe of anything built on top of it is
deflated against the real number of looks.

A conditional IC difference that is large but statistically indistinguishable from zero
is reported as exactly that. The pipeline's correct output on a factor with no
conditional structure is "no conditional structure".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import polars as pl
from scipy.stats import norm as _norm

from ..config import Config
from ..validation.ic import newey_west_se, rank_ic, summarize_ic
from .regimes import RegimeSet


@dataclass(frozen=True)
class ConditionalIC:
    """IC summary within one regime bucket."""

    factor: str
    regime_column: str
    bucket: str
    n_dates: int
    mean_ic: float
    std_ic: float
    ic_ir: float
    t_stat: float
    hit_rate: float

    def to_dict(self) -> dict:
        return {
            "factor": self.factor, "regime_column": self.regime_column,
            "bucket": self.bucket, "n_dates": self.n_dates, "mean_ic": self.mean_ic,
            "std_ic": self.std_ic, "ic_ir": self.ic_ir,
            "t_stat_newey_west": self.t_stat, "hit_rate": self.hit_rate,
            "deflated": False,
        }


def conditional_ic(
    panel: pl.DataFrame,
    fwd: pl.DataFrame,
    horizon: int,
    regimes: RegimeSet,
    regime_column: str,
    *,
    factor: str = "factor",
    min_names: int = 10,
    min_dates: int = 30,
) -> list[ConditionalIC]:
    """IC summaries per regime bucket.

    The IC series is computed **once** over all dates and then partitioned by regime,
    rather than recomputed per bucket. Same numbers, but it makes it structurally
    impossible for the buckets to disagree with the unconditional series they came from.

    Buckets with fewer than ``min_dates`` observations are dropped: a mean IC over 12
    days is not a regime effect.
    """
    series = rank_ic(panel, fwd, horizon, min_names=min_names)
    if series.is_empty():
        return []
    labelled = regimes.attach(series, regime_column)
    out: list[ConditionalIC] = []
    for bucket in sorted(set(labelled["regime"].to_list())):
        sub = labelled.filter(pl.col("regime") == bucket)
        if sub.height < min_dates:
            continue
        s = summarize_ic(sub.drop("regime"), factor=factor, horizon=horizon)
        out.append(
            ConditionalIC(
                factor=factor, regime_column=regime_column, bucket=str(bucket),
                n_dates=s.n_dates, mean_ic=s.mean, std_ic=s.std, ic_ir=s.ic_ir,
                t_stat=s.t_stat, hit_rate=s.hit_rate,
            )
        )
    return out


def conditional_ic_table(
    panels: Mapping[str, pl.DataFrame],
    fwd: pl.DataFrame,
    horizon: int,
    regimes: RegimeSet,
    regime_columns: Sequence[str],
    *,
    min_names: int = 10,
    min_dates: int = 30,
) -> pl.DataFrame:
    """Tidy conditional-IC table over every (factor, regime column, bucket)."""
    rows = []
    for name, panel in panels.items():
        for col in regime_columns:
            if col not in regimes.columns:
                continue
            for entry in conditional_ic(
                panel, fwd, horizon, regimes, col,
                factor=name, min_names=min_names, min_dates=min_dates,
            ):
                rows.append(entry.to_dict())
    if not rows:
        return pl.DataFrame(
            schema={
                "factor": pl.Utf8, "regime_column": pl.Utf8, "bucket": pl.Utf8,
                "n_dates": pl.Int64, "mean_ic": pl.Float64, "std_ic": pl.Float64,
                "ic_ir": pl.Float64, "t_stat_newey_west": pl.Float64,
                "hit_rate": pl.Float64, "deflated": pl.Boolean,
            }
        )
    return pl.DataFrame(rows).sort(["factor", "regime_column", "bucket"])


# ---------------------------------------------------------------------------
# Declared experiments
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConditioningExperiment:
    """A hypothesis declared before it is tested.

    Attributes
    ----------
    factor:
        Factor whose IC is expected to vary.
    regime_column:
        Conditioning variable.
    favourable / unfavourable:
        The buckets in which the factor's IC is expected to be higher / lower. Stated in
        advance, so a reversed result is a refutation and not a rebranded discovery.
    rationale:
        Why the hypothesis was proposed. Printed in the report next to the result.
    """

    factor: str
    regime_column: str
    favourable: str
    unfavourable: str
    rationale: str = ""

    @property
    def name(self) -> str:
        return f"{self.factor}|{self.regime_column}|{self.favourable}-vs-{self.unfavourable}"


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PowerAnalysis:
    """How large a conditioning effect this design could actually have caught.

    **Why this is part of the result and not a footnote.** "The difference is not
    distinguishable from zero" has two very different causes: there is no effect, or
    the test cannot see one. Only the first is a finding. A test with 20% power that
    reports nothing has learned nothing, and calling that "no conditional structure"
    lets an absence of evidence read as evidence of absence -- the same class of error
    as letting generated numbers be described as real.

    **Calibrated to the gate the pipeline actually uses.** :attr:`ExperimentResult.supported`
    requires ``t >= gate_t_threshold`` with the sign declared in advance, so the honest
    calculation is one-sided against *that* gate, not against a conventional 1.96.
    Under a true effect ``delta`` the statistic is approximately ``Normal(delta / se, 1)``,
    so the probability of declaring support is ``Phi(delta / se - gate)``. Inverting
    gives the effect size detected with any chosen probability.

    **Adequacy is judged against a declared effect, never the observed one.** Power
    evaluated at the effect that happened to be observed is a strictly monotone function
    of the test statistic, so it carries no information the t-stat did not already carry,
    and using it to decide adequacy would make "adequately powered null" unreachable by
    construction. Adequacy is therefore assessed against :attr:`reference_effect` --
    ``reference_effect_fraction`` of the factor's own unconditional IC, declared in
    config alongside the hypothesis direction. That makes it a property of the *design*
    (sample size, autocorrelation, regime balance), which is what the question
    "could this study have seen it?" is actually about.

    Attributes
    ----------
    se_difference:
        Newey-West standard error of the between-bucket IC difference. Inherits the
        autocorrelation adjustment, so overlapping labels do not inflate the apparent
        precision -- and therefore do not understate the MDE.
    reference_effect:
        The effect size worth detecting: ``reference_effect_fraction * |unconditional_ic|``.
        A regime effect smaller than a fraction of the factor's whole average IC is
        unlikely to survive the extra turnover the overlay creates, so it is not the
        thing the study is trying to find.
    detectable_50:
        Effect size at which support is a coin flip (``gate * se``). Below this the test
        is more likely to miss a real effect than to find it.
    mde:
        Minimum effect reliably detected: ``(gate + z_target) * se``.
    power_at_reference:
        Probability the design would declare support if the true effect equalled
        :attr:`reference_effect`. This is the number that decides :attr:`adequate`.
    power_at_observed:
        **Descriptive only.** Probability of declaring support had the true effect been
        as large as the observed difference, computed on ``abs(difference)`` so it speaks
        to magnitude and stays interpretable when the sign came out unpredicted. Monotone
        in ``t``, so it adds no independent evidence -- reported because it answers
        "was the difference we saw even resolvable here?", not used for any decision.
    mde_over_unconditional_ic:
        :attr:`mde` as a multiple of the factor's own unconditional mean IC. Above 1
        means the smallest reliably detectable regime effect exceeds the entire average
        effect being conditioned, which is not a plausible size for a real one.
    adequate:
        Whether :attr:`power_at_reference` clears ``power_target``. ``False`` whenever
        power is unknown: an unquantified design must not be reported as a powered one.
    """

    se_difference: float
    gate_t_threshold: float
    power_target: float
    reference_effect_fraction: float
    reference_effect: float
    detectable_50: float
    mde: float
    power_at_reference: float
    observed_difference: float
    power_at_observed: float
    unconditional_ic: float
    mde_over_unconditional_ic: float
    adequate: bool

    def to_dict(self) -> dict:
        return {
            "se_difference": self.se_difference,
            "gate_t_threshold": self.gate_t_threshold,
            "power_target": self.power_target,
            "reference_effect_fraction": self.reference_effect_fraction,
            "reference_effect": self.reference_effect,
            "detectable_at_50pct_power": self.detectable_50,
            "mde_at_target_power": self.mde,
            "power_at_reference_effect": self.power_at_reference,
            "observed_difference_abs": self.observed_difference,
            "power_at_observed_effect_descriptive_only": self.power_at_observed,
            "unconditional_mean_ic": self.unconditional_ic,
            "mde_over_unconditional_ic": self.mde_over_unconditional_ic,
            "adequately_powered": self.adequate,
        }


def power_analysis(
    *,
    difference: float,
    se_difference: float,
    gate_t_threshold: float,
    unconditional_ic: float,
    power_target: float = 0.80,
    reference_effect_fraction: float = 0.5,
) -> PowerAnalysis:
    """Minimum detectable effect and design power for one declared experiment.

    See :class:`PowerAnalysis` for the derivation and for why adequacy is judged against
    a declared reference effect rather than the observed one. A non-positive or
    non-finite ``se_difference``, or an unusable unconditional IC, yields NaN power with
    ``adequate=False``: "power unknown" and "power adequate" must never render the same.
    """
    gate = float(gate_t_threshold)
    target = float(power_target)
    fraction = float(reference_effect_fraction)
    se = float(se_difference)
    observed = abs(float(difference)) if np.isfinite(difference) else float("nan")
    uncond = float(unconditional_ic)
    denom = abs(uncond)
    reference = fraction * denom if np.isfinite(denom) else float("nan")

    def _power(effect: float) -> float:
        if not (np.isfinite(effect) and np.isfinite(se) and se > 0.0):
            return float("nan")
        return float(_norm.cdf(effect / se - gate))

    if not (np.isfinite(se) and se > 0.0):
        return PowerAnalysis(
            se_difference=float("nan"), gate_t_threshold=gate, power_target=target,
            reference_effect_fraction=fraction, reference_effect=reference,
            detectable_50=float("nan"), mde=float("nan"),
            power_at_reference=float("nan"), observed_difference=observed,
            power_at_observed=float("nan"), unconditional_ic=uncond,
            mde_over_unconditional_ic=float("nan"), adequate=False,
        )

    z_target = float(_norm.ppf(target))
    mde = (gate + z_target) * se
    power_ref = _power(reference)
    return PowerAnalysis(
        se_difference=se, gate_t_threshold=gate, power_target=target,
        reference_effect_fraction=fraction, reference_effect=reference,
        detectable_50=gate * se, mde=mde,
        power_at_reference=power_ref,
        observed_difference=observed, power_at_observed=_power(observed),
        unconditional_ic=uncond,
        mde_over_unconditional_ic=(
            mde / denom if np.isfinite(denom) and denom > 0.0 else float("inf")
        ),
        adequate=bool(np.isfinite(power_ref) and power_ref >= target),
    )


@dataclass
class ExperimentResult:
    """Outcome of one declared experiment."""

    experiment: ConditioningExperiment
    horizon: int
    ic_favourable: float
    ic_unfavourable: float
    n_favourable: int
    n_unfavourable: int
    difference: float
    t_stat: float
    #: True when the difference is in the predicted direction AND |t| clears the gate.
    supported: bool
    gate_t_threshold: float
    per_bucket: list[ConditionalIC] = field(default_factory=list)
    #: Present whenever an IC series existed; ``None`` only when the test never ran.
    power: PowerAnalysis | None = None

    @property
    def verdict(self) -> str:
        """Categorical outcome.

        A null result is split by power. A design that could not have seen an effect of
        the size observed reports INCONCLUSIVE, because "we found nothing" is only a
        finding when the test was capable of finding something.
        """
        if self.supported:
            return "SUPPORTED"
        if not np.isfinite(self.t_stat):
            return "INCONCLUSIVE (insufficient data)"
        if self.difference < 0 and abs(self.t_stat) >= self.gate_t_threshold:
            return "REFUTED (difference significant in the OPPOSITE direction)"
        if self.power is not None and not self.power.adequate:
            p = self.power
            if not np.isfinite(p.power_at_reference):
                return "INCONCLUSIVE (test power could not be quantified)"
            return (
                f"INCONCLUSIVE (underpowered: {p.power_target:.0%} power needs "
                f"|difference| >= {p.mde:.5f} = "
                f"{p.mde_over_unconditional_ic:.1f}x the factor's unconditional IC, but "
                f"only {p.power_at_reference:.0%} power against the declared "
                f"{p.reference_effect_fraction:.0%}-of-IC effect of {p.reference_effect:.5f})"
            )
        return (
            "NOT SUPPORTED (difference not distinguishable from zero; design had "
            f"{self.power.power_at_reference:.0%} power against the declared reference "
            f"effect of {self.power.reference_effect:.5f})"
            if self.power is not None
            else "NOT SUPPORTED (difference not distinguishable from zero)"
        )

    def to_dict(self) -> dict:
        return {
            "name": self.experiment.name,
            "factor": self.experiment.factor,
            "regime_column": self.experiment.regime_column,
            "favourable_bucket": self.experiment.favourable,
            "unfavourable_bucket": self.experiment.unfavourable,
            "rationale": self.experiment.rationale,
            "horizon": self.horizon,
            "ic_favourable": self.ic_favourable,
            "ic_unfavourable": self.ic_unfavourable,
            "n_favourable": self.n_favourable,
            "n_unfavourable": self.n_unfavourable,
            "ic_difference": self.difference,
            "t_stat_difference": self.t_stat,
            "gate_t_threshold": self.gate_t_threshold,
            "supported": self.supported,
            "verdict": self.verdict,
            "power": self.power.to_dict() if self.power is not None else None,
        }


def _bucket_stats(series: pl.DataFrame, bucket: str) -> tuple[np.ndarray, float, float]:
    sub = series.filter(pl.col("regime") == bucket)
    values = np.asarray(sub["ic"].to_list(), dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return values, float("nan"), float("nan")
    se, _ = newey_west_se(values)
    return values, float(values.mean()), se


def run_experiment(
    experiment: ConditioningExperiment,
    panel: pl.DataFrame,
    fwd: pl.DataFrame,
    horizon: int,
    regimes: RegimeSet,
    *,
    gate_t_threshold: float = 1.5,
    min_names: int = 10,
    min_dates: int = 30,
    power_target: float = 0.80,
    reference_effect_fraction: float = 0.5,
) -> ExperimentResult:
    """Evaluate one declared hypothesis.

    The difference in mean IC between buckets is tested with Newey-West standard errors
    on each bucket, combined as independent samples. The autocorrelation adjustment
    matters here as much as anywhere: overlapping labels make consecutive daily ICs
    correlated, and the unadjusted difference test would be badly over-confident.

    Every result carries a :class:`PowerAnalysis`, so a null verdict is always
    accompanied by the effect size the test was capable of detecting. Without it a
    quiet test and a genuinely absent effect are indistinguishable in the output.
    """
    series = rank_ic(panel, fwd, horizon, min_names=min_names)
    per_bucket = conditional_ic(
        panel, fwd, horizon, regimes, experiment.regime_column,
        factor=experiment.factor, min_names=min_names, min_dates=min_dates,
    )
    if series.is_empty():
        return ExperimentResult(
            experiment=experiment, horizon=horizon,
            ic_favourable=float("nan"), ic_unfavourable=float("nan"),
            n_favourable=0, n_unfavourable=0, difference=float("nan"),
            t_stat=float("nan"), supported=False,
            gate_t_threshold=gate_t_threshold, per_bucket=per_bucket,
        )
    labelled = regimes.attach(series, experiment.regime_column)
    fav, mean_fav, se_fav = _bucket_stats(labelled, experiment.favourable)
    unf, mean_unf, se_unf = _bucket_stats(labelled, experiment.unfavourable)

    diff = mean_fav - mean_unf
    if np.isfinite(se_fav) and np.isfinite(se_unf) and (se_fav > 0 or se_unf > 0):
        se_diff = float(np.sqrt(se_fav ** 2 + se_unf ** 2))
        t_stat = diff / se_diff
    else:
        se_diff = float("nan")
        t_stat = float("nan")
    enough = fav.size >= min_dates and unf.size >= min_dates
    supported = bool(
        enough and np.isfinite(t_stat) and diff > 0 and t_stat >= gate_t_threshold
    )

    # The unconditional IC is the yardstick the MDE is expressed against, so it is taken
    # from the same series the buckets partition rather than recomputed.
    all_ic = np.asarray(series["ic"].to_list(), dtype=float)
    all_ic = all_ic[np.isfinite(all_ic)]
    uncond = float(all_ic.mean()) if all_ic.size else float("nan")

    return ExperimentResult(
        experiment=experiment, horizon=horizon,
        ic_favourable=mean_fav, ic_unfavourable=mean_unf,
        n_favourable=int(fav.size), n_unfavourable=int(unf.size),
        difference=float(diff), t_stat=float(t_stat), supported=supported,
        gate_t_threshold=gate_t_threshold, per_bucket=per_bucket,
        power=power_analysis(
            difference=float(diff), se_difference=se_diff,
            gate_t_threshold=gate_t_threshold, unconditional_ic=uncond,
            power_target=power_target,
            reference_effect_fraction=reference_effect_fraction,
        ),
    )


def declared_experiments(cfg: Config, regimes: RegimeSet) -> list[ConditioningExperiment]:
    """Build the experiment list from the declared ``factor_regime_map``.

    The map names a factor prefix and a regime column. The favourable and unfavourable
    buckets follow from the hypothesis, not from the data:

    * ``fii_regime``: momentum is expected to work better when FIIs are **buying**
      (``high``) than selling (``low``).
    * ``retail_extreme``: reversal is expected to work better when retail participation
      is **extreme** than when it is ``mid``.

    Factors are matched by prefix so that ``reversal`` in the map covers both
    ``reversal_5d`` and ``reversal_21d`` without the config having to enumerate the
    windows it already declares elsewhere.
    """
    mapping = cfg.get("signals.regime_overlay.factor_regime_map", {}) or {}
    rationale = {
        "fii_regime": (
            "Hypothesis: momentum is a flow-continuation effect, so it should pay more "
            "when the dominant institutional buyer is adding than when it is selling."
        ),
        "retail_extreme": (
            "Hypothesis: short-horizon reversal is compensation for absorbing "
            "uninformed pressure, so it should pay more when retail participation is at "
            "an extreme."
        ),
    }
    favourable = {"fii_regime": ("high", "low"), "retail_extreme": ("extreme", "mid")}
    out: list[ConditioningExperiment] = []
    for factor_prefix, regime_col in mapping.items():
        if regime_col not in regimes.columns:
            continue
        buckets = set(regimes.buckets(regime_col))
        fav, unf = favourable.get(regime_col, (None, None))
        if fav is None:
            ordered = sorted(buckets)
            if len(ordered) < 2:
                continue
            fav, unf = ordered[-1], ordered[0]
        if fav not in buckets or unf not in buckets:
            continue
        out.append(
            ConditioningExperiment(
                factor=str(factor_prefix), regime_column=str(regime_col),
                favourable=fav, unfavourable=unf,
                rationale=rationale.get(regime_col, "declared in config.yaml"),
            )
        )
    return out


def expand_experiments(
    experiments: Sequence[ConditioningExperiment], factor_names: Sequence[str]
) -> list[tuple[ConditioningExperiment, str]]:
    """Match declared factor prefixes to concrete factor names.

    Returns ``(experiment, factor_name)`` pairs. A declared prefix that matches nothing
    yields no pairs and is therefore visible as an absent row rather than a silent skip.
    """
    out = []
    for exp in experiments:
        for name in factor_names:
            if name == exp.factor or name.startswith(exp.factor):
                out.append((exp, name))
    return out


# ---------------------------------------------------------------------------
# The conditional strategy
# ---------------------------------------------------------------------------

def conditional_panel(
    panel: pl.DataFrame,
    regimes: RegimeSet,
    regime_column: str,
    active_buckets: Sequence[str],
    *,
    flat_outside: bool = True,
) -> pl.DataFrame:
    """Restrict a factor panel to the sessions where its regime is favourable.

    With ``flat_outside`` (the default) the panel keeps every date but zeroes the signal
    outside the active buckets, so the backtest holds nothing on those days. That is the
    honest comparison against the unconditional strategy: a version that simply dropped
    the dates would be measured over a shorter, self-selected sample and would look
    better for that reason alone.
    """
    if panel.is_empty():
        return panel
    labels = regimes.labels(regime_column)
    joined = panel.join(labels, on="date", how="left")
    active = joined.with_columns(pl.col("regime").is_in(list(active_buckets)).fill_null(False).alias("_on"))
    if flat_outside:
        return active.with_columns(
            pl.when(pl.col("_on")).then(pl.col("value")).otherwise(0.0).alias("value")
        ).drop(["regime", "_on"])
    return active.filter(pl.col("_on")).drop(["regime", "_on"])


def regime_exposure(
    regimes: RegimeSet, regime_column: str, active_buckets: Sequence[str]
) -> float:
    """Fraction of *defined* sessions on which the conditional strategy is invested.

    Reported alongside every conditional result: a strategy that is only in the market
    30% of the time has a mechanically lower total return, and comparing its raw return
    to an always-invested strategy without saying so would be misleading.
    """
    labels = regimes.labels(regime_column)
    if labels.is_empty():
        return float("nan")
    on = labels.filter(pl.col("regime").is_in(list(active_buckets)))
    return on.height / labels.height
