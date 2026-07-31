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

    @property
    def verdict(self) -> str:
        if self.supported:
            return "SUPPORTED"
        if not np.isfinite(self.t_stat):
            return "INCONCLUSIVE (insufficient data)"
        if self.difference < 0 and abs(self.t_stat) >= self.gate_t_threshold:
            return "REFUTED (difference significant in the OPPOSITE direction)"
        return "NOT SUPPORTED (difference not distinguishable from zero)"

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
) -> ExperimentResult:
    """Evaluate one declared hypothesis.

    The difference in mean IC between buckets is tested with Newey-West standard errors
    on each bucket, combined as independent samples. The autocorrelation adjustment
    matters here as much as anywhere: overlapping labels make consecutive daily ICs
    correlated, and the unadjusted difference test would be badly over-confident.
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
        t_stat = diff / np.sqrt(se_fav ** 2 + se_unf ** 2)
    else:
        t_stat = float("nan")
    enough = fav.size >= min_dates and unf.size >= min_dates
    supported = bool(
        enough and np.isfinite(t_stat) and diff > 0 and t_stat >= gate_t_threshold
    )
    return ExperimentResult(
        experiment=experiment, horizon=horizon,
        ic_favourable=mean_fav, ic_unfavourable=mean_unf,
        n_favourable=int(fav.size), n_unfavourable=int(unf.size),
        difference=float(diff), t_stat=float(t_stat), supported=supported,
        gate_t_threshold=gate_t_threshold, per_bucket=per_bucket,
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
