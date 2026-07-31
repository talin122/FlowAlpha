"""Multiple-testing honesty: Deflated Sharpe Ratio, PBO, and a cumulative trial registry.

The problem
-----------
A Sharpe ratio quoted without a trial count is not a result. If you evaluate 50
strategies on the same data, the best one has an inflated Sharpe *even when none of
them has any edge*, because you selected on the noise. The expected maximum Sharpe
under the null grows roughly with ``sqrt(2 * log(N))``.

Three tools here, all from Bailey & Lopez de Prado:

:func:`deflated_sharpe_ratio`
    The probability that the observed Sharpe exceeds what the *best of N trials* would
    produce under the null, accounting for the return distribution's skew and kurtosis
    as well as the sample length.

:class:`TrialRegistry`
    The trial count has to be *cumulative across the project*, not per-script. A
    registry persisted at ``results/trial_registry.json`` counts every strategy this
    repository has ever evaluated, so re-running with a tweak inflates the count rather
    than resetting it.

:func:`pbo_cscv`
    Probability of Backtest Overfitting via combinatorially symmetric
    cross-validation: split the return series into ``S`` blocks, take every balanced
    train/test partition, select the best configuration in-sample, and measure how
    often it lands below median out-of-sample.

IC t-statistics are **not** deflated by any of this. They are reported raw
(Newey-West adjusted for autocorrelation only), and labelled as such.
"""

from __future__ import annotations

import datetime as _dt
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import stats

TRADING_DAYS = 252
TRIAL_REGISTRY_FILENAME = "trial_registry.json"


# ---------------------------------------------------------------------------
# Sharpe
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: Sequence[float] | np.ndarray,
    *,
    periods_per_year: int = TRADING_DAYS,
    risk_free: float = 0.0,
    annualize: bool = True,
) -> float:
    """Sharpe ratio of a return series.

    ``risk_free`` is a per-period rate, subtracted before the ratio is formed.
    Annualised by ``sqrt(periods_per_year)`` unless told otherwise -- the square-root
    scaling assumes serial independence, which daily equity returns roughly satisfy and
    overlapping-horizon strategy returns do not; that is why the backtest engine
    reports non-overlapping daily P&L.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    excess = arr - risk_free
    sd = float(excess.std(ddof=1))
    # A *relative* degeneracy test, not `sd > 0`. The standard deviation of a
    # numerically constant series is not exactly zero -- np.full(100, 0.001) gives
    # 2.2e-19 -- and dividing by that yields a Sharpe of 7e16, which then sails into
    # a report as a spectacular result. Compare against the series' own scale.
    scale = max(abs(float(excess.mean())), float(np.abs(excess).max()), 1e-300)
    if not (sd > 1e-12 * scale):
        return float("nan")
    sr = float(excess.mean()) / sd
    return sr * math.sqrt(periods_per_year) if annualize else sr


def expected_max_sharpe(n_trials: int, *, variance: float = 1.0) -> float:
    """Expected maximum of ``n_trials`` iid standard-normal Sharpe estimates.

    Uses the standard extreme-value approximation with the Euler-Mascheroni
    correction. This is the quantity a raw Sharpe has to beat before it means anything.
    """
    n = max(1, int(n_trials))
    if n == 1:
        return 0.0
    gamma = 0.5772156649015329
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * math.e))
    return math.sqrt(variance) * ((1.0 - gamma) * z1 + gamma * z2)


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Outcome of a DSR computation, with every input recorded.

    The inputs are kept because a deflated Sharpe is only interpretable alongside the
    trial count it was deflated against.
    """

    sharpe_annual: float
    sharpe_per_period: float
    n_obs: int
    n_trials: int
    skew: float
    kurtosis: float
    expected_max_sharpe_per_period: float
    dsr: float
    benchmark_sr_annual: float

    def to_dict(self) -> dict:
        return {
            "sharpe_annual": self.sharpe_annual,
            "sharpe_per_period": self.sharpe_per_period,
            "n_obs": self.n_obs,
            "n_trials": self.n_trials,
            "skew": self.skew,
            "kurtosis_excess": self.kurtosis,
            "expected_max_sharpe_per_period": self.expected_max_sharpe_per_period,
            "deflated_sharpe_probability": self.dsr,
            "benchmark_sr_annual": self.benchmark_sr_annual,
        }


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skew: float,
    kurtosis_excess: float,
) -> float:
    """Probability that the true (per-period) Sharpe exceeds ``benchmark_sr``.

    The denominator is the Sharpe estimator's standard error under non-normality:
    negative skew and fat tails both make a given Sharpe less reliable, so a strategy
    that earns its Sharpe by selling tails is penalised, correctly.
    """
    if n_obs < 2 or not math.isfinite(observed_sr):
        return float("nan")
    denom_sq = (
        1.0
        - skew * observed_sr
        + ((kurtosis_excess + 3.0) - 1.0) / 4.0 * observed_sr ** 2
    )
    if denom_sq <= 0:
        return float("nan")
    se = math.sqrt(denom_sq / (n_obs - 1))
    if se <= 0:
        return float("nan")
    return float(stats.norm.cdf((observed_sr - benchmark_sr) / se))


def deflated_sharpe_ratio(
    returns: Sequence[float] | np.ndarray,
    n_trials: int,
    *,
    periods_per_year: int = TRADING_DAYS,
    benchmark_sr_annual: float = 0.0,
    trial_sharpe_variance: float | None = None,
) -> DeflatedSharpeResult:
    """Deflate a Sharpe ratio against ``n_trials`` evaluated strategies.

    The benchmark is raised from ``benchmark_sr_annual`` to the expected maximum
    Sharpe of ``n_trials`` draws under the null, then the probabilistic Sharpe ratio is
    evaluated against that raised bar. A DSR of 0.95 means "even allowing for the fact
    that we looked ``n_trials`` times, there is a 95% chance the true Sharpe beats the
    benchmark".

    ``trial_sharpe_variance`` is the cross-trial variance of per-period Sharpe
    estimates. When unknown it defaults to the estimator's own sampling variance
    ``1 / (n_obs - 1)``, which is the conventional fallback and is conservative for a
    set of related strategies.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    sr_period = sharpe_ratio(arr, periods_per_year=periods_per_year, annualize=False)
    sr_annual = sr_period * math.sqrt(periods_per_year) if math.isfinite(sr_period) else float("nan")
    if n < 3 or not math.isfinite(sr_period):
        return DeflatedSharpeResult(
            sharpe_annual=sr_annual, sharpe_per_period=sr_period, n_obs=n,
            n_trials=int(n_trials), skew=float("nan"), kurtosis=float("nan"),
            expected_max_sharpe_per_period=float("nan"), dsr=float("nan"),
            benchmark_sr_annual=benchmark_sr_annual,
        )

    skew = float(stats.skew(arr, bias=False))
    kurt = float(stats.kurtosis(arr, bias=False))  # excess
    variance = trial_sharpe_variance if trial_sharpe_variance is not None else 1.0 / (n - 1)
    emax = expected_max_sharpe(n_trials, variance=variance)
    benchmark_period = benchmark_sr_annual / math.sqrt(periods_per_year)
    bar = benchmark_period + emax
    dsr = probabilistic_sharpe_ratio(sr_period, bar, n, skew, kurt)
    return DeflatedSharpeResult(
        sharpe_annual=sr_annual, sharpe_per_period=sr_period, n_obs=n,
        n_trials=int(n_trials), skew=skew, kurtosis=kurt,
        expected_max_sharpe_per_period=emax, dsr=dsr,
        benchmark_sr_annual=benchmark_sr_annual,
    )


# ---------------------------------------------------------------------------
# Trial registry
# ---------------------------------------------------------------------------

@dataclass
class TrialRegistry:
    """Cumulative count of every strategy this repository has evaluated.

    Persisted, because the honest denominator for multiple-testing correction spans the
    whole research project rather than the current script invocation. Re-running an
    experiment with one parameter changed adds a trial; it does not start over.

    Trials are keyed by name so that re-evaluating the *same* configuration does not
    double-count -- but its ``count`` rises, which is itself informative: a strategy
    evaluated 40 times has been tuned 40 times.
    """

    path: Path
    trials: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, results_dir: str | Path) -> "TrialRegistry":
        path = Path(results_dir) / TRIAL_REGISTRY_FILENAME
        if not path.exists():
            return cls(path=path, trials={})
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls(path=path, trials={})
        return cls(path=path, trials=dict(raw.get("trials", {})))

    @property
    def n_trials(self) -> int:
        """Total evaluations, counting repeats.

        Repeats count because each one is another look at the same data.
        """
        return sum(int(t.get("count", 1)) for t in self.trials.values())

    @property
    def n_distinct(self) -> int:
        return len(self.trials)

    def register(
        self,
        name: str,
        *,
        kind: str = "factor",
        detail: Mapping | None = None,
        timestamp: _dt.datetime | None = None,
    ) -> int:
        """Record one evaluation and return the new cumulative trial count."""
        ts = (timestamp or _dt.datetime.now(_dt.timezone.utc)).isoformat()
        entry = self.trials.get(name)
        if entry is None:
            self.trials[name] = {
                "kind": kind, "count": 1, "first_seen": ts, "last_seen": ts,
                "detail": dict(detail or {}),
            }
        else:
            entry["count"] = int(entry.get("count", 1)) + 1
            entry["last_seen"] = ts
            if detail:
                entry["detail"] = dict(detail)
        return self.n_trials

    def register_many(self, names: Iterable[str], *, kind: str = "factor") -> int:
        for name in names:
            self.register(name, kind=kind)
        return self.n_trials

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "n_trials": self.n_trials,
                    "n_distinct": self.n_distinct,
                    "trials": self.trials,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return self.path

    def summary(self) -> str:
        return (
            f"{self.n_trials} cumulative trials across {self.n_distinct} distinct "
            f"configurations (registry: {self.path.name})"
        )


# ---------------------------------------------------------------------------
# PBO via CSCV
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PBOResult:
    """Probability of backtest overfitting and the logits behind it."""

    pbo: float
    n_partitions: int
    n_combinations: int
    n_configs: int
    logits: tuple[float, ...]
    median_oos_rank: float

    def to_dict(self) -> dict:
        return {
            "pbo": self.pbo, "n_partitions": self.n_partitions,
            "n_combinations": self.n_combinations, "n_configs": self.n_configs,
            "median_oos_rank": self.median_oos_rank,
        }


def pbo_cscv(
    returns_matrix: np.ndarray,
    *,
    n_partitions: int = 16,
    metric=None,
    max_combinations: int | None = 20_000,
) -> PBOResult:
    """Probability of backtest overfitting, via combinatorially symmetric CV.

    Parameters
    ----------
    returns_matrix:
        ``(n_periods, n_configs)`` array of strategy returns. Needs at least two
        configurations: PBO measures how selection among alternatives degrades
        out-of-sample, so a single strategy has nothing to select between.
    n_partitions:
        Number of contiguous blocks ``S``. Must be even, since every partition splits
        the blocks into equal halves. Reduced automatically if the series is too short
        to give each block at least two observations.
    metric:
        Ranking metric, defaulting to the non-annualised Sharpe. Any callable taking a
        1-D return array works.
    max_combinations:
        Cap on the number of balanced partitions evaluated -- ``C(16, 8)`` is 12,870
        but ``C(24, 12)`` is 2.7 million. When the cap binds, a deterministic evenly
        spaced subset is used and ``n_combinations`` reports what was actually run, so
        a truncated sweep is never mistaken for an exhaustive one.

    Returns
    -------
    ``PBOResult`` where ``pbo`` is the fraction of partitions in which the
    in-sample-best configuration ranked below the out-of-sample median.
    """
    matrix = np.asarray(returns_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (periods x configs)")
    n_periods, n_configs = matrix.shape
    if n_configs < 2:
        raise ValueError(
            "PBO needs at least two configurations: it measures the cost of selecting "
            "among alternatives, and there is nothing to select between here"
        )
    if metric is None:
        metric = lambda x: sharpe_ratio(x, annualize=False)  # noqa: E731

    S = int(n_partitions)
    if S % 2 != 0:
        raise ValueError("n_partitions must be even so each partition splits in half")
    while S > 2 and n_periods // S < 2:
        S -= 2
    if n_periods // S < 2:
        raise ValueError(
            f"series of {n_periods} periods is too short for CSCV with {n_partitions} partitions"
        )

    edges = np.linspace(0, n_periods, S + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(S)]

    all_combos = list(itertools.combinations(range(S), S // 2))
    if max_combinations is not None and len(all_combos) > max_combinations:
        step = len(all_combos) / max_combinations
        combos = [all_combos[int(i * step)] for i in range(max_combinations)]
    else:
        combos = all_combos

    logits: list[float] = []
    below = 0
    for train_blocks in combos:
        test_blocks = [i for i in range(S) if i not in train_blocks]
        train_idx = np.concatenate([blocks[i] for i in train_blocks])
        test_idx = np.concatenate([blocks[i] for i in test_blocks])

        is_scores = np.array([metric(matrix[train_idx, j]) for j in range(n_configs)])
        oos_scores = np.array([metric(matrix[test_idx, j]) for j in range(n_configs)])
        if not np.isfinite(is_scores).any() or not np.isfinite(oos_scores).any():
            continue
        best = int(np.nanargmax(np.where(np.isfinite(is_scores), is_scores, -np.inf)))

        finite = np.isfinite(oos_scores)
        ranks = stats.rankdata(np.where(finite, oos_scores, -np.inf))
        # Relative rank in (0, 1]; 1.0 is best out-of-sample.
        omega = float(ranks[best] / (n_configs + 1))
        if omega <= 0.5:
            below += 1
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(math.log(omega / (1 - omega)))

    n_eval = len(logits)
    pbo = (below / n_eval) if n_eval else float("nan")
    median_rank = float(np.median([1 / (1 + math.exp(-l)) for l in logits])) if logits else float("nan")
    return PBOResult(
        pbo=pbo, n_partitions=S, n_combinations=n_eval, n_configs=n_configs,
        logits=tuple(logits), median_oos_rank=median_rank,
    )
