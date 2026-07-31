"""REINFORCE alpha miner.

Search loop: the policy emits a legal RPN token sequence, the formula is evaluated into a
cross-sectional signal, the reward is its information coefficient (pooled or
regime-robust) minus a novelty penalty against formulas already found, and the policy is
updated.

Multiple-testing discipline
--------------------------
A miner is a multiple-testing machine: 20,000 episodes is 20,000 looks at the same data.
So **every candidate the miner keeps registers as a trial** in the cumulative registry,
and the Deflated Sharpe of anything built on a mined formula is deflated against that
count. Without this, mined alphas would carry the most inflated Sharpes in the project
while appearing to be its best results.

``regime_robust`` reward
-----------------------
Rewards the *minimum* IC across flow regime buckets rather than the pooled IC. A formula
that works only when FIIs are buying scores by its worst regime, so the search is pushed
towards effects that hold across the flow cycle instead of ones that ride a single regime.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import polars as pl

from ..config import Config
from ..factors.base import FactorContext
from ..validation.ic import rank_ic, summarize_ic
from ..validation.neutralization import treat_cross_section
from .grammar import (
    Formula,
    GrammarError,
    Token,
    build_vocabulary,
    evaluate,
    is_complete,
    legal_next,
    terminal_matrices,
)
from .policy import BasePolicy, GRUPolicy, Rollout, make_policy

REWARDS = ("pooled", "regime_robust")

#: Reward given to a formula that could not be scored -- degenerate signal, or an IC
#: series shorter than ``min_dates``. It must be NEGATIVE. Using 0.0 made an unscorable
#: formula look better than every formula with a genuine negative IC, so the top-k filled
#: with candidates that had never been evaluated at all, and REINFORCE was mildly
#: encouraged to produce them.
UNSCORABLE_REWARD = -1.0


class MiningError(Exception):
    """Raised for an unusable mining configuration."""


@dataclass
class Candidate:
    """One mined formula and its evaluation."""

    formula: Formula
    infix: str
    reward: float
    ic: float
    ic_t_stat: float
    n_dates: int
    novelty_penalty: float
    episode: int
    seed: int
    per_regime_ic: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "formula_rpn": str(self.formula),
            "formula": self.infix,
            "length": self.formula.length,
            "reward": self.reward,
            "ic": self.ic,
            "ic_t_stat_newey_west": self.ic_t_stat,
            "n_dates": self.n_dates,
            "novelty_penalty": self.novelty_penalty,
            "episode": self.episode,
            "seed": self.seed,
            "per_regime_ic": dict(self.per_regime_ic),
            "deflated": False,
        }


@dataclass
class MiningResult:
    """Outcome of a mining run."""

    candidates: list[Candidate]
    episodes: int
    seeds: list[int]
    policy: str
    vocabulary: str
    reward_kind: str
    n_illegal: int
    n_degenerate: int
    diagnostics: list[dict] = field(default_factory=list)
    #: Formulas that produced a signal but whose IC series was too short to score.
    n_unscorable: int = 0

    def to_dict(self) -> dict:
        return {
            "episodes": self.episodes,
            "seeds": list(self.seeds),
            "policy": self.policy,
            "vocabulary": self.vocabulary,
            "reward": self.reward_kind,
            "n_illegal_sequences": self.n_illegal,
            "n_degenerate_signals": self.n_degenerate,
            "n_unscorable_signals": self.n_unscorable,
            "candidates": [c.to_dict() for c in self.candidates],
            "diagnostics_tail": self.diagnostics[-10:],
        }


def action_space_size(vocab: Sequence[Token]) -> int:
    """Vocabulary plus one explicit STOP action.

    STOP is a real action, not an implicit length cap. Without it the policy is forced to
    emit exactly ``max_formula_len`` tokens every episode, so the whole search consists of
    maximum-length formulas and the length in ``max_formula_len`` stops being a cap and
    becomes a requirement. STOP is masked out unless the expression already reduces to a
    single value, so it can never produce an unparseable formula.
    """
    return len(vocab) + 1


def sample_formula(
    policy: BasePolicy, vocab: Sequence[Token], max_len: int, *, min_len: int = 2
) -> tuple[Formula, Rollout]:
    """Sample one legal, complete formula.

    The legality mask makes every episode productive: the policy cannot emit a sequence
    that fails to parse, so no episodes are burned on syntax.
    """
    policy.reset_episode()
    stop_index = len(vocab)
    tokens: list[str] = []
    rollout = Rollout()
    for _ in range(max_len):
        token_mask = legal_next(tokens, vocab, max_len)
        can_stop = len(tokens) >= min_len and is_complete(tokens, vocab)
        mask = np.concatenate([token_mask, np.array([can_stop])])
        if not mask.any():
            break
        action, log_prob, entropy = policy.sample(rollout.actions, mask)
        rollout.actions.append(action)
        rollout.log_probs.append(log_prob)
        rollout.entropies.append(entropy)
        if action == stop_index:
            break
        tokens.append(vocab[action].name)
    return Formula(tokens=tuple(tokens)), rollout


def signal_from_formula(
    formula: Formula,
    terminals: Mapping[str, np.ndarray],
    ctx: FactorContext,
    vocab: Sequence[Token],
) -> pl.DataFrame | None:
    """Evaluate and standardise a formula into a factor panel, or None if unusable.

    Numpy warnings are suppressed here rather than globally: a search over thousands of
    random formulas will constantly divide by zero and take the mean of empty slices, and
    those are expected outcomes handled by the NaN checks below -- not conditions a human
    needs to see thousands of times.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            matrix = evaluate(formula, terminals, vocab)
    except (GrammarError, ValueError, FloatingPointError):
        return None
    if matrix.shape != (ctx.n_sessions, ctx.n_symbols):
        # A flow-only formula is constant within a date; broadcasting keeps the shape,
        # but a genuinely mis-shaped result is a grammar bug and is dropped.
        return None
    if not np.isfinite(matrix).any():
        return None
    panel = ctx.to_long(matrix)
    if panel.is_empty():
        return None
    return treat_cross_section(panel, winsor=(0.01, 0.99))


def formula_reward(
    panel: pl.DataFrame,
    fwd: pl.DataFrame,
    horizon: int,
    *,
    kind: str = "pooled",
    regimes=None,
    regime_column: str = "fii_regime",
    min_dates: int = 60,
) -> tuple[float, float, float, int, dict]:
    """Reward for one candidate.

    Returns ``(reward, ic, t_stat, n_dates, per_regime_ic)``. The reward is the pooled
    mean IC, or the **minimum** across regime buckets when ``kind`` is ``regime_robust``.
    """
    series = rank_ic(panel, fwd, horizon)
    if series.is_empty() or series.height < min_dates:
        return float("nan"), float("nan"), float("nan"), series.height, {}
    summary = summarize_ic(series, factor="mined", horizon=horizon)
    per_regime: dict = {}
    reward = summary.mean
    if kind == "regime_robust":
        if regimes is None or regime_column not in getattr(regimes, "columns", ()):
            raise MiningError(
                "reward='regime_robust' needs a RegimeSet with the requested column; "
                "refusing to fall back to pooled, which would mislabel the objective"
            )
        labelled = regimes.attach(series, regime_column)
        if labelled.is_empty():
            return float("nan"), summary.mean, summary.t_stat, series.height, {}
        for bucket in sorted(set(labelled["regime"].to_list())):
            sub = labelled.filter(pl.col("regime") == bucket)
            if sub.height < max(20, min_dates // 4):
                continue
            per_regime[str(bucket)] = float(np.nanmean(sub["ic"].to_numpy()))
        reward = min(per_regime.values()) if per_regime else float("nan")
    return float(reward), summary.mean, summary.t_stat, series.height, per_regime


def novelty_penalty(
    formula: Formula, kept: Sequence[Candidate], lam: float
) -> float:
    """Penalty for resembling a formula already kept.

    Similarity is token-set Jaccard overlap. Crude but effective for this grammar, and
    cheap enough to evaluate every episode. Its purpose is to stop the policy converging
    on one sequence and re-discovering it thousands of times, which would inflate the
    trial count without exploring anything.
    """
    if not kept or lam <= 0:
        return 0.0
    mine = set(formula.tokens)
    best = 0.0
    for candidate in kept:
        other = set(candidate.formula.tokens)
        union = mine | other
        if not union:
            continue
        best = max(best, len(mine & other) / len(union))
    return float(lam * best)


def mine(
    ctx: FactorContext,
    fwd: pl.DataFrame,
    cfg: Config,
    *,
    flow_features: pl.DataFrame | None = None,
    regimes=None,
    episodes: int | None = None,
    seeds: Sequence[int] | None = None,
    policy_kind: str | None = None,
    top_k: int = 10,
    batch_size: int = 16,
    verbose: bool = False,
) -> MiningResult:
    """Run the miner.

    Parameters come from the ``mining`` block of ``config.yaml``. ``episodes`` overrides
    it for smoke runs (``mining.smoke_episodes``).
    """
    mcfg = cfg.get("mining", {}) or {}
    vocabulary = str(mcfg.get("vocab", "flow_augmented"))
    reward_kind = str(mcfg.get("reward", "pooled"))
    if reward_kind not in REWARDS:
        raise MiningError(f"unknown reward {reward_kind!r}; expected one of {REWARDS}")
    max_len = int(mcfg.get("max_formula_len", 20))
    lam = float(mcfg.get("novelty_lambda", 0.2))
    n_episodes = int(episodes if episodes is not None else mcfg.get("episodes", 20000))
    seed_list = [int(s) for s in (seeds if seeds is not None else mcfg.get("seeds", [0]))]
    policy_name = str(policy_kind or mcfg.get("policy", "gru"))
    horizon = int(cfg.get("signals.composite.ic_horizon", 5))

    vocab = build_vocabulary(vocabulary)
    terminals = terminal_matrices(ctx, flow_features)

    kept: list[Candidate] = []
    diagnostics: list[dict] = []
    n_illegal = n_degenerate = n_unscorable = 0

    for seed in seed_list:
        policy = make_policy(policy_name, action_space_size(vocab), cfg, seed=seed)
        episode_terms: list = []
        batch_rewards: list[float] = []

        for episode in range(n_episodes):
            formula, rollout = sample_formula(policy, vocab, max_len)
            if not is_complete(formula.tokens, vocab):
                n_illegal += 1
                if isinstance(policy, GRUPolicy):
                    episode_terms.append(policy.finish_episode())
                    batch_rewards.append(0.0)
                continue
            panel = signal_from_formula(formula, terminals, ctx, vocab)
            scored = False
            if panel is None:
                n_degenerate += 1
                reward = UNSCORABLE_REWARD
                ic = t_stat = float("nan")
                n_dates = 0
                per_regime: dict = {}
                penalty = 0.0
            else:
                raw_reward, ic, t_stat, n_dates, per_regime = formula_reward(
                    panel, fwd, horizon, kind=reward_kind,
                    regimes=regimes, regime_column="fii_regime",
                )
                penalty = novelty_penalty(formula, kept, lam)
                scored = bool(np.isfinite(raw_reward))
                if scored:
                    reward = raw_reward - penalty
                else:
                    # Too few dates to form an IC. Not a zero-IC formula -- an unmeasured
                    # one, which must not be allowed to outrank measured ones.
                    n_unscorable += 1
                    reward = UNSCORABLE_REWARD

            if isinstance(policy, GRUPolicy):
                episode_terms.append(policy.finish_episode())
                batch_rewards.append(float(reward))
                if len(episode_terms) >= batch_size:
                    diag = policy.update_batch(episode_terms, batch_rewards)
                    diag.update({"seed": seed, "episode": episode})
                    diagnostics.append(diag)
                    episode_terms, batch_rewards = [], []

            # Keep only formulas that were ACTUALLY scored.
            if scored and np.isfinite(ic):
                kept.append(
                    Candidate(
                        formula=formula, infix=formula.infix(vocab), reward=float(reward),
                        ic=float(ic), ic_t_stat=float(t_stat), n_dates=int(n_dates),
                        novelty_penalty=float(penalty), episode=episode, seed=seed,
                        per_regime_ic=per_regime,
                    )
                )
                kept.sort(key=lambda c: (-c.reward, c.formula.length))
                del kept[top_k:]

            if verbose and episode and episode % max(1, n_episodes // 10) == 0:
                best = kept[0].reward if kept else float("nan")
                print(f"    seed {seed} episode {episode}/{n_episodes} best reward {best:.5f}",
                      flush=True)

        if isinstance(policy, GRUPolicy) and episode_terms:
            policy.update_batch(episode_terms, batch_rewards)

    return MiningResult(
        candidates=kept, episodes=n_episodes, seeds=seed_list,
        policy=policy_name, vocabulary=vocabulary, reward_kind=reward_kind,
        n_illegal=n_illegal, n_degenerate=n_degenerate, diagnostics=diagnostics,
        n_unscorable=n_unscorable,
    )


def register_candidates(result: MiningResult, registry) -> int:
    """Register every kept candidate as a trial.

    A miner is a multiple-testing machine; without this the mined alphas would carry the
    most inflated Sharpes in the project while looking like its best results. Returns the
    new cumulative trial count.
    """
    for candidate in result.candidates:
        registry.register(
            f"mined:{candidate.formula}",
            kind="mined",
            detail={
                "reward": candidate.reward, "ic": candidate.ic,
                "episodes": result.episodes, "policy": result.policy,
                "vocabulary": result.vocabulary, "reward_kind": result.reward_kind,
            },
        )
    return registry.n_trials
