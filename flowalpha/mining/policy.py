"""Sequence policies for the alpha miner.

Two implementations behind one interface:

:class:`GRUPolicy`
    A GRU trained with REINFORCE: moving-average baseline, entropy bonus, reward
    clipping. Requires the ``mining`` extra (``pip install -e ".[mining]"``).

:class:`RandomPolicy`
    A uniform sampler over the legal-token mask. Not a fallback that pretends to be the
    real thing -- it is a **declared control**. If the GRU's best formulas are no better
    than random search over the same grammar, the policy is not adding anything, and
    running the control is the only way to know. It also lets the mining tests run
    without a 2 GB dependency.

:func:`make_policy` returns the requested policy and **raises** when torch is missing
rather than silently downgrading to random. A random search reported as a trained policy
would misdescribe the search that produced the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


class PolicyError(Exception):
    """Raised when a policy cannot be constructed as requested."""


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class Rollout:
    """One sampled sequence and the bookkeeping REINFORCE needs."""

    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)


class BasePolicy:
    """Interface shared by both policies."""

    name = "base"
    trainable = False

    def __init__(self, n_actions: int, *, seed: int = 0) -> None:
        self.n_actions = int(n_actions)
        self.rng = np.random.default_rng(seed)

    def sample(self, prefix: Sequence[int], mask: np.ndarray) -> tuple[int, float, float]:
        """Return ``(action, log_prob, entropy)`` for one step."""
        raise NotImplementedError

    def update(self, rollouts: Sequence[Rollout], rewards: Sequence[float]) -> dict:
        """Apply one learning step. Returns diagnostics. No-op for a fixed policy."""
        return {}

    def reset_episode(self) -> None:
        pass


class RandomPolicy(BasePolicy):
    """Uniform over the legal tokens. The declared control, not a stand-in."""

    name = "random"
    trainable = False

    def sample(self, prefix: Sequence[int], mask: np.ndarray) -> tuple[int, float, float]:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            raise PolicyError("no legal action available; the grammar mask is empty")
        action = int(self.rng.choice(legal))
        p = 1.0 / legal.size
        return action, float(np.log(p)), float(np.log(legal.size))


class GRUPolicy(BasePolicy):
    """GRU policy trained with REINFORCE.

    Stabilisers, all from ``config.yaml`` and all load-bearing for a search this noisy:

    * **Moving-average baseline** on the reward, so the gradient reflects whether a
      formula beat the recent average rather than whether the reward was positive. Without
      it, an all-positive reward scale reinforces every sequence.
    * **Entropy bonus**, because a REINFORCE policy over a discrete vocabulary collapses
      onto one sequence within a few hundred episodes otherwise.
    * **Reward clipping**, since an IC that happens to be computed over a handful of dates
      can be enormous, and one such episode would otherwise dominate the whole update.
    """

    name = "gru"
    trainable = True

    def __init__(
        self,
        n_actions: int,
        *,
        seed: int = 0,
        hidden_size: int = 64,
        embedding_size: int = 32,
        learning_rate: float = 1e-3,
        baseline_momentum: float = 0.99,
        entropy_coef: float = 0.01,
        reward_clip: float = 1.0,
    ) -> None:
        super().__init__(n_actions, seed=seed)
        if not torch_available():
            raise PolicyError(
                "GRUPolicy needs torch. Install the mining extra: pip install -e '.[mining]'. "
                "Refusing to substitute RandomPolicy silently -- a random search reported as "
                "a trained policy would misdescribe how the numbers were produced."
            )
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self._torch = torch
        # +1 action index reserved as the start-of-sequence embedding.
        self.embed = nn.Embedding(n_actions + 1, embedding_size)
        self.gru = nn.GRUCell(embedding_size, hidden_size)
        self.head = nn.Linear(hidden_size, n_actions)
        self.hidden_size = hidden_size
        self.params = list(self.embed.parameters()) + list(self.gru.parameters()) + list(self.head.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=learning_rate)
        self.baseline_momentum = float(baseline_momentum)
        self.entropy_coef = float(entropy_coef)
        self.reward_clip = float(reward_clip)
        self.baseline: float | None = None
        self._hidden = None
        self._step_log_probs: list = []
        self._step_entropies: list = []

    def reset_episode(self) -> None:
        torch = self._torch
        self._hidden = torch.zeros(1, self.hidden_size)
        self._last_action = torch.tensor([self.n_actions])  # start token
        self._step_log_probs = []
        self._step_entropies = []

    def sample(self, prefix: Sequence[int], mask: np.ndarray) -> tuple[int, float, float]:
        torch = self._torch
        if self._hidden is None:
            self.reset_episode()
        emb = self.embed(self._last_action)
        self._hidden = self.gru(emb, self._hidden)
        logits = self.head(self._hidden).squeeze(0)
        mask_t = torch.tensor(np.asarray(mask, dtype=bool))
        if not bool(mask_t.any()):
            raise PolicyError("no legal action available; the grammar mask is empty")
        logits = logits.masked_fill(~mask_t, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        self._step_log_probs.append(log_prob)
        self._step_entropies.append(entropy)
        self._last_action = action.detach().reshape(1)
        return int(action.item()), float(log_prob.item()), float(entropy.item())

    def finish_episode(self):
        """Return the episode's summed log-prob and entropy tensors."""
        torch = self._torch
        if not self._step_log_probs:
            return torch.zeros(()), torch.zeros(())
        return torch.stack(self._step_log_probs).sum(), torch.stack(self._step_entropies).sum()

    def update_batch(self, episode_terms: Sequence[tuple], rewards: Sequence[float]) -> dict:
        """One REINFORCE step over a batch of episodes."""
        torch = self._torch
        if not episode_terms:
            return {}
        clipped = np.clip(np.asarray(rewards, dtype=float), -self.reward_clip, self.reward_clip)
        clipped = np.nan_to_num(clipped, nan=0.0)
        mean_reward = float(clipped.mean())
        if self.baseline is None:
            self.baseline = mean_reward
        else:
            m = self.baseline_momentum
            self.baseline = m * self.baseline + (1.0 - m) * mean_reward

        loss = torch.zeros(())
        for (log_prob, entropy), reward in zip(episode_terms, clipped):
            advantage = float(reward) - float(self.baseline)
            loss = loss - advantage * log_prob - self.entropy_coef * entropy
        loss = loss / len(episode_terms)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 5.0)
        self.optimizer.step()
        return {
            "loss": float(loss.item()),
            "baseline": float(self.baseline),
            "mean_reward_clipped": mean_reward,
        }


def make_policy(kind: str, n_actions: int, cfg=None, *, seed: int = 0) -> BasePolicy:
    """Construct a policy by name.

    ``gru`` raises when torch is absent; ``random`` always works and is labelled as the
    control in every output it appears in.
    """
    kind = str(kind).lower()
    if kind == "random":
        return RandomPolicy(n_actions, seed=seed)
    if kind == "gru":
        mining = (cfg.get("mining", {}) if cfg is not None else {}) or {}
        return GRUPolicy(
            n_actions,
            seed=seed,
            learning_rate=float(mining.get("learning_rate", 1e-3)),
            baseline_momentum=float(mining.get("baseline_momentum", 0.99)),
            entropy_coef=float(mining.get("entropy_coef", 0.01)),
            reward_clip=float(mining.get("reward_clip", 1.0)),
        )
    raise PolicyError(f"unknown policy {kind!r}; expected 'gru' or 'random'")
