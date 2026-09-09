"""Portfolio-level risk overlays: volatility targeting and drawdown control.

These sit *on top of* the position-level constraints in :mod:`flowalpha.signals.portfolio`
(name caps, sector caps, gross and net exposure limits). Those bound what any single
position or sector may be at a point in time. These bound what the *book as a whole* is
exposed to, as a function of how the strategy has actually been performing.

Causality
---------
Both overlays are functions of **realised returns only**. At the moment the scale for
session ``t`` is requested, the controller has been fed returns for sessions
``<= t`` and nothing else. That is not a convention to be respected by callers, it is
the reason the classes are stateful and fed one step at a time: there is no array of
future returns in scope to accidentally index into.
``tests/test_risk.py::test_a_future_shock_cannot_change_any_earlier_scale`` injects a
shock and asserts no earlier scale moves.

Why volatility targeting is not free
------------------------------------
Rescaling the book is trading, and trading costs money. The engine applies the scale
through the ordinary sleeve-rebalance path so the turnover is charged by the same cost
model as everything else. A vol-targeting implementation that adjusts exposure without
paying for it manufactures Sharpe out of nothing;
``tests/test_risk.py::test_volatility_targeting_increases_turnover`` pins that the cost
is actually incurred.

Why volatility targeting is not an improvement
----------------------------------------------
Scaling exposure by trailing volatility changes the *path* of returns. It raises Sharpe
only if volatility is both predictable and informative about future returns. Applied to
a strategy with a negative edge it scales the losses. The pipeline reports before/after
rather than assuming a direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..validation.deflated_sharpe import TRADING_DAYS

#: Scale returned when no estimate is possible. Deliberately 1.0 (leave the book alone)
#: rather than 0.0 (flatten it): "I cannot measure this yet" must not silently become a
#: position decision.
NEUTRAL_SCALE = 1.0


@dataclass(frozen=True)
class VolTargetConfig:
    """Parameters for :class:`VolatilityTargeter`."""

    enabled: bool = False
    #: Annualised volatility the book is scaled towards, e.g. 0.10 for 10%.
    target_annual_vol: float = 0.10
    #: Trailing window, in sessions, used to estimate realised volatility.
    window: int = 63
    #: Minimum observations before any scaling is applied. Below this the scale is
    #: NEUTRAL_SCALE -- an unmeasurable volatility must not become a leverage decision.
    min_obs: int = 63
    #: Bounds on the leverage multiple. The upper bound matters most: a quiet patch can
    #: produce an arbitrarily small denominator, and an uncapped ratio would lever the
    #: book into the next volatility regime just in time to be hurt by it.
    max_leverage_multiple: float = 2.0
    min_leverage_multiple: float = 0.25

    @classmethod
    def from_mapping(cls, node: Mapping[str, Any] | None) -> "VolTargetConfig":
        node = dict(node or {})
        return cls(
            enabled=bool(node.get("enabled", False)),
            target_annual_vol=float(node.get("target_annual_vol", 0.10)),
            window=int(node.get("window", 63)),
            min_obs=int(node.get("min_obs", 63)),
            max_leverage_multiple=float(node.get("max_leverage_multiple", 2.0)),
            min_leverage_multiple=float(node.get("min_leverage_multiple", 0.25)),
        )

    def validate(self) -> None:
        if self.target_annual_vol <= 0:
            raise ValueError("target_annual_vol must be positive")
        if self.window < 2:
            raise ValueError("vol window must be at least 2 sessions")
        if self.min_obs < 2:
            raise ValueError("vol min_obs must be at least 2 sessions")
        if not (0 < self.min_leverage_multiple <= self.max_leverage_multiple):
            raise ValueError(
                "require 0 < min_leverage_multiple <= max_leverage_multiple "
                f"(got {self.min_leverage_multiple}, {self.max_leverage_multiple})"
            )

    @property
    def target_daily_vol(self) -> float:
        return self.target_annual_vol / math.sqrt(TRADING_DAYS)


@dataclass(frozen=True)
class DrawdownConfig:
    """Parameters for :class:`DrawdownController`."""

    enabled: bool = False
    #: De-risk once the drawdown is worse than this, as a positive fraction (0.10 = -10%).
    threshold: float = 0.10
    #: Exposure multiple while de-risked.
    derisk_to: float = 0.5
    #: Restore full exposure once the drawdown recovers to shallower than this. Must be
    #: strictly smaller than ``threshold``: with a single boundary the controller
    #: switches on and off on consecutive sessions around it, paying turnover each time
    #: for no change in risk. The gap is the hysteresis band.
    restore_threshold: float = 0.05

    @classmethod
    def from_mapping(cls, node: Mapping[str, Any] | None) -> "DrawdownConfig":
        node = dict(node or {})
        return cls(
            enabled=bool(node.get("enabled", False)),
            threshold=float(node.get("threshold", 0.10)),
            derisk_to=float(node.get("derisk_to", 0.5)),
            restore_threshold=float(node.get("restore_threshold", 0.05)),
        )

    def validate(self) -> None:
        if not (0.0 < self.threshold < 1.0):
            raise ValueError("drawdown threshold must be in (0, 1)")
        if not (0.0 <= self.derisk_to <= 1.0):
            raise ValueError("derisk_to must be in [0, 1]")
        if not (0.0 <= self.restore_threshold < self.threshold):
            raise ValueError(
                "require 0 <= restore_threshold < threshold so the controller has a "
                f"hysteresis band (got {self.restore_threshold}, {self.threshold})"
            )


class VolatilityTargeter:
    """Scale exposure so trailing realised volatility approaches a target.

    Fed one realised return at a time via :meth:`update`; :meth:`scale` reports the
    multiple to apply *next*, using only what has been fed so far.
    """

    def __init__(self, cfg: VolTargetConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        self._returns: list[float] = []
        self.last_realized_vol: float = float("nan")

    def update(self, ret: float) -> None:
        """Record one realised return. Non-finite values are skipped, not zero-filled:
        a missing day is an absence of information, and booking it as a 0.0 return would
        understate volatility and lever the book up in response."""
        if np.isfinite(ret):
            self._returns.append(float(ret))

    def scale(self) -> float:
        if not self.cfg.enabled:
            return NEUTRAL_SCALE
        if len(self._returns) < self.cfg.min_obs:
            self.last_realized_vol = float("nan")
            return NEUTRAL_SCALE
        window = np.asarray(self._returns[-self.cfg.window:], dtype=float)
        vol = float(window.std(ddof=1)) if window.size > 1 else float("nan")
        self.last_realized_vol = vol
        # A degenerate volatility gives an unbounded ratio, so return the neutral scale
        # rather than the cap: the cap is a risk decision, and this is a measurement
        # failure.
        #
        # The test must be RELATIVE. A constant series does not produce std == 0.0 but
        # something around 1e-18, so an absolute `vol > 0` check passes and the ratio
        # pins to max_leverage_multiple -- levering the book to its limit precisely
        # because nothing has been happening. This is the same defect the audit found in
        # `sharpe_ratio`, which returned 7.28e16 on a constant series, and it is fixed
        # the same way.
        magnitude = max(
            abs(float(window.mean())), float(np.abs(window).max()), 1e-300
        )
        if not (np.isfinite(vol) and vol > 1e-12 * magnitude):
            return NEUTRAL_SCALE
        raw = self.cfg.target_daily_vol / vol
        return float(
            np.clip(raw, self.cfg.min_leverage_multiple, self.cfg.max_leverage_multiple)
        )


class DrawdownController:
    """De-risk after a drawdown, restore after a recovery, with hysteresis.

    Unlike :class:`VolatilityTargeter` -- which is fed the *unlevered* return series so
    its estimate is not contaminated by its own scaling -- this is fed the **actual**
    net return of the book. That feedback is the point: the controller exists to react
    to real losses, and de-risking is supposed to shrink subsequent ones.
    """

    def __init__(self, cfg: DrawdownConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        self._equity = 1.0
        self._peak = 1.0
        self._derisked = False
        self.last_drawdown: float = 0.0

    def update(self, ret: float) -> None:
        if not np.isfinite(ret):
            return
        self._equity *= 1.0 + float(ret)
        self._peak = max(self._peak, self._equity)
        self.last_drawdown = (
            self._equity / self._peak - 1.0 if self._peak > 0 else float("nan")
        )
        dd = self.last_drawdown
        if not np.isfinite(dd):
            return
        if self._derisked:
            if dd >= -self.cfg.restore_threshold:
                self._derisked = False
        elif dd <= -self.cfg.threshold:
            self._derisked = True

    def scale(self) -> float:
        if not self.cfg.enabled:
            return NEUTRAL_SCALE
        return self.cfg.derisk_to if self._derisked else NEUTRAL_SCALE

    @property
    def derisked(self) -> bool:
        return self._derisked


@dataclass
class RiskOverlay:
    """Both overlays, combined multiplicatively.

    The two answer different questions -- "is the book too volatile?" and "is the book
    losing money?" -- and a book can be both. Multiplying lets each bind independently;
    taking the min would let a mild vol signal mask an active drawdown stop.
    """

    vol: VolatilityTargeter
    drawdown: DrawdownController
    #: Per-session diagnostics, appended in lockstep with the engine's daily rows.
    history: list[dict] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg) -> "RiskOverlay":
        node = cfg.get("backtest.risk", {}) or {}
        return cls(
            vol=VolatilityTargeter(VolTargetConfig.from_mapping(node.get("volatility_target"))),
            drawdown=DrawdownController(DrawdownConfig.from_mapping(node.get("drawdown_control"))),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.vol.cfg.enabled or self.drawdown.cfg.enabled)

    def scale(self) -> float:
        return float(self.vol.scale() * self.drawdown.scale())

    def record(self, scale: float) -> None:
        self.history.append(
            {
                "risk_scale": float(scale),
                "realized_vol": float(self.vol.last_realized_vol),
                "drawdown": float(self.drawdown.last_drawdown),
                "derisked": bool(self.drawdown.derisked),
            }
        )

    def update(self, *, unit_return: float, net_return: float) -> None:
        """Feed one session's outcome.

        ``unit_return`` is the book at scale 1.0 and drives the volatility estimate, so
        the estimator does not chase its own leverage. ``net_return`` is what the book
        actually earned and drives the drawdown state, because that is the equity curve
        the control exists to protect.
        """
        self.vol.update(unit_return)
        self.drawdown.update(net_return)

    def summary(self) -> dict:
        if not self.history:
            return {"enabled": self.enabled, "n_sessions": 0}
        scales = np.asarray([h["risk_scale"] for h in self.history], dtype=float)
        derisked = np.asarray([h["derisked"] for h in self.history], dtype=bool)
        return {
            "enabled": self.enabled,
            "volatility_target_enabled": self.vol.cfg.enabled,
            "drawdown_control_enabled": self.drawdown.cfg.enabled,
            "n_sessions": int(scales.size),
            "mean_risk_scale": float(np.nanmean(scales)),
            "min_risk_scale": float(np.nanmin(scales)),
            "max_risk_scale": float(np.nanmax(scales)),
            "sessions_derisked": int(derisked.sum()),
            "fraction_derisked": float(derisked.mean()),
            "target_annual_vol": self.vol.cfg.target_annual_vol if self.vol.cfg.enabled else None,
        }


def max_drawdown_from_returns(returns: np.ndarray) -> float:
    """Worst peak-to-trough decline of the compounded series, as a negative fraction."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = equity / peak - 1.0
    return float(np.nanmin(dd))


def time_under_water(returns: np.ndarray) -> int:
    """Longest run of sessions spent below a previous equity peak."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    under = equity < peak
    longest = run = 0
    for flag in under:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return int(longest)
