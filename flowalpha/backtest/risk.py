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


#: Targeting modes. See :class:`VolTargetConfig`.
ABSOLUTE = "absolute"
RELATIVE = "relative"
TARGET_MODES = (ABSOLUTE, RELATIVE)


@dataclass(frozen=True)
class VolTargetConfig:
    """Parameters for :class:`VolatilityTargeter`.

    Choosing ``target_mode``
    -----------------------
    ``absolute`` scales towards a fixed annualised volatility. It is only a *risk
    control* if that number is at or below the strategy's own volatility; set above it,
    the overlay is permanent leverage wearing a risk-control label, and leverage
    multiplies drawdown roughly linearly.

    That is not hypothetical. This project shipped ``target_annual_vol: 0.10`` against
    factor books whose realised volatility is 4.0%-8.2%, so every one of them was levered
    1.22x-2.51x at all times -- two never de-levered in 1,900 sessions -- and the maximum
    drawdown grew on 7/7 factors in almost exact proportion to the mean leverage applied.

    ``relative`` targets the strategy's *own* long-run volatility instead of a constant:
    ``scale = long_run_vol / short_run_vol``. It is centred on 1.0 by construction, so it
    stabilises volatility rather than inflating it, and it isolates the question of
    whether vol timing helps from the separate question of whether leverage helps.
    """

    enabled: bool = False
    #: ``absolute`` (scale towards ``target_annual_vol``) or ``relative`` (scale towards
    #: the strategy's own trailing ``long_window`` volatility).
    target_mode: str = ABSOLUTE
    #: Annualised volatility targeted in ``absolute`` mode, e.g. 0.10 for 10%.
    target_annual_vol: float = 0.10
    #: Trailing window, in sessions, used to estimate current realised volatility.
    window: int = 63
    #: Baseline window for ``relative`` mode. Long enough that it is not chasing the same
    #: moves as ``window``, or the ratio is noise over noise and pinned near 1.0.
    long_window: int = 504
    #: Minimum observations before any scaling. Below this the scale is NEUTRAL_SCALE --
    #: an unmeasurable volatility must not become a leverage decision.
    min_obs: int = 63
    #: Bounds on the leverage multiple. The upper bound matters most: a quiet patch can
    #: produce an arbitrarily small denominator, and an uncapped ratio would lever the
    #: book into the next volatility regime just in time to be hurt by it.
    max_leverage_multiple: float = 2.0
    min_leverage_multiple: float = 0.25
    #: Clamp the scale at 1.0 so the overlay can only ever *reduce* exposure. A risk
    #: control that cannot add risk, and the only configuration whose worst case is
    #: bounded by the unlevered strategy.
    de_lever_only: bool = False

    @classmethod
    def from_mapping(cls, node: Mapping[str, Any] | None) -> "VolTargetConfig":
        node = dict(node or {})
        return cls(
            enabled=bool(node.get("enabled", False)),
            target_mode=str(node.get("target_mode", ABSOLUTE)),
            target_annual_vol=float(node.get("target_annual_vol", 0.10)),
            window=int(node.get("window", 63)),
            long_window=int(node.get("long_window", 504)),
            min_obs=int(node.get("min_obs", 63)),
            max_leverage_multiple=float(node.get("max_leverage_multiple", 2.0)),
            min_leverage_multiple=float(node.get("min_leverage_multiple", 0.25)),
            de_lever_only=bool(node.get("de_lever_only", False)),
        )

    def validate(self) -> None:
        if self.target_mode not in TARGET_MODES:
            raise ValueError(f"target_mode must be one of {TARGET_MODES}")
        if self.target_annual_vol <= 0:
            raise ValueError("target_annual_vol must be positive")
        if self.window < 2:
            raise ValueError("vol window must be at least 2 sessions")
        if self.min_obs < 2:
            raise ValueError("vol min_obs must be at least 2 sessions")
        if self.target_mode == RELATIVE and self.long_window <= self.window:
            raise ValueError(
                "relative mode needs long_window > window, otherwise the baseline "
                f"tracks the same moves it is measured against (got {self.long_window} "
                f"<= {self.window})"
            )
        if not (0 < self.min_leverage_multiple <= self.max_leverage_multiple):
            raise ValueError(
                "require 0 < min_leverage_multiple <= max_leverage_multiple "
                f"(got {self.min_leverage_multiple}, {self.max_leverage_multiple})"
            )

    @property
    def target_daily_vol(self) -> float:
        return self.target_annual_vol / math.sqrt(TRADING_DAYS)

    @property
    def effective_min_obs(self) -> int:
        """Observations needed before a scale can be formed.

        Relative mode also needs the long baseline, so requiring only ``min_obs`` would
        scale off a baseline built from a handful of points.
        """
        return max(self.min_obs, self.long_window) if self.target_mode == RELATIVE else self.min_obs


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
        #: The volatility being targeted at the last :meth:`scale` call -- a constant in
        #: absolute mode, the trailing long-run estimate in relative mode. Recorded so a
        #: reader can see whether the overlay was de-levering or levering, and why.
        self.last_target_vol: float = float("nan")

    def update(self, ret: float) -> None:
        """Record one realised return. Non-finite values are skipped, not zero-filled:
        a missing day is an absence of information, and booking it as a 0.0 return would
        understate volatility and lever the book up in response."""
        if np.isfinite(ret):
            self._returns.append(float(ret))

    def _std(self, n: int) -> float:
        w = np.asarray(self._returns[-n:], dtype=float)
        if w.size < 2:
            return float("nan")
        vol = float(w.std(ddof=1))
        # Relative degeneracy test, as in `sharpe_ratio`: a constant series gives an sd
        # around 1e-18 rather than 0.0, which an absolute `> 0` check would accept.
        magnitude = max(abs(float(w.mean())), float(np.abs(w).max()), 1e-300)
        return vol if vol > 1e-12 * magnitude else float("nan")

    def scale(self) -> float:
        if not self.cfg.enabled:
            return NEUTRAL_SCALE
        if len(self._returns) < self.cfg.effective_min_obs:
            self.last_realized_vol = float("nan")
            return NEUTRAL_SCALE
        vol = self._std(self.cfg.window)
        self.last_realized_vol = vol
        # A degenerate volatility gives an unbounded ratio, so return the neutral scale
        # rather than the cap: the cap is a risk decision, and this is a measurement
        # failure. `_std` already applies the relative degeneracy test.
        if not np.isfinite(vol):
            return NEUTRAL_SCALE

        if self.cfg.target_mode == RELATIVE:
            target = self._std(self.cfg.long_window)
            if not np.isfinite(target):
                return NEUTRAL_SCALE
        else:
            target = self.cfg.target_daily_vol
        self.last_target_vol = target

        raw = target / vol
        upper = min(self.cfg.max_leverage_multiple, 1.0) if self.cfg.de_lever_only \
            else self.cfg.max_leverage_multiple
        return float(np.clip(raw, self.cfg.min_leverage_multiple, upper))


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
                "target_vol": float(self.vol.last_target_vol),
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
        # The single most diagnostic number: time spent ABOVE 1.0 is time the "risk
        # control" was adding risk. An overlay that levers more often than it de-levers
        # is leverage, and its drawdown behaviour should be read as such.
        levered = float((scales > 1.0 + 1e-12).mean())
        return {
            "enabled": self.enabled,
            "volatility_target_enabled": self.vol.cfg.enabled,
            "drawdown_control_enabled": self.drawdown.cfg.enabled,
            "target_mode": self.vol.cfg.target_mode if self.vol.cfg.enabled else None,
            "de_lever_only": self.vol.cfg.de_lever_only if self.vol.cfg.enabled else None,
            "n_sessions": int(scales.size),
            "mean_risk_scale": float(np.nanmean(scales)),
            "min_risk_scale": float(np.nanmin(scales)),
            "max_risk_scale": float(np.nanmax(scales)),
            "fraction_levered_above_1x": levered,
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
