"""Portfolio construction: quintile long/short with real-world constraints.

All constraints come from ``signals.construction``. They are **not** applied as a
sequence of independent passes, because they interact: capping one name pushes weight into
other names and other sectors, and a later rescale can undo an earlier cap. Instead each
leg of the book is projected onto its whole constraint set at once
(:func:`_project_side`), and the result is then **verified** and reported.

1. **Quintile long/short** (or long-only) selection, equal weight within each leg.
2. **Per-name caps** -- ``max_weight``, tightened by ``adv_participation_cap`` against
   trailing ADV wherever an ADV estimate exists, so the model portfolio cannot hold a
   position it could not build. Names with no ADV are counted and reported, never assigned
   an invented one.
3. **Projection** of each leg onto {per-name caps, ``sector_cap`` share cap,
   ``gross_leverage`` target} by nested water-filling.
4. **rebalance_band** -- names whose target moves less than this from yesterday are left
   alone, so the book does not churn on noise.
5. **Verification** -- every constraint is re-checked on the final weights and any
   violation is reported in ``warnings`` and in ``constraints_applied["satisfied"]``.

The trade list carries a per-trade cost in basis points, which is the number a human
actually needs in order to decide whether a trade is worth doing.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import polars as pl

from ..backtest.costs import CostModel
from ..config import Config
from .state import Holdings

TRADE_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.Utf8,
    "prev_weight": pl.Float64,
    "target_weight": pl.Float64,
    "delta_weight": pl.Float64,
    "side": pl.Utf8,
    "adv": pl.Float64,
    "participation": pl.Float64,
    "cost_bps": pl.Float64,
}


@dataclass
class Portfolio:
    """A model portfolio and the trades needed to reach it."""

    as_of: _dt.date
    #: (symbol, weight, sector, adv)
    holdings: pl.DataFrame
    trades: pl.DataFrame
    gross: float
    net: float
    n_long: int
    n_short: int
    turnover: float
    total_cost_bps: float
    notional: float
    constraints_applied: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_holdings(self, source: str = "") -> Holdings:
        return Holdings.from_frame(
            self.holdings.select("symbol", "weight"), as_of=self.as_of, source=source
        )

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "gross": self.gross, "net": self.net,
            "n_long": self.n_long, "n_short": self.n_short,
            "n_positions": self.holdings.height,
            "turnover": self.turnover,
            "total_cost_bps": self.total_cost_bps,
            "notional": self.notional,
            "constraints": dict(self.constraints_applied),
            "warnings": list(self.warnings),
        }


def _cap_and_redistribute(weights: np.ndarray, cap: float, *, max_iter: int = 50) -> np.ndarray:
    """Clip |weight| to ``cap`` and push the excess onto uncapped names.

    Preserves gross exposure and the long/short split. A plain clip would quietly
    shrink the book -- a 3% cap on a 50-name equal-weight leg would take gross from
    1.0 to 0.75 without saying so.
    """
    out = weights.astype(float).copy()
    if cap <= 0:
        return out
    for _ in range(max_iter):
        for sign in (1.0, -1.0):
            side = np.sign(out) == sign
            if not side.any():
                continue
            over = side & (np.abs(out) > cap + 1e-15)
            if not over.any():
                continue
            excess = float(np.abs(out[over]).sum() - cap * over.sum())
            out[over] = sign * cap
            room = side & (np.abs(out) < cap - 1e-15)
            if not room.any():
                break  # nowhere left to put it; gross falls, and the caller is told
            headroom = cap - np.abs(out[room])
            share = headroom / headroom.sum()
            out[room] = out[room] + sign * np.minimum(excess * share, headroom)
        if np.all(np.abs(out) <= cap + 1e-12):
            break
    return out


def _sector_shares(weights: np.ndarray, sectors: np.ndarray) -> dict[str, float]:
    """Each sector's share of total gross exposure."""
    gross = float(np.abs(weights).sum())
    if gross <= 0:
        return {}
    return {
        str(s): float(np.abs(weights[sectors == s]).sum()) / gross
        for s in np.unique(sectors)
    }


def _water_fill(magnitudes: np.ndarray, caps: np.ndarray, target: float) -> np.ndarray:
    """Distribute ``target`` across items proportional to ``magnitudes``, each capped.

    Items that hit their cap are fixed and removed from the pool, and the remainder is
    redistributed among those still below cap -- repeating until the target is placed or
    every item is capped. Exact and terminating (one pass per item at worst), unlike a
    repeated proportional rescale.

    When the caps cannot absorb ``target`` the result sums to the total capacity, which is
    the most that can be held; the caller detects the shortfall and reports it.
    """
    mag = np.asarray(magnitudes, dtype=float).copy()
    caps = np.asarray(caps, dtype=float)
    out = np.zeros_like(mag)
    free = mag > 0
    remaining = float(target)
    for _ in range(mag.size + 1):
        if not free.any() or remaining <= 1e-15:
            break
        pool = float(mag[free].sum())
        if pool <= 0:
            break
        scale = remaining / pool
        proposed = mag * scale
        over = free & (proposed > caps + 1e-15)
        if not over.any():
            out[free] = proposed[free]
            remaining = 0.0
            break
        out[over] = caps[over]
        remaining -= float(caps[over].sum())
        free = free & ~over
    return out


def _project_side(
    weights: np.ndarray,
    sectors: np.ndarray,
    name_caps: np.ndarray,
    *,
    sector_cap: float,
    target_gross: float,
) -> tuple[np.ndarray, bool, float]:
    """Project ONE side of the book onto its constraint set.

    Enforces, simultaneously and exactly:

    * ``|w_i| <= name_caps[i]``  (per-name cap: max_weight, tightened by ADV where known)
    * each sector's gross <= ``sector_cap * target_gross``
    * total gross == ``target_gross``

    Returns ``(weights, feasible, capacity)``.

    **Why per side.** An earlier version capped sectors across the whole book at once,
    scaling every name in a sector by the same factor regardless of which leg it sat on.
    Capping a long-heavy sector then removed more long exposure than short, and a
    nominally dollar-neutral portfolio silently acquired a directional bet -- measured at
    -13% net on a 4-long/4-short example. Projecting each leg onto its own target gross
    makes dollar-neutrality structural. It also preserves the sector cap for the combined
    book, since a weighted average of two per-side shares each below the cap is itself
    below the cap.

    **Why nested water-filling rather than iterating the two caps.** The per-name and
    per-sector caps interact: redistributing away from a name at its cap pushes weight into
    other sectors, which can re-breach the sector cap, which pushes it back. Iterating that
    cycles. Allocating to sectors first (each sector's capacity being
    ``min(n_names * name_cap, sector_cap * target)``) and then within each sector to names
    is a single exact pass, because the second step can never overflow what the first
    allotted.
    """
    mag = np.abs(np.asarray(weights, dtype=float))
    signs = np.sign(np.asarray(weights, dtype=float))
    if target_gross <= 0 or not (mag > 0).any():
        return np.zeros_like(mag), True, 0.0

    selected = mag > 0
    unique = list(dict.fromkeys(sectors[selected].tolist()))
    # A share-of-gross cap presupposes at least two buckets to spread across. With one
    # sector its share is definitionally 1.0, so the cap is inert -- NOT infeasible.
    # Treating it as binding would cap the leg at `sector_cap * target` and silently run
    # the whole book at a fraction of its intended size.
    group_cap = (
        sector_cap * target_gross if (0.0 < sector_cap < 1.0 and len(unique) > 1) else np.inf
    )

    sector_mag, sector_capacity = [], []
    for s in unique:
        member = selected & (sectors == s)
        sector_mag.append(float(mag[member].sum()))
        # A sector can hold at most the sum of its members' own caps, and at most its
        # share cap.
        sector_capacity.append(min(float(name_caps[member].sum()), group_cap))
    allotment = _water_fill(np.array(sector_mag), np.array(sector_capacity), target_gross)

    out = np.zeros_like(mag)
    for s, allot in zip(unique, allotment):
        member = selected & (sectors == s)
        out[member] = _water_fill(mag[member], name_caps[member], float(allot))

    capacity = float(np.array(sector_capacity).sum())
    feasible = capacity >= target_gross - 1e-9
    return signs * out, feasible, capacity


def build_portfolio(
    signal: pl.DataFrame,
    prices: pl.DataFrame,
    cfg: Config,
    *,
    as_of_date: _dt.date,
    prior: Holdings | None = None,
    sectors: Mapping[str, str] | None = None,
    adv: pl.DataFrame | None = None,
    notional: float = 1_000_000_000.0,
    cost_model: CostModel | None = None,
) -> Portfolio:
    """Build the model portfolio for one date and the trade list to reach it.

    ``notional`` defaults to Rs 100 crore so that ADV participation and impact costs are
    meaningful out of the box; a portfolio report with a notional of 1.0 would show every
    constraint as non-binding.
    """
    ccfg = cfg["signals"]["construction"]
    top_n = int(ccfg.get("top_n", 50))
    max_weight = float(ccfg.get("max_weight", 1.0))
    sector_cap = float(ccfg.get("sector_cap", 1.0))
    adv_cap = float(ccfg.get("adv_participation_cap", 1.0))
    band = float(ccfg.get("rebalance_band", 0.0))
    leverage = float(ccfg.get("gross_leverage", 1.0))
    long_only = bool(ccfg.get("long_only", False))
    model = cost_model or CostModel.from_config(cfg)
    prior = prior or Holdings.empty()
    warnings: list[str] = []
    applied: dict = {}

    today = signal.filter(pl.col("date") == as_of_date).filter(pl.col("value").is_not_null())
    if today.is_empty():
        return _empty_portfolio(as_of_date, notional, ["no signal available for this date"])

    ranked = today.sort("value", descending=True)
    # Selection is by rank order only: quintile long/short weights do not depend on
    # the magnitude of the signal, just its ordering.
    symbols = ranked["symbol"].to_list()
    n = len(symbols)

    # 1. selection and base magnitudes
    k = min(top_n, n // 2 if not long_only else n)
    if k < 1:
        return _empty_portfolio(as_of_date, notional, [f"only {n} names available; need at least 2"])
    weights = np.zeros(n)
    if long_only:
        weights[:k] = 1.0 / k
    else:
        weights[:k] = 0.5 / k
        weights[-k:] = -0.5 / k
    applied["selection"] = {"n_candidates": n, "top_n": k, "long_only": long_only}

    sector_arr = np.array([str((sectors or {}).get(s, "NA")) for s in symbols], dtype=object)
    sector_list = sector_arr.tolist()

    # 2. per-name caps: max_weight, tightened by the ADV participation limit where an ADV
    # estimate exists. Folding ADV in here rather than clipping afterwards means the
    # redistribution respects it instead of fighting it.
    adv_map: dict[str, float] = {}
    if adv is not None and not adv.is_empty():
        day_adv = adv.filter(pl.col("date") == as_of_date)
        adv_map = {
            str(s): float(v)
            for s, v in zip(day_adv["symbol"].to_list(), day_adv["adv"].to_list())
            if v is not None and np.isfinite(float(v))
        }
    name_caps = np.full(n, max_weight if max_weight > 0 else np.inf, dtype=float)
    n_adv_bound = n_missing_adv = 0
    for i, sym in enumerate(symbols):
        if abs(weights[i]) <= 0:
            continue
        a = adv_map.get(sym)
        if a is None or a <= 0:
            n_missing_adv += 1
            continue
        if adv_cap < 1.0 and notional > 0:
            limit = adv_cap * a / notional
            if limit < name_caps[i]:
                name_caps[i] = limit
                n_adv_bound += 1
    applied["adv_participation_cap"] = {
        "cap": adv_cap, "n_binding": n_adv_bound, "n_without_adv": n_missing_adv,
    }
    if n_missing_adv:
        warnings.append(
            f"{n_missing_adv} selected names have no ADV estimate, so the participation "
            "cap could not be applied to them (no ADV is invented)"
        )

    # 3. project each leg onto its own constraint set, so dollar-neutrality is structural
    # rather than something the sector cap can quietly undo.
    if long_only:
        legs = [(weights > 0, leverage)]
    else:
        legs = [(weights > 0, leverage / 2.0), (weights < 0, leverage / 2.0)]
    projected = np.zeros(n)
    leg_report = []
    all_feasible = True
    for mask, target in legs:
        side = np.where(mask, weights, 0.0)
        out, feasible, capacity = _project_side(
            side, sector_arr, name_caps, sector_cap=sector_cap, target_gross=target,
        )
        projected += out
        all_feasible = all_feasible and feasible
        leg_report.append(
            {"target_gross": target, "achieved_gross": float(np.abs(out).sum()),
             "capacity": capacity, "feasible": feasible}
        )
    weights = projected

    # If one leg could not reach its target while the other could, the book would carry an
    # unintended directional bet -- long 0.40 against short 0.50 is a 10% net market
    # position nobody asked for. Dollar-neutrality is the stronger commitment, so the
    # larger leg is trimmed to match the smaller. Scaling one leg down uniformly cannot
    # breach that leg's per-name caps (they are upper bounds) or its sector shares (which
    # are scale-invariant), so no constraint is disturbed.
    leg_balanced = True
    if not long_only:
        long_gross = float(weights[weights > 0].sum())
        short_gross = float(-weights[weights < 0].sum())
        if abs(long_gross - short_gross) > 1e-12 and min(long_gross, short_gross) > 0:
            smaller = min(long_gross, short_gross)
            if long_gross > smaller:
                weights[weights > 0] *= smaller / long_gross
            else:
                weights[weights < 0] *= smaller / short_gross
            leg_balanced = False
            warnings.append(
                f"legs reached different gross ({long_gross:.4f} long vs {short_gross:.4f} "
                f"short) because one leg's constraints could not absorb its target; the "
                f"larger leg was trimmed to {smaller:.4f} to keep the book dollar-neutral, "
                "so total gross is below the leverage target"
            )
    applied["projection"] = {
        "legs": leg_report, "feasible": all_feasible, "legs_balanced": leg_balanced,
    }
    applied["max_weight"] = {"cap": max_weight}
    applied["sector_cap"] = {"cap": sector_cap, "n_sectors": len(set(sector_list))}
    if len(set(sector_list)) <= 1 and 0.0 < sector_cap < 1.0:
        warnings.append(
            "sector_cap is inert: every name carries the same sector label "
            f"({sector_list[0] if sector_list else 'none'}), so there is nothing to cap"
        )
    if not all_feasible:
        warnings.append(
            f"constraints are jointly infeasible at gross {leverage:.2f}: max_weight "
            f"{max_weight:.2%} and sector_cap {sector_cap:.0%} (and any ADV limits) cannot "
            "absorb the target. Gross is left BELOW target rather than breaching a declared "
            "limit; loosen a cap or widen top_n."
        )

    # 4. rebalance band, against yesterday's actual book. Applied last, because its whole
    # purpose is to deviate from the freshly-computed target in order to avoid churn.
    prev = np.array([prior.weights.get(s, 0.0) for s in symbols], dtype=float)
    n_held = 0
    if band > 0:
        small = np.abs(weights - prev) < band
        n_held = int(small.sum())
        weights = np.where(small, prev, weights)
    applied["rebalance_band"] = {"band": band, "n_left_alone": n_held}

    # 5. verify every declared constraint on the FINAL weights and report the result.
    # Nothing here silently claims a constraint holds: the band step above can move a
    # weight back to yesterday's value, and yesterday's book was built against yesterday's
    # caps and ADV.
    final_gross = float(np.abs(weights).sum())
    max_abs = float(np.abs(weights).max()) if n else 0.0
    shares = _sector_shares(weights, sector_arr)
    worst_share = max(shares.values()) if shares else 0.0
    satisfied = {
        "max_weight": bool(max_abs <= max_weight + 1e-9),
        "sector_cap": bool(sector_cap >= 1.0 or len(set(sector_list)) <= 1
                           or worst_share <= sector_cap + 1e-9),
        "gross_leverage": bool(
            abs(final_gross - leverage) <= 1e-6 if (all_feasible and leg_balanced) else False
        ),
        "dollar_neutral": bool(long_only or abs(float(weights.sum())) <= 1e-6),
    }
    applied["satisfied"] = satisfied
    applied["gross_leverage"] = {
        "target": leverage, "final_gross": final_gross,
        "worst_sector_share": worst_share, "max_abs_weight": max_abs,
    }
    for name, ok in satisfied.items():
        if not ok and name == "gross_leverage" and not (all_feasible and leg_balanced):
            continue  # already explained by the infeasibility / leg-trim warning
        if not ok:
            warnings.append(
                f"constraint {name} is NOT satisfied on the final book "
                f"(gross {final_gross:.4f}, max |w| {max_abs:.4f}, worst sector share "
                f"{worst_share:.3f}, net {float(weights.sum()):+.4f})"
            )

    keep = np.abs(weights) > 1e-12
    holdings = pl.DataFrame(
        {
            "symbol": [s for s, k_ in zip(symbols, keep) if k_],
            "weight": weights[keep],
            "sector": [s for s, k_ in zip(sector_list, keep) if k_],
            "adv": [adv_map.get(s, float("nan")) for s, k_ in zip(symbols, keep) if k_],
        },
        schema={"symbol": pl.Utf8, "weight": pl.Float64, "sector": pl.Utf8, "adv": pl.Float64},
    ).sort("weight", descending=True)

    trades = _build_trades(symbols, prev, weights, adv_map, model, notional, prior)
    turnover = float(trades["delta_weight"].abs().sum()) if not trades.is_empty() else 0.0
    total_cost = (
        float((trades["delta_weight"].abs() * trades["cost_bps"]).sum())
        if not trades.is_empty() else 0.0
    )

    return Portfolio(
        as_of=as_of_date,
        holdings=holdings,
        trades=trades,
        gross=float(np.abs(weights).sum()),
        net=float(weights.sum()),
        n_long=int((weights > 1e-12).sum()),
        n_short=int((weights < -1e-12).sum()),
        turnover=turnover,
        total_cost_bps=total_cost,
        notional=notional,
        constraints_applied=applied,
        warnings=warnings,
    )


def _build_trades(
    symbols: list[str],
    prev: np.ndarray,
    target: np.ndarray,
    adv_map: dict[str, float],
    model: CostModel,
    notional: float,
    prior: Holdings,
) -> pl.DataFrame:
    """Trade list including exits of names that have left the selection.

    Names in yesterday's book but absent from today's candidate set must be sold to
    zero. Omitting them would understate turnover and leave the state file describing a
    book that was never actually held.
    """
    rows = []
    seen = set()
    for i, sym in enumerate(symbols):
        seen.add(sym)
        delta = float(target[i] - prev[i])
        if abs(delta) <= 1e-12:
            continue
        rows.append(_trade_row(sym, float(prev[i]), float(target[i]), delta, adv_map, model, notional))
    for sym, weight in prior.weights.items():
        if sym in seen or abs(weight) <= 1e-12:
            continue
        rows.append(_trade_row(sym, float(weight), 0.0, -float(weight), adv_map, model, notional))
    if not rows:
        return pl.DataFrame(schema=TRADE_SCHEMA)
    return pl.DataFrame(rows, schema=TRADE_SCHEMA).sort("delta_weight", descending=True)


def _trade_row(
    symbol: str,
    prev_w: float,
    target_w: float,
    delta: float,
    adv_map: dict[str, float],
    model: CostModel,
    notional: float,
) -> dict:
    value = abs(delta) * notional
    a = adv_map.get(symbol)
    a = a if (a is not None and np.isfinite(a) and a > 0) else None
    buy = value if delta > 0 else 0.0
    sell = value if delta < 0 else 0.0
    fixed_bps = 1e4 * model.fixed_cost(buy, sell) / value if value > 0 else 0.0
    impact_bps = model.impact_bps(value, a)
    return {
        "symbol": symbol,
        "prev_weight": prev_w,
        "target_weight": target_w,
        "delta_weight": delta,
        "side": "BUY" if delta > 0 else "SELL",
        "adv": float(a) if a is not None else float("nan"),
        "participation": (value / a) if a else float("nan"),
        "cost_bps": fixed_bps + impact_bps,
    }


def _empty_portfolio(as_of: _dt.date, notional: float, warnings: list[str]) -> Portfolio:
    return Portfolio(
        as_of=as_of,
        holdings=pl.DataFrame(
            schema={"symbol": pl.Utf8, "weight": pl.Float64, "sector": pl.Utf8, "adv": pl.Float64}
        ),
        trades=pl.DataFrame(schema=TRADE_SCHEMA),
        gross=0.0, net=0.0, n_long=0, n_short=0, turnover=0.0,
        total_cost_bps=0.0, notional=notional, warnings=warnings,
    )
