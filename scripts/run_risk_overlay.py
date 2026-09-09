#!/usr/bin/env python
"""Risk overlays: does volatility targeting or drawdown control improve anything?

Runs each factor under four configurations -- no overlay, volatility targeting alone,
drawdown control alone, and both -- and reports net Sharpe, drawdown, turnover and cost
side by side.

What this is NOT
----------------
Volatility targeting does not mechanically improve Sharpe. It rescales exposure over
time, which changes the *path* of returns; it raises Sharpe only if volatility is both
predictable and informative about subsequent returns. Applied to a strategy with a
negative edge it scales the losses and nothing else. Drawdown control is stronger still:
it can only ever reduce participation, so on a strategy that eventually recovers it
locks in losses.

Both are therefore evaluated as hypotheses to be refuted, not features to be shipped.
Each configuration is a fresh look at the same data and registers as its own trial, so
the Deflated Sharpe of anything built on top is deflated against the real number of
looks. Running this script four ways and quoting the best would be exactly the search
the trial registry exists to price.
"""

from __future__ import annotations

import argparse
import sys

from _cli import bootstrap, say
from _pipeline import FIXED_COST_ONLY_NOTIONAL, load_pipeline, relative, write_json

from flowalpha.backtest.engine import run_backtest
from flowalpha.backtest.risk import (
    DrawdownConfig,
    DrawdownController,
    RiskOverlay,
    VolatilityTargeter,
    VolTargetConfig,
)
from flowalpha.config import new_run_dir
from flowalpha.validation.deflated_sharpe import deflated_sharpe_ratio

#: The configurations compared. Declared up front, as a fixed set, so the comparison is
#: not a search that stops when something looks good.
VARIANTS = ("none", "vol_target", "drawdown", "both")


def _overlay(cfg, variant: str) -> RiskOverlay:
    """Build an overlay for one variant from the config's parameters.

    The *parameters* come from config; only the enabled flags vary here. A variant that
    also retuned the window or the threshold would be a different hypothesis, and
    picking among those is the multiple-testing problem this file warns about.
    """
    node = cfg.get("backtest.risk", {}) or {}
    vol = VolTargetConfig.from_mapping(node.get("volatility_target"))
    dd = DrawdownConfig.from_mapping(node.get("drawdown_control"))
    vol = VolTargetConfig(**{**vol.__dict__, "enabled": variant in ("vol_target", "both")})
    dd = DrawdownConfig(**{**dd.__dict__, "enabled": variant in ("drawdown", "both")})
    return RiskOverlay(vol=VolatilityTargeter(vol), drawdown=DrawdownController(dd))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notional", type=float, default=FIXED_COST_ONLY_NOTIONAL,
                        help="book size in rupees (default: capacity-independent)")
    parser.add_argument("--no-register", action="store_true",
                        help="evaluate without incrementing the trial registry")
    args = parser.parse_args(argv)

    cfg, prov = bootstrap("run_risk_overlay")
    run_dir = new_run_dir(cfg, "risk_overlay")
    pipe = load_pipeline(cfg)

    node = cfg.get("backtest.risk", {}) or {}
    vt = VolTargetConfig.from_mapping(node.get("volatility_target"))
    dd = DrawdownConfig.from_mapping(node.get("drawdown_control"))
    say("")
    say(f"volatility target : {vt.target_annual_vol:.1%} annual, {vt.window}d window, "
        f"leverage in [{vt.min_leverage_multiple}, {vt.max_leverage_multiple}]")
    say(f"drawdown control  : de-risk to {dd.derisk_to:.0%} below -{dd.threshold:.0%}, "
        f"restore above -{dd.restore_threshold:.0%}")
    say(f"notional          : Rs {args.notional:,.0f}")
    say("")
    say("NOTE: neither overlay mechanically improves Sharpe. Vol targeting reshapes the")
    say("      return path; drawdown control can only reduce participation. Both are")
    say("      tested here as hypotheses, and each variant registers as its own trial.")

    rows = []
    for name in sorted(pipe.panels):
        panel = pipe.panels[name]
        say("")
        say(f"{name}")
        say(f"  {'variant':<12} {'net SR':>8} {'maxDD':>9} {'TUW':>6} {'calmar':>8} "
            f"{'turnover':>9} {'cost bps':>9} {'DSR':>6}")
        for variant in VARIANTS:
            res = run_backtest(
                panel, pipe.prices, cfg, notional=args.notional,
                risk=_overlay(cfg, variant),
            )
            if not args.no_register:
                pipe.registry.register(
                    f"{name}|risk={variant}", kind="risk_overlay",
                    detail={"notional": args.notional, "variant": variant},
                )
            dsr = deflated_sharpe_ratio(
                res.net_returns, max(1, pipe.registry.n_trials),
                benchmark_sr_annual=float(
                    cfg.get("validation.deflated_sharpe.benchmark_sr", 0.0)
                ),
            )
            prob = float(dsr.dsr)
            say(f"  {variant:<12} {res.net_sharpe:>8.3f} {res.max_drawdown:>9.4f} "
                f"{res.time_under_water_days:>6} {res.calmar:>8.3f} "
                f"{res.annual_turnover:>9.1f} {res.total_cost_bps:>9.1f} {prob:>6.3f}")
            rows.append(
                {
                    "factor": name, "variant": variant,
                    "net_sharpe": res.net_sharpe, "gross_sharpe": res.gross_sharpe,
                    "max_drawdown": res.max_drawdown,
                    "time_under_water_days": res.time_under_water_days,
                    "calmar": res.calmar, "annual_turnover": res.annual_turnover,
                    "cost_bps_per_year": res.total_cost_bps,
                    "deflated_sharpe_prob": prob,
                    "risk": res.risk_detail,
                }
            )

    if not args.no_register:
        pipe.registry.save()

    # Did any overlay beat its own no-overlay baseline? Reported as a count, per factor,
    # rather than by quoting the best cell in the table.
    base = {r["factor"]: r for r in rows if r["variant"] == "none"}
    improved = {
        v: sum(
            1 for r in rows
            if r["variant"] == v and r["net_sharpe"] > base[r["factor"]]["net_sharpe"]
        )
        for v in VARIANTS if v != "none"
    }
    shallower = {
        v: sum(
            1 for r in rows
            if r["variant"] == v and r["max_drawdown"] > base[r["factor"]]["max_drawdown"]
        )
        for v in VARIANTS if v != "none"
    }
    n = len(base)
    say("")
    say("VERDICT")
    for v in VARIANTS[1:]:
        say(f"  {v:<12} improved net Sharpe on {improved[v]}/{n} factors; "
            f"reduced max drawdown on {shallower[v]}/{n}")
    say("")
    say("  A reduced drawdown with an unchanged or worse Sharpe is not an improvement in")
    say("  the signal -- it is less participation. Read the two columns together.")
    say(f"trial registry: {pipe.registry.summary()}")

    payload = {
        "provenance_label": prov.label(),
        "notional": args.notional,
        "variants": list(VARIANTS),
        "volatility_target": vt.__dict__,
        "drawdown_control": dd.__dict__,
        "rows": rows,
        "n_improved_net_sharpe": improved,
        "n_reduced_max_drawdown": shallower,
        "n_factors": n,
        "trial_registry": {
            "n_trials": pipe.registry.n_trials, "n_distinct": pipe.registry.n_distinct,
        },
        "caveats": list(prov.caveats),
    }
    write_json(payload, run_dir / "risk_overlay.json")
    write_json(payload, cfg.path("results") / "risk_overlay.json")
    say(f"wrote {relative(run_dir / 'risk_overlay.json')} and results/risk_overlay.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
