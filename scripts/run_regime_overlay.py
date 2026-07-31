#!/usr/bin/env python
"""Regime overlay: gate factors on/off by trailing conditional IC t-stat.

The factor-to-regime map is declared in ``config.yaml`` and is a stated hypothesis, not
the result of a search over pairs. Gates are point-in-time: the decision on session t
uses only IC observations whose labels had realised by t.

Reports the overlay composite against the ungated composite, gross and net.
"""

from __future__ import annotations

import argparse
import sys

from _cli import bootstrap, say
from _pipeline import (
    FIXED_COST_ONLY_NOTIONAL,
    evaluate_panel,
    load_pipeline,
    relative,
    write_json,
)
from flowalpha.conditioning.regimes import RegimeSet
from flowalpha.config import new_run_dir
from flowalpha.signals.composite import build_composite, compute_weights
from flowalpha.signals.regime_overlay import compute_gates, gate_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notional", type=float, default=1e9)
    args = parser.parse_args(argv)

    cfg, prov = bootstrap("run_regime_overlay")
    if not bool(cfg.get("signals.regime_overlay.enabled", True)):
        say("signals.regime_overlay.enabled is false; nothing to do")
        return 0

    run_dir = new_run_dir(cfg, "regime_overlay")
    pipe = load_pipeline(cfg)

    try:
        regimes = RegimeSet.from_store(pipe.store, cfg)
    except FileNotFoundError as exc:
        say(f"FATAL: {exc}")
        return 1

    overlay = compute_gates(pipe.panels, pipe.fwd, cfg, regimes, sessions=pipe.sessions)
    say("")
    for line in gate_summary(overlay):
        say(line)

    # Ungated composite, for the comparison.
    base_weights = compute_weights(pipe.panels, pipe.fwd, cfg, sessions=pipe.sessions)
    base_signal = build_composite(pipe.panels, base_weights, cfg)
    gated_weights = compute_weights(overlay.panels, pipe.fwd, cfg, sessions=pipe.sessions)
    gated_signal = build_composite(overlay.panels, gated_weights, cfg)

    payload = {
        "provenance_label": prov.label(),
        "overlay": overlay.to_dict(),
        "gate_decisions_sample": overlay.decisions.tail(20).to_dicts(),
        "variants": {},
    }

    for label, signal in (("composite_ungated", base_signal), ("composite_overlay", gated_signal)):
        say("")
        if signal.is_empty():
            say(f"--- {label}: empty signal")
            payload["variants"][label] = {"empty": True}
            continue
        card, result = evaluate_panel(
            pipe, label, signal, kind="overlay",
            hypothesis="regime-gated composite" if "overlay" in label else "ungated composite",
            notional=args.notional,
        )
        fixed_card, _ = evaluate_panel(pipe, label, signal, kind="overlay",
                                       notional=FIXED_COST_ONLY_NOTIONAL, register=False)
        say(f"--- {label}")
        for line in card.summary_lines():
            say(line)
        say(f"    fixed-costs-only net SR={fixed_card.backtest['net_sharpe']:+.2f}")
        result.daily.write_parquet(run_dir / f"{label}_daily.parquet")
        payload["variants"][label] = {
            "card": card.to_dict(), "fixed_cost_only": fixed_card.backtest,
        }

    ung = payload["variants"].get("composite_ungated", {}).get("fixed_cost_only") or {}
    ovl = payload["variants"].get("composite_overlay", {}).get("fixed_cost_only") or {}
    if ung and ovl:
        better = ovl.get("net_sharpe", float("nan")) > ung.get("net_sharpe", float("nan"))
        say("")
        say(f"VERDICT: regime overlay {'improves' if better else 'does NOT improve'} the "
            f"composite on fixed-cost-only net Sharpe "
            f"({ovl.get('net_sharpe'):+.2f} vs {ung.get('net_sharpe'):+.2f})")
        payload["overlay_improves_composite"] = bool(better)

    overlay.decisions.write_parquet(run_dir / "gate_decisions.parquet")
    pipe.registry.save()
    payload["trial_registry"] = {
        "n_trials": pipe.registry.n_trials, "n_distinct": pipe.registry.n_distinct,
    }
    payload["caveats"] = list(prov.caveats)
    write_json(payload, run_dir / "regime_overlay.json")
    write_json(payload, cfg.path("results") / "regime_overlay.json")
    say(f"trial registry: {pipe.registry.summary()}")
    say(f"wrote {relative(run_dir / 'regime_overlay.json')} and results/regime_overlay.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
