#!/usr/bin/env python
"""Composite: IC/IR-weighted blend of the factor library.

Weights use only IC observations whose labels had already realised as of the weighting
date. The comparison against an equal-weight composite is printed alongside, because
equal weight cannot be overfit and is therefore the bar an IC-weighted scheme has to
clear to justify its extra freedom.
"""

from __future__ import annotations

import argparse
import sys

from _cli import bootstrap, say
from _pipeline import (
    FIXED_COST_ONLY_NOTIONAL,
    capacity_note,
    evaluate_panel,
    load_pipeline,
    relative,
    write_json,
)
from flowalpha.config import new_run_dir
from flowalpha.signals.composite import build_composite, compute_weights, weight_history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notional", type=float, default=1e9)
    args = parser.parse_args(argv)

    cfg, prov = bootstrap("run_composite")
    run_dir = new_run_dir(cfg, "composite")
    pipe = load_pipeline(cfg)

    ccfg = cfg["signals"]["composite"]
    say("")
    say(f"composite scheme={ccfg['scheme']} ic_window={ccfg['ic_window']} "
        f"ic_horizon={ccfg['ic_horizon']} min_coverage={ccfg['min_coverage']} "
        f"drop_negative_ic={ccfg['drop_negative_ic']}")
    say("weights use only IC observations whose labels had realised by the weighting date")

    payload: dict = {
        "provenance_label": prov.label(),
        "config": dict(ccfg),
        "variants": {},
    }

    for scheme in (str(ccfg["scheme"]), "equal"):
        scheme_cfg = cfg.with_overrides(
            signals={**cfg["signals"], "composite": {**ccfg, "scheme": scheme}}
        )
        weights = compute_weights(pipe.panels, pipe.fwd, scheme_cfg, sessions=pipe.sessions)
        signal = build_composite(pipe.panels, weights, scheme_cfg)
        label = f"composite_{scheme}"
        say("")
        say(f"--- {label}: {weights.weights['date'].n_unique() if not weights.weights.is_empty() else 0} "
            f"weighted dates, {signal.height} signal rows")
        if weights.n_dropped_negative:
            say(f"    {weights.n_dropped_negative} factor-dates dropped for non-positive trailing IC")
        if signal.is_empty():
            say("    composite is empty; nothing to evaluate")
            payload["variants"][label] = {"empty": True, "weights": weights.to_dict()}
            continue

        card, result = evaluate_panel(
            pipe, label, signal,
            hypothesis=f"IC-weighted ({scheme}) blend of {len(pipe.panels)} factors",
            kind="composite", notional=args.notional,
        )
        for line in card.summary_lines():
            say(line)
        fixed_card, fixed_result = evaluate_panel(
            pipe, label, signal, kind="composite",
            notional=FIXED_COST_ONLY_NOTIONAL, register=False,
        )
        say(f"    fixed-costs-only net SR={fixed_card.backtest['net_sharpe']:+.2f} "
            f"(capacity-independent)")

        history = weight_history(weights)
        if not history.is_empty():
            history.write_csv(run_dir / f"{label}_weights.csv")
            tail = history.tail(1).to_dicts()[0]
            say("    latest weights: " + ", ".join(
                f"{k}={v:.3f}" for k, v in tail.items() if k != "date" and abs(float(v)) > 1e-9
            ))
        result.daily.write_parquet(run_dir / f"{label}_daily.parquet")
        payload["variants"][label] = {
            "weights": weights.to_dict(),
            "card": card.to_dict(),
            "fixed_cost_only": fixed_card.backtest,
            "capacity": capacity_note(result, pipe.prices, cfg),
            "latest_weights": history.tail(1).to_dicts()[0] if not history.is_empty() else {},
        }

    pipe.registry.save()
    say("")
    say(f"trial registry: {pipe.registry.summary()}")
    payload["trial_registry"] = {
        "n_trials": pipe.registry.n_trials, "n_distinct": pipe.registry.n_distinct,
    }
    payload["caveats"] = list(prov.caveats)

    write_json(payload, run_dir / "composite.json")
    write_json(payload, cfg.path("results") / "composite.json")
    say(f"wrote {relative(run_dir / 'composite.json')} and results/composite.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
