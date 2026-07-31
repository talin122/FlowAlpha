#!/usr/bin/env python
"""Baseline: evaluate every factor unconditionally.

Produces the IC table (with Newey-West t-statistics, reported RAW), the gross and net
backtest for each factor, the Deflated Sharpe against the cumulative trial registry, and
PBO across the factor set.

This is the number every conditional result has to beat. If regime conditioning does not
improve on this after costs, that is the finding.
"""

from __future__ import annotations

import argparse
import sys

from _cli import bootstrap, say
from _pipeline import (
    capacity_note,
    evaluate_library_two_scales,
    load_pipeline,
    pbo_over_factors,
    relative,
    write_json,
)
from flowalpha.backtest.costs import CostModel, describe
from flowalpha.config import new_run_dir
from flowalpha.signals.factor_card import cards_to_frame
from flowalpha.validation.ic import ic_table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notional", type=float, default=1e9,
                        help="book size in rupees for impact costs (default Rs 100 crore)")
    parser.add_argument("--no-register", action="store_true",
                        help="evaluate without incrementing the trial registry")
    args = parser.parse_args(argv)

    cfg, prov = bootstrap("run_baseline")
    run_dir = new_run_dir(cfg, "baseline")
    say(f"run directory: {relative(run_dir)}")

    pipe = load_pipeline(cfg)
    say("")
    say("cost model:")
    for line in describe(CostModel.from_config(cfg)):
        say(line)
    say("")

    cards, results, fixed_cards, fixed_results = evaluate_library_two_scales(
        pipe, args.notional
    )
    if not args.no_register:
        pipe.registry.save()
    say("")
    say("fixed-costs-only net Sharpe (capacity-independent: STT + stamp duty + brokerage,")
    say("  no market impact -- the bar a signal must clear before size matters):")
    for card in fixed_cards:
        bt = card.backtest or {}
        say(f"  {card.name:<22} gross SR={bt.get('gross_sharpe', float('nan')):+.2f} "
            f"net SR={bt.get('net_sharpe', float('nan')):+.2f} "
            f"cost={bt.get('total_cost_bps', float('nan')):.0f}bps/yr")
    say("")
    say(f"trial registry: {pipe.registry.summary()}")

    table = ic_table(pipe.panels, pipe.fwd, pipe.horizons)
    say("")
    say("IC table (t-stats are Newey-West adjusted and NOT deflated):")
    for row in table.iter_rows(named=True):
        say(f"  {row['factor']:<22} h={row['horizon']:<3} "
            f"mean_ic={row['mean_ic']:+.5f} IR={row['ic_ir']:+.3f} "
            f"t={row['t_stat_newey_west']:+.2f} n={row['n_dates']}")

    headline_h = int(cfg.get("signals.composite.ic_horizon", pipe.horizons[0]))
    summary_table = cards_to_frame(cards, headline_h)

    pbo = pbo_over_factors(results, cfg)
    if pbo:
        if "error" in pbo:
            say(f"PBO not computed: {pbo['error']}")
        else:
            say("")
            say(f"PBO across {pbo['n_configs']} factors: {pbo['pbo']:.3f} "
                f"({pbo['n_combinations']} partitions of {pbo['n_partitions']})")
            say("  (probability the in-sample-best factor lands below median out-of-sample)")

    capacity = {
        card.name: capacity_note(results[card.name], pipe.prices, cfg)
        for card in cards if card.name in results
    }

    payload = {
        "provenance_label": prov.label(),
        "fixed_cost_only_summary": cards_to_frame(fixed_cards, headline_h).to_dicts(),
        "as_of": cfg["dates"]["end"],
        "n_sessions": pipe.ctx.n_sessions,
        "n_symbols": pipe.ctx.n_symbols,
        "shares_available": pipe.shares_available,
        "skipped_factors": pipe.skipped,
        "notional": args.notional,
        "trial_registry": {
            "n_trials": pipe.registry.n_trials,
            "n_distinct": pipe.registry.n_distinct,
        },
        "ic_table": table.to_dicts(),
        "summary_table": summary_table.to_dicts(),
        "cards": [c.to_dict() for c in cards],
        "pbo": pbo,
        "capacity": capacity,
        "caveats": list(prov.caveats),
    }
    write_json(payload, run_dir / "baseline.json")
    write_json(payload, cfg.path("results") / "baseline.json")
    table.write_csv(run_dir / "ic_table.csv")
    summary_table.write_csv(run_dir / "factor_summary.csv")

    for name, result in results.items():
        if result.n_days:
            result.daily.write_parquet(run_dir / f"backtest_{name}.parquet")

    say("")
    say(f"wrote {relative(run_dir / 'baseline.json')} and results/baseline.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
