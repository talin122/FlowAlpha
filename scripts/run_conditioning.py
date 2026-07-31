#!/usr/bin/env python
"""Conditioning: does factor performance depend on institutional order flow?

This is the study. Three outputs:

1. A conditional IC table: every factor's IC within every regime bucket.
2. The **declared** experiments -- momentum on FII flow, reversal on retail extremes --
   with the direction stated in advance and a verdict of SUPPORTED / NOT SUPPORTED /
   REFUTED.
3. A conditional strategy per supported experiment: trade the factor only in its
   favourable regime, flat otherwise, backtested against the unconditional version
   **after costs**.

A conditional strategy that does not beat its unconditional counterpart after costs is
reported as not beating it. That is the correct output of this pipeline, not a failure of
it.
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
from flowalpha.conditioning.analysis import (
    conditional_ic_table,
    conditional_panel,
    declared_experiments,
    expand_experiments,
    regime_exposure,
    run_experiment,
)
from flowalpha.conditioning.regimes import RegimeSet
from flowalpha.config import new_run_dir
from flowalpha.factors.flow_features import REGIME_COLUMNS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notional", type=float, default=1e9)
    parser.add_argument("--horizon", type=int, default=None,
                        help="IC horizon for conditioning (default: signals.composite.ic_horizon)")
    args = parser.parse_args(argv)

    cfg, prov = bootstrap("run_conditioning")
    run_dir = new_run_dir(cfg, "conditioning")
    pipe = load_pipeline(cfg)

    try:
        regimes = RegimeSet.from_store(pipe.store, cfg)
    except FileNotFoundError as exc:
        say(f"FATAL: {exc}")
        say("Run scripts/build_flows.py to produce flow_features.parquet.")
        return 1
    if regimes.features.is_empty():
        say("FATAL: flow_features is empty; nothing to condition on")
        return 1

    horizon = int(args.horizon or cfg.get("signals.composite.ic_horizon", 5))
    say("")
    say("regime coverage:")
    for col, cov in regimes.summary().items():
        say(f"  {col:<18} defined={cov['n_defined']:<6} "
            f"last={cov.get('last_defined')} buckets={cov.get('buckets')}")

    cond = conditional_ic_table(
        pipe.panels, pipe.fwd, horizon, regimes,
        [c for c in REGIME_COLUMNS if c in regimes.columns],
    )
    say("")
    say(f"conditional IC at horizon {horizon} (t-stats Newey-West, NOT deflated):")
    for row in cond.iter_rows(named=True):
        say(f"  {row['factor']:<22} {row['regime_column']:<16} {row['bucket']:<9} "
            f"mean_ic={row['mean_ic']:+.5f} t={row['t_stat_newey_west']:+.2f} n={row['n_dates']}")
    cond.write_csv(run_dir / "conditional_ic.csv")

    # --- the declared experiments -----------------------------------------
    experiments = declared_experiments(cfg, regimes)
    pairs = expand_experiments(experiments, sorted(pipe.panels))
    say("")
    say(f"declared experiments (from signals.regime_overlay.factor_regime_map): {len(pairs)}")
    say("  directions are stated in advance; a reversed result is a refutation, not a finding")

    gate = float(cfg.get("signals.regime_overlay.gate_t_threshold", 1.5))
    experiment_results = []
    strategy_results = []

    for exp, factor_name in pairs:
        concrete = type(exp)(
            factor=factor_name, regime_column=exp.regime_column,
            favourable=exp.favourable, unfavourable=exp.unfavourable,
            rationale=exp.rationale,
        )
        result = run_experiment(
            concrete, pipe.panels[factor_name], pipe.fwd, horizon, regimes,
            gate_t_threshold=gate,
        )
        experiment_results.append(result.to_dict())
        say("")
        say(f"  {factor_name} | {exp.regime_column}: "
            f"IC[{exp.favourable}]={result.ic_favourable:+.5f} (n={result.n_favourable}) "
            f"vs IC[{exp.unfavourable}]={result.ic_unfavourable:+.5f} (n={result.n_unfavourable})")
        say(f"    difference={result.difference:+.5f}  t(NW)={result.t_stat:+.2f}  "
            f"-> {result.verdict}")

        # The conditional strategy is evaluated regardless of the verdict, so the
        # after-cost comparison is available even when the IC difference is weak.
        active = [exp.favourable]
        cond_signal = conditional_panel(
            pipe.panels[factor_name], regimes, exp.regime_column, active
        )
        exposure = regime_exposure(regimes, exp.regime_column, active)
        label = f"conditional_{factor_name}_{exp.regime_column}_{exp.favourable}"

        card_c, res_c = evaluate_panel(
            pipe, label, cond_signal, kind="conditional",
            hypothesis=f"{factor_name} traded only when {exp.regime_column}={exp.favourable}",
            notional=args.notional,
        )
        card_u, res_u = evaluate_panel(
            pipe, factor_name, pipe.panels[factor_name], kind="factor",
            notional=args.notional, register=False,
        )
        fixed_c, _ = evaluate_panel(pipe, label, cond_signal, kind="conditional",
                                    notional=FIXED_COST_ONLY_NOTIONAL, register=False)
        fixed_u, _ = evaluate_panel(pipe, factor_name, pipe.panels[factor_name],
                                    kind="factor", notional=FIXED_COST_ONLY_NOTIONAL,
                                    register=False)

        beat_net = (
            res_c.net_sharpe > res_u.net_sharpe
            if all(v == v for v in (res_c.net_sharpe, res_u.net_sharpe)) else False
        )
        beat_fixed = (
            fixed_c.backtest["net_sharpe"] > fixed_u.backtest["net_sharpe"]
            if all(v == v for v in (fixed_c.backtest["net_sharpe"], fixed_u.backtest["net_sharpe"]))
            else False
        )
        say(f"    invested on {exposure:.1%} of regime-defined sessions")
        say(f"    conditional   net SR={res_c.net_sharpe:+.2f} (fixed-only {fixed_c.backtest['net_sharpe']:+.2f}) "
            f"turnover={res_c.annual_turnover:.1f}x")
        say(f"    unconditional net SR={res_u.net_sharpe:+.2f} (fixed-only {fixed_u.backtest['net_sharpe']:+.2f}) "
            f"turnover={res_u.annual_turnover:.1f}x")
        say(f"    conditioning beats unconditional after costs: "
            f"{'YES' if beat_net else 'NO'} (at notional), "
            f"{'YES' if beat_fixed else 'NO'} (fixed costs only)")

        strategy_results.append(
            {
                "label": label,
                "factor": factor_name,
                "regime_column": exp.regime_column,
                "active_buckets": active,
                "exposure_fraction": exposure,
                "conditional": card_c.to_dict(),
                "unconditional": card_u.to_dict(),
                "conditional_fixed_cost_only": fixed_c.backtest,
                "unconditional_fixed_cost_only": fixed_u.backtest,
                "beats_unconditional_after_costs_at_notional": beat_net,
                "beats_unconditional_fixed_costs_only": beat_fixed,
            }
        )
        res_c.daily.write_parquet(run_dir / f"{label}_daily.parquet")

    pipe.registry.save()
    n_supported = sum(1 for r in experiment_results if r["supported"])
    n_beat = sum(1 for s in strategy_results if s["beats_unconditional_fixed_costs_only"])
    say("")
    say(f"VERDICT: {n_supported}/{len(experiment_results)} declared hypotheses supported "
        f"at |t| >= {gate}")
    say(f"         {n_beat}/{len(strategy_results)} conditional strategies beat their "
        f"unconditional counterpart on fixed-cost-only net Sharpe")
    say(f"trial registry: {pipe.registry.summary()}")

    payload = {
        "provenance_label": prov.label(),
        "horizon": horizon,
        "gate_t_threshold": gate,
        "regime_coverage": regimes.summary(),
        "conditional_ic": cond.to_dicts(),
        "experiments": experiment_results,
        "strategies": strategy_results,
        "n_supported": n_supported,
        "n_experiments": len(experiment_results),
        "n_conditional_beats_unconditional_fixed_costs": n_beat,
        "trial_registry": {
            "n_trials": pipe.registry.n_trials, "n_distinct": pipe.registry.n_distinct,
        },
        "caveats": list(prov.caveats),
    }
    write_json(payload, run_dir / "conditioning.json")
    write_json(payload, cfg.path("results") / "conditioning.json")
    say(f"wrote {relative(run_dir / 'conditioning.json')} and results/conditioning.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
