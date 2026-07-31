#!/usr/bin/env python
"""Optional RL alpha miner (requires the ``mining`` extra: torch).

Every candidate kept is registered as a trial, so the Deflated Sharpe of a mined formula
is deflated against the whole search rather than against one. Use ``--smoke`` for a fast
run at ``mining.smoke_episodes``, and ``--policy random`` to run the declared control: if
the GRU's best formulas are no better than uniform sampling over the same grammar, the
policy is not contributing.
"""

from __future__ import annotations

import argparse
import sys

from _cli import bootstrap, say
from _pipeline import evaluate_panel, load_pipeline, relative, write_json
from flowalpha.conditioning.regimes import RegimeSet
from flowalpha.config import new_run_dir
from flowalpha.mining.grammar import build_vocabulary, terminal_matrices
from flowalpha.mining.miner import mine, register_candidates, signal_from_formula
from flowalpha.mining.policy import PolicyError, torch_available


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="use mining.smoke_episodes")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--policy", default=None, choices=["gru", "random"])
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--notional", type=float, default=1e9)
    parser.add_argument("--evaluate-best", action="store_true",
                        help="backtest the best candidate and report its DSR")
    args = parser.parse_args(argv)

    cfg, prov = bootstrap("run_mining")
    mcfg = cfg["mining"]
    policy_kind = args.policy or str(mcfg.get("policy", "gru"))
    if policy_kind == "gru" and not torch_available():
        say("FATAL: the gru policy needs torch. Install the mining extra:")
        say("  pip install -e '.[mining]'")
        say("Refusing to fall back to random sampling: a random search reported as a "
            "trained policy would misdescribe how the numbers were produced.")
        say("To run the control deliberately: python scripts/run_mining.py --policy random")
        return 1

    episodes = args.episodes
    if episodes is None:
        episodes = int(mcfg["smoke_episodes"]) if args.smoke else int(mcfg["episodes"])

    run_dir = new_run_dir(cfg, f"mining_{policy_kind}")
    pipe = load_pipeline(cfg)

    flow_features = None
    regimes = None
    try:
        regimes = RegimeSet.from_store(pipe.store, cfg)
        flow_features = regimes.features
    except FileNotFoundError:
        say("WARNING: flow_features absent; flow terminals will be all-null")
        if str(mcfg.get("vocab")) == "flow_augmented":
            say("  the flow_augmented vocabulary will therefore behave as price_only")

    vocab = build_vocabulary(str(mcfg.get("vocab", "flow_augmented")))
    say("")
    say(f"policy={policy_kind} episodes={episodes} seeds={args.seeds or mcfg['seeds']}")
    say(f"vocabulary={mcfg['vocab']} ({len(vocab)} tokens) "
        f"max_formula_len={mcfg['max_formula_len']} reward={mcfg['reward']} "
        f"novelty_lambda={mcfg['novelty_lambda']}")
    if policy_kind == "random":
        say("NOTE: this is the declared CONTROL -- uniform sampling over the grammar, "
            "not a trained policy.")

    try:
        result = mine(
            pipe.ctx, pipe.fwd, cfg,
            flow_features=flow_features, regimes=regimes,
            episodes=episodes, seeds=args.seeds, policy_kind=policy_kind,
            top_k=args.top_k, verbose=True,
        )
    except PolicyError as exc:
        say(f"FATAL: {exc}")
        return 1

    say("")
    say(f"episodes run: {result.episodes} x {len(result.seeds)} seed(s); "
        f"{result.n_illegal} incomplete, {result.n_degenerate} degenerate signals")
    say(f"top {len(result.candidates)} candidates:")
    for i, c in enumerate(result.candidates, 1):
        say(f"  {i:2d}. reward={c.reward:+.5f} ic={c.ic:+.5f} t={c.ic_t_stat:+.2f} "
            f"n={c.n_dates} len={c.formula.length}")
        say(f"      {c.infix}")
        if c.per_regime_ic:
            say(f"      per-regime IC: {c.per_regime_ic}")

    n_trials = register_candidates(result, pipe.registry)
    pipe.registry.save()
    say("")
    say(f"registered {len(result.candidates)} mined candidates as trials")
    say(f"trial registry: {pipe.registry.summary()}")

    payload = result.to_dict()
    payload["provenance_label"] = prov.label()
    payload["trials_registered"] = len(result.candidates)
    payload["trial_registry"] = {
        "n_trials": n_trials, "n_distinct": pipe.registry.n_distinct,
    }
    payload["caveats"] = list(prov.caveats)

    if args.evaluate_best and result.candidates:
        best = result.candidates[0]
        terminals = terminal_matrices(pipe.ctx, flow_features)
        panel = signal_from_formula(best.formula, terminals, pipe.ctx, vocab)
        if panel is not None:
            card, bt = evaluate_panel(
                pipe, f"mined:{best.formula}", panel, kind="mined",
                hypothesis=f"mined formula: {best.infix}", notional=args.notional,
                register=False,  # already registered above
            )
            say("")
            for line in card.summary_lines():
                say(line)
            payload["best_evaluation"] = card.to_dict()
            if bt.n_days:
                bt.daily.write_parquet(run_dir / "best_candidate_daily.parquet")

    write_json(payload, run_dir / "mining.json")
    write_json(payload, cfg.path("results") / "mining.json")
    say(f"wrote {relative(run_dir / 'mining.json')} and results/mining.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
