#!/usr/bin/env python
"""Daily decision support: model portfolio, trade list, and a self-contained digest.

Produces ``results/daily/<date>/digest.html`` plus the machine-readable portfolio and
trade list. Yesterday's holdings are read from ``results/holdings.json`` so the trade
list reflects real deltas rather than a full build from cash every day.

The digest carries the provenance banner. On this project's data that banner says the
flow series is F&O contracts rather than cash rupees, which is exactly the kind of thing
a person acting on a trade list needs to know.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys

import polars as pl

from _cli import bootstrap, say
from _pipeline import load_pipeline, relative, write_json
from flowalpha.backtest.costs import rolling_adv
from flowalpha.conditioning.regimes import RegimeSet
from flowalpha.factors.base import _load_sectors
from flowalpha.reporting.report import (
    ReportDocument,
    esc,
    frame_to_table,
    kv_list,
)
from flowalpha.signals.composite import build_composite, compute_weights
from flowalpha.signals.portfolio import build_portfolio
from flowalpha.signals.regime_overlay import compute_gates, gate_summary
from flowalpha.signals.state import (
    archive_holdings,
    load_holdings,
    save_holdings,
)
from flowalpha.config import now_ist


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="as-of date (default: config dates.end)")
    parser.add_argument("--notional", type=float, default=1e9,
                        help="book size in rupees (default Rs 100 crore)")
    parser.add_argument("--no-overlay", action="store_true", help="skip the regime overlay")
    parser.add_argument("--no-state", action="store_true",
                        help="compute trades but do not persist today's holdings")
    args = parser.parse_args(argv)

    cfg, prov = bootstrap("daily_signals")
    as_of = _dt.date.fromisoformat(args.date) if args.date else cfg.end_date
    pipe = load_pipeline(cfg, as_of_date=as_of)
    as_of = pipe.sessions[-1]
    say(f"as-of session: {as_of}")

    regimes = None
    overlay = None
    panels = pipe.panels
    try:
        regimes = RegimeSet.from_store(pipe.store, cfg, as_of_date=as_of)
    except FileNotFoundError:
        say("WARNING: flow_features absent; regime overlay and regime reporting disabled")

    if regimes is not None and not args.no_overlay and bool(
        cfg.get("signals.regime_overlay.enabled", True)
    ):
        overlay = compute_gates(panels, pipe.fwd, cfg, regimes, sessions=pipe.sessions)
        panels = overlay.panels
        say("")
        for line in gate_summary(overlay):
            say(line)

    weights = compute_weights(panels, pipe.fwd, cfg, sessions=pipe.sessions)
    signal = build_composite(panels, weights, cfg)

    # A composite needs `min_coverage` factors with a usable trailing IC. When fewer
    # qualify, the model has no opinion today. The correct action is to hold the
    # existing book and say so -- NOT to fall back on a stale signal, which would put
    # a month-old view into a file labelled with today's date.
    today_weights = weights.weights.filter(pl.col("date") == as_of)
    min_coverage = int(cfg.get("signals.composite.min_coverage", 1))
    no_signal_reason = ""
    if signal.filter(pl.col("date") == as_of).is_empty():
        latest = signal["date"].max() if not signal.is_empty() else None
        no_signal_reason = (
            f"No composite signal for {as_of}: only {today_weights.height} factor(s) had a "
            f"usable positive trailing IC, against signals.composite.min_coverage = "
            f"{min_coverage}. The most recent session with a signal was {latest}. "
            "No trades are proposed and the existing book is held unchanged; a stale "
            "signal is not substituted."
        )
        say("")
        say("NO SIGNAL TODAY")
        say(f"  {no_signal_reason}")

    sectors = _load_sectors(cfg.path("reference"))
    adv = rolling_adv(pipe.prices, int(cfg["backtest"]["costs"]["adv_window"]))
    prior = load_holdings(cfg.path("results"))
    say("")
    say(f"prior holdings: {len(prior.weights)} positions, gross {prior.gross:.3f}, "
        f"as of {prior.as_of} ({prior.source or 'persisted state'})")

    if no_signal_reason:
        # Hold the prior book: same weights, zero trades.
        portfolio = build_portfolio(
            pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64}),
            pipe.prices, cfg, as_of_date=as_of, prior=prior,
            sectors=sectors, adv=adv, notional=args.notional,
        )
        portfolio.warnings.insert(0, no_signal_reason)
    else:
        portfolio = build_portfolio(
            signal, pipe.prices, cfg, as_of_date=as_of, prior=prior,
            sectors=sectors, adv=adv, notional=args.notional,
        )
    say(f"portfolio: {portfolio.n_long} long / {portfolio.n_short} short, "
        f"gross {portfolio.gross:.3f}, net {portfolio.net:+.4f}")
    say(f"turnover {portfolio.turnover:.4f} of notional, "
        f"estimated cost {portfolio.total_cost_bps:.1f} bps of notional")
    for warning in portfolio.warnings:
        say(f"  WARNING: {warning}")

    daily_dir = cfg.path("results") / "daily" / as_of.isoformat()
    daily_dir.mkdir(parents=True, exist_ok=True)
    portfolio.holdings.write_csv(daily_dir / "portfolio.csv")
    portfolio.trades.write_csv(daily_dir / "trades.csv")

    today_regimes = {}
    if regimes is not None:
        for col in regimes.columns:
            today_regimes[col] = regimes.label_on(col, as_of)
        say("")
        say("today's regimes: " + ", ".join(f"{k}={v}" for k, v in today_regimes.items()))

    # --- digest -----------------------------------------------------------
    doc = ReportDocument(
        title=f"FlowAlpha daily digest -- {as_of.isoformat()}",
        provenance=prov,
        generated_at=now_ist(),
        subtitle=(
            f"Decision support only. Notional Rs {args.notional / 1e7:,.0f} crore. "
            f"Composite scheme: {cfg.get('signals.composite.scheme')}."
        ),
    )
    if no_signal_reason:
        doc.add(
            "No signal today",
            f'<div class="banner warn">{esc(no_signal_reason)}</div>'
            + frame_to_table(today_weights.select("factor", "weight"), digits={"weight": 4}),
        )
    doc.add(
        "Today's flow regime",
        (
            kv_list([(k, v if v is not None else "undefined") for k, v in today_regimes.items()])
            if today_regimes
            else '<p class="note">Flow features unavailable; no regime labels for this session.</p>'
        ),
    )
    if overlay is not None:
        today_gates = overlay.decisions.filter(pl.col("date") == as_of)
        doc.add(
            "Factor gates (declared map, not data-mined)",
            "<p>A factor is ON when its trailing conditional IC t-statistic inside "
            f"today's regime clears {overlay.gate_t_threshold}. The factor-to-regime map is "
            "declared in <code>config.yaml</code>.</p>"
            + frame_to_table(
                today_gates.select("factor", "regime_column", "bucket", "t_stat", "n_obs", "on"),
                digits={"t_stat": 2},
            ),
        )
    doc.add(
        "Composite weights",
        frame_to_table(
            weights.weights.filter(pl.col("date") == as_of).select("factor", "weight"),
            digits={"weight": 4},
        ),
    )
    doc.add(
        "Model portfolio",
        kv_list(
            [
                ("as of", as_of.isoformat()),
                ("long positions", portfolio.n_long),
                ("short positions", portfolio.n_short),
                ("gross exposure", portfolio.gross),
                ("net exposure", portfolio.net),
                ("notional (Rs crore)", args.notional / 1e7),
            ]
        )
        + frame_to_table(portfolio.holdings, digits={"weight": 4, "adv": 0}, max_rows=200),
    )
    doc.add(
        "Trade list",
        "<p>Cost is the total of fixed costs (STT, stamp duty, brokerage) and "
        "square-root market impact against trailing ADV, expressed in basis points of "
        "each trade's own value. Names with no ADV estimate carry no impact charge and "
        "are shown with a blank participation.</p>"
        + kv_list(
            [
                ("turnover (fraction of notional)", portfolio.turnover),
                ("total cost (bps of notional)", portfolio.total_cost_bps),
                ("number of trades", portfolio.trades.height),
            ]
        )
        + frame_to_table(
            portfolio.trades,
            digits={"prev_weight": 4, "target_weight": 4, "delta_weight": 4,
                    "adv": 0, "participation": 4, "cost_bps": 1},
            max_rows=300,
        ),
    )
    if portfolio.warnings:
        doc.extra_limitations.extend(portfolio.warnings)
    doc.extra_limitations.append(
        "This digest is a research output, not investment advice. Backtested factor "
        "performance in this repository does not survive costs at this notional; see "
        "results/baseline.json."
    )
    digest_path = doc.write(daily_dir / "digest.html")
    say("")
    say(f"wrote {relative(digest_path)}")

    write_json(
        {
            "as_of": as_of.isoformat(),
            "provenance_label": prov.label(),
            "regimes": today_regimes,
            "portfolio": portfolio.to_dict(),
            "no_signal_reason": no_signal_reason or None,
            "weights": weights.to_dict(),
            "latest_weights": dict(
                zip(
                    weights.weights.filter(pl.col("date") == as_of)["factor"].to_list(),
                    weights.weights.filter(pl.col("date") == as_of)["weight"].to_list(),
                )
            ),
            "overlay": overlay.to_dict() if overlay else None,
        },
        daily_dir / "digest.json",
    )

    if no_signal_reason:
        # Preserve the prior book verbatim, including its own as-of date lineage.
        new_holdings = type(prior)(
            as_of=as_of, weights=dict(prior.weights),
            source=f"held unchanged on {as_of.isoformat()}: no signal (see digest)",
        )
    else:
        new_holdings = portfolio.to_holdings(source=f"daily_signals {as_of.isoformat()}")
    archive_holdings(new_holdings, daily_dir)
    if not args.no_state:
        path = save_holdings(new_holdings, cfg.path("results"))
        say(f"persisted holdings to {relative(path)} "
            f"(tomorrow's run computes real deltas from this)")
    else:
        say("holdings NOT persisted (--no-state)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
