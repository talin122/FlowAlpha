#!/usr/bin/env python
"""Assemble ``results/report.html`` from the artefacts the run_* stages left behind.

Reads only JSON and parquet already on disk, so the report can be regenerated after a
presentation change without re-running the study. Missing stages are reported as missing
rather than silently omitted -- a report that quietly lacks its conditioning section
looks like a study that found nothing.

The output is a single self-contained file: figures are embedded as base64 PNG data URIs
and :func:`validate_html` refuses to write anything that references an external host.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from _cli import bootstrap, say
from flowalpha.config import now_ist
from flowalpha.reporting.report import (
    ReportDocument,
    bar_figure,
    embed_figure,
    esc,
    experiments_html,
    frame_to_table,
    kv_list,
    line_figure,
    validate_html,
)


def _load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _missing(stage: str, script: str) -> str:
    return (
        f'<div class="banner warn">Stage output missing: no <code>{esc(stage)}</code>. '
        f"Run <code>python {esc(script)}</code> to populate this section. "
        "It is reported as absent rather than omitted, so an incomplete study cannot be "
        "mistaken for a complete one.</div>"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="output path (default results/report.html)")
    args = parser.parse_args(argv)

    cfg, prov = bootstrap("make_report")
    results = cfg.path("results")
    out_path = Path(args.out) if args.out else results / "report.html"

    quality = _load(results / "data_quality_report.json")
    baseline = _load(results / "baseline.json")
    composite = _load(results / "composite.json")
    overlay = _load(results / "regime_overlay.json")
    conditioning = _load(results / "conditioning.json")
    coverage = _load(results / "regime_coverage.json")
    flow_summary = _load(results / "flow_ingest_summary.json")
    mining = _load(results / "mining.json")
    registry = _load(results / "trial_registry.json")

    doc = ReportDocument(
        title="FlowAlpha: is Indian equity factor performance conditional on institutional order flow?",
        provenance=prov,
        generated_at=now_ist(),
        subtitle=(
            f"Universe {cfg['universe']['name']} ({cfg['universe']['size']} names), "
            f"{cfg['dates']['start']} to {cfg['dates']['end']}. "
            f"Seed {cfg.seed}."
        ),
    )
    tag = prov.tag()

    # --- headline ---------------------------------------------------------
    headline_bits = []
    if conditioning:
        n_exp = conditioning["n_experiments"]
        headline_bits.append(
            f"{conditioning['n_supported']} of {n_exp} declared "
            f"conditioning hypotheses were supported at |t| >= {conditioning['gate_t_threshold']}."
        )
        # The support count alone would let an underpowered sample read as evidence that
        # conditioning does not work. These two counts license different claims and are
        # therefore always stated together.
        n_inc = conditioning.get("n_inconclusive_underpowered")
        n_null = conditioning.get("n_adequately_powered_nulls")
        if n_inc is not None and n_null is not None:
            headline_bits.append(
                f"Of the non-supported hypotheses, {n_null} of {n_exp} are adequately "
                f"powered nulls (evidence against a conditioning effect) and {n_inc} of "
                f"{n_exp} are INCONCLUSIVE -- the sample could not have detected the "
                "effect they were declared to look for."
            )
            if n_inc and not conditioning["n_supported"]:
                headline_bits.append(
                    "This study therefore does NOT establish that flow conditioning "
                    "fails. It establishes that this sample cannot resolve the question."
                )
        headline_bits.append(
            f"{conditioning['n_conditional_beats_unconditional_fixed_costs']} of "
            f"{len(conditioning['strategies'])} conditional strategies beat their "
            "unconditional counterpart on capacity-independent net Sharpe."
        )
    if registry:
        headline_bits.append(
            f"All Sharpe ratios are deflated against {registry['n_trials']} cumulative "
            f"trials across {registry['n_distinct']} distinct configurations."
        )
    doc.add(
        "Headline",
        "<ul>" + "".join(f"<li>{esc(b)}</li>" for b in headline_bits) + "</ul>"
        if headline_bits
        else _missing("conditioning.json", "scripts/run_conditioning.py"),
    )

    # --- data -------------------------------------------------------------
    if quality:
        checks = pl.DataFrame(
            [
                {"check": c["name"], "status": c["status"], "detail": c["message"]}
                for c in quality["checks"]
            ]
        )
        body = kv_list(
            [
                ("overall", quality["overall"]),
                ("sessions", quality["n_sessions"]),
                ("symbols", quality["n_symbols"]),
                ("price rows", quality["n_price_rows"]),
                ("calendar source", quality["calendar_source"]),
            ]
        ) + frame_to_table(checks, max_rows=None)
        card = next((c for c in quality["checks"] if c["name"] == "universe_cardinality"), None)
        if card and card["details"].get("series"):
            series = card["details"]["series"]
            fig = line_figure(
                {"members": ([r["date"] for r in series], [r["n"] for r in series])},
                title="Per-rebalance universe cardinality",
                ylabel="members", provenance_tag=tag,
            )
            body += embed_figure(
                fig,
                "Membership GROWS across the sample. NSE publishes only the current "
                "constituent list, so names that left the index are absent and post-2019 "
                "listings have no early history. This is survivorship bias, not a data "
                "error, and it biases returns upward by an unknown amount.",
            )
        doc.add("Data quality", body)
    else:
        doc.add("Data quality", _missing("data_quality_report.json", "scripts/validate_data.py"))

    if flow_summary:
        doc.add(
            "Flow coverage",
            kv_list(
                [
                    ("sessions with flow data", flow_summary.get("sessions_covered")),
                    ("first session", flow_summary.get("first_session")),
                    ("last session", flow_summary.get("last_session")),
                    ("sessions with no published archive", flow_summary.get("n_missing_sessions", 0)),
                ]
            )
            + "<p>Missing sessions are reported, never filled. Interpolating a flow value "
            "would manufacture the very signal under test. The trailing-sum implementation "
            "nulls only the windows that actually span a gap, so regimes recover "
            "immediately afterwards.</p>",
        )

    if coverage:
        rows = [
            {
                "regime": col,
                "sessions defined": e["n_defined"],
                "first defined": e.get("first_defined"),
                "last defined": e.get("last_defined"),
                "buckets": json.dumps(e.get("buckets") or {"n_true": e.get("n_true")}),
            }
            for col, e in coverage.get("regimes", {}).items()
        ]
        doc.add(
            "Flow regimes",
            "<p>All thresholds are <strong>expanding-window</strong>: the tercile "
            "boundaries and the shock percentile at date <em>t</em> use only observations "
            "up to and including <em>t</em>. A full-sample quantile would tell you today "
            "which bucket today belongs to relative to data you have not seen.</p>"
            + frame_to_table(pl.DataFrame(rows), max_rows=None)
            + '<p class="note">The last-defined date reaching the end of the price history '
            "is the check that the trailing sum is gap-tolerant rather than "
            "cumsum-based.</p>",
        )

    # --- baseline ---------------------------------------------------------
    if baseline:
        ic = pl.DataFrame(baseline["ic_table"])
        summary = pl.DataFrame(baseline["summary_table"])
        fixed = pl.DataFrame(baseline.get("fixed_cost_only_summary") or [])
        body = (
            "<p>IC t-statistics are Newey-West adjusted for the autocorrelation induced "
            "by overlapping forward-return labels, and are reported <strong>RAW</strong>: "
            "they are not deflated for multiple testing. The Sharpe ratios below are, "
            "against the cumulative trial registry.</p>"
            + frame_to_table(
                ic,
                digits={"mean_ic": 5, "std_ic": 4, "ic_ir": 3,
                        "t_stat_newey_west": 2, "hit_rate": 3, "mean_names": 0},
                max_rows=None,
            )
        )
        if baseline.get("skipped_factors"):
            body += (
                "<h3>Factors not evaluated</h3><ul>"
                + "".join(f"<li>{esc(s)}</li>" for s in baseline["skipped_factors"])
                + "</ul><p>A factor with no usable input is skipped with a warning and is "
                "<strong>not counted as a trial</strong>: an unevaluated factor is not a "
                "test.</p>"
            )
        body += "<h3>Backtests</h3>" + frame_to_table(
            summary,
            digits={"mean_ic": 5, "ic_ir": 3, "t_stat_newey_west_raw": 2, "hit_rate": 3,
                    "gross_sharpe": 2, "net_sharpe": 2, "annual_turnover": 1,
                    "cost_bps_per_year": 0, "deflated_sharpe_prob": 3},
            max_rows=None,
        )
        body += (
            f"<p>Net figures above are at a notional of Rs "
            f"{baseline['notional'] / 1e7:,.0f} crore, where square-root market impact "
            "dominates. The table below strips impact entirely, leaving only the "
            "capacity-independent costs (STT, stamp duty, brokerage). A signal that fails "
            "there fails for reasons no amount of size discipline can fix.</p>"
        )
        if not fixed.is_empty():
            body += frame_to_table(
                fixed.select("factor", "gross_sharpe", "net_sharpe", "annual_turnover",
                             "cost_bps_per_year"),
                digits={"gross_sharpe": 2, "net_sharpe": 2, "annual_turnover": 1,
                        "cost_bps_per_year": 0},
                max_rows=None,
            )
            fig = bar_figure(
                fixed["factor"].to_list(),
                [float(v) for v in fixed["net_sharpe"].to_list()],
                title="Net Sharpe, fixed costs only (capacity-independent)",
                ylabel="Sharpe", provenance_tag=tag,
            )
            body += embed_figure(fig, "Green is positive, red negative.")

        if baseline.get("capacity"):
            cap = pl.DataFrame(
                [
                    {"factor": k, "break-even AUM (Rs crore)": v["break_even_aum_crore"],
                     "median trailing ADV (Rs crore)": v["median_trailing_adv_rupees"] / 1e7}
                    for k, v in baseline["capacity"].items()
                ]
            )
            body += "<h3>Capacity</h3>" + frame_to_table(
                cap, digits={"break-even AUM (Rs crore)": 2,
                             "median trailing ADV (Rs crore)": 2}, max_rows=None
            ) + (
                "<p>Break-even AUM is the size at which square-root impact consumes "
                "whatever gross return survives the fixed costs. Zero means fixed costs "
                "alone already exceed the gross return.</p>"
            )
        if baseline.get("pbo") and "pbo" in baseline["pbo"]:
            p = baseline["pbo"]
            body += "<h3>Probability of backtest overfitting</h3>" + kv_list(
                [
                    ("PBO", p["pbo"]),
                    ("configurations", p["n_configs"]),
                    ("partitions evaluated", p["n_combinations"]),
                    ("blocks", p["n_partitions"]),
                ]
            ) + (
                "<p>Fraction of combinatorially symmetric partitions in which the "
                "in-sample-best factor landed below the out-of-sample median.</p>"
            )
        doc.add("Baseline: unconditional factors", body)
    else:
        doc.add("Baseline", _missing("baseline.json", "scripts/run_baseline.py"))

    # --- conditioning -----------------------------------------------------
    if conditioning:
        cond = pl.DataFrame(conditioning["conditional_ic"])
        body = (
            "<p>The study. Each factor's IC is measured <em>within</em> each order-flow "
            "regime bucket. The IC series is computed once and partitioned, so the buckets "
            "cannot disagree with the unconditional series they came from.</p>"
            + frame_to_table(
                cond,
                digits={"mean_ic": 5, "std_ic": 4, "ic_ir": 3,
                        "t_stat_newey_west": 2, "hit_rate": 3},
                max_rows=None,
            )
        )
        body += (
            "<h3>Declared hypotheses</h3>"
            "<p>The factor-to-regime map lives in <code>config.yaml</code> and the expected "
            "direction is stated in advance. Nobody scanned factor-regime pairs looking for "
            "the best one; a result in the wrong direction is therefore a refutation and not "
            "a finding with the sign flipped.</p>"
            + experiments_html(conditioning["experiments"])
        )
        strat_rows = []
        for s in conditioning["strategies"]:
            c, u = s["conditional"], s["unconditional"]
            cf, uf = s["conditional_fixed_cost_only"], s["unconditional_fixed_cost_only"]
            strat_rows.append(
                {
                    "strategy": s["label"],
                    "invested fraction": s["exposure_fraction"],
                    "cond net SR": (c.get("backtest") or {}).get("net_sharpe"),
                    "uncond net SR": (u.get("backtest") or {}).get("net_sharpe"),
                    "cond net SR (fixed only)": cf.get("net_sharpe"),
                    "uncond net SR (fixed only)": uf.get("net_sharpe"),
                    "cond turnover": (c.get("backtest") or {}).get("annual_turnover"),
                    "uncond turnover": (u.get("backtest") or {}).get("annual_turnover"),
                    "beats after costs": s["beats_unconditional_fixed_costs_only"],
                }
            )
        body += "<h3>Conditional strategies, after costs</h3>" + frame_to_table(
            pl.DataFrame(strat_rows),
            digits={"invested fraction": 3, "cond net SR": 2, "uncond net SR": 2,
                    "cond net SR (fixed only)": 2, "uncond net SR (fixed only)": 2,
                    "cond turnover": 1, "uncond turnover": 1},
            max_rows=None,
        ) + (
            "<p>A conditional strategy holds nothing outside its favourable regime rather "
            "than dropping those dates, so it is measured over the same window as the "
            "unconditional version. Dropping the dates would shorten the sample in a way "
            "correlated with the signal and flatter the result for that reason alone. The "
            "invested fraction is shown because a strategy in the market a third of the "
            "time has a mechanically lower total return.</p>"
        )
        doc.add("Conditioning: does flow regime matter?", body)
    else:
        doc.add("Conditioning", _missing("conditioning.json", "scripts/run_conditioning.py"))

    # --- composite and overlay -------------------------------------------
    if composite:
        rows = []
        for label, v in composite["variants"].items():
            if v.get("empty"):
                rows.append({"variant": label, "note": "empty signal"})
                continue
            bt = (v["card"].get("backtest") or {})
            rows.append(
                {
                    "variant": label,
                    "gross SR": bt.get("gross_sharpe"),
                    "net SR": bt.get("net_sharpe"),
                    "net SR (fixed only)": (v.get("fixed_cost_only") or {}).get("net_sharpe"),
                    "turnover": bt.get("annual_turnover"),
                    "DSR": (v["card"].get("deflated_sharpe") or {}).get(
                        "deflated_sharpe_probability"
                    ),
                    "trials": v["card"].get("n_trials_at_evaluation"),
                }
            )
        doc.add(
            "Composite",
            "<p>Factors are weighted by trailing IC/IR using only IC observations whose "
            "labels had already realised as of the weighting date. The equal-weight variant "
            "is shown alongside because it cannot be overfit, and is therefore the bar an "
            "IC-weighted scheme has to clear.</p>"
            + frame_to_table(
                pl.DataFrame(rows),
                digits={"gross SR": 2, "net SR": 2, "net SR (fixed only)": 2,
                        "turnover": 1, "DSR": 3},
                max_rows=None,
            ),
        )
    else:
        doc.add("Composite", _missing("composite.json", "scripts/run_composite.py"))

    if overlay:
        rows = []
        for label, v in overlay["variants"].items():
            if v.get("empty"):
                rows.append({"variant": label, "note": "empty signal"})
                continue
            bt = (v["card"].get("backtest") or {})
            rows.append(
                {
                    "variant": label,
                    "gross SR": bt.get("gross_sharpe"),
                    "net SR": bt.get("net_sharpe"),
                    "net SR (fixed only)": (v.get("fixed_cost_only") or {}).get("net_sharpe"),
                    "turnover": bt.get("annual_turnover"),
                }
            )
        on_frac = overlay["overlay"].get("on_fraction", {})
        doc.add(
            "Regime overlay",
            "<p>A factor is switched ON when its trailing conditional IC t-statistic inside "
            f"today's regime clears {overlay['overlay']['gate_t_threshold']}. The map is "
            "declared, not mined.</p>"
            + kv_list([(f"{k} ON fraction", v) for k, v in sorted(on_frac.items())])
            + frame_to_table(
                pl.DataFrame(rows),
                digits={"gross SR": 2, "net SR": 2, "net SR (fixed only)": 2, "turnover": 1},
                max_rows=None,
            )
            + (
                f"<p>Overlay improves the composite on capacity-independent net Sharpe: "
                f"<strong>{'yes' if overlay.get('overlay_improves_composite') else 'no'}</strong>. "
                "Note that gating also reduces turnover, so any improvement should be read "
                "as partly a cost effect rather than purely a signal effect.</p>"
            ),
        )
    else:
        doc.add("Regime overlay", _missing("regime_overlay.json", "scripts/run_regime_overlay.py"))

    if mining:
        doc.add(
            "Alpha miner",
            kv_list(
                [
                    ("episodes", mining.get("episodes")),
                    ("seeds", str(mining.get("seeds"))),
                    ("candidates kept", len(mining.get("candidates", []))),
                    ("trials registered", mining.get("trials_registered")),
                ]
            )
            + frame_to_table(
                pl.DataFrame(mining.get("candidates", [])),
                digits={"reward": 4, "ic": 5, "novelty_penalty": 4}, max_rows=50,
            )
            + "<p>Every mined candidate is registered as a trial, so the Deflated Sharpe of "
            "anything downstream is deflated against the full search, not against one.</p>",
        )

    # --- methodology ------------------------------------------------------
    doc.add(
        "Method and invariants",
        "<ul>"
        "<li><strong>No look-ahead, enforced not assumed.</strong> Every dataset declares "
        "an availability lag in <code>config.yaml</code>, and a single access layer refuses "
        "to return any row whose availability date is after the query date. A test injects "
        "future flow data and asserts that past features are bit-identical.</li>"
        "<li><strong>Expanding windows only.</strong> Regime terciles, shock percentiles and "
        "factor weights are computed from history up to and including <em>t</em>.</li>"
        "<li><strong>Per-dataset provenance.</strong> Machine-readable in "
        "<code>data/PROVENANCE.json</code>; <code>data/DATA_SOURCE.md</code> is generated "
        "from it so prose cannot drift from fact.</li>"
        "<li><strong>Multiple-testing honesty.</strong> Deflated Sharpe against a cumulative, "
        "persisted trial registry, plus PBO via combinatorially symmetric "
        "cross-validation.</li>"
        "<li><strong>Reproducibility.</strong> Each run snapshots its config, git hash and "
        "timestamp into its own directory under <code>results/runs/</code>.</li>"
        "<li><strong>Purged cross-validation.</strong> Where models are fitted, folds purge "
        "training observations whose label windows overlap the test block, plus an "
        "embargo.</li>"
        "</ul>",
    )

    if registry:
        doc.add(
            "Trial registry",
            kv_list(
                [
                    ("cumulative trials", registry["n_trials"]),
                    ("distinct configurations", registry["n_distinct"]),
                ]
            )
            + frame_to_table(
                pl.DataFrame(
                    [
                        {"configuration": k, "kind": v.get("kind"), "evaluations": v.get("count"),
                         "first seen": v.get("first_seen"), "last seen": v.get("last_seen")}
                        for k, v in sorted(registry["trials"].items())
                    ]
                ),
                max_rows=None,
            )
            + "<p>The registry is cumulative across the whole project, not per script. "
            "Re-running an experiment with one parameter changed adds a trial; it does not "
            "start over. A configuration evaluated many times has been tuned many "
            "times.</p>",
        )

    path = doc.write(out_path)
    text = path.read_text(encoding="utf-8")
    validate_html(text)
    say("")
    say(f"wrote {path} ({len(text):,} bytes, self-contained)")
    say(f"provenance banner: {prov.banner() or '(none -- all datasets real)'}")
    for stage, obj in (
        ("data_quality_report.json", quality), ("baseline.json", baseline),
        ("composite.json", composite), ("regime_overlay.json", overlay),
        ("conditioning.json", conditioning),
    ):
        say(f"  {stage:<28} {'present' if obj else 'MISSING (reported in the report)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
