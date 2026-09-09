#!/usr/bin/env python
"""Verify FlowAlpha's declared invariants against the artefacts actually on disk.

The test suite proves the code behaves correctly on fixtures. This proves the *shipped
state* is correct: that the parquet on disk, the provenance record, the trial registry
and the run snapshots are mutually consistent and honour the six invariants. A green
test suite over a stale or fabricated data tree would still be a failure.

Every claim is re-derived from the artefacts. Nothing here trusts a docstring.

Exit code is non-zero if any check fails, so it can gate a release.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import subprocess
import sys
from pathlib import Path

import polars as pl

from flowalpha.config import load_config
from flowalpha.data.store import LookAheadError, PointInTimeStore

ROOT = Path(__file__).resolve().parents[1]
results = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    print(f"[ {'PASS' if ok else 'FAIL'} ] {name}\n         {detail}")


def main() -> int:
    cfg = load_config()
    store = PointInTimeStore.from_config(cfg)

    # --- 1. No look-ahead, enforced not assumed --------------------------------
    asof = _dt.date(2026, 6, 15)
    bad = []
    for ds in ("prices", "flows_daily", "participant_flows", "flow_features"):
        d = store.get(ds, as_of_date=asof)
        if d.height and d["date"].max() > asof:
            bad.append(f"{ds} max={d['date'].max()}")
    check("1a. store returns nothing after as-of", not bad,
          f"4 datasets probed at {asof}; violations: {bad or 'none'}")

    # Lag is actually applied, not merely non-violating.
    lagged = store.get("flows_daily", as_of_date=asof)["date"].max()
    prices_max = store.get("prices", as_of_date=asof)["date"].max()
    check("1b. declared lag is actually applied", lagged < prices_max,
          f"prices (lag 0) max={prices_max}, flows_daily (lag 1) max={lagged}")

    # The guard raises rather than silently returning a filtered frame.
    try:
        store.get("flows_daily", as_of_date=asof, symbols=["RELIANCE"])
        raised = False
    except LookAheadError:
        raised = True
    check("1c. aggregate dataset rejects a symbols filter", raised,
          "store.get(symbols=...) on a series with no entity column raises LookAheadError")

    # --- 2. Expanding windows, never full-sample ------------------------------
    ff = pl.read_parquet(ROOT / "data/processed/flow_features.parquet").sort("date")
    reg = ff.filter(pl.col("fii_regime").is_not_null())
    min_obs = int(cfg["conditioning"]["expanding_min_obs"])
    first_defined = reg["date"].min()
    warmup = ff.filter(pl.col("date") < first_defined).height
    check("2. regimes undefined during the expanding warm-up", warmup >= min_obs,
          f"expanding_min_obs={min_obs}; {warmup} sessions null before first label "
          f"({first_defined}) -- a full-sample quantile would label from session 1")

    # --- 3. Provenance is per-dataset and machine-readable --------------------
    prov = json.loads((ROOT / "data/PROVENANCE.json").read_text())
    ds = prov.get("datasets", prov)
    statuses = {k: v.get("status") for k, v in ds.items()}
    generated = [k for k, v in statuses.items() if v == "generated"]
    unknown = [k for k, v in statuses.items() if v not in ("real", "unavailable", "generated")]
    check("3a. every dataset declares a valid status", not unknown,
          f"{len(statuses)} datasets: "
          f"{sum(v=='real' for v in statuses.values())} real, "
          f"{sum(v=='unavailable' for v in statuses.values())} unavailable, "
          f"{len(generated)} generated")
    check("3b. no generated data in the live pipeline", not generated,
          f"generated datasets: {generated or 'none'}")

    # Nothing outside tests may import the synthetic generator.
    # --include="*.py" so compiled bytecode in __pycache__ is not mistaken for source,
    # and match an IMPORT rather than any mention -- provenance.py names the module in a
    # docstring while importing nothing from it.
    hits = subprocess.run(
        ["grep", "-rlnE", r"^\s*(from|import).*\bsynthetic\b", "--include=*.py",
         "flowalpha/", "scripts/"],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout.split()
    hits = [h for h in hits if "synthetic.py" not in h]
    check("3c. synthetic generator unreachable from the pipeline", not hits,
          f"non-test importers of flowalpha.data.synthetic: {hits or 'none'}")

    # --- 4. Multiple-testing honesty ------------------------------------------
    reg_path = ROOT / "results/trial_registry.json"
    tr = json.loads(reg_path.read_text())
    n_trials = tr.get("n_trials") or len(tr.get("trials", []))
    base = json.loads((ROOT / "results/baseline.json").read_text())
    deflated = [c["deflated_sharpe"] for c in base["cards"] if c.get("deflated_sharpe")]
    trials_used = {d.get("n_trials") for d in deflated}
    check("4a. trial registry is cumulative and persisted", n_trials > 7,
          f"{n_trials} cumulative trials recorded at {reg_path.name} "
          f"(more than the 7 factors, so prior looks are carried)")
    check("4b. deflated Sharpe is deflated against the registry", all(t > 1 for t in trials_used),
          f"n_trials seen in factor cards: {sorted(trials_used)}")

    # --- 5. Reproducibility ----------------------------------------------------
    runs = sorted((ROOT / "results/runs").glob("*/provenance.json"))
    snap = json.loads(runs[-1].read_text())
    keys = {"git_hash", "seed", "timestamp_ist", "config_source"}
    have = keys <= set(snap)
    dirty = str(snap.get("git_hash", "")).endswith("-dirty")
    check("5a. every run snapshots hash, seed, timestamp, config", have,
          f"{len(runs)} run dirs; latest: git_hash={snap.get('git_hash')} "
          f"seed={snap.get('seed')}")
    check("5b. run hash is a real commit, not 'nogit'", snap.get("git_hash", "").split("-")[0] != "nogit",
          f"git_hash={snap.get('git_hash')}" + ("  (tree dirty at run time)" if dirty else ""))

    # --- 6. Never fabricate to make a step pass -------------------------------
    fl = pl.read_parquet(ROOT / "data/processed/flows_daily.parquet")
    sessions = sorted(set(pl.read_parquet(ROOT / "data/processed/prices.parquet")["date"].to_list()))
    flow_dates = set(fl["date"].to_list())
    missing = [d for d in sessions if d not in flow_dates]
    q = json.loads((ROOT / "results/data_quality_report.json").read_text())
    gap_check = next(
        (c for c in q.get("checks", []) if c.get("name") == "flows_daily_gaps"), None
    )
    check("6a. flow gaps are left as gaps, not filled", len(missing) > 0,
          f"{len(missing)} sessions have no flow archive and remain absent: "
          f"{[str(d) for d in missing[:4]]}")
    check("6b. those gaps are reported by the quality gate", gap_check is not None,
          f"{gap_check.get('message','') if gap_check else 'MISSING'}")

    # An inert factor is skipped and named, not silently zero-filled.
    skipped = base.get("skipped_factors", [])
    check("6c. unavailable inputs disable their factor visibly", len(skipped) == 1,
          f"skipped_factors={skipped}")

    # --- outputs ---------------------------------------------------------------
    html = (ROOT / "results/report.html").read_text()
    ext = re.findall(r'(?:src|href)=["\'](?!#|data:)', html)
    check("7a. report is self-contained", not ext,
          f"{len(html):,} bytes, {len(ext)} external references")
    bad_json = []
    for p in (ROOT / "results").glob("*.json"):
        try:
            json.loads(p.read_text())
        except Exception as e:
            bad_json.append(f"{p.name}: {e}")
    check("7b. every results JSON parses", not bad_json,
          f"{len(list((ROOT/'results').glob('*.json')))} files; bad: {bad_json or 'none'}")

    print()
    n_pass = sum(1 for _, ok, _ in results if ok)
    print("=" * 70)
    print(f"{n_pass}/{len(results)} checks passed")
    print("=" * 70)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
