# FlowAlpha

A research pipeline testing whether Indian equity factor performance is **conditional on
institutional order flow** — whether momentum works better when FIIs are buying, and
whether short-term reversal works better when retail participation is extreme.

This is quantitative research code whose output could inform real capital allocation.
Correctness and honesty about data provenance matter more than features or speed. A result
that looks good because of a subtle look-ahead bug is worse than no result.

**The headline result on real data is inconclusive, and that is a finding about the design
rather than about the market.** No declared hypothesis is supported, but none of the three
tests had the power to detect the effect it was looking for: the smallest difference the
momentum test could reliably resolve is 2.1× momentum's entire unconditional IC. A
market-wide time-series regime test on 6½ years of NSE participant-flow data cannot answer
this question at any threshold. See [Results](#results).

---

## The invariants

These are the project's reason for existing. Violating any of them is a defect, however
good the numbers look.

### 1. No look-ahead, enforced not assumed

Every dataset declares an availability lag in `config.yaml`:

```yaml
availability:
  prices:            {lag_days: 0}          # EOD bars, public the same session
  flows_daily:       {lag_days: 1}          # published after the close
  shareholding:      {lag_calendar_days: 21} # filing deadline, wall-clock
```

`flowalpha/data/store.py` is the **only** sanctioned way research code reads data. Given a
row's *event date* it computes the row's *available-from* date — trading-day lags shift
along the trading calendar, so a one-trading-day lag over a long weekend is three calendar
days — and **refuses to return any row whose available-from date is after the query date**.
Research code is therefore *unable* to see data that was not public; the invariant does not
depend on anyone remembering it.

The warrant for this is `tests/test_store.py::test_injecting_future_flows_does_not_move_past_features`:
it builds features, mutates flow values *after* a cutoff, rebuilds, and asserts every
feature up to the cutoff is bit-identical. Two guard tests sit alongside it, proving that a
full-sample quantile and a centred rolling window *would* be caught — so the injection test
cannot pass for the wrong reason.

`flow_features` is registered at **lag 0 on purpose**: its construction already shifts the
underlying flow by one session, so the row stamped `d` is the as-of-`d` snapshot. Adding a
lag there would double-lag it.

### 2. Expanding windows, never full-sample

Regime terciles, the shock percentile and factor weights are computed from history up to
and including `t`. A full-sample quantile tells you today which bucket today belongs to
relative to data you have not observed — see
`flowalpha/factors/flow_features.py::expanding_bucket_labels`.

The composite's IC weights go further: an IC observation dated `t` is not knowable on `t`,
because its label is a forward return. Weights on session `t` therefore use only IC
observations whose labels had **already realised** (`usable_from <= t`). Filtering on the
IC's own date instead is the commonest way a composite described as point-in-time turns out
not to be.

### 3. Per-dataset provenance

A single "is this synthetic?" boolean is insufficient, and getting this wrong is how a
pipeline ends up claiming generated numbers are market data. The normal state of this
project is a *hybrid*.

`data/PROVENANCE.json` records every dataset as `real` / `synthetic` / `unavailable` with a
source and a note. `flowalpha/data/provenance.py` derives the run label
(`LIVE DATA`, `SYNTHETIC DATA`, `HYBRID: SYNTHETIC FLOWS_DAILY`,
`LIVE DATA (PARTIAL: … UNAVAILABLE)`), and when no record exists it reports
**`PROVENANCE UNRECORDED`** — never "live data". `data/DATA_SOURCE.md` is *generated* from
the record so the prose cannot drift from the fact. Every script header and both report
banners read this record; nothing infers provenance from a file's existence.

All console strings are folded to ASCII, because these print on Windows terminals under
cp1252 where one en-dash aborts the run at the last line of output.

### 4. Multiple-testing honesty

A Sharpe ratio quoted without a trial count is not a result. `results/trial_registry.json`
counts every strategy this repository has ever evaluated — **cumulative across the project,
not per script**. Deflated Sharpe deflates against that count; PBO via combinatorially
symmetric cross-validation measures how often the in-sample-best configuration lands below
the out-of-sample median. IC t-statistics are Newey–West adjusted for the autocorrelation
that overlapping labels induce, and are reported **RAW** — the column is literally named
`t_stat_newey_west_raw`.

A factor with no usable input is skipped with a printed warning and is **not** counted as a
trial: an unevaluated factor is not a test.

### 5. Reproducibility

Each run creates `results/runs/<YYYYmmdd_HHMMSS>_<label>/` containing a snapshot of
`config.yaml` plus `provenance.json` (timestamp, git hash — suffixed `-dirty` when the tree
has uncommitted changes, label, cwd, seed). One `seed` in config drives everything
stochastic.

### 6. Never fabricate data to make a step pass

Missing sessions are **reported, not filled**. The 4 sessions in this window with no
published NSE archive are listed by date; interpolating them would manufacture the very
signal under test.

---

## Data sources, and what they cost the study

Run `python scripts/fetch_universe_tickers.py` etc., then read
[`data/DATA_SOURCE.md`](data/DATA_SOURCE.md) — generated from the record, so it is always
current. The load-bearing caveats:

### Flow units: contracts, not rupees

This project's thesis is about flow, so the flow source determines what the research
measures. Verified endpoint availability:

| source | span | content |
|---|---|---|
| `nsccl/fao_participant_vol_DDMMYYYY.csv` | **2019 → present** | Client/DII/FII/Pro F&O long & short, in **contracts** |
| `api/fiidiiTradeReact` | **current day only** | FII/DII **cash**, ₹ crores |
| `content/fo/fii_stats_DD-MMM-YYYY.xls` | 2019 → present | FII F&O values; genuine legacy `.xls`, needs `xlrd` |
| `cm_participant_vol_*.csv` | — | **404, does not exist** |

Consequence: **daily cash-market FII/DII history is not freely obtainable.** Only the
participant-wise F&O file spans the full window, so `flows_daily` and `participant_flows`
are built from it and are denominated in **net futures contracts** (index + stock futures,
long minus short; option legs excluded because a long call and a short put are not
comparable exposures). Regime features are quantile- and sign-based so bucketing is
unaffected, but **"FII net" in this codebase means net futures positioning**, not cash.

The cash series lives in its own dataset (`fii_dii_cash`, ₹ crores) accumulated one session
at a time by `scripts/download_flows.py`. It will be short — that is expected, not a
failure. It is written to a **separate file** and never to `flows_daily.parquet`:
overwriting the full-window flow panel with a few months of cash data would null nearly
every regime label downstream, which reads as a modelling problem rather than a
file-handling one.

### Survivorship bias

NSE publishes only the **current** NIFTY 500 constituent list. Applied historically this
omits names that left the index and includes names that joined for their whole history.
Combined with post-2019 listings having no early history, per-rebalance cardinality
**grows**: **365 members in 2019 rising to 500 by 2026, median 423**. That is why
`universe.cardinality_band` is `[350, 600]` and not centred on 500 — a band centred on 500
could never pass with this data. Returns are biased upward by an unknown amount.

The downloader has **no hardcoded fallback universe**. If the constituent list cannot be
fetched it exits non-zero, because a silent fallback to a short built-in list would produce
a ~50-name universe while config declares 500, and every cross-sectional rank would then be
computed over a tenth of the intended breadth — a failure that looks like a working
pipeline.

### What Yahoo does not provide

* **delivery %** — absent. The delivery factor computes to an empty panel and is skipped
  with a printed warning. Recorded `unavailable`. (`parse_bhavcopy` exists because NSE
  bhavcopy *does* carry `DELIV_PER`; ingest those and the factor comes alive.)
* **shares outstanding** — absent. Written as 1.0, which degrades size to a *log-price*
  proxy. The factor is therefore named **`size_logprice_proxy`**, so the degradation is
  visible at every call site rather than inviting a small-cap reading the data cannot
  support.
* **dividend adjustment** is approximate: `adj_factor = adj_close / close` applied to
  open/high/low. `volume` is left raw so `turnover = close × volume` stays split-invariant.
* **sectors** come from the constituent file's Industry column and are a **current
  snapshot**, not point-in-time. Recorded `real` with that note; `factors.sector_neutralize`
  is `false` by default for this reason.

### Retail proxy

Retail participation is the `Client` category in NSE's F&O file, which includes HNIs and
other non-institutional accounts. It is a proxy for retail, not a measurement of it.

---

## Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test,flows_xls]"        # add ,mining for the RL miner (torch)

python scripts/fetch_universe_tickers.py
python scripts/download_yahoo.py --start 2019-01-01 --end 2026-07-30
python scripts/download_participant_flows.py     # ~1900 cached requests, resumable
python scripts/build_flows.py
python scripts/validate_data.py                  # must exit 0

python -m pytest                                 # 672 passed, 1 skipped

python scripts/run_baseline.py && python scripts/run_composite.py
python scripts/run_regime_overlay.py && python scripts/run_conditioning.py
python scripts/daily_signals.py && python scripts/make_report.py
```

Optional:

```bash
python scripts/download_flows.py                 # accumulate the cash FII/DII series
python scripts/run_mining.py --smoke             # RL alpha miner (needs the mining extra)
python scripts/run_mining.py --policy random     # the declared control
```

Downloaders are **incremental**: raw responses are cached and skipped on re-runs, sessions
with no published archive are remembered in `_missing.json` so they are not re-requested
five times each, and `--provenance-only` rewrites the provenance record from what is on
disk without touching the network.

Outputs: `results/report.html` (research) and `results/daily/<date>/digest.html` (decision
support). Both are **fully self-contained** — no external CSS, JS, fonts or images; figures
are embedded as base64 data URIs, and `validate_html()` refuses to write a file containing
`src="http` or `href="http`.

---

## Results

Real data, 500 NIFTY 500 names, 1875 sessions, 2019-01-01 → 2026-07-30. Provenance:
`LIVE DATA (PARTIAL: DELIVERY_PCT, SHARES_OUTSTANDING UNAVAILABLE)`.

### Unconditional factor ICs (Newey–West t, **raw, not deflated**)

| factor | h=1 | h=5 | h=21 |
|---|---|---|---|
| `reversal_5d` | **+0.0196** (t 7.6) | **+0.0281** (t 6.3) | +0.0090 (t 1.6) |
| `low_volatility` | **+0.0313** (t 8.3) | +0.0196 (t 2.5) | +0.0008 (t 0.1) |
| `illiquidity` | −0.0198 (t −6.6) | −0.0025 (t −0.4) | **+0.0352** (t 3.0) |
| `momentum` | +0.0054 (t 1.5) | +0.0172 (t 2.4) | +0.0283 (t 2.0) |
| `reversal_21d` | +0.0106 (t 3.8) | +0.0100 (t 1.9) | +0.0019 (t 0.2) |
| `size_logprice_proxy` | +0.0159 (t 3.9) | +0.0063 (t 0.7) | −0.0175 (t −1.0) |
| `volume_trend` | −0.0031 (t −1.6) | +0.0018 (t 0.5) | +0.0016 (t 0.3) |

`delivery_ratio` was **not evaluated** (no delivery data from Yahoo) and is not counted as
a trial.

### After costs

Net Sharpe is reported twice on purpose. Fixed costs (STT, stamp duty, brokerage) are
proportional to notional and therefore capacity-independent; square-root market impact is
not. A signal that fails the capacity-independent column fails for reasons no amount of
size discipline can fix.

| factor | gross SR | net SR (fixed only) | net SR @ ₹100 cr | turnover | break-even AUM |
|---|---|---|---|---|---|
| `illiquidity` | 1.85 | **+1.75** | +0.92 | 6.6×/yr | ₹1.4 cr |
| `momentum` | 1.00 | **+0.87** | −0.13 | 9.0×/yr | ₹0.2 cr |
| `volume_trend` | 0.88 | +0.08 | −7.68 | 34×/yr | 0 |
| `reversal_5d` | 0.77 | −0.69 | −13.65 | 80×/yr | 0 |
| `reversal_21d` | −0.17 | −0.74 | −6.08 | 38×/yr | 0 |
| `low_volatility` | −0.91 | −1.01 | −2.00 | 9.6×/yr | 0 |
| `size_logprice_proxy` | −1.15 | −1.17 | −1.37 | 2.0×/yr | 0 |

PBO across the 7 factors: 0.000. Highest DSR: `illiquidity` at 0.849 against 7 cumulative
trials; everything else ≤ 0.03.

The strongest surviving factor is **illiquidity with a break-even AUM of about ₹1.4 crore**
— which is close to a restatement of the illiquidity premium itself: the return is
compensation for capacity you do not have. `reversal_5d` has the best IC in the study and
the worst net Sharpe, because 80× annual turnover at 17.5 bps round-trip plus impact
consumes it entirely.

### The conditioning result — inconclusive, not negative

The factor→regime map is **declared in `config.yaml`** and the expected direction is stated
before the test, so a reversed result is a refutation rather than a finding with the sign
flipped.

| hypothesis | IC fav | IC unfav | difference | t | MDE(80%) | ×uncond IC | power at ref | verdict |
|---|---|---|---|---|---|---|---|---|
| momentum, `fii_regime` high vs low | +0.0132 | +0.0255 | **−0.0123** | −0.81 | 0.0357 | 2.1× | 17.5% | INCONCLUSIVE |
| `reversal_5d`, `retail_extreme` extreme vs mid | +0.0319 | +0.0233 | +0.0086 | +0.87 | 0.0231 | 0.8× | 47.0% | INCONCLUSIVE |
| `reversal_21d`, `retail_extreme` extreme vs mid | +0.0091 | −0.0027 | +0.0117 | +1.14 | 0.0241 | 2.4× | 15.6% | INCONCLUSIVE |

**0 of 3 declared hypotheses supported at |t| ≥ 1.5. 0 of 3 are adequately powered nulls.
3 of 3 are inconclusive. 0 of 3 conditional strategies beat their unconditional counterpart
on capacity-independent net Sharpe.**

**This study does not establish that flow conditioning fails — it establishes that this
sample cannot resolve the question.** Every observed difference is well inside the noise
band. The smallest effect the momentum test could reliably detect is 0.0357, which is
**2.1× momentum's entire unconditional IC**; the reversal_21d test needs 2.4×. An effect
that large is not a plausible size for a real one, so those two hypotheses are not merely
unproven here, they are untestable at this sample size.

Power is measured against a **pre-declared** reference effect — half the factor's own
unconditional IC, set in `conditioning.reference_effect_fraction`. That matters: power
evaluated at whatever effect happened to be observed is a monotone function of the t-stat,
carries no independent information, and would make "adequately powered null" unreachable by
construction. See `tests/test_power.py::test_adequacy_ignores_the_observed_difference_entirely`.

Sample size needed for 80% power against that reference effect, from the observed
Newey–West standard errors (`se ∝ n^-1/2`):

| hypothesis | sessions used | needed | multiple | ≈ years |
|---|---|---|---|---|
| momentum × `fii_regime` | 1,181 | 20,218 | 17.1× | 81 |
| `reversal_21d` × `retail_extreme` | 1,518 | 35,084 | 23.1× | 140 |
| `reversal_5d` × `retail_extreme` | 1,518 | 4,097 | 2.7× | 16 |

Only `reversal_5d` is within reach of any feasible data-collection effort, and 16 years of
daily participant-flow data does not exist — NSE's file begins in 2019. **A market-wide
time-series regime test cannot answer this question.** The escape route is cross-sectional:
a per-stock retail measure (bhavcopy `DELIV_PER`) turns ~1,500 regime observations into
~425 names × ~1,850 sessions of independent variation, testing the same hypothesis with
orders of magnitude more information. See [Known limitations](#known-limitations).

Momentum's conditional IC is *directionally opposite* to the hypothesis — higher when FIIs
are in their selling tercile (+0.026) than their buying tercile (+0.013) — but at 17.5%
power that sign carries no weight.

The regime overlay does improve the composite's capacity-independent net Sharpe (+0.11 vs
−0.81), but it also cuts turnover from 35× to 23× per year, so that improvement should be
read as substantially a **cost** effect rather than a signal effect. It is not a positive
result for the hypothesis.

**An inconclusive result honestly reported is the correct output of this pipeline.** The
machinery is not the reason: run against `flowalpha/data/synthetic.py`, whose DGP embeds
both conditional effects at a magnitude this design *can* resolve,
`tests/test_baseline_integration.py` shows the same code recovers them — momentum weaker in
the FII-selling tercile, reversal stronger when retail is extreme, both at the declared
direction and threshold. So the real-data result is a statement about the sample, not about
the code.

### Portfolio-level risk overlays — volatility targeting makes drawdowns worse

The position-level constraints (name caps, sector caps, gross and net exposure) bound what
any single position may be. Two *book-level* overlays were added and tested as hypotheses:
volatility targeting (scale exposure by trailing realised vol toward 10% annual) and
drawdown control (de-risk to 50% below −10%, restore above −5%, with a hysteresis band).
Both ship **disabled**; `scripts/run_risk_overlay.py` runs all four combinations, and every
one registers as its own trial.

At capacity-independent notional, across all seven factors:

| overlay | improved net Sharpe | reduced max drawdown | turnover multiple |
|---|---|---|---|
| volatility targeting | 3/7 | **0/7** | **1.58× – 2.42×** |
| drawdown control | 4/7 | 7/7 | 0.58× – 1.00× |
| both | 5/7 | 3/7 | — |

**Volatility targeting made the maximum drawdown worse on every single factor**, by up to
23 percentage points. The first explanation written here was that it "levers up in calm
regimes, and calm regimes precede the breaks." That was a plausible story asserted without
testing, and it is **not** the dominant mechanism. The real cause is duller and worse:

Every factor book realises **4.0%–8.2% annualised volatility**. The configured target was
**10%**. So on all seven the overlay demanded 1.22×–2.51× leverage *permanently* — mean
applied scale 1.33–1.96, and on `reversal_5d` and `volume_trend` the **minimum** scale was
1.000, meaning it never de-levered once in 1,900 sessions. That is not a risk control; it
is leverage wearing a risk-control label, and leverage multiplies drawdown.

Across all 35 variant × factor cells, mean applied leverage and the change in drawdown
correlate at **+0.906 — leverage alone explains 82% of the variance.**

#### The fix, and what it revealed

Two modes were added. `relative` targets the strategy's *own* trailing 504-session
volatility (`scale = long_run_vol / short_run_vol`), so it is centred on 1.0 by
construction; `de_lever_only` clamps the scale at 1.0 so the overlay can only ever reduce
exposure. `relative` is now the config default, and `absolute` carries a warning.

The fix works as designed — mean leverage falls from 1.33–1.96 to 1.09–1.15, and on
`illiquidity` net Sharpe goes **1.739 → 1.815 with the drawdown unchanged**, against the
broken variant's −0.2045. But with the leverage confound removed, the honest conclusion is
that **volatility timing adds nothing here**:

| variant | better Sharpe | shallower maxDD | mean leverage |
|---|---|---|---|
| `vol_absolute` (original, broken) | 3/7 | 0/7 | 1.33 – 1.96 |
| `vol_relative` (fixed) | 3/7 | 2/7 | 1.09 – 1.15 |
| `vol_delever` (can only reduce) | 1/7 | 5/7 | 0.96 – 0.98 |

So the original finding was right for the wrong reason. Volatility targeting did not fail
because vol timing is harmful; it failed because it was mis-parameterised into a leverage
machine. Corrected, it is simply neutral — a wash on Sharpe that costs 30–40% more
turnover. It stays off, now for an accurate reason.

Drawdown control reduces drawdown on 7/7, which is close to tautological — an overlay that
can only cut participation will shrink the drawdown on the path it was measured against.
The honest reading is the Sharpe column beside it: on the four factors where Sharpe also
improved, three have a *negative* edge, so what improved was the amount of a losing
strategy being traded. The one case worth noting is `illiquidity`, the only factor with a
real edge: net Sharpe 1.746 → 1.770 with the drawdown cut from −14.3% to −12.4% at
identical turnover and cost.

**That last result should not be trusted as-is.** A drawdown control is fitted to the
realised path by construction: it reduces the drawdown that actually happened. Nothing here
establishes it would help on an unseen path, and the 28 configurations evaluated are now in
the trial registry precisely so the next Deflated Sharpe accounts for having looked.

Drawdown, time-under-water and Calmar are now measured on **every** backtest whether or not
control is enabled — they were not before, and the numbers change the picture: at ₹100
crore, `reversal_5d` shows a **−99.65%** maximum drawdown across 1,869 of 1,869 sessions
under water.

### Regime coverage

All five regime columns are defined **through 2026-07-30**, the final price session:
`fii_regime` and `retail_regime` 1521 sessions each (from 2020-02-11), `retail_extreme`
1521, `flow_divergence` 1772, `flow_shock` 1619 (206 shock days).

That the last-defined date reaches the end of the price history is the check that the
trailing sum is gap-tolerant. A `np.cumsum` implementation propagates one NaN to every
later value, so the first unpublished session would kill every regime from that date
onward — regimes stopping dead two years before the data ends. See
`tests/test_flows.py::test_trailing_sum_recovers_after_a_gap`.

---

## Layout

```
flowalpha/
  config.py              Config, load_config, new_run_dir, now_ist, IST
  data/
    calendar.py          TradingCalendar (derived from observed sessions), ensure_holidays_file
    store.py             PointInTimeStore, DATASET_SPECS, LookAheadError   <- the keystone
    fetch.py             polite retrying NSE client (shared cross-thread delay)
    prices.py            parse_bhavcopy, process_prices
    nse_flows.py         participant/cash/deal parsers, ingest_flows
    quality.py           PASS/WARN/FAIL data checks
    provenance.py        per-dataset provenance, generated DATA_SOURCE.md
    synthetic.py         schema-exact fixtures with a KNOWN DGP
  factors/
    base.py              Factor ABC, FactorContext, rolling primitives
    library.py           the factor zoo + default_library(cfg)
    flow_features.py     gap-tolerant trailing sums, expanding regimes
  conditioning/
    regimes.py           RegimeSet
    analysis.py          conditional IC, declared experiments, conditional strategy
  validation/
    ic.py                forward_returns, rank_ic, Newey-West
    purged_cv.py         PurgedKFold (purging + embargo)
    deflated_sharpe.py   DSR, PBO via CSCV, cumulative TrialRegistry
    neutralization.py    winsorize, zscore, sector-neutralise
  backtest/
    costs.py             Indian costs + sqrt impact + break-even AUM
    engine.py            daily long/short with overlapping sleeves
  signals/
    composite.py, regime_overlay.py, portfolio.py, factor_card.py, state.py
  reporting/report.py    self-contained HTML + validate_html
  mining/                grammar.py, policy.py, miner.py (RL alpha miner)
scripts/                 one CLI per pipeline stage, all with provenance headers
tests/                   673 tests
```

Sign convention throughout: **a higher factor value means a higher expected return** under
that factor's own hypothesis. So reversal is the negative of the recent return and low
volatility is the negative of realised volatility. Without a uniform convention an IC table
becomes a table of signs the reader has to remember.

---

## Post-build audit

A full audit was run over the finished codebase. Eleven defects were found, each
reproduced with an executable probe before being fixed, and each now pinned by a
regression test in [tests/test_audit_regressions.py](tests/test_audit_regressions.py)
named for the symptom rather than the line.

**Severe — silently wrong portfolios**

1. **A single-sector book ran at 0.5% of its target gross.** A share-of-gross cap needs at
   least two buckets to mean anything; with one sector its share is definitionally 1.0. The
   code still capped capacity at `sector_cap × target`, so with the shipped
   `sector_cap: 0.25` and no sector labels the portfolio came out at gross 0.005 against a
   target of 1.0 — a 190× under-investment — while simultaneously reporting the cap as
   *inert*. The two warnings contradicted each other.
2. **The sector cap gave a market-neutral book a directional bet.** Sectors were scaled
   across both legs at once, so capping a long-heavy sector removed more long exposure than
   short: **−13% net** on a 4-long/4-short example. Each leg is now projected onto its own
   target gross, which makes dollar-neutrality structural.
3. **A guard tested a mathematically invariant quantity.** The final leverage rescale was
   gated on re-checking sector shares — but a share is `|w_sector| / |w_total|`, which is
   unchanged by multiplying every weight by a constant. The guard could only ever fire when
   the cap was already unsatisfiable, and then it suppressed the rescale for no reason. This
   was the mechanism behind (1).

These three shared one root cause: the constraints were applied as a sequence of
independent passes, and they interact. `max_weight` redistribution moves weight between
sectors; the sector cap moves weight between legs; the leverage rescale undoes both.
`signals/portfolio.py` now projects **each leg** onto its whole constraint set at once by
nested water-filling (sectors first, then names within each sector — exact and terminating,
where iterating the two caps cycles), then **verifies every constraint on the final book**
and reports the result in `constraints_applied["satisfied"]`. When the caps are jointly
infeasible — `max_weight 1% × top_n 2`, or fewer than `1/sector_cap` sectors on a leg — the
book is left below target gross and says so, because breaching a declared concentration
limit is worse than running a smaller book. If one leg cannot reach its target, the other
is trimmed to match, because an unintended 10% net market position is worse than being
under-invested.

**Important — misleading results or a false immutability guarantee**

4. **`Config` immutability was skin-deep.** `cfg["universe"]["size"] = 999` succeeded:
   only top-level assignment was blocked, leaving every nested dict writable. A run could
   mutate its own parameters mid-flight, and its run-directory `config.yaml` snapshot would
   then describe something other than what ran — quietly breaking the reproducibility
   invariant the snapshot exists to provide. Reads now hand out detached copies.
5. **The miner kept formulas it had never scored.** An IC series shorter than `min_dates`
   yields `NaN`, which was coerced to reward `0.0` — beating every formula with a genuine
   *negative* IC. So the top-k could fill with candidates that were never evaluated, and
   REINFORCE was mildly encouraged to produce them. Unscorable formulas now receive a
   negative reward and are never kept.
6. **Forward returns crossed per-name gaps.** The shift was by row position, not by
   session: for a name halted two sessions, `shift(-1)` spanned three sessions while still
   being labelled `fwd_1` — mislabelling the horizon precisely for the names whose data is
   least trustworthy. Now computed on a dense session grid, so a gap yields a **null** label.
   This panel has 128 gap-sessions across 77 of its 500 names, and the fix corrected **437
   of 798,773 labels (0.055%)**. Rare, but individually large: `AEGISLOG`'s `fwd_5` for
   2025-01-28 was recorded as +19.3% when the true 5-session return was +6.6%. Aggregate ICs
   moved by ~1e-5 and no gross Sharpe changed to four decimals.

**Latent and honesty fixes**

7. **`usable_from` was clamped to the last session**, which would mark an IC whose label had
   not realised as usable on the final date — the exact look-ahead that column exists to
   prevent. Unreachable given how the IC series terminates, but it is now `None` rather than
   a clamp.
8. **A held name with no next-session price was booked as a flat exit.** Contributing 0 is
   an assumption — a costless exit at the last observed price — not a measurement. The
   engine now counts and reports `unpriced_position_days` and `unpriced_gross_exposure`.
9. **`universe_cardinality`'s docstring claimed a market-cap ranking the code never
   performed.** The check reports a *count*, which is invariant to ranking. The docstring
   now says so and the result carries a `reconstruction` field stating that
   `method: mktcap_rank` is recorded but not applied.
10. **`store.get(symbols=...)` was a silent no-op on aggregate datasets**, letting a caller
    believe it had restricted the universe when it had not. Now an error.
11. **`is_complete` compared tokens by value, not position**, so a mid-sequence stack
    underflow was excused whenever the offending token happened to equal the last one.
    Unreachable under the legality mask, fixed anyway.

Also removed: one dead regex, one dead function parameter, one dead local, and nine unused
imports.

**What the audit did not find:** no look-ahead in the store, the factors, the regimes, the
composite weighting or the overlay gating; no error in the cost model, the Deflated Sharpe,
PBO, or the purged-CV purge/embargo arithmetic.

**The research findings survive every fix.** Re-running the full analysis afterwards:
conditioning still 0/3 supported (differences −0.01228 / +0.01174 / +0.00858 against
−0.01226 / +0.01171 / +0.00868 before), every verdict string identical, 0/3 conditional
strategies beating unconditional, overlay still improving the composite. That the numbers
barely moved is itself worth stating: the severe bugs were in **portfolio construction**,
which the research conclusions do not depend on — they would have corrupted a live book, not
this study.

## Known limitations

Beyond the data caveats above, and all surfaced automatically in the report's Limitations
section:

* **Survivorship bias** inflates returns by an unknown amount.
* **Flow is F&O positioning in contracts**, not cash rupees.
* **Retail is proxied** by the `Client` category.
* **Dividend adjustment is proportional**, not a reconstruction of each corporate action.
* **Sector labels are a current snapshot**, so sector caps and neutralisation inherit a
  mild classification look-ahead.
* **IC t-statistics are raw**, adjusted only for autocorrelation. Only Sharpe ratios are
  deflated.
* **The conditioning test is underpowered by construction, not by accident.** A regime
  defined on a market-wide daily series yields one observation per session, so the effective
  sample is ~1,500 regardless of how many names are in the panel. The two hypotheses whose
  MDE exceeds their own unconditional IC (momentum, `reversal_21d`) cannot be settled by
  collecting more of the same data — they need a **cross-sectional** conditioning variable.
  Per-stock delivery percentage (NSE bhavcopy `DELIV_PER`, currently unavailable on the
  Yahoo path) is the obvious candidate and would raise the effective sample by orders of
  magnitude. Until then, treat the flow-conditioning question as open.
* **The alpha miner is a multiple-testing machine.** Every candidate it keeps registers as a
  trial. `--policy random` runs the declared control: if the GRU's best formulas are no
  better than uniform sampling over the same grammar, the policy is not contributing.
  `make_policy("gru", …)` **raises** when torch is absent rather than silently downgrading,
  because a random search reported as a trained policy would misdescribe the numbers.
* **The daily digest holds rather than guesses.** When fewer than
  `signals.composite.min_coverage` factors have a usable positive trailing IC, the model has
  no opinion: the digest says so, proposes no trades, and holds the existing book. A stale
  signal is never substituted into a file stamped with today's date.
* **Backtested performance here does not survive costs at institutional size.** Nothing in
  this repository is investment advice.
