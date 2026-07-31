"""Schema-exact synthetic fixtures with a KNOWN data-generating process.

Purpose
-------
The validation machinery needs data with structure it *should* be able to recover. If
the only test of an IC pipeline is real data, a bug that halves every IC is invisible:
the answer looks plausible either way. Here the answer is known in advance.

This generator is **never** a silent substitute for real data. It marks every dataset
it writes as ``synthetic`` in the provenance record, which makes the run's label
``SYNTHETIC DATA`` (or ``HYBRID: ...`` if mixed with real files), and every script
header and report banner then says so.

The embedded effects
--------------------
Each is listed in :data:`KNOWN_EFFECTS` with the factor that should detect it:

1. **Persistent return driver.** A per-symbol AR(1) expected-return component with
   ``phi = 0.98``. Past 12-month return is informative about it, and it persists, so
   *momentum* should show positive IC.
2. **Transient shock that reverses.** Occasional jumps that are partially given back
   over the following days, so *short-horizon reversal* should show positive IC.
3. **Low-volatility premium.** Expected return carries ``-lambda * sigma_i``, so *low
   volatility* (which is signed as negative realised vol) should show positive IC.
4. **Delivery loads on the driver.** ``delivery_pct`` is a noisy monotone function of
   the persistent driver, so the *delivery* factor should show positive IC -- this is
   the only fixture in the project where that factor is not inert.
5. **Reversal strength scales with |retail flow|.** The give-back coefficient rises
   with the magnitude of trailing retail participation, so reversal IC should be
   *higher in the ``retail_extreme = extreme`` bucket* than in ``mid``.
6. **Momentum weakens when FIIs are selling.** The driver's contribution to returns is
   damped when trailing FII flow is negative, so momentum IC should be *lower in the
   ``fii_regime = low`` bucket* than in ``high``.

Effects 5 and 6 are the study's hypotheses. They are built in here so that the
conditioning machinery can be shown to find them when they exist -- which is what
makes a *negative* result on real data meaningful rather than merely unexplained.

Crucially, the flow-dependent modulations are keyed to **the same expanding-tercile
labels that** :mod:`flowalpha.factors.flow_features` **computes**, not to some other
summary of the same series. An earlier version keyed effect 6 to ``sign(trailing FII
flow)`` and effect 5 to a full-sample z-score; both are correlated with the tercile
labels but neither coincides with them, so the embedded effect and the measured bucket
disagreed and one of the two conditional effects came out inverted. Sharing the labelling
function makes the alignment structural rather than approximate.

Effect sizes here are deliberately far larger than anything in a real market. The point of
this fixture is to verify that the machinery *can* detect structure that is definitely
present, so the effects are made unmissable; nothing about their magnitude is a claim
about Indian equities.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from ..config import Config
from .calendar import TradingCalendar, ensure_holidays_file
from .nse_flows import (
    FLOWS_DAILY_SCHEMA,
    PARTICIPANT_FLOWS_SCHEMA,
)
from ..factors.flow_features import collapse_extreme, expanding_bucket_labels
from .prices import process_prices
from .provenance import (
    CAVEAT_MULTIPLE_TESTING,
    SYNTHETIC,
    Provenance,
)

#: What a correct pipeline should recover from this fixture. Referenced by the
#: integration tests so the claims and the assertions cannot drift apart.
KNOWN_EFFECTS: dict[str, str] = {
    "momentum": "positive IC from the persistent AR(1) driver",
    "reversal": "positive IC from the transient reversing shock",
    "low_volatility": "positive IC from the low-volatility premium",
    "delivery_ratio": "positive IC; delivery_pct loads on the persistent driver",
    "reversal_conditional": "reversal IC higher when retail_extreme == 'extreme'",
    "momentum_conditional": "momentum IC lower when fii_regime == 'low'",
}

SECTORS = ("Financial Services", "Information Technology", "Healthcare",
           "Consumer Goods", "Industrials", "Energy")


@dataclass(frozen=True)
class SyntheticSpec:
    """DGP parameters. Every one is documented because every one is load-bearing."""

    n_symbols: int = 120
    n_sessions: int = 900
    start_date: _dt.date = _dt.date(2015, 1, 5)
    seed: int = 7

    #: Persistence of the expected-return driver. High enough that a 12-month
    #: trailing return is informative about the next month.
    driver_phi: float = 0.98
    #: Innovation scale of the driver, in daily return units. Large enough that momentum
    #: is comfortably detectable over 900 sessions against the shock and idiosyncratic
    #: channels; see the note on effect sizes in the module docstring.
    driver_sigma: float = 0.0012

    # The shock channel has to be strong enough to dominate the *momentum* channel inside
    # a 5-day trailing window, or the reversal factor measures the persistent driver
    # instead and shows NEGATIVE IC. Both effects are supposed to be recoverable at the
    # horizons their factors are built for, so the two channels are separated by
    # magnitude: shocks are large and revert fast, the driver is small and persists.
    #: Probability per symbol-day of a transient shock.
    shock_prob: float = 0.12
    #: Scale of the transient shock, in daily return units.
    shock_scale: float = 0.05
    #: Give-back fraction at zero retail flow.
    reversal_base: float = 0.45
    #: Give-back fraction as |trailing retail flow| grows without bound. Kept strictly
    #: BELOW 1.0: a coefficient above 1 means the price overshoots on reversion, and the
    #: leftover negative residual then partially cancels the next shock inside the 5-day
    #: trailing window -- which weakens the reversal signal exactly in the extreme-retail
    #: bucket where it is supposed to be strongest, inverting the embedded effect.
    reversal_max: float = 0.90
    #: Steepness of the approach from base to max, in z-units of trailing retail flow.
    #: This is embedded effect 5.
    reversal_retail_beta: float = 1.0

    #: Coefficient on -sigma_i in DAILY expected return (the low-volatility premium).
    #: Applied per day, not annualised: dividing by 252 makes the premium ~1e-6 per day,
    #: which is four orders of magnitude below the idiosyncratic noise and therefore
    #: undetectable at any sample size this fixture can reach.
    lowvol_lambda: float = 0.06
    #: Cross-sectional spread of per-symbol daily volatility.
    vol_low: float = 0.010
    vol_high: float = 0.030

    #: Momentum damping when trailing FII flow is negative. Embedded effect 6:
    #: below 1 means the driver pays less in FII-selling regimes.
    momentum_fii_damp: float = 0.25

    #: Delivery percentage = base + slope * z(driver) + noise, clipped to [5, 95].
    delivery_base: float = 55.0
    delivery_slope: float = 12.0
    delivery_noise: float = 6.0

    #: Flow process: AR(1) in daily net contracts.
    flow_phi: float = 0.85
    fii_flow_sigma: float = 12_000.0
    dii_flow_sigma: float = 9_000.0
    retail_flow_sigma: float = 25_000.0
    pro_flow_sigma: float = 8_000.0

    #: Trailing window used by the DGP's modulations. Kept equal to
    #: `conditioning.regime_window` by `spec_from_config` so the effect the features
    #: look for is the effect that is actually there.
    flow_window: int = 21

    #: Weekday holidays to drop, as a fraction, so the calendar has real gaps.
    holiday_fraction: float = 0.04
    #: Sessions for which no flow archive is published, so gap handling is exercised.
    n_flow_gaps: int = 3

    #: Observations required before an expanding tercile label is emitted. MUST match
    #: `conditioning.expanding_min_obs` in the config used to read this fixture, or the
    #: DGP's labels and the features' labels will disagree on the early sample.
    expanding_min_obs: int = 126


def spec_from_config(cfg: Config, **overrides) -> SyntheticSpec:
    """Build a spec from ``config.yaml``'s ``synthetic`` block."""
    scfg = cfg.get("synthetic", {}) or {}
    base = {
        "n_symbols": int(scfg.get("n_symbols", 120)),
        "seed": int(scfg.get("seed", 7)),
        "flow_window": int(cfg.get("conditioning.regime_window", 21)),
        "expanding_min_obs": int(cfg.get("conditioning.expanding_min_obs", 126)),
    }
    base.update(overrides)
    return SyntheticSpec(**base)


@dataclass
class SyntheticData:
    """The generated fixture, plus the ground truth used to build it."""

    prices: pl.DataFrame
    flows_daily: pl.DataFrame
    participant_flows: pl.DataFrame
    sectors: pl.DataFrame
    shares: pl.DataFrame
    calendar: TradingCalendar
    spec: SyntheticSpec
    #: (n_sessions, n_symbols) persistent driver, for tests that want ground truth.
    driver: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 0)))
    #: (n_sessions,) trailing FII flow through t-1, as the DGP used it.
    fii_trailing: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    #: (n_sessions,) trailing retail flow through t-1, in z-units.
    retail_trailing_z: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))


def _sessions(spec: SyntheticSpec, rng: np.random.Generator) -> list[_dt.date]:
    """Weekdays from ``start_date``, with a random subset removed as holidays."""
    days: list[_dt.date] = []
    day = spec.start_date
    target = int(spec.n_sessions / (1.0 - spec.holiday_fraction)) + 10
    while len(days) < target:
        if day.weekday() < 5:
            days.append(day)
        day += _dt.timedelta(days=1)
    n_holidays = int(spec.holiday_fraction * len(days))
    holiday_idx = set(rng.choice(len(days), size=n_holidays, replace=False).tolist())
    kept = [d for i, d in enumerate(days) if i not in holiday_idx]
    return kept[: spec.n_sessions]


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(n)
    eps = rng.normal(0.0, sigma, size=n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def _trailing_shifted_sum(daily: np.ndarray, window: int) -> np.ndarray:
    """Trailing sum of the PREVIOUS ``window`` sessions, i.e. through ``t-1``.

    Matches what :func:`flowalpha.factors.flow_features.build_flow_features` computes,
    so an effect keyed to this series is genuinely recoverable point-in-time.
    """
    shifted = np.concatenate([[np.nan], daily[:-1]])
    out = np.full(daily.shape[0], np.nan)
    for i in range(window, daily.shape[0] + 1):
        chunk = shifted[i - window : i]
        if not np.isnan(chunk).any():
            out[i - 1] = chunk.sum()
    return out


def generate(spec: SyntheticSpec | None = None) -> SyntheticData:
    """Generate the fixture. Deterministic in ``spec.seed``."""
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)
    sessions = _sessions(spec, rng)
    n, m = len(sessions), spec.n_symbols
    symbols = [f"SYN{i:04d}" for i in range(m)]

    # --- flow processes -----------------------------------------------------
    flows = {
        "FII": _ar1(n, spec.flow_phi, spec.fii_flow_sigma, rng),
        "DII": _ar1(n, spec.flow_phi, spec.dii_flow_sigma, rng),
        "Client": _ar1(n, spec.flow_phi, spec.retail_flow_sigma, rng),
        "Pro": _ar1(n, spec.flow_phi, spec.pro_flow_sigma, rng),
    }
    fii_trailing = _trailing_shifted_sum(flows["FII"], spec.flow_window)
    retail_trailing = _trailing_shifted_sum(flows["Client"], spec.flow_window)
    retail_scale = np.nanstd(retail_trailing)
    retail_trailing_z = retail_trailing / (retail_scale if retail_scale > 0 else 1.0)

    # Label the flow series with EXACTLY the function the features use, so the embedded
    # effects are keyed to the buckets the study measures rather than to a correlated
    # proxy for them.
    fii_labels = expanding_bucket_labels(
        fii_trailing, n_buckets=3, min_obs=spec.expanding_min_obs
    )
    retail_labels = [
        collapse_extreme(x)
        for x in expanding_bucket_labels(
            retail_trailing, n_buckets=3, min_obs=spec.expanding_min_obs
        )
    ]

    # Embedded effect 6: the driver pays less when FIIs are in their selling tercile.
    # Undated (None) labels get the neutral multiplier, matching the features, which emit
    # no label there and so drop those dates from any conditional comparison.
    mom_by_label = {
        "high": 1.0,
        "mid": 0.5 * (1.0 + spec.momentum_fii_damp),
        "low": spec.momentum_fii_damp,
        None: 1.0,
    }
    momentum_mult = np.array([mom_by_label[label] for label in fii_labels], dtype=float)

    # Embedded effect 5: shocks revert harder when retail participation is extreme.
    # `reversal_max` stays strictly below 1.0: a coefficient above 1 overshoots on
    # reversion, and the leftover negative residual then partially cancels the next shock
    # inside the 5-day trailing window -- weakening the reversal signal in exactly the
    # bucket where it is supposed to be strongest.
    rev_by_label = {
        "extreme": spec.reversal_max,
        "mid": spec.reversal_base,
        None: 0.5 * (spec.reversal_base + spec.reversal_max),
    }
    reversal_coef = np.array([rev_by_label[label] for label in retail_labels], dtype=float)

    # --- per-symbol return construction ------------------------------------
    sigma_i = rng.uniform(spec.vol_low, spec.vol_high, size=m)
    driver = np.zeros((n, m))
    for j in range(m):
        driver[:, j] = _ar1(n, spec.driver_phi, spec.driver_sigma, rng)

    shock_hit = rng.random((n, m)) < spec.shock_prob
    shocks = np.where(shock_hit, rng.normal(0.0, spec.shock_scale, size=(n, m)), 0.0)
    prev_shocks = np.vstack([np.zeros((1, m)), shocks[:-1]])

    idio = rng.normal(0.0, 1.0, size=(n, m)) * sigma_i[None, :]

    returns = (
        driver * momentum_mult[:, None]                    # effects 1 + 6
        - spec.lowvol_lambda * sigma_i[None, :]            # effect 3
        + shocks                                           # effect 2 (impulse)
        - reversal_coef[:, None] * prev_shocks             # effect 2 + 5 (give-back)
        + idio
    )

    levels = 100.0 * np.exp(np.cumsum(np.log1p(returns), axis=0))
    base_price = rng.uniform(0.4, 8.0, size=m)
    levels = levels * base_price[None, :]

    # Effect 4: delivery percentage loads on the persistent driver.
    driver_z = (driver - driver.mean(axis=1, keepdims=True)) / np.where(
        driver.std(axis=1, keepdims=True) > 0, driver.std(axis=1, keepdims=True), 1.0
    )
    delivery = np.clip(
        spec.delivery_base + spec.delivery_slope * driver_z
        + rng.normal(0.0, spec.delivery_noise, size=(n, m)),
        5.0, 95.0,
    )

    log_vol_base = rng.uniform(np.log(2e5), np.log(5e7), size=m)
    volume = np.exp(log_vol_base[None, :] + rng.normal(0.0, 0.35, size=(n, m)))

    # --- assemble frames ---------------------------------------------------
    dates_col = np.repeat(np.array(sessions, dtype="object"), m).tolist()
    syms_col = np.tile(np.array(symbols, dtype="object"), n).tolist()
    close = levels.reshape(-1)
    raw_prices = pl.DataFrame(
        {
            "date": dates_col,
            "symbol": syms_col,
            "open": close * 0.998,
            "high": close * 1.008,
            "low": close * 0.992,
            "close": close,
            # No dividends in the DGP, so adj_close == close and adj_factor == 1.
            "adj_close": close,
            "volume": volume.reshape(-1),
            "delivery_pct": delivery.reshape(-1),
        },
        schema={
            "date": pl.Date, "symbol": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
            "low": pl.Float64, "close": pl.Float64, "adj_close": pl.Float64,
            "volume": pl.Float64, "delivery_pct": pl.Float64,
        },
    )
    prices = process_prices(raw_prices)

    # A few sessions with no published flow archive, so the gap handling in
    # flow_features is exercised by the integration fixture too.
    gap_idx = set(
        rng.choice(np.arange(spec.flow_window + 5, n - 5), size=spec.n_flow_gaps, replace=False).tolist()
    )
    part_rows, daily_rows = [], []
    for i, day in enumerate(sessions):
        if i in gap_idx:
            continue
        for participant, series in flows.items():
            net = float(series[i])
            long_leg = abs(net) + 5_000.0
            short_leg = long_leg - net
            part_rows.append(
                {
                    "date": day, "participant": participant,
                    "fut_index_long": long_leg * 0.35, "fut_index_short": short_leg * 0.35,
                    "fut_stock_long": long_leg * 0.65, "fut_stock_short": short_leg * 0.65,
                    "opt_index_call_long": 0.0, "opt_index_put_long": 0.0,
                    "opt_index_call_short": 0.0, "opt_index_put_short": 0.0,
                    "opt_stock_call_long": 0.0, "opt_stock_put_long": 0.0,
                    "opt_stock_call_short": 0.0, "opt_stock_put_short": 0.0,
                    "total_long": long_leg, "total_short": short_leg,
                    "net_futures": net,
                }
            )
            if participant in ("FII", "DII"):
                daily_rows.append(
                    {
                        "date": day, "participant": participant, "net_futures": net,
                        "gross_futures_long": long_leg, "gross_futures_short": short_leg,
                    }
                )
    participant_flows = pl.DataFrame(part_rows, schema=PARTICIPANT_FLOWS_SCHEMA).sort(
        ["date", "participant"]
    )
    flows_daily = pl.DataFrame(daily_rows, schema=FLOWS_DAILY_SCHEMA).sort(["date", "participant"])

    sectors = pl.DataFrame(
        {"symbol": symbols, "sector": [SECTORS[i % len(SECTORS)] for i in range(m)]},
        schema={"symbol": pl.Utf8, "sector": pl.Utf8},
    )
    shares = pl.DataFrame(
        {"symbol": symbols, "shares": rng.uniform(1e7, 5e9, size=m)},
        schema={"symbol": pl.Utf8, "shares": pl.Float64},
    )

    return SyntheticData(
        prices=prices,
        flows_daily=flows_daily,
        participant_flows=participant_flows,
        sectors=sectors,
        shares=shares,
        calendar=TradingCalendar(sessions),
        spec=spec,
        driver=driver,
        fii_trailing=fii_trailing,
        retail_trailing_z=retail_trailing_z,
    )


def write_tree(
    data: SyntheticData,
    root: Path,
    *,
    processed_subdir: str = "processed",
    reference_subdir: str = "reference",
    write_provenance: bool = True,
) -> dict[str, Path]:
    """Write the fixture as a schema-exact pipeline input under ``root``.

    When ``write_provenance`` is set (the default), a provenance record marking every
    dataset ``synthetic`` is written to ``root/PROVENANCE.json``, so anything that
    reads this tree reports ``SYNTHETIC DATA`` rather than claiming live data.
    """
    root = Path(root)
    processed = root / processed_subdir
    reference = root / reference_subdir
    processed.mkdir(parents=True, exist_ok=True)
    reference.mkdir(parents=True, exist_ok=True)

    paths = {
        "prices": processed / "prices.parquet",
        "flows_daily": processed / "flows_daily.parquet",
        "participant_flows": processed / "participant_flows.parquet",
        "sectors": reference / "sectors.csv",
        "shares_outstanding": reference / "shares_outstanding.csv",
        "universe_tickers": reference / "universe_tickers.csv",
    }
    data.prices.write_parquet(paths["prices"])
    data.flows_daily.write_parquet(paths["flows_daily"])
    data.participant_flows.write_parquet(paths["participant_flows"])
    data.sectors.write_csv(paths["sectors"])
    data.shares.write_csv(paths["shares_outstanding"])
    data.sectors.select("symbol").write_csv(paths["universe_tickers"])
    sessions_path, holidays_path = ensure_holidays_file(data.calendar, reference)
    paths["trading_sessions"] = sessions_path
    paths["holidays"] = holidays_path

    if write_provenance:
        write_synthetic_provenance(data, root)
    return paths


def write_synthetic_provenance(data: SyntheticData, data_dir: Path) -> Provenance:
    """Mark every generated dataset ``synthetic`` in ``data_dir/PROVENANCE.json``.

    Overwrites whatever was there for these dataset names. A tree that mixes generated
    and real files then reports ``HYBRID: SYNTHETIC ...``, which is the accurate
    description -- and the legacy all-synthetic marker is removed automatically.
    """
    prov = Provenance.load(data_dir)
    source = (
        f"flowalpha.data.synthetic.generate(seed={data.spec.seed}, "
        f"n_symbols={data.spec.n_symbols}, n_sessions={data.spec.n_sessions})"
    )
    note = "GENERATED from a known DGP. Not market data. See flowalpha/data/synthetic.py."
    for name in (
        "prices", "flows_daily", "participant_flows", "flow_features",
        "delivery_pct", "shares_outstanding", "sectors", "universe", "trading_calendar",
    ):
        prov.mark(name, SYNTHETIC, source, note=note)
    prov.add_caveat(
        "SYNTHETIC RUN: all inputs are generated from the DGP in "
        "flowalpha/data/synthetic.py. Results validate that the pipeline recovers "
        "effects that are known to be present; they say nothing about the market."
    )
    prov.add_caveat(
        "Known embedded effects: " + "; ".join(f"{k} -> {v}" for k, v in KNOWN_EFFECTS.items())
    )
    prov.add_caveat(CAVEAT_MULTIPLE_TESTING)
    prov.save(data_dir)
    return prov
