"""The factor zoo.

Sign convention: **a higher factor value means a higher expected return** under the
factor's own hypothesis. So reversal is the *negative* of the recent return, and low
volatility is the *negative* of realised volatility. Without a uniform convention,
an IC table becomes a table of signs the reader has to remember, and a composite that
sums factors becomes meaningless.

Each factor states its hypothesis in a class attribute. Those strings appear in
reports, so a reader can see what was being bet on rather than inferring it from a
formula.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..config import Config
from .base import (
    Factor,
    FactorContext,
    empty_panel,
    rolling_mean,
    rolling_std,
    shift_rows,
)


class Momentum(Factor):
    """Cross-sectional price momentum, skipping the most recent month.

    ``value[t] = P[t - skip] / P[t - skip - lookback] - 1``

    The skip is the point of the factor: the most recent month of returns carries
    short-horizon reversal, which is a *different* effect with the opposite sign.
    Including it contaminates momentum with its own opposite.
    """

    hypothesis = "Names that rose over the past year (excluding the last month) keep rising."
    requires = ("adj_close",)

    def __init__(self, lookback: int = 252, skip: int = 21, name: str | None = None) -> None:
        if lookback <= 0 or skip < 0:
            raise ValueError("lookback must be positive and skip non-negative")
        self.lookback = int(lookback)
        self.skip = int(skip)
        self.name = name or "momentum"

    def compute(self, ctx: FactorContext) -> pl.DataFrame:
        if not self.is_available(ctx) or ctx.n_sessions <= self.lookback + self.skip:
            return empty_panel()
        end = shift_rows(ctx.adj_close, self.skip)
        start = shift_rows(ctx.adj_close, self.skip + self.lookback)
        with np.errstate(divide="ignore", invalid="ignore"):
            value = np.where(start > 0, end / start - 1.0, np.nan)
        return ctx.to_long(value)


class Reversal(Factor):
    """Short-horizon reversal: the negative of the trailing ``window``-day return.

    Sign is flipped so that a high value means "expected to outperform", consistent
    with every other factor here.
    """

    hypothesis = "Recent losers bounce; recent winners give back, over days to a month."
    requires = ("adj_close",)

    def __init__(self, window: int = 5, name: str | None = None) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = int(window)
        self.name = name or f"reversal_{self.window}d"

    def compute(self, ctx: FactorContext) -> pl.DataFrame:
        if not self.is_available(ctx) or ctx.n_sessions <= self.window:
            return empty_panel()
        past = shift_rows(ctx.adj_close, self.window)
        with np.errstate(divide="ignore", invalid="ignore"):
            trailing = np.where(past > 0, ctx.adj_close / past - 1.0, np.nan)
        return ctx.to_long(-trailing)


class LowVolatility(Factor):
    """Negative realised volatility of daily returns over ``window`` sessions.

    High value = low volatility, so the low-volatility anomaly predicts positive IC.
    """

    hypothesis = "Low-volatility names earn more than their risk justifies."
    requires = ("adj_close",)

    def __init__(self, window: int = 60, name: str | None = None) -> None:
        self.window = int(window)
        self.name = name or "low_volatility"

    def compute(self, ctx: FactorContext) -> pl.DataFrame:
        if not self.is_available(ctx) or ctx.n_sessions <= self.window:
            return empty_panel()
        # min_valid at 60% of the window: a name that listed mid-window still has a
        # meaningful volatility estimate, and demanding a full window would throw
        # away the whole early sample for every recent listing.
        vol = rolling_std(ctx.returns, self.window, min_valid=max(2, int(0.6 * self.window)))
        return ctx.to_long(-vol)


class AmihudIlliquidity(Factor):
    """Amihud illiquidity: mean of ``|return| / rupee turnover`` over ``window``.

    Left in its natural sign -- a high value means *illiquid* -- because the
    hypothesis under test is an illiquidity premium. If that premium is absent in this
    universe the IC will simply be negative, which is a result rather than a bug.
    """

    hypothesis = "Illiquid names pay a premium for the illiquidity they impose."
    requires = ("adj_close", "turnover")

    def __init__(self, window: int = 21, name: str | None = None) -> None:
        self.window = int(window)
        self.name = name or "illiquidity"

    def compute(self, ctx: FactorContext) -> pl.DataFrame:
        if not self.is_available(ctx) or ctx.n_sessions <= self.window:
            return empty_panel()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(ctx.turnover > 0, np.abs(ctx.returns) / ctx.turnover, np.nan)
        illiq = rolling_mean(ratio, self.window, min_valid=max(2, int(0.6 * self.window)))
        # Scale for readability; monotone, so ranks and therefore IC are unchanged.
        return ctx.to_long(illiq * 1e9)


class DeliveryRatio(Factor):
    """Mean delivery percentage over ``window`` sessions.

    **Inert on Yahoo-sourced data**, which carries no delivery field: this returns an
    empty panel, and consumers skip it with a printed warning rather than counting it
    as a trial. The factor is kept because NSE bhavcopy ingestion does provide the
    field, and because an absent factor should be visibly absent.
    """

    hypothesis = "High delivery-based volume means conviction buying, not churn."
    requires = ("delivery_pct",)

    def __init__(self, window: int = 21, name: str | None = None) -> None:
        self.window = int(window)
        self.name = name or "delivery_ratio"

    def compute(self, ctx: FactorContext) -> pl.DataFrame:
        if not self.is_available(ctx) or ctx.n_sessions <= self.window:
            return empty_panel()
        avg = rolling_mean(ctx.delivery_pct, self.window, min_valid=max(2, int(0.6 * self.window)))
        return ctx.to_long(avg)


class Size(Factor):
    """Log market capitalisation, or log price when share counts are unavailable.

    The name changes with the input, on purpose. With ``shares_available=False`` the
    factor is ``log(price)`` and is called ``size_logprice_proxy``, because calling a
    log-price series "size" invites a reader to believe a small-cap result that the
    data cannot support. Price level and market cap are only loosely related: a
    high-priced small company and a low-priced large one are both common.
    """

    hypothesis = "Smaller companies out-earn larger ones (so expect NEGATIVE IC here)."
    requires = ("close",)

    def __init__(self, *, shares_available: bool = False, name: str | None = None) -> None:
        self.shares_available = bool(shares_available)
        self.name = name or ("size_logmktcap" if shares_available else "size_logprice_proxy")

    def compute(self, ctx: FactorContext) -> pl.DataFrame:
        if not self.is_available(ctx):
            return empty_panel()
        shares = ctx.shares_vector if self.shares_available else np.ones(ctx.n_symbols)
        with np.errstate(divide="ignore", invalid="ignore"):
            mktcap = ctx.close * shares[None, :]
            value = np.where(mktcap > 0, np.log(mktcap), np.nan)
        return ctx.to_long(value)


class VolumeTrend(Factor):
    """Log ratio of recent to prior average volume.

    ``value[t] = log(mean(V[t-w+1 .. t]) / mean(V[t-2w+1 .. t-w]))``

    Rising participation as a precursor to continued moves. Uses raw (unadjusted)
    volume: the log *ratio* of two means is invariant to a split factor applied to
    both, so adjusting would change nothing except where the split sits inside the
    window -- and there the unadjusted ratio is the one that reflects what actually
    traded.
    """

    hypothesis = "Names attracting rising volume continue to attract flow."
    requires = ("volume",)

    def __init__(self, window: int = 21, name: str | None = None) -> None:
        self.window = int(window)
        self.name = name or "volume_trend"

    def compute(self, ctx: FactorContext) -> pl.DataFrame:
        if not self.is_available(ctx) or ctx.n_sessions <= 2 * self.window:
            return empty_panel()
        min_valid = max(2, int(0.6 * self.window))
        recent = rolling_mean(ctx.volume, self.window, min_valid=min_valid)
        prior = shift_rows(recent, self.window)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where((prior > 0) & (recent > 0), recent / prior, np.nan)
            value = np.log(ratio)
        return ctx.to_long(value)


def default_library(cfg: Config, *, shares_available: bool = False) -> list[Factor]:
    """The configured factor set.

    Every window and lookback comes from ``config.yaml``; nothing here is tuned in
    code. ``shares_available`` controls the size factor's identity -- pass True only
    when a genuine share-count series is present, or the resulting factor will be
    named for a quantity it does not measure.
    """
    fcfg = cfg["factors"]
    mom = fcfg["momentum"]
    factors: list[Factor] = [
        Momentum(lookback=int(mom["lookback"]), skip=int(mom["skip"])),
    ]
    for window in fcfg["reversal"]["windows"]:
        factors.append(Reversal(window=int(window)))
    factors += [
        LowVolatility(window=int(fcfg["volatility"]["window"])),
        AmihudIlliquidity(window=int(fcfg["liquidity"]["window"])),
        DeliveryRatio(window=int(fcfg["delivery"]["window"])),
        Size(shares_available=shares_available),
        VolumeTrend(window=int(fcfg["volume_trend"]["window"])),
    ]
    return factors


def build_panels(
    factors: list[Factor],
    ctx: FactorContext,
    *,
    verbose: bool = True,
) -> tuple[dict[str, pl.DataFrame], list[str]]:
    """Compute every factor's standardised panel.

    Factors with no usable input are **skipped with a printed warning** and returned
    in the second element. They are not silently dropped, and -- importantly -- they
    are not counted as trials, because an unevaluated factor is not a test.
    """
    panels: dict[str, pl.DataFrame] = {}
    skipped: list[str] = []
    for factor in factors:
        panel = factor.panel(ctx)
        if panel.is_empty():
            reason = factor.unavailable_reason(ctx) if not factor.is_available(ctx) else (
                f"{factor.name}: insufficient history "
                f"({ctx.n_sessions} sessions) for its window"
            )
            skipped.append(reason)
            if verbose:
                print(f"WARNING: skipping factor -- {reason}", flush=True)
            continue
        panels[factor.name] = panel
    return panels, skipped
