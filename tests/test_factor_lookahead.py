"""Per-factor look-ahead tests.

The store test proves the *access layer* cannot leak. These prove each factor's own
arithmetic cannot either: a rolling window that accidentally centres itself, or a
shift in the wrong direction, would still pass a store-level test.

Method: compute the factor on the full panel, then recompute on a panel whose values
after a cutoff have been replaced with nonsense, and require every value up to the
cutoff to be bit-identical.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import polars as pl
import pytest

from flowalpha.factors.base import FactorContext
from flowalpha.factors.library import (
    AmihudIlliquidity,
    DeliveryRatio,
    LowVolatility,
    Momentum,
    Reversal,
    Size,
    VolumeTrend,
    default_library,
)

from conftest import make_config

CUTOFF_IDX = 60
N_SESSIONS = 100


def _build_ctx(tmp_path, *, poison_after: int | None = None) -> FactorContext:
    sessions = []
    day = _dt.date(2022, 1, 3)
    while len(sessions) < N_SESSIONS:
        if day.weekday() < 5:
            sessions.append(day)
        day += _dt.timedelta(days=1)

    rng = np.random.default_rng(4242)
    n_sym = 5
    steps = rng.normal(0.0004, 0.014, size=(N_SESSIONS, n_sym))
    adj_close = 100.0 * np.cumprod(1.0 + steps, axis=0)
    volume = 1e5 * (1.0 + rng.random((N_SESSIONS, n_sym)))
    delivery = 30.0 + 40.0 * rng.random((N_SESSIONS, n_sym))

    if poison_after is not None:
        adj_close[poison_after + 1 :, :] = 1e9
        volume[poison_after + 1 :, :] = 1e12
        delivery[poison_after + 1 :, :] = 99.0

    turnover = adj_close * volume
    cfg = make_config(tmp_path, sessions)
    return FactorContext(
        sessions=sessions,
        symbols=[f"S{i}" for i in range(n_sym)],
        adj_close=adj_close, close=adj_close.copy(),
        volume=volume, turnover=turnover, delivery_pct=delivery,
        cfg=cfg, as_of_date=sessions[-1],
        shares={f"S{i}": float(i + 1) for i in range(n_sym)},
    )


FACTORS = [
    Momentum(lookback=20, skip=5),
    Reversal(window=5),
    Reversal(window=21),
    LowVolatility(window=20),
    AmihudIlliquidity(window=10),
    DeliveryRatio(window=10),
    Size(shares_available=True),
    VolumeTrend(window=10),
]


@pytest.mark.parametrize("factor", FACTORS, ids=[f.name for f in FACTORS])
def test_factor_values_before_cutoff_are_immune_to_future_data(tmp_path, factor):
    clean = _build_ctx(tmp_path / "clean")
    poisoned = _build_ctx(tmp_path / "poisoned", poison_after=CUTOFF_IDX)
    cutoff = clean.sessions[CUTOFF_IDX]

    a = factor.compute(clean).filter(pl.col("date") <= cutoff).sort(["symbol", "date"])
    b = factor.compute(poisoned).filter(pl.col("date") <= cutoff).sort(["symbol", "date"])
    assert not a.is_empty(), f"{factor.name} produced nothing before the cutoff"
    assert a.equals(b), f"{factor.name} leaked future data into past values"


@pytest.mark.parametrize("factor", FACTORS, ids=[f.name for f in FACTORS])
def test_poisoning_actually_changes_something_after_the_cutoff(tmp_path, factor):
    """Guard against a vacuous pass: the injection must be detectable somewhere."""
    clean = _build_ctx(tmp_path / "clean")
    poisoned = _build_ctx(tmp_path / "poisoned", poison_after=CUTOFF_IDX)
    a = factor.compute(clean)
    b = factor.compute(poisoned)
    assert not a.equals(b), f"{factor.name} ignored the injection entirely"


def test_standardised_panels_are_also_immune(tmp_path):
    """Cross-sectional treatment must not reintroduce leakage.

    It cannot in principle -- winsorisation and z-scoring are within-date -- but a
    single missing ``.over("date")`` would turn them into full-sample operations, so
    the property is asserted rather than assumed.
    """
    clean = _build_ctx(tmp_path / "clean")
    poisoned = _build_ctx(tmp_path / "poisoned", poison_after=CUTOFF_IDX)
    cutoff = clean.sessions[CUTOFF_IDX]
    for factor in default_library(clean.cfg, shares_available=True):
        a = factor.panel(clean).filter(pl.col("date") <= cutoff).sort(["symbol", "date"])
        b = factor.panel(poisoned).filter(pl.col("date") <= cutoff).sort(["symbol", "date"])
        assert a.equals(b), f"{factor.name} standardised panel leaked"


def test_a_centred_window_would_fail_this_test(tmp_path):
    """Guard on the guard: demonstrate the test can actually catch leakage."""
    clean = _build_ctx(tmp_path / "clean")
    poisoned = _build_ctx(tmp_path / "poisoned", poison_after=CUTOFF_IDX)
    cutoff_row = CUTOFF_IDX

    def centred_mean(matrix, window=11):
        half = window // 2
        out = np.full_like(matrix, np.nan)
        for i in range(half, matrix.shape[0] - half):
            out[i] = np.nanmean(matrix[i - half : i + half + 1], axis=0)
        return out

    a = centred_mean(clean.adj_close)[:cutoff_row + 1]
    b = centred_mean(poisoned.adj_close)[:cutoff_row + 1]
    assert not np.allclose(a, b, equal_nan=True), (
        "a centred window must be detectable as look-ahead, or these tests prove nothing"
    )
