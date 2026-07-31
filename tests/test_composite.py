"""IC-weighted composite, with emphasis on its point-in-time discipline."""

from __future__ import annotations

import datetime as _dt

import numpy as np
import polars as pl
import pytest

from flowalpha.signals.composite import (
    CompositeError,
    build_composite,
    compute_weights,
    weight_history,
)
from flowalpha.validation.ic import forward_returns

from conftest import make_config

D = _dt.date


def _sessions(n: int = 200) -> list[_dt.date]:
    out, day = [], D(2022, 1, 3)
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += _dt.timedelta(days=1)
    return out


SESSIONS = _sessions()
SYMBOLS = [f"S{i:02d}" for i in range(30)]


def _panel(values: dict[tuple[_dt.date, str], float]) -> pl.DataFrame:
    rows = [{"date": d, "symbol": s, "value": v} for (d, s), v in values.items()]
    return pl.DataFrame(rows, schema={"date": pl.Date, "symbol": pl.Utf8, "value": pl.Float64})


def _fixture(seed: int = 0, *, good_strength: float = 0.9):
    """Two factors: one genuinely predictive of the FORWARD return, one pure noise.

    The ordering matters. ``good`` on date ``d`` must drive the step from ``d`` to
    ``d+1``, so that it predicts ``fwd_1[d]``. A version that drove ``d``'s own step
    would predict the *contemporaneous* return and carry no forward information at all --
    which the composite would then correctly downweight, making the test assert the
    opposite of what it means to.
    """
    rng = np.random.default_rng(seed)
    good, bad, prices = {}, {}, []
    levels = {s: 100.0 for s in SYMBOLS}
    pending: dict[str, float] | None = None
    for day in SESSIONS:
        for s in SYMBOLS:
            if pending is not None:
                step = good_strength * 0.01 * pending[s] + 0.01 * float(rng.normal())
                levels[s] *= 1.0 + step
            prices.append({"date": day, "symbol": s, "adj_close": levels[s]})
        signal_today = {s: float(rng.normal()) for s in SYMBOLS}
        for s in SYMBOLS:
            good[(day, s)] = signal_today[s]
            bad[(day, s)] = float(rng.normal())
        pending = signal_today
    price_frame = pl.DataFrame(
        prices, schema={"date": pl.Date, "symbol": pl.Utf8, "adj_close": pl.Float64}
    )
    return _panel(good), _panel(bad), price_frame


@pytest.fixture(scope="module")
def fixture():
    return _fixture()


@pytest.fixture
def cfg(tmp_path):
    base = make_config(tmp_path, SESSIONS)
    return base.with_overrides(
        signals={
            **base["signals"],
            "composite": {
                "scheme": "ic_ir", "ic_window": 60, "ic_horizon": 1,
                "min_coverage": 1, "drop_negative_ic": True,
            },
        }
    )


# --- weights ---------------------------------------------------------------

def test_unknown_scheme_rejected(cfg, fixture):
    good, bad, prices = fixture
    bad_cfg = cfg.with_overrides(
        signals={**cfg["signals"], "composite": {**cfg["signals"]["composite"], "scheme": "vibes"}}
    )
    with pytest.raises(CompositeError, match="unknown composite scheme"):
        compute_weights({"g": good}, forward_returns(prices, [1]), bad_cfg)


def test_weights_sum_to_one_in_absolute_value(cfg, fixture):
    good, bad, prices = fixture
    fwd = forward_returns(prices, [1])
    w = compute_weights({"good": good, "bad": bad}, fwd, cfg, sessions=SESSIONS)
    assert not w.weights.is_empty()
    for day in set(w.weights["date"].to_list()):
        total = w.weights.filter(pl.col("date") == day)["weight"].abs().sum()
        assert float(total) == pytest.approx(1.0)


def test_predictive_factor_gets_more_weight_than_noise(cfg, fixture):
    good, bad, prices = fixture
    fwd = forward_returns(prices, [1])
    w = compute_weights({"good": good, "bad": bad}, fwd, cfg, sessions=SESSIONS)
    last = w.weights.filter(pl.col("date") == max(w.weights["date"].to_list()))
    weights = dict(zip(last["factor"].to_list(), last["weight"].to_list()))
    assert weights.get("good", 0.0) > weights.get("bad", 0.0)


def test_no_weights_before_the_window_can_fill(cfg, fixture):
    good, bad, prices = fixture
    fwd = forward_returns(prices, [1])
    w = compute_weights({"good": good}, fwd, cfg, sessions=SESSIONS)
    assert w.weights.filter(pl.col("date") == SESSIONS[0]).is_empty()


def test_weights_are_point_in_time_immune_to_future_data(cfg):
    """Weights on date t must not change when data after t changes."""
    good, bad, prices = _fixture(seed=1)
    fwd = forward_returns(prices, [1])
    cutoff = SESSIONS[120]

    a = compute_weights({"good": good, "bad": bad}, fwd, cfg, sessions=SESSIONS)

    poisoned = prices.with_columns(
        pl.when(pl.col("date") > cutoff).then(pl.lit(1e6)).otherwise(pl.col("adj_close")).alias("adj_close")
    )
    b = compute_weights(
        {"good": good, "bad": bad}, forward_returns(poisoned, [1]), cfg, sessions=SESSIONS
    )
    upto_a = a.weights.filter(pl.col("date") <= cutoff).sort(["date", "factor"])
    upto_b = b.weights.filter(pl.col("date") <= cutoff).sort(["date", "factor"])
    assert not upto_a.is_empty()
    assert upto_a.equals(upto_b)


def test_weights_ignore_ic_whose_label_has_not_realised(tmp_path):
    """With horizon h, the IC dated t only becomes usable at t+h.

    Setting ic_window to 1 makes the point sharp: with a window of one session, the only
    candidate IC observation for date t is t's own, which is not yet knowable, so no
    weight can be produced at all.
    """
    base = make_config(tmp_path, SESSIONS)
    cfg = base.with_overrides(
        signals={
            **base["signals"],
            "composite": {"scheme": "ic_mean", "ic_window": 1, "ic_horizon": 5,
                          "min_coverage": 1, "drop_negative_ic": False},
        }
    )
    good, _, prices = _fixture(seed=2)
    w = compute_weights({"good": good}, forward_returns(prices, [5]), cfg, sessions=SESSIONS)
    assert w.weights.is_empty()


def test_equal_scheme_gives_identical_weights(cfg, fixture):
    good, bad, prices = fixture
    eq = cfg.with_overrides(
        signals={**cfg["signals"], "composite": {**cfg["signals"]["composite"], "scheme": "equal"}}
    )
    w = compute_weights({"good": good, "bad": bad}, forward_returns(prices, [1]), eq,
                        sessions=SESSIONS)
    last = w.weights.filter(pl.col("date") == max(w.weights["date"].to_list()))
    assert last["weight"].to_list() == pytest.approx([0.5, 0.5])


def test_equal_scheme_never_drops_a_factor(cfg, fixture):
    good, bad, prices = fixture
    eq = cfg.with_overrides(
        signals={**cfg["signals"], "composite": {**cfg["signals"]["composite"], "scheme": "equal"}}
    )
    w = compute_weights({"good": good, "bad": bad}, forward_returns(prices, [1]), eq,
                        sessions=SESSIONS)
    assert w.n_dropped_negative == 0


def test_drop_negative_ic_removes_factors(cfg, fixture):
    good, bad, prices = fixture
    fwd = forward_returns(prices, [1])
    dropping = compute_weights({"good": good, "bad": bad}, fwd, cfg, sessions=SESSIONS)
    keep_cfg = cfg.with_overrides(
        signals={**cfg["signals"],
                 "composite": {**cfg["signals"]["composite"], "drop_negative_ic": False}}
    )
    keeping = compute_weights({"good": good, "bad": bad}, fwd, keep_cfg, sessions=SESSIONS)
    assert dropping.n_dropped_negative > 0
    assert keeping.n_dropped_negative == 0
    assert dropping.weights.height < keeping.weights.height


def test_nan_ic_from_a_flat_factor_does_not_empty_the_composite(cfg, fixture):
    """A gated factor is deliberately flat on its OFF dates, so its IC is undefined
    there. One NaN weight used to null the weighted sum for EVERY name on that date,
    emptying the whole composite -- a failure that reads as "no signal"."""
    good, bad, prices = fixture
    flat = good.with_columns(pl.lit(0.0).alias("value"))
    fwd = forward_returns(prices, [1])
    w = compute_weights({"good": good, "flat": flat}, fwd, cfg, sessions=SESSIONS)
    assert not w.weights.is_empty()
    assert np.isfinite(np.asarray(w.weights["weight"].to_list(), dtype=float)).all()
    signal = build_composite({"good": good, "flat": flat}, w, cfg)
    assert not signal.is_empty()


def test_no_factors_gives_typed_empty_weights(cfg, fixture):
    _, _, prices = fixture
    w = compute_weights({}, forward_returns(prices, [1]), cfg, sessions=SESSIONS)
    assert w.weights.is_empty()
    assert w.weights.schema["date"] == pl.Date
    assert w.factors == ()


def test_weights_to_dict_records_the_scheme(cfg, fixture):
    good, _, prices = fixture
    w = compute_weights({"good": good}, forward_returns(prices, [1]), cfg, sessions=SESSIONS)
    d = w.to_dict()
    assert d["scheme"] == "ic_ir" and d["ic_window"] == 60 and d["ic_horizon"] == 1


def test_weights_on_helper(cfg, fixture):
    good, bad, prices = fixture
    w = compute_weights({"good": good, "bad": bad}, forward_returns(prices, [1]), cfg,
                        sessions=SESSIONS)
    day = max(w.weights["date"].to_list())
    assert set(w.on(day)) <= {"good", "bad"}
    assert w.on(SESSIONS[0]) == {}


# --- composite signal ------------------------------------------------------

def test_composite_is_standardised_per_date(cfg, fixture):
    good, bad, prices = fixture
    fwd = forward_returns(prices, [1])
    w = compute_weights({"good": good, "bad": bad}, fwd, cfg, sessions=SESSIONS)
    signal = build_composite({"good": good, "bad": bad}, w, cfg)
    for day in list(set(signal["date"].to_list()))[:5]:
        vals = np.asarray(
            signal.filter(pl.col("date") == day)["value"].to_list(), dtype=float
        )
        assert abs(float(vals.mean())) < 1e-9
        assert float(vals.std(ddof=1)) == pytest.approx(1.0)


def test_min_coverage_excludes_thinly_scored_names(tmp_path, fixture):
    """A stock scored from one factor is not comparable with one scored from seven."""
    good, bad, prices = fixture
    base = make_config(tmp_path, SESSIONS)
    strict = base.with_overrides(
        signals={**base["signals"],
                 "composite": {"scheme": "equal", "ic_window": 60, "ic_horizon": 1,
                               "min_coverage": 2, "drop_negative_ic": False}}
    )
    fwd = forward_returns(prices, [1])
    # Remove one symbol from the second factor entirely.
    thin = bad.filter(pl.col("symbol") != "S00")
    w = compute_weights({"good": good, "bad": thin}, fwd, strict, sessions=SESSIONS)
    signal = build_composite({"good": good, "bad": thin}, w, strict)
    assert "S00" not in set(signal["symbol"].to_list())
    assert signal.height > 0


def test_composite_with_no_weights_is_typed_empty(cfg, fixture):
    good, _, prices = fixture
    empty_w = compute_weights({}, forward_returns(prices, [1]), cfg, sessions=SESSIONS)
    signal = build_composite({"good": good}, empty_w, cfg)
    assert signal.is_empty() and signal.schema["symbol"] == pl.Utf8


def test_composite_with_no_panels_is_typed_empty(cfg, fixture):
    good, _, prices = fixture
    w = compute_weights({"good": good}, forward_returns(prices, [1]), cfg, sessions=SESSIONS)
    assert build_composite({}, w, cfg).is_empty()


def test_composite_beats_the_noise_factor_on_ic(cfg, fixture):
    """The composite should end up closer to the good factor than to the bad one."""
    from flowalpha.validation.ic import rank_ic, summarize_ic

    good, bad, prices = fixture
    fwd = forward_returns(prices, [1])
    w = compute_weights({"good": good, "bad": bad}, fwd, cfg, sessions=SESSIONS)
    signal = build_composite({"good": good, "bad": bad}, w, cfg)
    ic_comp = summarize_ic(rank_ic(signal, fwd, 1), factor="c", horizon=1).mean
    ic_bad = summarize_ic(rank_ic(bad, fwd, 1), factor="b", horizon=1).mean
    assert ic_comp > ic_bad


# --- history ---------------------------------------------------------------

def test_weight_history_is_wide_and_null_filled(cfg, fixture):
    good, bad, prices = fixture
    w = compute_weights({"good": good, "bad": bad}, forward_returns(prices, [1]), cfg,
                        sessions=SESSIONS)
    history = weight_history(w)
    assert "date" in history.columns
    assert history.null_count().sum_horizontal().to_list()[0] == 0
    assert history["date"].is_sorted()


def test_weight_history_of_empty_weights(cfg, fixture):
    _, _, prices = fixture
    empty = compute_weights({}, forward_returns(prices, [1]), cfg, sessions=SESSIONS)
    assert weight_history(empty).is_empty()
