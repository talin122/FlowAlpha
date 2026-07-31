"""Factor cards: the Sharpe and its trial count must be inseparable."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from flowalpha.conditioning.analysis import ConditionalIC
from flowalpha.signals.factor_card import FactorCard, cards_to_frame
from flowalpha.validation.deflated_sharpe import deflated_sharpe_ratio
from flowalpha.validation.ic import ICSummary


def _ic(horizon: int, mean: float = 0.02) -> ICSummary:
    return ICSummary(
        factor="f", horizon=horizon, n_dates=1000, mean=mean, std=0.1,
        ic_ir=mean / 0.1, t_stat=mean / 0.003, nw_lags=4, hit_rate=0.55, mean_names=400,
    )


def _dsr(n_trials: int = 7):
    rng = np.random.default_rng(0)
    return deflated_sharpe_ratio(0.001 + 0.01 * rng.normal(size=1000), n_trials)


@pytest.fixture
def card() -> FactorCard:
    return FactorCard(
        name="momentum",
        hypothesis="Names that rose keep rising.",
        ic=[_ic(1, 0.005), _ic(5, 0.017), _ic(21, 0.028)],
        conditional_ic=[
            ConditionalIC("momentum", "fii_regime", "high", 589, 0.013, 0.1, 0.13, 1.2, 0.53),
            ConditionalIC("momentum", "fii_regime", "low", 592, 0.025, 0.1, 0.25, 2.1, 0.56),
        ],
        backtest={"gross_sharpe": 1.0, "net_sharpe": -0.13, "annual_turnover": 9.0,
                  "total_cost_bps": 713.0},
        dsr=_dsr(7),
        n_trials=7,
        notes=["delivery data unavailable"],
    )


def test_ic_at_selects_the_horizon(card):
    assert card.ic_at(5).mean == pytest.approx(0.017)
    assert card.ic_at(99) is None


def test_headline_ic_is_the_shortest_horizon(card):
    assert card.headline_ic.horizon == 1


def test_headline_ic_of_an_empty_card():
    assert FactorCard(name="x", hypothesis="").headline_ic is None


def test_to_dict_keeps_trial_count_with_the_sharpe(card):
    """A Sharpe without its trial count is not a result."""
    d = card.to_dict()
    assert d["n_trials_at_evaluation"] == 7
    assert d["deflated_sharpe"]["n_trials"] == 7
    assert d["backtest"]["net_sharpe"] == pytest.approx(-0.13)


def test_to_dict_marks_raw_ic_as_not_deflated(card):
    d = card.to_dict()
    for entry in d["ic"]:
        assert entry["deflated"] is False
    for entry in d["conditional_ic"]:
        assert entry["deflated"] is False


def test_summary_lines_are_ascii(card):
    text = "\n".join(card.summary_lines())
    text.encode("ascii")


def test_summary_lines_label_raw_ic(card):
    text = "\n".join(card.summary_lines())
    assert "RAW, not deflated" in text
    assert "cumulative" in text


def test_summary_lines_include_hypothesis_and_notes(card):
    text = "\n".join(card.summary_lines())
    assert "Names that rose keep rising." in text
    assert "delivery data unavailable" in text


def test_summary_lines_include_conditional_ic(card):
    text = "\n".join(card.summary_lines())
    assert "cond IC fii_regime=high" in text
    assert "cond IC fii_regime=low" in text


def test_summary_lines_of_a_bare_card():
    text = "\n".join(FactorCard(name="x", hypothesis="h").summary_lines())
    assert "x" in text and "h" in text


def test_nan_values_render_as_na():
    card = FactorCard(
        name="x", hypothesis="h",
        ic=[ICSummary("x", 1, 0, float("nan"), float("nan"), float("nan"),
                      float("nan"), 0, float("nan"), float("nan"))],
    )
    text = "\n".join(card.summary_lines())
    assert "n/a" in text
    assert "nan" not in text.lower().replace("n/a", "")


def test_cards_to_frame_pairs_sharpe_with_its_correction(card):
    frame = cards_to_frame([card], horizon=5)
    assert frame.height == 1
    row = frame.to_dicts()[0]
    assert row["factor"] == "momentum"
    assert row["n_trials"] == 7
    assert math.isfinite(row["deflated_sharpe_prob"])
    assert row["t_stat_newey_west_raw"] == pytest.approx(card.ic_at(5).t_stat)


def test_cards_to_frame_column_names_say_raw(card):
    """The column name itself carries the caveat, so a copied cell keeps it."""
    frame = cards_to_frame([card], horizon=5)
    assert "t_stat_newey_west_raw" in frame.columns
    assert "deflated_sharpe_prob" in frame.columns


def test_cards_to_frame_missing_horizon_is_nan(card):
    row = cards_to_frame([card], horizon=99).to_dicts()[0]
    assert math.isnan(row["mean_ic"])
    assert row["n_dates"] == 0


def test_cards_to_frame_empty_keeps_schema():
    frame = cards_to_frame([], horizon=5)
    assert frame.is_empty()
    assert frame.schema["factor"] == pl.Utf8
    assert "n_trials" in frame.columns


def test_cards_to_frame_is_sorted(card):
    other = FactorCard(name="aaa", hypothesis="h", ic=[_ic(5)])
    frame = cards_to_frame([card, other], horizon=5)
    assert frame["factor"].to_list() == ["aaa", "momentum"]


def test_card_without_backtest_still_renders():
    card = FactorCard(name="x", hypothesis="h", ic=[_ic(5)])
    frame = cards_to_frame([card], horizon=5)
    assert math.isnan(frame["net_sharpe"][0])
    assert card.to_dict()["backtest"] is None
