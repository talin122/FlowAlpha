"""Alpha miner: grammar, causality, policies, and trial registration."""

from __future__ import annotations

import datetime as _dt

import numpy as np
import polars as pl
import pytest

from flowalpha.mining.grammar import (
    FLOW_TERMINALS,
    PRICE_TERMINALS,
    Formula,
    GrammarError,
    build_vocabulary,
    evaluate,
    is_complete,
    legal_next,
    terminal_matrices,
)
from flowalpha.mining.miner import (
    MiningError,
    action_space_size,
    formula_reward,
    mine,
    novelty_penalty,
    register_candidates,
    sample_formula,
    signal_from_formula,
)
from flowalpha.mining.policy import (
    PolicyError,
    RandomPolicy,
    make_policy,
    torch_available,
)
from flowalpha.validation.deflated_sharpe import TrialRegistry
from flowalpha.validation.ic import forward_returns

from conftest import make_config

D = _dt.date
requires_torch = pytest.mark.skipif(not torch_available(), reason="mining extra not installed")


# --- grammar ---------------------------------------------------------------

def test_vocabularies_differ_in_flow_terminals():
    price = {t.name for t in build_vocabulary("price_only") if t.kind == "terminal"}
    flow = {t.name for t in build_vocabulary("flow_augmented") if t.kind == "terminal"}
    assert price == set(PRICE_TERMINALS)
    assert flow == set(PRICE_TERMINALS) | set(FLOW_TERMINALS)
    assert flow > price


def test_unknown_vocabulary_rejected():
    with pytest.raises(GrammarError, match="unknown vocabulary"):
        build_vocabulary("everything")


def test_vocabulary_order_is_stable():
    """The order IS the policy's action space; a checkpoint means nothing otherwise."""
    a = [t.name for t in build_vocabulary("flow_augmented")]
    b = [t.name for t in build_vocabulary("flow_augmented")]
    assert a == b


def test_every_operator_is_causal():
    """There is deliberately no operator that can see the future.

    A mined formula is only trustworthy if the grammar cannot express look-ahead, so the
    absence of a forward-shift is asserted rather than assumed.
    """
    names = {t.name for t in build_vocabulary("flow_augmented")}
    for name in names:
        assert not name.startswith("lead")
        assert not name.startswith("future")
    assert any(n.startswith("delay") for n in names)


def test_legal_next_prevents_unparseable_sequences():
    vocab = build_vocabulary("price_only")
    mask = legal_next([], vocab, 10)
    for token, allowed in zip(vocab, mask):
        if allowed:
            assert token.arity == 0  # nothing on the stack yet


def test_legal_next_allows_a_unary_after_one_terminal():
    vocab = build_vocabulary("price_only")
    mask = legal_next(["close"], vocab, 10)
    allowed = {t.name for t, ok in zip(vocab, mask) if ok}
    assert "neg" in allowed
    assert "add" not in allowed  # needs two operands


def test_legal_next_allows_a_binary_after_two_terminals():
    vocab = build_vocabulary("price_only")
    allowed = {t.name for t, ok in zip(vocab, legal_next(["close", "volume"], vocab, 10)) if ok}
    assert "add" in allowed


def test_legal_next_respects_the_length_budget():
    vocab = build_vocabulary("price_only")
    # With one slot left and two values on the stack, only a binary op can close it.
    allowed = {t.name for t, ok in zip(vocab, legal_next(["close", "volume"], vocab, 3)) if ok}
    assert allowed and all(
        next(t.arity for t in vocab if t.name == n) == 2 for n in allowed
    )


def test_is_complete():
    vocab = build_vocabulary("price_only")
    assert is_complete(["close"], vocab)
    assert is_complete(["close", "neg"], vocab)
    assert is_complete(["close", "volume", "add"], vocab)
    assert not is_complete(["close", "volume"], vocab)
    assert not is_complete([], vocab)


def test_formula_infix_is_readable():
    f = Formula(tokens=("close", "volume", "add", "neg"), vocabulary="price_only")
    assert f.infix() == "neg(add(close, volume))"
    assert str(f) == "close volume add neg"
    assert f.length == 4


def test_formula_infix_of_a_broken_sequence_falls_back_to_rpn():
    f = Formula(tokens=("add",), vocabulary="price_only")
    assert f.infix() == "add"


# --- evaluation ------------------------------------------------------------

@pytest.fixture
def ctx(tmp_path):
    from flowalpha.factors.base import FactorContext

    sessions = []
    day = D(2022, 1, 3)
    while len(sessions) < 200:
        if day.weekday() < 5:
            sessions.append(day)
        day += _dt.timedelta(days=1)
    rng = np.random.default_rng(3)
    n, m = len(sessions), 20
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.015, size=(n, m)), axis=0)
    volume = 1e5 * (1.0 + rng.random((n, m)))
    return FactorContext(
        sessions=sessions, symbols=[f"S{i:02d}" for i in range(m)],
        adj_close=close, close=close.copy(), volume=volume,
        turnover=close * volume, delivery_pct=np.full((n, m), np.nan),
        cfg=make_config(tmp_path, sessions), as_of_date=sessions[-1],
    )


def test_evaluate_a_simple_formula(ctx):
    vocab = build_vocabulary("price_only")
    terminals = terminal_matrices(ctx)
    got = evaluate(Formula(("close", "neg"), "price_only"), terminals, vocab)
    np.testing.assert_allclose(got, -ctx.adj_close)


def test_evaluate_a_binary_formula(ctx):
    vocab = build_vocabulary("price_only")
    terminals = terminal_matrices(ctx)
    got = evaluate(Formula(("close", "volume", "mul"), "price_only"), terminals, vocab)
    np.testing.assert_allclose(got, ctx.adj_close * ctx.volume)


def test_evaluate_delay_is_backward_looking(ctx):
    vocab = build_vocabulary("price_only")
    terminals = terminal_matrices(ctx)
    got = evaluate(Formula(("close", "delay5"), "price_only"), terminals, vocab)
    assert np.isnan(got[:5]).all()
    np.testing.assert_allclose(got[5:], ctx.adj_close[:-5])


def test_evaluate_rejects_unknown_tokens(ctx):
    with pytest.raises(GrammarError, match="unknown token"):
        evaluate(Formula(("nope",), "price_only"), terminal_matrices(ctx))


def test_evaluate_rejects_incomplete_formulas(ctx):
    with pytest.raises(GrammarError, match="single value"):
        evaluate(Formula(("close", "volume"), "price_only"), terminal_matrices(ctx))


def test_evaluate_rejects_missing_operands(ctx):
    with pytest.raises(GrammarError, match="needs"):
        evaluate(Formula(("add",), "price_only"), terminal_matrices(ctx))


def test_flow_terminals_are_null_without_flow_features(ctx):
    terminals = terminal_matrices(ctx, flow_features=None)
    for name in FLOW_TERMINALS:
        assert np.isnan(terminals[name]).all()


def test_flow_terminals_are_broadcast_across_symbols(ctx):
    from flowalpha.factors.flow_features import FLOW_FEATURES_SCHEMA

    n = ctx.n_sessions
    features = pl.DataFrame(
        {
            "date": ctx.sessions,
            "fii_flow_21d": np.arange(float(n)),
            "dii_flow_21d": [None] * n,
            "retail_flow_21d": [None] * n,
            "fii_daily": [None] * n,
            "retail_daily": [None] * n,
            "fii_regime": [None] * n,
            "retail_regime": [None] * n,
            "retail_extreme": [None] * n,
            "flow_divergence": [None] * n,
            "flow_shock": [None] * n,
        },
        schema=FLOW_FEATURES_SCHEMA,
    )
    terminals = terminal_matrices(ctx, flow_features=features)
    fii = terminals["fii_flow"]
    assert fii.shape == (ctx.n_sessions, ctx.n_symbols)
    # Market-wide series: constant within a date.
    assert np.allclose(fii, fii[:, [0]])


def test_signal_from_formula_standardises(ctx):
    vocab = build_vocabulary("price_only")
    panel = signal_from_formula(
        Formula(("close",), "price_only"), terminal_matrices(ctx), ctx, vocab
    )
    assert panel is not None
    day = ctx.sessions[100]
    vals = np.asarray(panel.filter(pl.col("date") == day)["value"].to_list(), dtype=float)
    assert abs(float(vals.mean())) < 1e-9


def test_signal_from_an_all_null_formula_is_none(ctx):
    vocab = build_vocabulary("flow_augmented")
    panel = signal_from_formula(
        Formula(("fii_flow",), "flow_augmented"), terminal_matrices(ctx), ctx, vocab
    )
    assert panel is None


# --- policies --------------------------------------------------------------

def test_random_policy_samples_only_legal_actions():
    policy = RandomPolicy(5, seed=0)
    mask = np.array([False, True, False, True, False])
    for _ in range(20):
        action, log_prob, entropy = policy.sample([], mask)
        assert mask[action]
        assert log_prob < 0 and entropy > 0


def test_random_policy_raises_on_an_empty_mask():
    with pytest.raises(PolicyError, match="no legal action"):
        RandomPolicy(3, seed=0).sample([], np.zeros(3, dtype=bool))


def test_random_policy_is_deterministic_in_its_seed():
    mask = np.ones(5, dtype=bool)
    a = [RandomPolicy(5, seed=1).sample([], mask)[0] for _ in range(5)]
    b = [RandomPolicy(5, seed=1).sample([], mask)[0] for _ in range(5)]
    assert a == b


def test_random_policy_is_labelled_a_control_not_a_fallback():
    assert RandomPolicy(3).name == "random"
    assert RandomPolicy(3).trainable is False


def test_make_policy_rejects_unknown_names():
    with pytest.raises(PolicyError, match="unknown policy"):
        make_policy("magic", 5)


@pytest.mark.skipif(torch_available(), reason="torch is installed")
def test_gru_policy_refuses_to_downgrade_silently():
    """A random search reported as a trained policy would misdescribe the numbers."""
    with pytest.raises(PolicyError, match="torch"):
        make_policy("gru", 5)


@requires_torch
def test_gru_policy_samples_only_legal_actions(tmp_path):
    cfg = make_config(tmp_path, [D(2022, 1, 3)])
    policy = make_policy("gru", 6, cfg, seed=0)
    policy.reset_episode()
    mask = np.array([False, True, True, False, False, False])
    for _ in range(10):
        action, _, _ = policy.sample([], mask)
        assert mask[action]


@requires_torch
def test_gru_policy_update_changes_parameters(tmp_path):
    cfg = make_config(tmp_path, [D(2022, 1, 3)])
    policy = make_policy("gru", 6, cfg, seed=0)
    before = [p.detach().clone() for p in policy.params]
    terms = []
    for _ in range(4):
        policy.reset_episode()
        policy.sample([], np.ones(6, dtype=bool))
        terms.append(policy.finish_episode())
    diag = policy.update_batch(terms, [0.5, -0.2, 0.1, 0.3])
    assert "loss" in diag and "baseline" in diag
    assert any((a != b).any() for a, b in zip(before, policy.params))


@requires_torch
def test_gru_policy_clips_rewards(tmp_path):
    """One episode with an enormous IC would otherwise dominate the whole update."""
    cfg = make_config(tmp_path, [D(2022, 1, 3)])
    policy = make_policy("gru", 6, cfg, seed=0)
    terms = []
    for _ in range(2):
        policy.reset_episode()
        policy.sample([], np.ones(6, dtype=bool))
        terms.append(policy.finish_episode())
    diag = policy.update_batch(terms, [1e9, -1e9])
    assert abs(diag["mean_reward_clipped"]) <= policy.reward_clip


# --- sampling --------------------------------------------------------------

def test_action_space_includes_a_stop_action():
    """Without STOP the policy is forced to emit max_formula_len tokens every episode,
    so the cap becomes a requirement and the search only ever sees long formulas."""
    vocab = build_vocabulary("price_only")
    assert action_space_size(vocab) == len(vocab) + 1


def test_sampled_formulas_are_always_complete():
    vocab = build_vocabulary("price_only")
    policy = RandomPolicy(action_space_size(vocab), seed=4)
    for _ in range(50):
        formula, rollout = sample_formula(policy, vocab, 12)
        assert is_complete(formula.tokens, vocab)
        assert len(rollout.actions) >= len(formula.tokens)


def test_sampled_formulas_vary_in_length():
    vocab = build_vocabulary("price_only")
    policy = RandomPolicy(action_space_size(vocab), seed=5)
    lengths = {sample_formula(policy, vocab, 14)[0].length for _ in range(60)}
    assert len(lengths) > 1
    assert min(lengths) < 14


def test_sampled_formulas_respect_the_length_cap():
    vocab = build_vocabulary("price_only")
    policy = RandomPolicy(action_space_size(vocab), seed=6)
    for _ in range(40):
        assert sample_formula(policy, vocab, 8)[0].length <= 8


# --- reward ----------------------------------------------------------------

def _panel_and_fwd(ctx):
    vocab = build_vocabulary("price_only")
    panel = signal_from_formula(
        Formula(("close", "delta5", "neg"), "price_only"),
        terminal_matrices(ctx), ctx, vocab,
    )
    prices = pl.DataFrame(
        {
            "date": np.repeat(np.array(ctx.sessions, dtype="object"), ctx.n_symbols).tolist(),
            "symbol": np.tile(np.array(ctx.symbols, dtype="object"), ctx.n_sessions).tolist(),
            "adj_close": ctx.adj_close.reshape(-1),
        },
        schema={"date": pl.Date, "symbol": pl.Utf8, "adj_close": pl.Float64},
    )
    return panel, forward_returns(prices, [5])


def test_pooled_reward_is_the_mean_ic(ctx):
    panel, fwd = _panel_and_fwd(ctx)
    reward, ic, t_stat, n_dates, per_regime = formula_reward(panel, fwd, 5, kind="pooled")
    assert reward == pytest.approx(ic)
    assert n_dates > 0
    assert per_regime == {}


def test_reward_of_a_too_short_series_is_nan(ctx):
    panel, fwd = _panel_and_fwd(ctx)
    reward, *_ = formula_reward(panel, fwd, 5, kind="pooled", min_dates=10_000)
    assert np.isnan(reward)


def test_regime_robust_reward_needs_regimes(ctx):
    """Falling back to pooled would mislabel the objective."""
    panel, fwd = _panel_and_fwd(ctx)
    with pytest.raises(MiningError, match="regime_robust"):
        formula_reward(panel, fwd, 5, kind="regime_robust", regimes=None)


def test_regime_robust_reward_is_the_worst_bucket(ctx):
    from flowalpha.conditioning.regimes import RegimeSet
    from flowalpha.factors.flow_features import FLOW_FEATURES_SCHEMA

    panel, fwd = _panel_and_fwd(ctx)
    n = ctx.n_sessions
    features = pl.DataFrame(
        {
            "date": ctx.sessions,
            **{c: [None] * n for c in FLOW_FEATURES_SCHEMA if c not in ("date", "fii_regime")},
            "fii_regime": ["high" if i % 2 == 0 else "low" for i in range(n)],
        },
        schema=FLOW_FEATURES_SCHEMA,
    )
    regimes = RegimeSet(features=features)
    reward, ic, _, _, per_regime = formula_reward(
        panel, fwd, 5, kind="regime_robust", regimes=regimes, min_dates=40
    )
    assert set(per_regime) == {"high", "low"}
    assert reward == pytest.approx(min(per_regime.values()))
    assert reward <= ic + 1e-12


def test_novelty_penalty_grows_with_similarity():
    from flowalpha.mining.miner import Candidate

    kept = [
        Candidate(
            formula=Formula(("close", "volume", "add"), "price_only"),
            infix="", reward=0.0, ic=0.0, ic_t_stat=0.0, n_dates=0,
            novelty_penalty=0.0, episode=0, seed=0,
        )
    ]
    identical = novelty_penalty(Formula(("close", "volume", "add"), "price_only"), kept, 1.0)
    different = novelty_penalty(Formula(("returns", "neg"), "price_only"), kept, 1.0)
    assert identical == pytest.approx(1.0)
    assert different < identical
    assert novelty_penalty(Formula(("close",), "price_only"), [], 1.0) == 0.0
    assert novelty_penalty(Formula(("close",), "price_only"), kept, 0.0) == 0.0


# --- end to end ------------------------------------------------------------

def test_mine_smoke_with_the_random_control(ctx, tmp_path):
    prices = pl.DataFrame(
        {
            "date": np.repeat(np.array(ctx.sessions, dtype="object"), ctx.n_symbols).tolist(),
            "symbol": np.tile(np.array(ctx.symbols, dtype="object"), ctx.n_sessions).tolist(),
            "adj_close": ctx.adj_close.reshape(-1),
        },
        schema={"date": pl.Date, "symbol": pl.Utf8, "adj_close": pl.Float64},
    )
    fwd = forward_returns(prices, [5])
    cfg = ctx.cfg.with_overrides(
        mining={**ctx.cfg["mining"], "vocab": "price_only", "max_formula_len": 6,
                "reward": "pooled"}
    )
    result = mine(ctx, fwd, cfg, episodes=15, seeds=[0], policy_kind="random", top_k=3)
    assert result.policy == "random"
    assert result.vocabulary == "price_only"
    assert len(result.candidates) <= 3
    assert result.episodes == 15


def test_unknown_reward_rejected(ctx):
    cfg = ctx.cfg.with_overrides(mining={**ctx.cfg["mining"], "reward": "vibes"})
    with pytest.raises(MiningError, match="unknown reward"):
        mine(ctx, pl.DataFrame(schema={"date": pl.Date}), cfg, episodes=1,
             policy_kind="random")


def test_every_candidate_is_registered_as_a_trial(tmp_path, ctx):
    """A miner is a multiple-testing machine; without this, mined alphas would carry the
    most inflated Sharpes in the project while looking like its best results."""
    from flowalpha.mining.miner import Candidate, MiningResult

    result = MiningResult(
        candidates=[
            Candidate(formula=Formula(("close",), "price_only"), infix="close",
                      reward=0.01, ic=0.01, ic_t_stat=1.0, n_dates=100,
                      novelty_penalty=0.0, episode=1, seed=0),
            Candidate(formula=Formula(("volume", "neg"), "price_only"), infix="neg(volume)",
                      reward=0.005, ic=0.005, ic_t_stat=0.5, n_dates=100,
                      novelty_penalty=0.0, episode=2, seed=0),
        ],
        episodes=100, seeds=[0], policy="random", vocabulary="price_only",
        reward_kind="pooled", n_illegal=0, n_degenerate=0,
    )
    registry = TrialRegistry.load(tmp_path)
    before = registry.n_trials
    after = register_candidates(result, registry)
    assert after == before + 2
    assert any(k.startswith("mined:") for k in registry.trials)
    assert registry.trials["mined:close"]["kind"] == "mined"


def test_mining_result_serialises():
    from flowalpha.mining.miner import MiningResult

    result = MiningResult(candidates=[], episodes=10, seeds=[0], policy="random",
                          vocabulary="price_only", reward_kind="pooled",
                          n_illegal=1, n_degenerate=2)
    d = result.to_dict()
    assert d["n_illegal_sequences"] == 1
    assert d["n_degenerate_signals"] == 2
    assert d["candidates"] == []


@requires_torch
def test_mine_smoke_with_the_gru_policy(ctx):
    prices = pl.DataFrame(
        {
            "date": np.repeat(np.array(ctx.sessions, dtype="object"), ctx.n_symbols).tolist(),
            "symbol": np.tile(np.array(ctx.symbols, dtype="object"), ctx.n_sessions).tolist(),
            "adj_close": ctx.adj_close.reshape(-1),
        },
        schema={"date": pl.Date, "symbol": pl.Utf8, "adj_close": pl.Float64},
    )
    fwd = forward_returns(prices, [5])
    cfg = ctx.cfg.with_overrides(
        mining={**ctx.cfg["mining"], "vocab": "price_only", "max_formula_len": 6,
                "reward": "pooled"}
    )
    result = mine(ctx, fwd, cfg, episodes=12, seeds=[0], policy_kind="gru",
                  top_k=2, batch_size=4)
    assert result.policy == "gru"
    assert result.diagnostics  # at least one update happened
