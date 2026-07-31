"""Tokenised formula grammar for the alpha miner.

Formulas are reverse-Polish token sequences over a small vocabulary of price and flow
primitives. RPN rather than an expression tree because a policy emitting one token at a
time can be kept legal with a single arity check, so the search never wastes episodes on
unparseable strings.

Two vocabularies:

``price_only``
    Price, volume and turnover primitives. The control: whatever the miner finds here
    cannot be a flow effect.
``flow_augmented``
    Adds the point-in-time flow features. This is where a genuine
    flow-conditional alpha would have to show up.

Every operator is **causal**. There is no operator that can see the future: rolling
windows look backwards, ``delay`` shifts information later, and there is deliberately no
inverse. That property is what lets a mined formula be trusted at all, and it is asserted
by the mining look-ahead test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from ..factors.base import FactorContext, rolling_mean, rolling_std, shift_rows

#: Rolling-window lengths the grammar may use. Kept small and explicit: an unbounded
#: window parameter turns every operator into a free parameter and the search into a
#: much larger multiple-testing problem than the trial count would suggest.
WINDOWS: tuple[int, ...] = (5, 10, 21, 63)


class GrammarError(Exception):
    """Raised for an illegal token sequence or an unknown vocabulary."""


@dataclass(frozen=True)
class Token:
    """One vocabulary entry.

    ``arity`` is how many operands it consumes from the stack; terminals have arity 0.
    """

    name: str
    arity: int
    fn: Callable | None = None
    kind: str = "op"  # "terminal" | "op"


# ---------------------------------------------------------------------------
# Operators. Every one is causal by construction.
# ---------------------------------------------------------------------------

def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) > 1e-12, a / b, np.nan)


def _rank_cs(x: np.ndarray) -> np.ndarray:
    """Cross-sectional rank, scaled to [-0.5, 0.5]. Within-date, so no look-ahead."""
    out = np.full_like(x, np.nan)
    for i in range(x.shape[0]):
        row = x[i]
        finite = np.isfinite(row)
        n = int(finite.sum())
        if n < 2:
            continue
        order = np.argsort(np.argsort(row[finite]))
        out[i, finite] = order / (n - 1) - 0.5
    return out


def _zscore_cs(x: np.ndarray) -> np.ndarray:
    mean = np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=1, keepdims=True)
    std = np.nanstd(np.where(np.isfinite(x), x, np.nan), axis=1, keepdims=True)
    return _safe_div(x - mean, np.where(std > 0, std, np.nan))


def _delta(x: np.ndarray, window: int) -> np.ndarray:
    return x - shift_rows(x, window)


def _ts_rank(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing rank of the latest value within its own window. Backward-looking only."""
    n, m = x.shape
    out = np.full((n, m), np.nan)
    if n < window:
        return out
    view = np.lib.stride_tricks.sliding_window_view(x, window, axis=0)
    finite = np.isfinite(view)
    counts = finite.sum(axis=-1)
    last = view[..., -1]
    less = (np.where(finite, view, np.inf) < last[..., None]).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ranked = np.where(counts > 1, less / (counts - 1) - 0.5, np.nan)
    out[window - 1 :] = np.where(np.isfinite(last), ranked, np.nan)
    return out


_UNARY_OPS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "neg": lambda x: -x,
    "abs": np.abs,
    "sign": np.sign,
    "log": lambda x: np.where(x > 0, np.log(np.where(x > 0, x, np.nan)), np.nan),
    "rank": _rank_cs,
    "zscore": _zscore_cs,
}

_BINARY_OPS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": _safe_div,
}


def _window_ops() -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    out: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    for w in WINDOWS:
        out[f"mean{w}"] = lambda x, w=w: rolling_mean(x, w, min_valid=max(2, int(0.6 * w)))
        out[f"std{w}"] = lambda x, w=w: rolling_std(x, w, min_valid=max(2, int(0.6 * w)))
        out[f"delta{w}"] = lambda x, w=w: _delta(x, w)
        out[f"tsrank{w}"] = lambda x, w=w: _ts_rank(x, w)
        out[f"delay{w}"] = lambda x, w=w: shift_rows(x, w)
    return out


# ---------------------------------------------------------------------------
# Terminals
# ---------------------------------------------------------------------------

PRICE_TERMINALS: tuple[str, ...] = ("close", "volume", "turnover", "returns")
FLOW_TERMINALS: tuple[str, ...] = (
    "fii_flow", "dii_flow", "retail_flow", "fii_daily", "flow_shock_num",
)

VOCABULARIES: dict[str, tuple[str, ...]] = {
    "price_only": PRICE_TERMINALS,
    "flow_augmented": PRICE_TERMINALS + FLOW_TERMINALS,
}


def build_vocabulary(kind: str = "flow_augmented") -> list[Token]:
    """Ordered token list for a vocabulary. The order is the policy's action space, so
    it must be stable across runs for a checkpoint to mean anything."""
    if kind not in VOCABULARIES:
        raise GrammarError(f"unknown vocabulary {kind!r}; expected one of {sorted(VOCABULARIES)}")
    tokens = [Token(name=t, arity=0, kind="terminal") for t in VOCABULARIES[kind]]
    for name, fn in sorted(_UNARY_OPS.items()):
        tokens.append(Token(name=name, arity=1, fn=fn))
    for name, fn in sorted(_window_ops().items()):
        tokens.append(Token(name=name, arity=1, fn=fn))
    for name, fn in sorted(_BINARY_OPS.items()):
        tokens.append(Token(name=name, arity=2, fn=fn))
    return tokens


@dataclass
class Formula:
    """An RPN token sequence."""

    tokens: tuple[str, ...]
    vocabulary: str = "flow_augmented"

    def __str__(self) -> str:
        return " ".join(self.tokens)

    @property
    def length(self) -> int:
        return len(self.tokens)

    def infix(self, vocab: Sequence[Token] | None = None) -> str:
        """Human-readable rendering. Reports appear in infix; the policy emits RPN."""
        table = {t.name: t for t in (vocab or build_vocabulary(self.vocabulary))}
        stack: list[str] = []
        for name in self.tokens:
            token = table.get(name)
            if token is None:
                return str(self)
            if token.arity == 0:
                stack.append(name)
            elif token.arity == 1 and stack:
                stack.append(f"{name}({stack.pop()})")
            elif token.arity == 2 and len(stack) >= 2:
                b, a = stack.pop(), stack.pop()
                stack.append(f"{name}({a}, {b})")
            else:
                return str(self)
        return stack[-1] if stack else str(self)


def legal_next(
    tokens: Sequence[str], vocab: Sequence[Token], max_len: int
) -> np.ndarray:
    """Boolean mask over ``vocab`` of tokens that may legally come next.

    A token is legal when the stack holds enough operands for its arity, and when there
    remain enough steps to close the expression to a single value. Masking rather than
    rejecting keeps every episode productive.
    """
    depth = 0
    for name in tokens:
        arity = next((t.arity for t in vocab if t.name == name), 0)
        depth += 1 - arity
    remaining = max_len - len(tokens)
    mask = np.zeros(len(vocab), dtype=bool)
    for i, token in enumerate(vocab):
        if token.arity > depth:
            continue
        new_depth = depth + 1 - token.arity
        # Each remaining step can reduce depth by at most 1 (a binary op).
        if new_depth - 1 > remaining - 1:
            continue
        mask[i] = True
    return mask


def is_complete(tokens: Sequence[str], vocab: Sequence[Token]) -> bool:
    """True when the sequence evaluates to exactly one value."""
    if not tokens:
        return False
    depth = 0
    arity = {t.name: t.arity for t in vocab}
    last = len(tokens) - 1
    for i, name in enumerate(tokens):
        depth += 1 - arity.get(name, 0)
        # Compare POSITION, not value: a repeated token equal to the final one would
        # otherwise excuse a mid-sequence stack underflow.
        if depth < 1 and i != last:
            return False
    return depth == 1


def terminal_matrices(
    ctx: FactorContext, flow_features=None
) -> dict[str, np.ndarray]:
    """Materialise every terminal as a (sessions x symbols) matrix.

    Flow terminals are broadcast across symbols: they are market-wide series, and the
    grammar's cross-sectional operators turn them into per-name signals only in
    combination with a per-name term. On their own they are constant within a date and
    therefore carry no cross-sectional information -- which is correct, not a defect.

    The flow values come from the ``flow_features`` panel, whose construction already
    applied the one-session flow lag.
    """
    out: dict[str, np.ndarray] = {
        "close": ctx.adj_close,
        "volume": ctx.volume,
        "turnover": ctx.turnover,
        "returns": ctx.returns,
    }
    n, m = ctx.n_sessions, ctx.n_symbols
    if flow_features is not None and not flow_features.is_empty():
        index = {d: i for i, d in enumerate(ctx.sessions)}
        mapping = {
            "fii_flow": "fii_flow_21d",
            "dii_flow": "dii_flow_21d",
            "retail_flow": "retail_flow_21d",
            "fii_daily": "fii_daily",
        }
        for terminal, column in mapping.items():
            if column not in flow_features.columns:
                continue
            series = np.full(n, np.nan)
            for day, value in zip(
                flow_features["date"].to_list(), flow_features[column].to_list()
            ):
                pos = index.get(day)
                if pos is not None and value is not None:
                    series[pos] = float(value)
            out[terminal] = np.repeat(series[:, None], m, axis=1)
        if "flow_shock" in flow_features.columns:
            series = np.full(n, np.nan)
            for day, value in zip(
                flow_features["date"].to_list(), flow_features["flow_shock"].to_list()
            ):
                pos = index.get(day)
                if pos is not None and value is not None:
                    series[pos] = 1.0 if value else 0.0
            out["flow_shock_num"] = np.repeat(series[:, None], m, axis=1)
    for terminal in FLOW_TERMINALS:
        out.setdefault(terminal, np.full((n, m), np.nan))
    return out


def evaluate(
    formula: Formula,
    terminals: Mapping[str, np.ndarray],
    vocab: Sequence[Token] | None = None,
) -> np.ndarray:
    """Evaluate an RPN formula into a (sessions x symbols) matrix."""
    table = {t.name: t for t in (vocab or build_vocabulary(formula.vocabulary))}
    stack: list[np.ndarray] = []
    for name in formula.tokens:
        token = table.get(name)
        if token is None:
            raise GrammarError(f"unknown token {name!r}")
        if token.arity == 0:
            if name not in terminals:
                raise GrammarError(f"terminal {name!r} not available in this context")
            stack.append(np.asarray(terminals[name], dtype=float))
        elif token.arity == 1:
            if not stack:
                raise GrammarError(f"{name} needs one operand")
            with np.errstate(all="ignore"):
                stack.append(token.fn(stack.pop()))
        else:
            if len(stack) < 2:
                raise GrammarError(f"{name} needs two operands")
            b, a = stack.pop(), stack.pop()
            with np.errstate(all="ignore"):
                stack.append(token.fn(a, b))
    if len(stack) != 1:
        raise GrammarError(
            f"formula does not reduce to a single value (stack depth {len(stack)})"
        )
    result = stack[0]
    return np.where(np.isfinite(result), result, np.nan)
