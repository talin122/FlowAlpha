"""Purged k-fold cross-validation over dates.

Why ordinary k-fold is wrong here
---------------------------------
The label for date ``t`` is a forward return spanning ``t`` to ``t + h``. A train
observation at ``t - 1`` therefore *contains* most of the same future as a test
observation at ``t``. Plain k-fold puts them in different folds and calls the result
out-of-sample, which it is not: the model has already seen the answer.

Two corrections, both from Lopez de Prado:

**Purging.** Drop from the training set any observation whose label window overlaps
any test observation's label window.

**Embargo.** Additionally drop training observations within ``embargo_days`` on either
side of the test block. Serial correlation in features means an observation just
outside the label window still carries information about it.

The splitter works over **dates**, not rows, because a cross-sectional panel has many
rows per date and splitting rows would put the same date on both sides.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np


class PurgedCVError(Exception):
    """Raised when a split is impossible for the given dates and parameters."""


@dataclass(frozen=True)
class Split:
    """One fold: the test dates, the surviving train dates, and what was removed."""

    fold: int
    train_dates: tuple[_dt.date, ...]
    test_dates: tuple[_dt.date, ...]
    purged_dates: tuple[_dt.date, ...]
    embargoed_dates: tuple[_dt.date, ...]

    @property
    def n_train(self) -> int:
        return len(self.train_dates)

    @property
    def n_test(self) -> int:
        return len(self.test_dates)


class PurgedKFold:
    """K-fold over an ordered date index with purging and an embargo.

    Parameters
    ----------
    n_splits:
        Number of contiguous test blocks.
    label_horizon:
        Length of the label window in sessions. An observation at ``t`` has a label
        spanning ``[t, t + label_horizon]``.
    embargo_days:
        Extra sessions removed from training on each side of the test block.

    Examples
    --------
    >>> cv = PurgedKFold(n_splits=3, label_horizon=5, embargo_days=2)
    >>> for split in cv.split(dates):  # doctest: +SKIP
    ...     train, test = split.train_dates, split.test_dates
    """

    def __init__(self, n_splits: int = 6, label_horizon: int = 1, embargo_days: int = 0) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if label_horizon < 1:
            raise ValueError("label_horizon must be at least 1 session")
        if embargo_days < 0:
            raise ValueError("embargo_days cannot be negative")
        self.n_splits = int(n_splits)
        self.label_horizon = int(label_horizon)
        self.embargo_days = int(embargo_days)

    def split(self, dates: Sequence[_dt.date]) -> Iterator[Split]:
        """Yield one :class:`Split` per fold, in chronological order of test block."""
        ordered = sorted(set(dates))
        n = len(ordered)
        if n < self.n_splits:
            raise PurgedCVError(
                f"cannot make {self.n_splits} folds from {n} dates"
            )
        bounds = np.linspace(0, n, self.n_splits + 1).astype(int)
        h, e = self.label_horizon, self.embargo_days

        for fold in range(self.n_splits):
            lo, hi = int(bounds[fold]), int(bounds[fold + 1])
            if hi <= lo:
                continue
            test_idx = list(range(lo, hi))

            # A train index i is purged when its label window [i, i+h] overlaps the
            # test block's label span [lo, hi-1+h].
            purged: set[int] = set()
            for i in range(n):
                if lo <= i < hi:
                    continue
                if i + h >= lo and i <= hi - 1 + h:
                    purged.add(i)

            embargoed: set[int] = set()
            for i in range(max(0, lo - e), lo):
                embargoed.add(i)
            for i in range(hi, min(n, hi + e)):
                embargoed.add(i)
            embargoed -= purged

            train_idx = [
                i for i in range(n)
                if not (lo <= i < hi) and i not in purged and i not in embargoed
            ]
            yield Split(
                fold=fold,
                train_dates=tuple(ordered[i] for i in train_idx),
                test_dates=tuple(ordered[i] for i in test_idx),
                purged_dates=tuple(ordered[i] for i in sorted(purged)),
                embargoed_dates=tuple(ordered[i] for i in sorted(embargoed)),
            )

    def splits(self, dates: Sequence[_dt.date]) -> list[Split]:
        return list(self.split(dates))

    @classmethod
    def from_config(cls, cfg, *, label_horizon: int) -> "PurgedKFold":
        pk = cfg["validation"]["purged_kfold"]
        return cls(
            n_splits=int(pk["n_splits"]),
            label_horizon=int(label_horizon),
            embargo_days=int(pk["embargo_days"]),
        )


def leakage_check(
    split: Split, label_horizon: int, all_dates: Sequence[_dt.date]
) -> bool:
    """True when no train label window overlaps any test label window.

    Used by the tests as an independent verification of the splitter. Positions are
    taken from the **full** date index passed in, not from the union of the split's own
    dates: purged dates leave holes in that union, and re-indexing over the holes would
    shift every position and quietly validate a broken splitter.
    """
    if not split.train_dates or not split.test_dates:
        return True
    index = {d: i for i, d in enumerate(sorted(set(all_dates)))}
    test_positions = [index[d] for d in split.test_dates]
    test_lo, test_hi = min(test_positions), max(test_positions)
    for d in split.train_dates:
        i = index[d]
        # Train label window [i, i+h] vs test label span [test_lo, test_hi+h].
        if i + label_horizon >= test_lo and i <= test_hi + label_horizon:
            return False
    return True
