"""Purged k-fold: no train label window may overlap a test label window."""

from __future__ import annotations

import datetime as _dt

import pytest

from flowalpha.validation.purged_cv import (
    PurgedCVError,
    PurgedKFold,
    leakage_check,
)

from conftest import make_config

DATES = [_dt.date(2022, 1, 3) + _dt.timedelta(days=i) for i in range(100)]


def test_folds_partition_the_test_dates():
    cv = PurgedKFold(n_splits=5, label_horizon=1, embargo_days=0)
    splits = cv.splits(DATES)
    assert len(splits) == 5
    covered = [d for s in splits for d in s.test_dates]
    assert sorted(covered) == DATES
    assert len(covered) == len(set(covered))


def test_test_blocks_are_contiguous_and_chronological():
    cv = PurgedKFold(n_splits=4, label_horizon=1)
    splits = cv.splits(DATES)
    for s in splits:
        idx = [DATES.index(d) for d in s.test_dates]
        assert idx == list(range(min(idx), max(idx) + 1))
    firsts = [s.test_dates[0] for s in splits]
    assert firsts == sorted(firsts)


def test_train_and_test_never_share_a_date():
    cv = PurgedKFold(n_splits=5, label_horizon=5, embargo_days=3)
    for s in cv.split(DATES):
        assert not (set(s.train_dates) & set(s.test_dates))


def test_no_label_overlap_survives_purging():
    """The property the splitter exists to guarantee, verified independently."""
    for horizon in (1, 5, 21):
        cv = PurgedKFold(n_splits=5, label_horizon=horizon, embargo_days=0)
        for s in cv.split(DATES):
            assert leakage_check(s, horizon, DATES), f"leak at horizon {horizon}, fold {s.fold}"


def test_purge_count_grows_with_the_label_horizon():
    """A longer label overlaps more training data, so more must be dropped."""
    counts = []
    for horizon in (1, 5, 21):
        cv = PurgedKFold(n_splits=5, label_horizon=horizon, embargo_days=0)
        counts.append(sum(len(s.purged_dates) for s in cv.split(DATES)))
    assert counts[0] < counts[1] < counts[2]


def test_purging_removes_dates_on_both_sides_of_an_interior_fold():
    cv = PurgedKFold(n_splits=5, label_horizon=5, embargo_days=0)
    interior = cv.splits(DATES)[2]
    lo = DATES.index(interior.test_dates[0])
    hi = DATES.index(interior.test_dates[-1])
    purged_idx = [DATES.index(d) for d in interior.purged_dates]
    assert any(i < lo for i in purged_idx)
    assert any(i > hi for i in purged_idx)


def test_embargo_removes_additional_dates():
    a = PurgedKFold(n_splits=5, label_horizon=1, embargo_days=0).splits(DATES)
    b = PurgedKFold(n_splits=5, label_horizon=1, embargo_days=5).splits(DATES)
    assert sum(s.n_train for s in b) < sum(s.n_train for s in a)
    assert all(len(s.embargoed_dates) > 0 for s in b[1:-1])


def test_embargo_dates_are_disjoint_from_purged_dates():
    """Reported separately so the cost of each correction is visible."""
    cv = PurgedKFold(n_splits=5, label_horizon=5, embargo_days=5)
    for s in cv.split(DATES):
        assert not (set(s.purged_dates) & set(s.embargoed_dates))


def test_first_and_last_folds_have_one_sided_purging():
    cv = PurgedKFold(n_splits=5, label_horizon=5, embargo_days=2)
    splits = cv.splits(DATES)
    first_purged = [DATES.index(d) for d in splits[0].purged_dates]
    assert all(i > DATES.index(splits[0].test_dates[-1]) for i in first_purged)


def test_dates_are_deduped_and_sorted():
    messy = [DATES[5], DATES[0], DATES[5], DATES[3]]
    cv = PurgedKFold(n_splits=2, label_horizon=1)
    splits = cv.splits(messy)
    covered = sorted(d for s in splits for d in s.test_dates)
    assert covered == [DATES[0], DATES[3], DATES[5]]


def test_too_few_dates_raises():
    cv = PurgedKFold(n_splits=10, label_horizon=1)
    with pytest.raises(PurgedCVError, match="cannot make"):
        cv.splits(DATES[:5])


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=1)
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=3, label_horizon=0)
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=3, embargo_days=-1)


def test_from_config(tmp_path):
    cfg = make_config(tmp_path, DATES)
    cv = PurgedKFold.from_config(cfg, label_horizon=5)
    assert cv.n_splits == 4 and cv.embargo_days == 2 and cv.label_horizon == 5


def test_leakage_check_detects_a_deliberate_leak():
    """Guard on the guard: the checker must be able to fail."""
    cv = PurgedKFold(n_splits=5, label_horizon=5, embargo_days=0)
    split = cv.splits(DATES)[2]
    leaky = type(split)(
        fold=split.fold,
        train_dates=split.train_dates + (split.test_dates[0] - _dt.timedelta(days=1),),
        test_dates=split.test_dates,
        purged_dates=(),
        embargoed_dates=(),
    )
    assert not leakage_check(leaky, 5, DATES)


def test_a_plain_kfold_would_fail_the_leakage_check():
    """Demonstrating why the purging exists at all."""
    cv = PurgedKFold(n_splits=5, label_horizon=5, embargo_days=0)
    split = cv.splits(DATES)[2]
    unpurged = type(split)(
        fold=split.fold,
        train_dates=tuple(d for d in DATES if d not in set(split.test_dates)),
        test_dates=split.test_dates,
        purged_dates=(), embargoed_dates=(),
    )
    assert not leakage_check(unpurged, 5, DATES)
