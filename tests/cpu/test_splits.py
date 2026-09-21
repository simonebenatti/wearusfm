import numpy as np
import pytest

from wearusfm.harness.splits import (
    SubjectSplit,
    check_no_cross_dataset_subject_overlap,
    masks_from_split,
    split_subjects,
)


def test_split_subjects_no_overlap_and_covers_all():
    subjects = [f"s{i:02d}" for i in range(20)]
    split = split_subjects(subjects, ratios=(0.7, 0.1, 0.2), seed=0)
    all_assigned = [*split.train, *split.val, *split.test]
    assert sorted(all_assigned) == sorted(subjects)
    assert len(set(all_assigned)) == len(all_assigned)


def test_split_subjects_is_deterministic():
    subjects = [f"s{i:02d}" for i in range(15)]
    a = split_subjects(subjects, seed=42)
    b = split_subjects(subjects, seed=42)
    assert a == b


def test_split_subjects_different_seeds_can_differ():
    subjects = [f"s{i:02d}" for i in range(30)]
    a = split_subjects(subjects, seed=0)
    b = split_subjects(subjects, seed=1)
    assert a != b


def test_split_subjects_rejects_bad_ratios():
    with pytest.raises(ValueError):
        split_subjects(["a", "b", "c"], ratios=(0.5, 0.5, 0.5))


def test_split_subjects_rejects_too_few_subjects():
    with pytest.raises(ValueError):
        split_subjects(["a", "b"])


def test_subject_split_rejects_overlap_across_sets():
    with pytest.raises(ValueError):
        SubjectSplit(train=("a", "b"), val=("b",), test=("c",))


def test_masks_from_split_assigns_correctly():
    split = SubjectSplit(train=("a", "b"), val=("c",), test=("d",))
    subject_ids = np.array(["a", "a", "b", "c", "d", "d"])
    masks = masks_from_split(subject_ids, split)
    assert masks["train"].tolist() == [True, True, True, False, False, False]
    assert masks["val"].tolist() == [False, False, False, True, False, False]
    assert masks["test"].tolist() == [False, False, False, False, True, True]


def test_cross_dataset_subject_overlap_detected():
    overlaps = check_no_cross_dataset_subject_overlap(
        {"db1": ["s1", "s2", "s3"], "db2": ["s2", "s4"], "db3": ["s5"]}
    )
    assert overlaps == [("db1", "db2", "s2")]


def test_cross_dataset_subject_overlap_empty_when_disjoint():
    overlaps = check_no_cross_dataset_subject_overlap({"db1": ["s1"], "db2": ["s2"]})
    assert overlaps == []
