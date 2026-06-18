"""Unit tests for the ApproachSelector 3-phase policy (pure logic, no model load)."""
import pytest

from rag.approach_selection import ApproachSelector


def _selector() -> ApproachSelector:
    return ApproachSelector(approaches=["rule_based", "example_based"], baseline="rule_based")


def test_phase1_baseline_only_below_explore_min():
    s = _selector()
    for count in (0, 1, 2):
        assert s.select(example_count=count, selection_index=0) == "rule_based"
        assert s.select(example_count=count, selection_index=7) == "rule_based"


def test_phase2_rotation_between_explore_and_exploit():
    s = _selector()
    assert s.select(example_count=3, selection_index=0) == "rule_based"
    assert s.select(example_count=3, selection_index=1) == "example_based"
    assert s.select(example_count=8, selection_index=2) == "rule_based"
    assert s.select(example_count=8, selection_index=3) == "example_based"


def test_phase3_without_scores_falls_back_to_rotation():
    s = _selector()
    assert s.select(example_count=20, selection_index=0, scores=None) == "rule_based"
    assert s.select(example_count=20, selection_index=1, scores={}) == "example_based"


def test_phase3_exploits_best_with_periodic_runner_up():
    s = _selector()
    scores = {"rule_based": 0.2, "example_based": 0.9}
    # Most selections take the best-scoring approach.
    assert s.select(example_count=9, selection_index=0, scores=scores) == "example_based"
    assert s.select(example_count=9, selection_index=1, scores=scores) == "example_based"
    # Every 3rd selection (index % 3 == 2) probes the runner-up.
    assert s.select(example_count=9, selection_index=2, scores=scores) == "rule_based"
    assert s.select(example_count=9, selection_index=5, scores=scores) == "rule_based"


def test_is_contrast_lesson_only_in_exploit_phase_on_every_third_index():
    s = _selector()
    scores = {"rule_based": 0.2, "example_based": 0.9}

    assert not s.is_contrast_lesson(example_count=2, selection_index=2, scores=scores)
    assert not s.is_contrast_lesson(example_count=8, selection_index=2, scores=scores)
    assert not s.is_contrast_lesson(example_count=9, selection_index=2, scores=None)

    assert not s.is_contrast_lesson(example_count=9, selection_index=0, scores=scores)
    assert not s.is_contrast_lesson(example_count=9, selection_index=1, scores=scores)
    assert s.is_contrast_lesson(example_count=9, selection_index=2, scores=scores)
    assert not s.is_contrast_lesson(example_count=9, selection_index=3, scores=scores)
    assert s.is_contrast_lesson(example_count=9, selection_index=5, scores=scores)
    assert s.is_contrast_lesson(example_count=9, selection_index=8, scores=scores)


def test_invalid_construction():
    with pytest.raises(ValueError):
        ApproachSelector(approaches=[], baseline="rule_based")
    with pytest.raises(ValueError):
        ApproachSelector(approaches=["rule_based"], baseline="missing")
