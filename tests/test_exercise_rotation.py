"""Tests for deterministic exercise-type rotation."""

from core.exercise_rotation import exercise_type_for_selection_index


def test_block_rotation_two_mcq_then_two_fill_blank():
    assert exercise_type_for_selection_index(0) == "multiple_choice"
    assert exercise_type_for_selection_index(1) == "multiple_choice"
    assert exercise_type_for_selection_index(2) == "fill_blank"
    assert exercise_type_for_selection_index(3) == "fill_blank"
    assert exercise_type_for_selection_index(4) == "multiple_choice"
    assert exercise_type_for_selection_index(5) == "multiple_choice"
    assert exercise_type_for_selection_index(6) == "fill_blank"
    assert exercise_type_for_selection_index(7) == "fill_blank"
