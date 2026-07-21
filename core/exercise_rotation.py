"""Deterministic exercise-type rotation."""

from __future__ import annotations

from typing import Literal

ExerciseType = Literal["multiple_choice", "fill_blank"]

_EXERCISE_TYPES: tuple[ExerciseType, ExerciseType] = (
    "multiple_choice",
    "fill_blank",
)

# Consecutive lessons per type before switching (2 MCQ → 2 fill_blank → …).
# Keeps MCQ/fill_blank counts roughly even when exploit phase sticks to one
# teaching approach for many lessons on the same mistake_type.
EXERCISE_ROTATION_BLOCK_SIZE = 2


def exercise_type_for_selection_index(selection_index: int) -> ExerciseType:
    """Pick drill format from the per-mistake_type lesson sequence (0-based).

    Block rotation on ``selection_index``: ``BLOCK_SIZE`` lessons of one type,
    then ``BLOCK_SIZE`` of the other (default 2+2). Extends to more types via
    ``_EXERCISE_TYPES`` and the same block formula.
    """
    n = len(_EXERCISE_TYPES)
    block = max(0, int(selection_index)) // EXERCISE_ROTATION_BLOCK_SIZE
    return _EXERCISE_TYPES[block % n]
