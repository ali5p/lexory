"""
Mistake types excluded from Qdrant mistake_examples (and matching LT pipeline behavior).

All logic that means "no mistake_examples point" / skip example ingest must go through
`skip_example_for_qdrant` so it stays in sync with scoring (delta=0) for the same set.
The exercise_attempt source is handled inside `skip_example_for_qdrant`, not in the frozenset.
"""

from __future__ import annotations

from typing import Any, Mapping

# Type-based: same rule family as languagetool_pipeline (e.g. no rule_message for other/style;
# "unlisted" = no mapping in assets).
MISTAKE_TYPES_EXCLUDED_FOR_QDRANT: frozenset[str] = frozenset(
    (
        "other",
        "style",
        "unlisted",
    )
)

_EXERCISE_SOURCE = "exercise_attempt"


def skip_example_for_qdrant(event: Mapping[str, Any]) -> bool:
    """True if we do not create/update mistake_examples (only occurrences path)."""
    if event.get("source") == _EXERCISE_SOURCE:
        return True
    return event.get("mistake_type") in MISTAKE_TYPES_EXCLUDED_FOR_QDRANT


def delta_for_ingest_mistake_event(event: Mapping[str, Any]) -> float:
    """
    Scoring for SQL-backed mistake events (ingest, non-exercise parts with same taxonomy).
    Excluded Qdrant types get 0; others +1. Independent of source (e.g. exercise uses
    process_exercise_attempt rules for +1/-0.5 as well, but for each per-hit row use
    `delta_for_exercise_lt_hit` / penalties separately).
    """
    mt = event.get("mistake_type")
    if mt in MISTAKE_TYPES_EXCLUDED_FOR_QDRANT:
        return 0.0
    return 1.0


def delta_for_exercise_lt_hit(event: Mapping[str, Any]) -> float:
    """+1 per LanguageTool row unless the type is in the Qdrant-excluded set (then 0)."""
    return delta_for_ingest_mistake_event(event)


def delta_for_exercise_missed_target() -> float:
    return -0.5
