"""
Mistake types excluded from Qdrant mistake_examples (and matching LT pipeline behavior).

All logic that means "no mistake_examples point" / skip example ingest must go through
`skip_example_for_qdrant` so it stays in sync with scoring (delta=0) for the same set.
"""

from __future__ import annotations

from typing import Any, Mapping

# Type-based: same rule family as languagetool_pipeline (e.g. no rule_message for other/style;
# "unlisted" = no mapping in assets; "unmapped" = mapped explicitly to low-pedagogical-value
# bucket in assets/languagetool_to_mistaketype.json — proper names, brands, generic AI rules,
# very rare niche rules).
MISTAKE_TYPES_EXCLUDED_FOR_QDRANT: frozenset[str] = frozenset(
    (
        "other",
        "style",
        "unlisted",
        "unmapped",
    )
)


def skip_example_for_qdrant(event: Mapping[str, Any]) -> bool:
    """True if we do not create/update mistake_examples (only occurrences path)."""
    return event.get("mistake_type") in MISTAKE_TYPES_EXCLUDED_FOR_QDRANT


def delta_for_ingest_mistake_event(event: Mapping[str, Any]) -> float:
    """Scoring for SQL-backed mistake events from free-text /submit."""
    mt = event.get("mistake_type")
    if mt in MISTAKE_TYPES_EXCLUDED_FOR_QDRANT:
        return 0.0
    return 1.0


def delta_for_exercise_correct() -> float:
    return -0.5


def delta_for_exercise_wrong() -> float:
    return 1.0
