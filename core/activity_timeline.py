"""Per-user activity timeline: submit sessions and exercise attempts in order.

Each ``/submit`` is one submit activity (``session_id``). Each ``POST /exercises/{id}/answer``
call is one exercise activity (``exercise_attempt_id``). Activities are sorted by
``occurred_at`` and assigned a zero-based ``activity_index`` for session-window
statistics (last N activities, not calendar time).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

ActivityKind = Literal["submit", "exercise"]


class UserActivity(BaseModel):
    """One row on the user's ordered activity timeline."""

    activity_index: int = Field(
        description="Zero-based order by occurred_at (0 = earliest activity)."
    )
    kind: ActivityKind
    activity_id: str = Field(
        description="session_id for submit; exercise_attempt_id for exercise."
    )
    occurred_at: str = Field(description="ISO timestamp for sorting and display.")
    origin_session_id: Optional[str] = Field(
        default=None,
        description="For exercises: session_id of the submit that produced the lesson.",
    )
    lesson_artifact_id: Optional[str] = Field(
        default=None,
        description="Set for exercise activities only.",
    )


def build_activity_timeline(
    *,
    submit_sessions: list[tuple[str, str]],
    exercise_attempts: list[tuple[str, str, Optional[str], Optional[str]]],
) -> list[UserActivity]:
    """Merge submit and exercise rows into a sorted, indexed timeline.

    Args:
        submit_sessions: (session_id, occurred_at ISO) — earliest event per submit.
        exercise_attempts: (exercise_attempt_id, occurred_at ISO,
            lesson_artifact_id, origin_session_id).
    """
    rows: list[tuple[datetime, ActivityKind, str, Optional[str], Optional[str]]] = []

    seen_submit: set[str] = set()
    for session_id, occurred_at in submit_sessions:
        if not session_id or session_id in seen_submit:
            continue
        seen_submit.add(session_id)
        rows.append(
            (_parse_occurred_at(occurred_at), "submit", session_id, None, None)
        )

    for attempt_id, occurred_at, artifact_id, origin_session_id in exercise_attempts:
        if not attempt_id:
            continue
        rows.append(
            (
                _parse_occurred_at(occurred_at),
                "exercise",
                attempt_id,
                artifact_id or None,
                origin_session_id or None,
            )
        )

    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    return [
        UserActivity(
            activity_index=i,
            kind=kind,
            activity_id=activity_id,
            occurred_at=occurred_at.isoformat(),
            lesson_artifact_id=artifact_id,
            origin_session_id=origin_session_id,
        )
        for i, (occurred_at, kind, activity_id, artifact_id, origin_session_id) in enumerate(
            rows
        )
    ]


def _parse_occurred_at(value: str) -> datetime:
    """Parse ISO timestamps stored in PostgreSQL string columns."""
    text = (value or "").strip()
    if not text:
        return datetime.min.replace(tzinfo=timezone.utc)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)
