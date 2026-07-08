"""Unit tests for per-user activity timeline merge and indexing."""

from core.activity_timeline import build_activity_timeline


def test_empty_timeline():
    assert build_activity_timeline(submit_sessions=[], exercise_attempts=[]) == []


def test_submits_only_sorted_and_indexed():
    timeline = build_activity_timeline(
        submit_sessions=[
            ("sess-b", "2026-01-02T10:00:00+00:00"),
            ("sess-a", "2026-01-01T10:00:00+00:00"),
        ],
        exercise_attempts=[],
    )
    assert [a.activity_index for a in timeline] == [0, 1]
    assert [a.kind for a in timeline] == ["submit", "submit"]
    assert [a.activity_id for a in timeline] == ["sess-a", "sess-b"]
    assert all(a.lesson_artifact_id is None for a in timeline)
    assert all(a.origin_session_id is None for a in timeline)


def test_exercises_carry_artifact_and_origin_session():
    timeline = build_activity_timeline(
        submit_sessions=[],
        exercise_attempts=[
            (
                "ex-1",
                "2026-01-03T12:00:00+00:00",
                "art-99",
                "sess-parent",
            ),
        ],
    )
    assert len(timeline) == 1
    row = timeline[0]
    assert row.kind == "exercise"
    assert row.activity_id == "ex-1"
    assert row.lesson_artifact_id == "art-99"
    assert row.origin_session_id == "sess-parent"


def test_mixed_timeline_merges_and_orders():
    timeline = build_activity_timeline(
        submit_sessions=[
            ("sess-1", "2026-01-01T09:00:00+00:00"),
            ("sess-2", "2026-01-03T09:00:00+00:00"),
        ],
        exercise_attempts=[
            ("ex-1", "2026-01-02T09:00:00+00:00", "art-1", "sess-1"),
        ],
    )
    assert [(a.kind, a.activity_id) for a in timeline] == [
        ("submit", "sess-1"),
        ("exercise", "ex-1"),
        ("submit", "sess-2"),
    ]
    assert [a.activity_index for a in timeline] == [0, 1, 2]


def test_duplicate_submit_session_id_is_deduped():
    timeline = build_activity_timeline(
        submit_sessions=[
            ("sess-a", "2026-01-01T10:00:00+00:00"),
            ("sess-a", "2026-01-01T11:00:00+00:00"),
        ],
        exercise_attempts=[],
    )
    assert len(timeline) == 1
    assert timeline[0].activity_id == "sess-a"


def test_same_timestamp_tie_breaks_by_kind_then_id():
    ts = "2026-01-01T10:00:00+00:00"
    timeline = build_activity_timeline(
        submit_sessions=[("sess-z", ts)],
        exercise_attempts=[("ex-a", ts, "art-1", "sess-z")],
    )
    assert [(a.kind, a.activity_id) for a in timeline] == [
        ("exercise", "ex-a"),
        ("submit", "sess-z"),
    ]


def test_skips_blank_exercise_attempt_id():
    timeline = build_activity_timeline(
        submit_sessions=[],
        exercise_attempts=[("", "2026-01-01T10:00:00+00:00", "art-1", "sess-1")],
    )
    assert timeline == []
