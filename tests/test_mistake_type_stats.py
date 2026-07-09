"""Unit tests for activity-index mistake_type stats computation."""

from core.activity_timeline import UserActivity, build_activity_timeline
from core.mistake_type_stats import (
    MistakeTypeStatsConfig,
    ScoringEventRow,
    compute_mistake_type_stats,
)


def _timeline(submit_count: int) -> list[UserActivity]:
    sessions = [
        (f"sess-{i}", f"2026-01-{i + 1:02d}T10:00:00+00:00")
        for i in range(submit_count)
    ]
    return build_activity_timeline(submit_sessions=sessions, exercise_attempts=[])


def test_empty_timeline_still_emits_lifetime_only_rows():
    rows = compute_mistake_type_stats(
        user_id="u1",
        timeline=[],
        scoring_events=[ScoringEventRow(mistake_type="x", delta=1.0, session_or_exercise_id="a")],
        text_to_session={},
    )
    assert len(rows) == 1
    assert rows[0].lifetime_score == 1.0
    assert rows[0].recent_burden == 0.0
    assert rows[0].is_new is False


def test_unmapped_events_still_count_toward_lifetime():
    timeline = _timeline(2)
    rows = compute_mistake_type_stats(
        user_id="u1",
        timeline=timeline,
        scoring_events=[
            ScoringEventRow(mistake_type="articles", delta=1.0, session_or_exercise_id="orphan-id"),
            ScoringEventRow(mistake_type="articles", delta=1.0, session_or_exercise_id="sess-0"),
        ],
        text_to_session={},
    )
    assert len(rows) == 1
    assert rows[0].lifetime_score == 2.0
    assert rows[0].recent_burden == 1.0


def test_maps_user_text_id_to_submit_activity():
    timeline = _timeline(3)
    text_id = "text-abc"
    text_to_session = {text_id: "sess-1"}

    rows = compute_mistake_type_stats(
        user_id="u1",
        timeline=timeline,
        scoring_events=[
            ScoringEventRow(mistake_type="articles", delta=1.0, session_or_exercise_id=text_id),
            ScoringEventRow(mistake_type="articles", delta=1.0, session_or_exercise_id=text_id),
            ScoringEventRow(mistake_type="articles", delta=-0.5, session_or_exercise_id="sess-2"),
        ],
        text_to_session=text_to_session,
        config=MistakeTypeStatsConfig(recent_k=2, new_n=2),
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.mistake_type == "articles"
    assert row.first_activity_index == 1
    assert row.lifetime_score == 1.5
    assert row.recent_burden == 1.5
    assert row.historical_burden == 0.0


def test_new_mistake_type_flag():
    timeline = _timeline(5)
    rows = compute_mistake_type_stats(
        user_id="u1",
        timeline=timeline,
        scoring_events=[
            ScoringEventRow(mistake_type="prepositions", delta=1.0, session_or_exercise_id="sess-4"),
        ],
        text_to_session={},
        config=MistakeTypeStatsConfig(new_n=2),
    )
    assert rows[0].is_new is True
    assert rows[0].first_activity_index == 4


def test_relapsed_detection():
    cfg = MistakeTypeStatsConfig(recent_k=2, relapse_q=2, relapse_threshold=0.5)
    timeline = _timeline(6)
    events = [
        ScoringEventRow(mistake_type="verbs", delta=2.0, session_or_exercise_id="sess-0"),
        ScoringEventRow(mistake_type="verbs", delta=2.0, session_or_exercise_id="sess-1"),
        ScoringEventRow(mistake_type="verbs", delta=1.0, session_or_exercise_id="sess-5"),
    ]
    rows = compute_mistake_type_stats(
        user_id="u1",
        timeline=timeline,
        scoring_events=events,
        text_to_session={},
        config=cfg,
    )
    row = rows[0]
    assert row.is_relapsed is True
    assert row.recent_burden >= cfg.relapse_threshold
    assert row.historical_burden > 0


def test_improving_detection():
    cfg = MistakeTypeStatsConfig(improve_m=2, improve_min_drop=0.5)
    timeline = _timeline(4)
    events = [
        ScoringEventRow(mistake_type="tenses", delta=2.0, session_or_exercise_id="sess-0"),
        ScoringEventRow(mistake_type="tenses", delta=2.0, session_or_exercise_id="sess-1"),
        ScoringEventRow(mistake_type="tenses", delta=-0.5, session_or_exercise_id="sess-3"),
    ]
    rows = compute_mistake_type_stats(
        user_id="u1",
        timeline=timeline,
        scoring_events=events,
        text_to_session={},
        config=cfg,
    )
    assert rows[0].is_improving is True


def test_exercise_activity_id_maps_directly():
    timeline = build_activity_timeline(
        submit_sessions=[("sess-0", "2026-01-01T10:00:00+00:00")],
        exercise_attempts=[
            ("ex-1", "2026-01-02T10:00:00+00:00", "art-1", "sess-0"),
        ],
    )
    rows = compute_mistake_type_stats(
        user_id="u1",
        timeline=timeline,
        scoring_events=[
            ScoringEventRow(mistake_type="articles", delta=-0.5, session_or_exercise_id="ex-1"),
        ],
        text_to_session={},
    )
    assert len(rows) == 1
    assert rows[0].first_activity_index == 1
    assert rows[0].recent_burden == -0.5
