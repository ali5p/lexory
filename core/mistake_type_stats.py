"""Per-user mistake_type stats from activity-index windows (not calendar time).

Used to materialize ``user_mistake_type_stats`` for internal priority ranking
(recent burden, new MT, relapsed, improving, lifetime).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

from core.activity_timeline import UserActivity


@dataclass(frozen=True)
class MistakeTypeStatsConfig:
    """Session-index window sizes and priority weights."""

    recent_k: int = 5
    new_n: int = 5
    improve_m: int = 3
    relapse_q: int = 3
    relapse_threshold: float = 0.5
    quiet_threshold: float = 0.25
    improve_min_drop: float = 0.5
    w_recent: float = 1.0
    w_lifetime: float = 0.5
    w_new: float = 2.0
    w_relapsed: float = 3.0
    w_historical: float = 0.3
    w_improving: float = 1.0


class ScoringEventRow(BaseModel):
    mistake_type: str
    delta: float
    session_or_exercise_id: str


class MistakeTypeStatsRow(BaseModel):
    user_id: str
    mistake_type: str
    first_activity_index: int = Field(
        description="Earliest activity index where this MT scored (0 if unmapped)."
    )
    lifetime_score: float = Field(description="Clamped SUM(delta) over all scoring events.")
    recent_burden: float = Field(description="SUM(delta) in the last recent_k activities.")
    historical_burden: float = Field(
        description="SUM(delta) in activities before the recent window."
    )
    is_new: bool = Field(description="First seen within the last new_n activities.")
    is_improving: bool = Field(description="Burden dropped over the last improve_m vs prior.")
    is_relapsed: bool = Field(
        description="Was quiet then active again in the recent window."
    )
    priority_score: float = Field(description="Weighted rank for internal recommendation.")
    total_activity_count: int = Field(
        description="Number of activities on the user's timeline at compute time."
    )


def compute_mistake_type_stats(
    *,
    user_id: str,
    timeline: list[UserActivity],
    scoring_events: list[ScoringEventRow],
    text_to_session: dict[str, str],
    config: MistakeTypeStatsConfig | None = None,
) -> list[MistakeTypeStatsRow]:
    """Compute per-MT stats for one user from timeline + scoring events."""
    cfg = config or MistakeTypeStatsConfig()
    if not scoring_events:
        return []

    all_deltas_by_type: dict[str, list[float]] = {}
    mapped_by_type: dict[str, list[tuple[int, float]]] = {}

    if timeline:
        activity_index_by_id = {a.activity_id: a.activity_index for a in timeline}
        max_idx = timeline[-1].activity_index
        recent_start = max(0, max_idx - cfg.recent_k + 1)
    else:
        activity_index_by_id = {}
        max_idx = -1
        recent_start = 0

    for event in scoring_events:
        mt = (event.mistake_type or "").strip()
        if not mt:
            continue
        all_deltas_by_type.setdefault(mt, []).append(event.delta)
        if not timeline:
            continue
        idx = _event_activity_index(
            event.session_or_exercise_id, activity_index_by_id, text_to_session
        )
        if idx is not None:
            mapped_by_type.setdefault(mt, []).append((idx, event.delta))

    rows: list[MistakeTypeStatsRow] = []
    for mt in sorted(all_deltas_by_type):
        lifetime_score = max(0.0, sum(all_deltas_by_type[mt]))
        indexed_deltas = mapped_by_type.get(mt, [])

        if indexed_deltas:
            first_idx = min(i for i, _ in indexed_deltas)
            recent_burden = _burden_in_range(indexed_deltas, recent_start, max_idx)
            historical_burden = _burden_in_range(indexed_deltas, 0, recent_start - 1)
            is_new = first_idx >= max(0, max_idx - cfg.new_n + 1)
            is_improving = _is_improving(indexed_deltas, max_idx, cfg)
            is_relapsed = _is_relapsed(
                first_idx=first_idx,
                max_idx=max_idx,
                recent_burden=recent_burden,
                historical_burden=historical_burden,
                indexed_deltas=indexed_deltas,
                cfg=cfg,
            )
            improve_delta = (
                _improvement_delta(indexed_deltas, max_idx, cfg.improve_m)
                if is_improving
                else 0.0
            )
            priority = (
                cfg.w_recent * recent_burden
                + cfg.w_lifetime * lifetime_score
                + cfg.w_new * (1.0 if is_new else 0.0)
                + cfg.w_relapsed * (1.0 if is_relapsed else 0.0)
                + cfg.w_historical
                * (historical_burden if recent_burden <= cfg.quiet_threshold else 0.0)
                - cfg.w_improving * improve_delta
            )
        else:
            first_idx = 0
            recent_burden = 0.0
            historical_burden = 0.0
            is_new = False
            is_improving = False
            is_relapsed = False
            priority = cfg.w_lifetime * lifetime_score

        rows.append(
            MistakeTypeStatsRow(
                user_id=user_id,
                mistake_type=mt,
                first_activity_index=first_idx,
                lifetime_score=lifetime_score,
                recent_burden=recent_burden,
                historical_burden=historical_burden,
                is_new=is_new,
                is_improving=is_improving,
                is_relapsed=is_relapsed,
                priority_score=priority,
                total_activity_count=len(timeline),
            )
        )

    rows.sort(key=lambda r: (-r.priority_score, -r.recent_burden, r.mistake_type))
    return rows


def _event_activity_index(
    session_or_exercise_id: str,
    activity_index_by_id: dict[str, int],
    text_to_session: dict[str, str],
) -> Optional[int]:
    key = (session_or_exercise_id or "").strip()
    if not key:
        return None
    if key in activity_index_by_id:
        return activity_index_by_id[key]
    session_id = text_to_session.get(key)
    if session_id and session_id in activity_index_by_id:
        return activity_index_by_id[session_id]
    return None


def _burden_in_range(
    indexed_deltas: list[tuple[int, float]], start: int, end: int
) -> float:
    if end < start:
        return 0.0
    return sum(d for i, d in indexed_deltas if start <= i <= end)


def _is_improving(
    indexed_deltas: list[tuple[int, float]], max_idx: int, cfg: MistakeTypeStatsConfig
) -> bool:
    m = cfg.improve_m
    if max_idx < 2 * m - 1:
        return False
    recent_start = max_idx - m + 1
    prior_start = max_idx - 2 * m + 1
    prior_end = max_idx - m
    recent = _burden_in_range(indexed_deltas, recent_start, max_idx)
    prior = _burden_in_range(indexed_deltas, prior_start, prior_end)
    return prior >= cfg.quiet_threshold and recent < prior - cfg.improve_min_drop


def _improvement_delta(
    indexed_deltas: list[tuple[int, float]], max_idx: int, m: int
) -> float:
    recent_start = max_idx - m + 1
    prior_start = max_idx - 2 * m + 1
    prior_end = max_idx - m
    recent = _burden_in_range(indexed_deltas, recent_start, max_idx)
    prior = _burden_in_range(indexed_deltas, prior_start, prior_end)
    return max(0.0, prior - recent)


def _is_relapsed(
    *,
    first_idx: int,
    max_idx: int,
    recent_burden: float,
    historical_burden: float,
    indexed_deltas: list[tuple[int, float]],
    cfg: MistakeTypeStatsConfig,
) -> bool:
    if recent_burden < cfg.relapse_threshold:
        return False
    if first_idx > max(0, max_idx - cfg.recent_k - cfg.relapse_q):
        return False
    if historical_burden <= 0:
        return False

    quiet_start = max(0, max_idx - cfg.recent_k - cfg.relapse_q + 1)
    quiet_end = max(0, max_idx - cfg.recent_k)
    quiet_burden = _burden_in_range(indexed_deltas, quiet_start, quiet_end)
    return quiet_burden <= cfg.quiet_threshold
