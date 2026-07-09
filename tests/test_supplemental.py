"""Tests for supplemental practice throttle logic."""

from unittest.mock import AsyncMock, patch

import pytest

from core.activity_timeline import UserActivity
from storage.repositories import should_offer_supplemental_practice


def _submit_timeline(count: int) -> list[UserActivity]:
    return [
        UserActivity(
            activity_index=i,
            kind="submit",
            activity_id=f"sess-{i}",
            occurred_at=f"2026-01-{i + 1:02d}T10:00:00+00:00",
        )
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_supplemental_off_when_not_nth_submit():
    session = AsyncMock()
    with (
        patch(
            "storage.repositories.get_user_activity_timeline",
            new=AsyncMock(return_value=_submit_timeline(2)),
        ),
        patch(
            "storage.repositories.top_priority_mistake_types",
            new=AsyncMock(return_value=["verbs"]),
        ),
    ):
        allowed = await should_offer_supplemental_practice(
            session,
            "user-1",
            exclude_mistake_types={"articles"},
            every_n_submits=3,
        )
    assert allowed is False


@pytest.mark.asyncio
async def test_supplemental_on_nth_submit_with_uncovered_priority_mt():
    session = AsyncMock()
    with (
        patch(
            "storage.repositories.get_user_activity_timeline",
            new=AsyncMock(return_value=_submit_timeline(3)),
        ),
        patch(
            "storage.repositories.top_priority_mistake_types",
            new=AsyncMock(return_value=["verbs", "articles"]),
        ),
    ):
        allowed = await should_offer_supplemental_practice(
            session,
            "user-1",
            exclude_mistake_types={"articles"},
            every_n_submits=3,
        )
    assert allowed is True


@pytest.mark.asyncio
async def test_supplemental_on_every_third_submit_even_if_recent_had_one():
    """No cooldown: 6th submit is eligible the same as 3rd."""
    session = AsyncMock()
    with (
        patch(
            "storage.repositories.get_user_activity_timeline",
            new=AsyncMock(return_value=_submit_timeline(6)),
        ),
        patch(
            "storage.repositories.top_priority_mistake_types",
            new=AsyncMock(return_value=["verbs"]),
        ),
    ):
        allowed = await should_offer_supplemental_practice(
            session,
            "user-1",
            exclude_mistake_types=set(),
            every_n_submits=3,
        )
    assert allowed is True


@pytest.mark.asyncio
async def test_supplemental_off_when_priority_already_covered():
    session = AsyncMock()
    with (
        patch(
            "storage.repositories.get_user_activity_timeline",
            new=AsyncMock(return_value=_submit_timeline(3)),
        ),
        patch(
            "storage.repositories.top_priority_mistake_types",
            new=AsyncMock(return_value=["articles"]),
        ),
    ):
        allowed = await should_offer_supplemental_practice(
            session,
            "user-1",
            exclude_mistake_types={"articles"},
            every_n_submits=3,
        )
    assert allowed is False
