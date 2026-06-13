"""Approach selection policy for lesson generation.

Chooses which teaching approach (e.g. ``rule_based`` vs ``example_based``) to use
for a given ``(user, mistake_type)``, based on how many examples the user has
accumulated for that type. Kept separate from ``RAGService`` so the policy is
unit-testable in isolation and easy to evolve.

Phases (by ``example_count``):
  * ``< EXPLORE_MIN``  -> baseline only (not enough material to teach by example).
  * ``EXPLORE_MIN..EXPLOIT_MIN - 1`` -> rotate across all approaches (exploration).
  * ``>= EXPLOIT_MIN`` -> exploit by score, with periodic exploration of the
    runner-up. ``scores`` maps approach name -> effectiveness (higher = better
    post-lesson outcomes). When ``scores`` is ``None`` or incomplete, falls
    back to rotation.

``selection_index`` is a per-``(user, mistake_type)`` monotonic counter (the number
of prior lessons generated for that type). It drives deterministic rotation and
the exploit-phase exploration cadence.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence


class ApproachSelector:
    EXPLORE_MIN = 3
    EXPLOIT_MIN = 9
    EXPLORE_EVERY = 3  # in exploit phase, every Nth selection probes the runner-up

    def __init__(self, approaches: Sequence[str], baseline: str):
        if not approaches:
            raise ValueError("approaches must not be empty")
        if baseline not in approaches:
            raise ValueError(f"baseline {baseline!r} must be one of {list(approaches)}")
        self._approaches = list(approaches)
        self._baseline = baseline

    @property
    def approaches(self) -> list[str]:
        return list(self._approaches)

    def select(
        self,
        *,
        example_count: int,
        selection_index: int,
        scores: Optional[Mapping[str, float]] = None,
    ) -> str:
        """Return the approach name to use for this lesson."""
        if example_count < self.EXPLORE_MIN:
            return self._baseline
        if example_count < self.EXPLOIT_MIN or not scores:
            return self._rotate(selection_index)
        return self._exploit(selection_index, scores)

    def _rotate(self, selection_index: int) -> str:
        idx = selection_index % len(self._approaches)
        return self._approaches[idx]

    def _exploit(self, selection_index: int, scores: Mapping[str, float]) -> str:
        ranked = sorted(
            self._approaches,
            key=lambda name: (scores.get(name, 0.0), name),
            reverse=True,
        )
        explore_turn = selection_index % self.EXPLORE_EVERY == self.EXPLORE_EVERY - 1
        if explore_turn and len(ranked) > 1:
            return ranked[1]
        return ranked[0]
