"""Strategy base class.

A Strategy is a class that attempts to solve a game given:
  - an open GameSession
  - the game's GameProfile
  - a Budget

Strategies must:
  1. Respect the budget (check .expired() before each action)
  2. Track levels completed and add new ones to result.new_levels
  3. Handle GAME_OVER by calling sess.soft_reset() (which costs a life)
  4. Not modify the registry directly — the orchestrator does that

Strategies SHOULD:
  - Use self.log() for human-readable progress
  - Return a complete StrategyResult even on failure
"""
from __future__ import annotations

import time
from typing import List

from meta.budget import Budget
from meta.common import GameSession
from meta.profile import GameProfile
from meta.result import StrategyResult


class Strategy:
    """Override .name, .confidence(), and .run()."""
    name: str = "base"

    # ── Overrides ────────────────────────────────────────────────────────────
    def confidence(self, profile: GameProfile) -> float:
        """Return 0.0-1.0 how likely this strategy is to work on this game."""
        return 0.0

    def run(self, sess: GameSession, profile: GameProfile, budget: Budget) -> StrategyResult:
        """Execute the strategy. Must return a StrategyResult (never None).

        The base run() creates the result scaffolding — subclasses call
        self._start() and self._finalize() around their logic.
        """
        raise NotImplementedError

    # ── Helpers for subclasses ───────────────────────────────────────────────
    def _start(self, sess: GameSession, budget: Budget) -> StrategyResult:
        budget.bind(steps_at_start=sess.steps_taken, lives_at_start=sess.lives_used)
        return StrategyResult(
            game_id=sess.game_id,
            strategy=self.name,
            started_at=time.time(),
        )

    def _finalize(self, res: StrategyResult, sess: GameSession, budget: Budget,
                  stopped_reason: str = "") -> StrategyResult:
        res.elapsed = time.time() - res.started_at
        res.max_levels_completed = max(res.max_levels_completed, sess.max_levels_completed)
        res.won_any_level = res.max_levels_completed > 0
        res.steps = budget.steps_used(sess.steps_taken)
        res.lives_used = budget.lives_used(sess.lives_used)
        if stopped_reason:
            res.stopped_reason = stopped_reason
        elif not res.stopped_reason or res.stopped_reason == "not_run":
            res.stopped_reason = "completed"
        return res

    def _note_level(self, res: StrategyResult, sess: GameSession) -> None:
        """Call this after any step to record newly-completed levels."""
        lv = sess.levels_completed
        if lv > res.max_levels_completed:
            for n in range(res.max_levels_completed + 1, lv + 1):
                if n not in res.new_levels:
                    res.new_levels.append(n)
            res.max_levels_completed = lv

    # Used by __init__ subclasses when they want a tagged log buffer
    _log_buf: list = None

    def log(self, msg: str) -> None:
        if self._log_buf is None:
            self._log_buf = []
        line = f"[{self.name}] {msg}"
        self._log_buf.append(line)
        print(line, flush=True)

    def log_text(self) -> str:
        return "\n".join(self._log_buf or [])
