"""cursor_walk — walk the cursor and sample reachable states.

Mechanic family (slider/navigation games):
  - Actions 3 and 4 move a cursor
  - Actions 1 and 2 commit/change state at current cursor position
  - The puzzle involves positioning a cursor correctly before acting

Algorithm:
  1. Cycle cursor via action 4 to enumerate positions until it wraps (or a cap).
  2. At each cursor position, try action 1 then action 2 and note outcomes.
  3. Spend remaining budget stress-testing any positions that produced
     unique state transitions.

Confidence: profile.looks_like_cursor_game; extra nudge if ACTION5 is a no-op
(rules out grid_click).
"""
from __future__ import annotations

import numpy as np

from meta.budget import Budget
from meta.common import GameSession
from meta.profile import GameProfile
from meta.result import StrategyResult
from meta.strategies.base import Strategy


class CursorWalk(Strategy):
    name = "cursor_walk"

    MAX_POSITIONS = 16  # cap on distinct cursor positions we'll probe

    def confidence(self, profile: GameProfile) -> float:
        c = float(profile.looks_like_cursor_game)
        # Bonus if grid_click is weak (ACTION5 likely a no-op)
        if profile.looks_like_grid_click < 0.1:
            c += 0.1
        return min(c, 1.0)

    def run(self, sess: GameSession, profile: GameProfile, budget: Budget) -> StrategyResult:
        res = self._start(sess, budget)
        r = sess.reset()
        if r.frame is None:
            res.stopped_reason = "reset_failed"
            return self._finalize(res, sess, budget)

        fwd_cursor = 4
        fwd_value = 1

        # Enumerate distinct cursor positions
        positions = []  # list of (pos_index, cursor_indicator_hash)
        positions.append((0, _indicator_hash(r.frame)))
        for i in range(1, self.MAX_POSITIONS):
            if budget.expired(steps=sess.steps_taken, lives=sess.lives_used): break
            rr = sess.step(fwd_cursor)
            self._note_level(res, sess)
            if not rr.ok or rr.frame is None:
                break
            h = _indicator_hash(rr.frame)
            if any(h == p[1] for p in positions):
                break  # wrapped
            positions.append((i, h))

        self.log(f"found {len(positions)} distinct cursor positions")
        res.details["cursor_positions"] = len(positions)

        # At each position, try values
        rr = sess.reset()
        if rr.frame is None:
            return self._finalize(res, sess, budget, stopped_reason="reset_failed")

        for pos in range(len(positions)):
            if budget.expired(steps=sess.steps_taken, lives=sess.lives_used): break
            # walk cursor forward pos times
            for _ in range(pos):
                if not sess.step(fwd_cursor).ok: break
                self._note_level(res, sess)
            # press value action up to 12 times
            for _ in range(12):
                if budget.expired(steps=sess.steps_taken, lives=sess.lives_used): break
                r = sess.step(fwd_value)
                self._note_level(res, sess)
                if r.terminal:
                    rr2 = sess.soft_reset()
                    if rr2.frame is None or rr2.terminal:
                        return self._finalize(res, sess, budget, stopped_reason="terminal_unrecoverable")
                    break

        return self._finalize(res, sess, budget)


def _indicator_hash(frame: np.ndarray) -> int:
    """Hash rows 46-51 (typical cursor indicator zone) — cheap signature."""
    if frame is None or frame.shape[0] < 52:
        return 0
    return int(frame[46:52].tobytes().__hash__())
