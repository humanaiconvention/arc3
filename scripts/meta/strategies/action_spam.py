"""action_spam — press each available action N times, see if anything wins.

This is the cheapest possible strategy. It's an always-on baseline that
catches games where a single button wins (rare, but possible) and provides
a cheap sanity check that the session + orchestrator plumbing works.

Confidence: always 0.1 (tiny but nonzero — we run it everywhere).
"""
from __future__ import annotations

from meta.budget import Budget
from meta.common import GameSession
from meta.profile import GameProfile
from meta.result import StrategyResult
from meta.strategies.base import Strategy


class ActionSpam(Strategy):
    name = "action_spam"

    # How many times to press each action before giving up on it.
    PRESSES_PER_ACTION = 10

    def confidence(self, profile: GameProfile) -> float:
        return 0.1  # always runs as a baseline

    def run(self, sess: GameSession, profile: GameProfile, budget: Budget) -> StrategyResult:
        res = self._start(sess, budget)

        # Fresh reset to start from known state.
        r = sess.reset()
        if r.frame is None:
            res.stopped_reason = "reset_failed"
            return self._finalize(res, sess, budget)

        actions = profile.available_actions or [1, 2, 3, 4]
        self.log(f"Spamming {actions} x {self.PRESSES_PER_ACTION}")

        for act_id in actions:
            for press in range(self.PRESSES_PER_ACTION):
                if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                    return self._finalize(
                        res, sess, budget,
                        stopped_reason=f"budget: {budget.why_expired(sess.steps_taken, sess.lives_used)}",
                    )

                r = sess.step(act_id)

                if r.won_any:
                    self._note_level(res, sess)

                if r.terminal:
                    # Try to continue via soft_reset (costs a life)
                    rr = sess.soft_reset()
                    if rr.frame is None:
                        return self._finalize(res, sess, budget, stopped_reason="terminal_unrecoverable")
                    # Don't keep spamming into a likely-failing action — break inner
                    break

                if r.exception:
                    # 500 server error etc. — skip this action
                    self.log(f"  action {act_id} exception: {r.exception}")
                    break

        return self._finalize(res, sess, budget)
