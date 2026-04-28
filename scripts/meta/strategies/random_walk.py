"""random_walk — baseline floor.

Picks random available actions. Useful for:
  - Catching trivial games where a short random sequence wins.
  - Establishing a floor performance every game should beat.
  - Sampling reachable states when characterization is inconclusive.

Confidence: 0.15 (always a bit of budget, but below serious strategies).
"""
from __future__ import annotations

import random

from meta.budget import Budget
from meta.common import GameSession
from meta.profile import GameProfile
from meta.result import StrategyResult
from meta.strategies.base import Strategy


class RandomWalk(Strategy):
    name = "random_walk"

    MAX_STEPS = 1000

    def confidence(self, profile: GameProfile) -> float:
        # Always a small baseline, extra nudge if the game looks static
        # (random actions are as good as anything for static games).
        base = 0.15
        if profile.looks_like_static > 0.5:
            base += 0.1
        return base

    def run(self, sess: GameSession, profile: GameProfile, budget: Budget) -> StrategyResult:
        res = self._start(sess, budget)
        r = sess.reset()
        if r.frame is None:
            res.stopped_reason = "reset_failed"
            return self._finalize(res, sess, budget)

        actions = profile.available_actions or [1, 2, 3, 4]
        rng = random.Random(0xA3C3)
        steps_done = 0

        # Record action sequence so replay_known can replay a winner.
        # We keep the sequence short after each win so replay doesn't retry
        # the same randomness forever.
        sequence = []  # list of (act_id, level_after)
        level_at_win = {}  # {level: sequence_index_just_after_win}

        while steps_done < self.MAX_STEPS:
            if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                break
            act = rng.choice(actions)
            if act in (5, 6, 7):
                data = {"x": rng.randint(0, 63), "y": rng.randint(0, 63)}
                r = sess.step_with_data(act, data)
                sequence.append({"act": act, "data": data})
            else:
                r = sess.step(act)
                sequence.append(act)
            lv_before = res.max_levels_completed
            self._note_level(res, sess)
            steps_done += 1
            if res.max_levels_completed > lv_before:
                # New level reached at this step index
                level_at_win[res.max_levels_completed] = len(sequence)
            if r.terminal:
                rr = sess.soft_reset()
                if rr.frame is None or rr.terminal:
                    res.stopped_reason = "terminal_unrecoverable"
                    break

        res.details["steps_done"] = steps_done
        if level_at_win:
            # Record the sequence up to (and including) the step that won L1
            if 1 in level_at_win:
                res.details["win_sequence"] = sequence[: level_at_win[1]]
                res.details["win_rng_seed"] = 0xA3C3
        return self._finalize(res, sess, budget)
