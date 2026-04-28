"""mover_toggle_walk — directed strategy for games with mover/toggle structure.

Hypothesis: some games are organised around a small set of "mover" actions
that change game state and a small set of "toggle" actions that flip a
binary state (commit/select/edit-mode). The win is reached by alternating:

    mover × N  →  toggle  →  mover × M  →  toggle  →  …

This generalises the g50t hypothesis from the original handoff: "after
act4×5 'right-drop', act5 at #2 undoes drop and converts much of original
val1 piece to val2." The state-conditional probe confirmed g50t's acts 2/4
unfreeze the puzzle (mover) and act 5 is a self-inverse toggle.

Confidence: requires both `dominant_actions` and `toggle_actions` non-empty
on the profile. Returns 0 otherwise — no specific structure to direct.

Cost: per (mover, n_before, m_after) combo: ~1 + n_before + |toggles| × (1 + m_after) ~ 10 calls.
With 2 movers × 2 n_before × 2 m_after = 8 combos × ~10 calls = ~24s + budget left
for cycle iteration.
"""
from __future__ import annotations

import itertools

from meta.budget import Budget
from meta.common import GameSession
from meta.profile import GameProfile
from meta.result import StrategyResult
from meta.strategies.base import Strategy


class MoverToggleWalk(Strategy):
    name = "mover_toggle_walk"

    N_BEFORE_CHOICES = (3, 5)
    M_AFTER_CHOICES = (1, 3)
    MAX_CYCLES_PER_COMBO = 4    # alternations of toggle+mover-burst within one attempt

    def confidence(self, profile: GameProfile) -> float:
        if not profile.dominant_actions:
            return 0.0
        if not profile.toggle_actions:
            return 0.0
        # Need at least one nav-style mover (the toggle is the inflection,
        # the mover does the work).
        if not any(a in (1, 2, 3, 4) for a in profile.dominant_actions):
            return 0.0
        return 0.45

    def run(self, sess: GameSession, profile: GameProfile, budget: Budget) -> StrategyResult:
        res = self._start(sess, budget)
        res.details = {}

        movers = [a for a in profile.dominant_actions if a in (1, 2, 3, 4)]
        toggles = list(profile.toggle_actions or [])
        if not movers or not toggles:
            res.stopped_reason = "missing_mover_or_toggle"
            return self._finalize(res, sess, budget)

        res.details.update({
            "movers": movers,
            "toggles": toggles,
        })
        self.log(f"movers={movers}  toggles={toggles}")

        attempts = 0
        for mover, n_before, m_after in itertools.product(
            movers, self.N_BEFORE_CHOICES, self.M_AFTER_CHOICES
        ):
            for toggle in toggles:
                if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                    res.details["attempts"] = attempts
                    res.stopped_reason = "budget_expired"
                    return self._finalize(res, sess, budget)
                attempts += 1

                won, win_seq = self._try_cycle(
                    sess, mover, n_before, toggle, m_after, res
                )
                if won:
                    res.details.update({
                        "win_sequence": win_seq,
                        "win_mover": mover,
                        "win_n_before": n_before,
                        "win_toggle": toggle,
                        "win_m_after": m_after,
                        "attempts": attempts,
                    })
                    res.stopped_reason = "mover_toggle_win"
                    self.log(
                        f"WIN attempt #{attempts}: "
                        f"mover={mover}×{n_before} toggle={toggle} after={m_after}"
                    )
                    return self._finalize(res, sess, budget)

        res.details["attempts"] = attempts
        res.stopped_reason = "exhausted_no_win"
        return self._finalize(res, sess, budget)

    def _try_cycle(self, sess: GameSession, mover: int, n_before: int,
                   toggle: int, m_after: int,
                   res: StrategyResult) -> tuple[bool, list]:
        """One attempt: reset, mover×n_before, then up to MAX_CYCLES_PER_COMBO
        of (toggle, mover×m_after). Returns (won, sequence)."""
        rr = sess.reset()
        if rr.frame is None:
            return False, []
        seq: list = []

        for _ in range(n_before):
            r = sess.step(mover)
            seq.append(mover)
            self._note_level(res, sess)
            if res.max_levels_completed > 0:
                return True, seq
            if not r.ok or r.terminal:
                return False, seq

        for _ in range(self.MAX_CYCLES_PER_COMBO):
            r = sess.step(toggle)
            seq.append(toggle)
            self._note_level(res, sess)
            if res.max_levels_completed > 0:
                return True, seq
            if not r.ok or r.terminal:
                return False, seq
            for _ in range(m_after):
                r = sess.step(mover)
                seq.append(mover)
                self._note_level(res, sess)
                if res.max_levels_completed > 0:
                    return True, seq
                if not r.ok or r.terminal:
                    return False, seq

        return False, seq
