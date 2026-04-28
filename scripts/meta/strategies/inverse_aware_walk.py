"""inverse_aware_walk — random walk that prunes inverse-pair undos.

Uses profile.inverse_pairs and profile.toggle_actions (derived by characterize)
to do smarter exploration than RandomWalk:

  1. Never apply an action immediately after its inverse (would undo the
     previous step — wasted budget).
  2. Periodically inject a toggle action (e.g. re86's act5) as a commit.
  3. Otherwise uniform-random over remaining actions.

Confidence: 0.40 if profile.inverse_pairs is non-empty (meaningful action
graph structure exists). Otherwise 0 — degenerate case is just random_walk.

This strategy is the proof-of-concept that the new derived signals
(inverse_pairs, toggle_actions) are actionable, not just descriptive.
"""
from __future__ import annotations

import random

from meta.budget import Budget
from meta.common import GameSession
from meta.profile import GameProfile
from meta.result import StrategyResult
from meta.strategies.base import Strategy


class InverseAwareWalk(Strategy):
    name = "inverse_aware_walk"

    MAX_STEPS = 1000
    TOGGLE_PERIOD = 8           # inject a toggle action every N steps
    SOFT_RESET_AFTER = 60       # restart after this many steps with no progress
    RNG_SEED = 41923            # match random_walk's seed for comparability
    DOMINANT_WEIGHT = 4         # weight multiplier for profile.dominant_actions

    def confidence(self, profile: GameProfile) -> float:
        if not profile.inverse_pairs:
            return 0.0
        # Need at least 2 actions in the available set
        avail = profile.available_actions or []
        if len(avail) < 2:
            return 0.0
        # Skip coord-only games (clicks need coords; this strategy doesn't probe)
        if any(a in avail for a in [5, 6, 7]) and not any(a in avail for a in [1, 2, 3, 4]):
            return 0.0
        return 0.40

    def run(self, sess: GameSession, profile: GameProfile, budget: Budget) -> StrategyResult:
        res = self._start(sess, budget)
        res.details = {}

        # Build inverse map: action → set of actions that undo it. Some games
        # (g50t, tu93) have triangular inverse subgroups (every member of a
        # set undoes every other), so each action can have multiple inverses.
        forbidden_after: dict = {}
        for a, b in profile.inverse_pairs:
            forbidden_after.setdefault(a, set()).add(b)
            forbidden_after.setdefault(b, set()).add(a)

        toggles = list(profile.toggle_actions or [])
        # Walk space excludes coord-only clicks (no default position to use)
        avail = [a for a in (profile.available_actions or []) if a in (1, 2, 3, 4, 5)]
        if not avail:
            res.stopped_reason = "no_walk_actions"
            return self._finalize(res, sess, budget)

        # Bias toward dominant actions when sampling. profile.dominant_actions
        # is the set of actions that actually move state — for static games
        # this comes from post-prelude probing, otherwise from initial-state
        # action_effects.
        dominant_set = set(int(a) for a in (profile.dominant_actions or []) if a in avail)
        res.details["dominant_set"] = sorted(dominant_set)

        rng = random.Random(self.RNG_SEED)

        rr = sess.reset()
        if rr.frame is None:
            res.stopped_reason = "reset_failed"
            return self._finalize(res, sess, budget)

        last_act: int = None
        step_seq: list = []
        steps_since_progress = 0
        prior_max_levels = res.max_levels_completed

        for step_i in range(self.MAX_STEPS):
            if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                res.stopped_reason = "budget_expired"
                break

            # Periodic toggle injection — commit whatever state we've built up.
            if toggles and (step_i + 1) % self.TOGGLE_PERIOD == 0:
                act = rng.choice(toggles)
            else:
                # Random with inverse-pruning AND dominant-action bias.
                forbidden = forbidden_after.get(last_act, set())
                choices = [a for a in avail if a not in forbidden]
                if not choices:
                    choices = avail  # everything forbidden — fall back
                if dominant_set:
                    weights = [self.DOMINANT_WEIGHT if a in dominant_set else 1
                               for a in choices]
                    act = rng.choices(choices, weights=weights, k=1)[0]
                else:
                    act = rng.choice(choices)

            r = sess.step(act)
            step_seq.append(act)
            self._note_level(res, sess)

            if res.max_levels_completed > prior_max_levels:
                prior_max_levels = res.max_levels_completed
                steps_since_progress = 0
                # Record win sequence so future strategies can replay-derive.
                res.details.setdefault("win_sequence", list(step_seq))
                res.details["win_step_count"] = step_i + 1
                res.stopped_reason = "level_won"
                self.log(f"WIN at step {step_i+1}: act={act} levels={res.max_levels_completed}")
                break

            steps_since_progress += 1
            if r.terminal:
                rr = sess.soft_reset()
                if rr.frame is None or rr.terminal:
                    res.stopped_reason = "terminal_unrecoverable"
                    break
                last_act = None
                step_seq = []
                continue

            if steps_since_progress >= self.SOFT_RESET_AFTER:
                rr = sess.reset()
                if rr.frame is None:
                    res.stopped_reason = "reset_failed_midwalk"
                    break
                last_act = None
                step_seq = []
                steps_since_progress = 0
                continue

            last_act = act

        res.details.setdefault("steps_done", sess.steps_taken)
        res.details.setdefault("inverse_pairs_used", profile.inverse_pairs)
        res.details.setdefault("toggles_used", toggles)
        return self._finalize(res, sess, budget)
