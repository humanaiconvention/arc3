"""target_state_match — find target↔state region pairs and drive state to match.

Many ARC-AGI-3 games display a target pattern (unchanging) and a state pattern
(changes under actions). Win when state == target. This strategy:

1. Detect regions of the frame with significant content.
2. Classify each region:
   - STATIC: unchanged by any probe action
   - DYNAMIC: changes under at least one action
3. Find STATIC↔DYNAMIC pairs where regions have the same shape and a
   "matching possibility" (non-trivial similarity).
4. For each pair, greedily press actions that affect the dynamic region
   while checking similarity to the target; accept moves that decrease
   hamming distance.

This is a greedy local search — not optimal but covers many target-match
games without game-specific code.

Confidence: non-zero when at least one static/dynamic pair exists in the
frame. Read partly from profile.mimic_target reconnaissance (if available).
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from meta.budget import Budget
from meta.common import GameSession
from meta.profile import GameProfile
from meta.result import StrategyResult
from meta.strategies.base import Strategy


class TargetStateMatch(Strategy):
    name = "target_state_match"

    # Only consider regions that pass this minimum non-zero-cell count (filters
    # empty blocks).
    MIN_ACTIVE_CELLS = 3
    # Max regions to probe (limits time)
    MAX_REGIONS = 24
    # Max outer iterations of the greedy search
    MAX_GREEDY_ITERS = 30
    # Per-iteration max actions tried on a single pair
    MAX_ACTIONS_PER_ITER = 7
    # Rows that change on every action (life timer, scorecard)
    NOISE_ROWS = frozenset([59, 60, 63])

    def confidence(self, profile: GameProfile) -> float:
        # Use the fact that mimic_target classifies regions; if it has both
        # static and dynamic, we're eligible. Since mimic_target runs separately
        # we don't have its data here — approximate from GameProfile:
        # non-empty set of dynamic rows + non-static rows = plausible.
        if profile.n_dynamic_rows == 0:
            return 0.0
        # Modest default; we don't want to out-rank specific strategies
        base = 0.25
        # Bonus if high-entropy regions exist (there's actual content to match)
        if len(profile.high_entropy_regions) >= 2:
            base += 0.1
        return min(base, 0.5)

    # ── Main ────────────────────────────────────────────────────────────────
    def run(self, sess: GameSession, profile: GameProfile, budget: Budget) -> StrategyResult:
        res = self._start(sess, budget)
        res.details = {}

        r = sess.reset()
        if r.frame is None:
            res.stopped_reason = "reset_failed"
            return self._finalize(res, sess, budget)
        base_frame = r.frame.copy()

        # 1. Extract candidate regions (8x8 tiles with enough content)
        regions = self._extract_regions(base_frame)
        if not regions:
            res.stopped_reason = "no_regions"
            return self._finalize(res, sess, budget)

        # 2. Probe which regions change under which actions
        actions = profile.available_actions or [1, 2, 3, 4]
        reactivity = self._probe_reactivity(sess, base_frame, regions, actions, budget)
        if reactivity is None:
            res.stopped_reason = "reactivity_probe_failed"
            return self._finalize(res, sess, budget)

        # 3. Classify regions
        static_idxs = [i for i in range(len(regions)) if not any(i in reactivity[a] for a in actions)]
        dynamic_idxs = [i for i in range(len(regions)) if any(i in reactivity[a] for a in actions)]
        self.log(f"regions: {len(regions)} total, {len(static_idxs)} static, {len(dynamic_idxs)} dynamic")
        res.details["n_regions"] = len(regions)
        res.details["n_static"] = len(static_idxs)
        res.details["n_dynamic"] = len(dynamic_idxs)

        if not static_idxs or not dynamic_idxs:
            res.stopped_reason = "no_pair"
            return self._finalize(res, sess, budget)

        # 4. Find same-shape pairs
        pairs = []
        for s in static_idxs:
            ss = regions[s]
            for d in dynamic_idxs:
                dd = regions[d]
                if (ss[1]-ss[0], ss[3]-ss[2]) == (dd[1]-dd[0], dd[3]-dd[2]):
                    pairs.append((s, d))
        if not pairs:
            res.stopped_reason = "no_shape_matched_pairs"
            return self._finalize(res, sess, budget)
        self.log(f"candidate (target, state) pairs: {len(pairs)}")
        res.details["n_pairs"] = len(pairs)

        # 5. Greedy match attempt on each pair
        for s_idx, d_idx in pairs[: self.MAX_REGIONS]:
            if budget.expired(steps=sess.steps_taken, lives=sess.lives_used): break
            got = self._greedy_match(sess, res, budget,
                                      target_region=regions[s_idx],
                                      state_region=regions[d_idx],
                                      reactivity=reactivity,
                                      actions=actions)
            if res.max_levels_completed > 0:
                res.details["win_pair"] = [s_idx, d_idx]
                break
            if got == "stuck":
                continue

        return self._finalize(res, sess, budget)

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _extract_regions(self, frame: np.ndarray) -> list:
        """8x8-tile regions with MIN_ACTIVE_CELLS distinct from background."""
        h, w = frame.shape
        bg = self._background_value(frame)
        step = 8
        regions = []
        for r0 in range(0, h - step + 1, step):
            for c0 in range(0, w - step + 1, step):
                # Skip tiles that are mostly in noise rows
                if any(nr in range(r0, r0 + step) for nr in self.NOISE_ROWS) and \
                   sum(1 for nr in self.NOISE_ROWS if r0 <= nr < r0 + step) >= step // 2:
                    continue
                sub = frame[r0:r0 + step, c0:c0 + step]
                active = int(np.sum(sub != bg))
                if active >= self.MIN_ACTIVE_CELLS:
                    regions.append([r0, r0 + step, c0, c0 + step])
        return regions[: self.MAX_REGIONS]

    def _background_value(self, frame: np.ndarray) -> int:
        """Most common value in the frame = background."""
        vals, counts = np.unique(frame, return_counts=True)
        return int(vals[np.argmax(counts)])

    def _probe_reactivity(self, sess: GameSession, base: np.ndarray,
                          regions: list, actions: list, budget: Budget) -> Optional[dict]:
        """For each action, return list of region indices that change."""
        out = {}
        for act in actions:
            if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                return None
            rr = sess.reset()
            if rr.frame is None: return None
            r = sess.step(act)
            self._note_level(sess=sess, res=None) if False else None  # no-op; we track elsewhere
            if not r.ok or r.frame is None:
                out[act] = []; continue
            changed = []
            for i, reg in enumerate(regions):
                r0, r1, c0, c1 = reg
                if not np.array_equal(base[r0:r1, c0:c1], r.frame[r0:r1, c0:c1]):
                    changed.append(i)
            out[act] = changed
        return out

    def _greedy_match(self, sess: GameSession, res: StrategyResult, budget: Budget,
                      target_region: list, state_region: list,
                      reactivity: dict, actions: list) -> str:
        """Press actions (that affect state_region) to minimize pixel hamming
        distance to target_region. Returns 'stuck' if no improvement, 'done'
        on budget expiry, or 'won' if level advanced.
        """
        rr = sess.reset()
        if rr.frame is None: return "stuck"

        def extract(frame, reg):
            r0, r1, c0, c1 = reg
            return frame[r0:r1, c0:c1]

        def dist():
            if sess.frame is None: return 1 << 30
            t = extract(rr.frame, target_region)
            s = extract(sess.frame, state_region)
            if t.shape != s.shape: return 1 << 30
            return int(np.sum(t != s))

        current_dist = dist()
        relevant_acts = [a for a in actions
                         if state_region_idx_in(state_region, reactivity, a)]
        if not relevant_acts:
            return "stuck"

        for _ in range(self.MAX_GREEDY_ITERS):
            if budget.expired(steps=sess.steps_taken, lives=sess.lives_used): return "done"
            improved = False
            for act in relevant_acts:
                if budget.expired(steps=sess.steps_taken, lives=sess.lives_used): return "done"
                r = sess.step(act)
                self._note_level(res, sess)
                if res.max_levels_completed > 0:
                    return "won"
                if not r.ok or r.frame is None:
                    # Recover if needed
                    rr = sess.soft_reset()
                    if rr.frame is None or rr.terminal: return "done"
                    current_dist = dist()
                    continue
                new_d = dist()
                if new_d < current_dist:
                    current_dist = new_d
                    improved = True
                    self.log(f"  improved: dist={current_dist}")
                    if current_dist == 0:
                        # State now equals target — game may not auto-win; keep trying other actions
                        pass
                else:
                    # Undo by pressing the inverse (crude: press same action K-1 times to cycle back)
                    # We don't know K. Accept the drift; next iteration may recover.
                    pass
            if not improved:
                return "stuck"
        return "done"


def state_region_idx_in(state_region: list, reactivity: dict, act: int) -> bool:
    """True if any region matching state_region's bounds is listed in reactivity[act].
    We approximate: an action is 'relevant' if it changes ANY region (safer to include)."""
    return len(reactivity.get(act, [])) > 0
