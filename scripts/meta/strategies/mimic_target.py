"""mimic_target — if the frame has distinctive target-like regions, try to match them.

Mechanic family (tr87-boxes / vc33 / cn04):
  - One part of the frame shows a "target" pattern
  - Another part shows the current state
  - The win condition is: current state matches target

Algorithm:
  1. Find regions of the frame with distinctive non-background content
     (excluding the cursor indicator row and life-timer row).
  2. Identify candidate (target_region, state_region) pairs by similarity of
     structure (same shape, overlapping value distributions).
  3. Attempt to drive the state region toward the target region by cycling
     actions 1/2 (if state region responds to them) while observing each
     action's effect on the state region.

This is a reconnaissance strategy: it records detailed observations so a
future pass can exploit them. It does NOT itself brute-force combos.

Confidence: small constant (0.2). Useful for collecting data on every game.
"""
from __future__ import annotations

import numpy as np

from meta.budget import Budget
from meta.common import GameSession
from meta.profile import GameProfile
from meta.result import StrategyResult
from meta.strategies.base import Strategy


class MimicTarget(Strategy):
    name = "mimic_target"

    def confidence(self, profile: GameProfile) -> float:
        return 0.2  # always run, but with modest budget

    def run(self, sess: GameSession, profile: GameProfile, budget: Budget) -> StrategyResult:
        res = self._start(sess, budget)
        r = sess.reset()
        if r.frame is None:
            res.stopped_reason = "reset_failed"
            return self._finalize(res, sess, budget)
        base_frame = r.frame.copy()

        # Identify non-background distinct regions
        regions = _find_regions(base_frame)
        self.log(f"detected {len(regions)} candidate regions")
        res.details["n_regions"] = len(regions)
        res.details["regions"] = regions[:20]

        # For each action, record which regions change
        reactivity: dict = {}
        for act in profile.available_actions or [1, 2, 3, 4]:
            if budget.expired(steps=sess.steps_taken, lives=sess.lives_used): break
            rr = sess.reset()
            if rr.frame is None: break
            f0 = rr.frame.copy()
            r = sess.step(act)
            self._note_level(res, sess)
            if not r.ok or r.frame is None: continue
            changed_regions = []
            for r_idx, reg in enumerate(regions):
                r0, r1, c0, c1 = reg
                before = f0[r0:r1, c0:c1]
                after = r.frame[r0:r1, c0:c1]
                if not np.array_equal(before, after):
                    changed_regions.append(r_idx)
            reactivity[act] = changed_regions
        res.details["action_reactivity"] = {str(a): list(r) for a, r in reactivity.items()}

        # Try: pair each "static" region (changes under 0 actions) with each
        # "dynamic" region and see if they could be target↔state pairs.
        static_idxs = [i for i in range(len(regions)) if not any(i in v for v in reactivity.values())]
        dynamic_idxs = [i for i in range(len(regions)) if any(i in v for v in reactivity.values())]
        self.log(f"static={len(static_idxs)}  dynamic={len(dynamic_idxs)}")
        res.details["static_regions"] = static_idxs
        res.details["dynamic_regions"] = dynamic_idxs

        res.stopped_reason = "reconnaissance_complete"
        return self._finalize(res, sess, budget)


def _find_regions(frame: np.ndarray) -> list:
    """Return list of [r0,r1,c0,c1] bounding rects of 'interesting' content.

    Uses a simple 4x4 tiling with a non-uniformity heuristic. Not perfect but
    fast and produces reasonable candidates on the typical 64x64 ARC frame.
    Skip the life-timer row (63) and the cursor indicator rows (46-51).
    """
    h, w = frame.shape
    skip_rows = set(list(range(46, 52)) + [63])
    regions = []
    rs, cs = max(1, h // 8), max(1, w // 8)
    for i in range(8):
        for j in range(8):
            r0, r1 = i * rs, min(h, (i + 1) * rs)
            c0, c1 = j * cs, min(w, (j + 1) * cs)
            if any(r in skip_rows for r in range(r0, r1)):
                continue
            sub = frame[r0:r1, c0:c1]
            u = len(np.unique(sub))
            if u >= 3:  # more than 2 values = likely interesting
                regions.append([int(r0), int(r1), int(c0), int(c1)])
    return regions
