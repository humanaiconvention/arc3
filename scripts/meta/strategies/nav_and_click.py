"""nav_and_click — alternate between navigation and clicking.

Mechanic family (many grid games where you must navigate a cursor/piece to
a target position and then commit):
  - Actions 3 and 4 move something (cursor, piece, selection)
  - Action 5 or 6 clicks/commits
  - Goal: find the right (navigation, click-target) pairs

Algorithm:
  1. Map click live-cells using a coarse grid (reuse grid_click logic in-line).
  2. For each live cell, try variants: click directly vs. navigate-then-click.
  3. Also try cursor cycles followed by click at varied positions.

Confidence: ACTION5 or ACTION6 + actions 3/4 both available and responsive.
"""
from __future__ import annotations

import numpy as np

from meta.budget import Budget
from meta.common import GameSession
from meta.profile import GameProfile
from meta.result import StrategyResult
from meta.strategies.base import Strategy


class NavAndClick(Strategy):
    name = "nav_and_click"

    GRID_STEP = 8  # matches grid_click coarse grid; fine enough to catch most buttons
    NOISE_ROWS = frozenset([59, 60, 63])

    def confidence(self, profile: GameProfile) -> float:
        avail = profile.available_actions or []
        has_click = 5 in avail or 6 in avail
        has_nav = 3 in avail and 4 in avail
        if not (has_click and has_nav):
            return 0.0
        # Both nav and click available — plausible for nav-and-click mechanics
        base = 0.3
        # Bonus if both produce non-trivial effects
        def eff(a): return profile.action_effects.get(str(a), {}).get("cells_changed", 0)
        if eff(3) > 0 and eff(5) + eff(6) > 0:
            base += 0.2
        return min(base, 0.7)

    def run(self, sess: GameSession, profile: GameProfile, budget: Budget) -> StrategyResult:
        res = self._start(sess, budget)
        res.details = {}

        avail = profile.available_actions or []
        click_act = 5 if 5 in avail else 6
        self.log(f"nav_and_click: click_act={click_act}")

        r = sess.reset()
        if r.frame is None:
            res.stopped_reason = "reset_failed"
            return self._finalize(res, sess, budget)
        h, w = r.frame.shape

        # Strategy A: cursor cycle + coarse click grid
        # For each cursor cycle position (0 to 4 or 5 presses of action 4),
        # try clicking each coarse grid position.
        positions = []
        for y in range(self.GRID_STEP // 2, h, self.GRID_STEP):
            for x in range(self.GRID_STEP // 2, w, self.GRID_STEP):
                positions.append((x, y))

        for cycle_presses in range(8):  # 0-7 nav presses covers most cursor-based games
            if budget.expired(steps=sess.steps_taken, lives=sess.lives_used): break
            for x, y in positions:
                if budget.expired(steps=sess.steps_taken, lives=sess.lives_used): break

                rr = sess.reset()
                if rr.frame is None: return self._finalize(res, sess, budget,
                                                           stopped_reason="reset_failed_mid_run")
                # Cycle cursor `cycle_presses` times
                ok = True
                for _ in range(cycle_presses):
                    r1 = sess.step(4)
                    self._note_level(res, sess)
                    if not r1.ok:
                        ok = False; break
                if not ok:
                    continue
                # Click
                r2 = sess.step_with_data(click_act, {"x": x, "y": y})
                self._note_level(res, sess)
                if res.max_levels_completed > 0:
                    self.log(f"WIN: cycle={cycle_presses} click=({x},{y})")
                    res.details["win_cycle_presses"] = cycle_presses
                    res.details["win_click"] = [x, y]
                    res.details["click_act"] = click_act
                    res.details["cycle_act"] = 4
                    return self._finalize(res, sess, budget)

        return self._finalize(res, sess, budget)
