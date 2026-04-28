"""grid_click — systematic ACTION5 clicks on grid positions.

Mechanic family (lp85, sp80):
  - Game has a playfield where clicking individual cells changes state
  - ACTION5 takes {x, y} click coordinates
  - Goal: find cells that produce state changes and exploit them

Algorithm:
  1. Build a coarse grid over the frame (step=8 by default).
  2. Click each grid position once; record which positions are "live" (produce
     non-trivial frame change) and which are "dead".
  3. Spend remaining budget cycling through live positions in an order biased
     toward recent successes.
  4. Detect and avoid damage cells (cells whose click reliably kills a life).

Confidence: profile.looks_like_grid_click (0 if ACTION5 is a no-op).
"""
from __future__ import annotations

import numpy as np

from meta.budget import Budget
from meta.common import GameSession
from meta.profile import GameProfile
from meta.result import StrategyResult
from meta.strategies.base import Strategy


class GridClick(Strategy):
    name = "grid_click"

    # Grid spacing for initial scan (cells). 8 → 64 positions on 64x64.
    GRID_STEP = 8
    # Minimum cells-changed (outside noise rows) to call a position "live"
    LIVE_THRESHOLD = 1

    # Rows that change on every action (life timer etc.) — excluded from diff
    NOISE_ROWS = frozenset([59, 60, 63])

    # Hook for subclasses to override GRID_STEP
    grid_step: int = 8

    def confidence(self, profile: GameProfile) -> float:
        # Base confidence from heuristic
        c = float(profile.looks_like_grid_click)
        # If ACTION5 or ACTION6 is the ONLY interesting action, bump confidence.
        # (e.g. lp85 has available_actions=[6] — it's click-driven by elimination.)
        avail = profile.available_actions or []
        if (5 in avail or 6 in avail) and not any(a in avail for a in [1, 2, 3, 4]):
            c = max(c, 0.6)
        return c

    def run(self, sess: GameSession, profile: GameProfile, budget: Budget) -> StrategyResult:
        res = self._start(sess, budget)
        res.details = {}

        r = sess.reset()
        if r.frame is None:
            res.stopped_reason = "reset_failed"
            return self._finalize(res, sess, budget)
        base_frame = r.frame.copy()

        # Pick a click action that's available (prefer 5, fall back to 6)
        avail = profile.available_actions or [5, 6]
        click_act = 5 if 5 in avail else (6 if 6 in avail else None)
        if click_act is None:
            self.log("No click action (5 or 6) available; skipping")
            res.stopped_reason = "no_click_action"
            return self._finalize(res, sess, budget)
        self.log(f"using click action {click_act}")
        res.details["click_action"] = click_act

        h, w = base_frame.shape
        # Phase 1: scan grid
        live_positions: list = []
        dead_positions: list = []
        terminal_count = 0
        DANGER_ABORT_THRESHOLD = 6  # abort if this many consecutive terminals in first 12 probes
        self.log(f"scanning {w//self.GRID_STEP}x{h//self.GRID_STEP} grid…")
        for y in range(self.GRID_STEP // 2, h, self.GRID_STEP):
            for x in range(self.GRID_STEP // 2, w, self.GRID_STEP):
                if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                    break

                # Early abort: if most probes so far are terminal (danger-field game)
                total_probed = len(live_positions) + len(dead_positions)
                if total_probed >= 12 and terminal_count >= DANGER_ABORT_THRESHOLD:
                    self.log(f"danger-field abort: {terminal_count}/{total_probed} probes terminal")
                    res.stopped_reason = "danger_field"
                    return self._finalize(res, sess, budget)

                rr = sess.reset()
                if rr.frame is None: break

                before = rr.frame.copy()
                r = sess.step_with_data(click_act, {"x": x, "y": y})
                self._note_level(res, sess)
                if r.frame is None:
                    dead_positions.append((x, y))
                    continue
                if r.terminal:
                    # This position causes GAME_OVER — don't exploit it in phase 2.
                    terminal_count += 1
                    dead_positions.append((x, y))
                    continue
                diff = before != r.frame
                for nr in self.NOISE_ROWS:
                    if nr < diff.shape[0]:
                        diff[nr, :] = False
                changed = int(np.sum(diff))
                if changed >= self.LIVE_THRESHOLD:
                    live_positions.append((x, y, changed))
                else:
                    dead_positions.append((x, y))

        live_positions.sort(key=lambda p: -p[2])  # most-active first
        self.log(f"live positions: {len(live_positions)} / {len(live_positions)+len(dead_positions)}")
        res.details["live_count"] = len(live_positions)
        res.details["live_positions"] = [list(p) for p in live_positions[:20]]

        if not live_positions:
            res.stopped_reason = "no_live_positions"
            return self._finalize(res, sess, budget)

        # Phase 1.5: burst test — for each live position, try N consecutive clicks.
        # Catches vc33-type games that need 3+ clicks in a row at the same spot.
        MAX_BURST = 8
        self.log(f"burst testing {len(live_positions)} positions (up to {MAX_BURST} clicks each)…")
        for bx, by, _ in live_positions:
            if budget.expired(steps=sess.steps_taken, lives=sess.lives_used): break
            rr = sess.reset()
            if rr.frame is None: break
            for n in range(MAX_BURST):
                if budget.expired(steps=sess.steps_taken, lives=sess.lives_used): break
                rb = sess.step_with_data(click_act, {"x": bx, "y": by})
                self._note_level(res, sess)
                if res.max_levels_completed > 0:
                    self.log(f"BURST WIN: ({bx},{by}) × {n+1} clicks")
                    res.details["win_burst_clicks"] = n + 1
                    res.details["win_click"] = [bx, by]
                    res.details["click_act"] = click_act
                    res.stopped_reason = "burst_win"
                    return self._finalize(res, sess, budget)
                if not rb.ok or rb.terminal:
                    break  # dangerous position or depleted; try next

        if res.max_levels_completed > 0:
            return self._finalize(res, sess, budget)

        # Phase 2: pound live positions in sequence, watching for wins / lives lost
        rr = sess.reset()
        if rr.frame is None:
            res.stopped_reason = "reset_failed_phase2"
            return self._finalize(res, sess, budget)

        cycle_count = 0
        while not budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
            cycle_count += 1
            for x, y, _act_est in live_positions:
                if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                    break
                r = sess.step_with_data(click_act, {"x": x, "y": y})
                self._note_level(res, sess)
                if r.terminal:
                    rr = sess.soft_reset()
                    if rr.frame is None or rr.terminal:
                        res.stopped_reason = "terminal_unrecoverable"
                        return self._finalize(res, sess, budget)
                    break  # restart the cycle

        res.details["cycle_count"] = cycle_count
        return self._finalize(res, sess, budget)
