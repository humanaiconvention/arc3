"""sequence_search — ordered click-sequence search for click-only games.

This is aimed at games where a single live position is not enough, but a short
ordered sequence of coordinate clicks may win the level.
"""
from __future__ import annotations

import itertools

import numpy as np

from meta.budget import Budget
from meta.common import GameSession
from meta.profile import GameProfile
from meta.result import StrategyResult
from meta.strategies.grid_click import GridClick


class SequenceSearch(GridClick):
    name = "sequence_search"

    GRID_STEP = 8
    MAX_CANDIDATES = 5
    MIN_CANDIDATES = 2
    MIN_SPACING = 8

    def confidence(self, profile: GameProfile) -> float:
        avail = set(profile.available_actions or [])
        has_click = bool(avail & {5, 6})
        if not avail or not has_click:
            return 0.0
        if avail.issubset({5, 6}):
            return max(0.55, float(profile.looks_like_grid_click) * 0.9)
        if avail & {1, 2, 3, 4}:
            # Mixed nav/click games can still be short ordered click puzzles
            # after navigation has no effect or a default cursor state matters.
            signal = max(float(profile.looks_like_grid_click), float(profile.looks_like_cursor_game))
            return max(0.20, min(0.45, signal * 0.6))
        return 0.0

    def run(self, sess: GameSession, profile: GameProfile, budget: Budget) -> StrategyResult:
        res = self._start(sess, budget)
        res.details = {}

        r = sess.reset()
        if r.frame is None:
            res.stopped_reason = "reset_failed"
            return self._finalize(res, sess, budget)

        avail = profile.available_actions or [5, 6]
        click_act = 5 if 5 in avail else (6 if 6 in avail else None)
        if click_act is None:
            res.stopped_reason = "no_click_action"
            return self._finalize(res, sess, budget)

        self.log(f"using click action {click_act}")
        res.details["click_action"] = click_act
        res.details["scan_step"] = self.GRID_STEP

        # Prefer the cached click heat-map from characterize. Falls back to a
        # local scan if the profile lacks one (older profiles, or probe was
        # disabled / aborted).
        cached_hm = [
            (int(e["x"]), int(e["y"]), int(e["cells_changed"]))
            for e in (profile.click_heatmap or [])
            if int(e.get("act", click_act)) == click_act
        ]
        if cached_hm:
            live_positions = sorted(cached_hm, key=lambda p: (-p[2], p[1], p[0]))
            res.details["live_source"] = "profile_heatmap"
        else:
            live_positions = self._scan_live_positions(sess, budget, click_act)
            res.details["live_source"] = "local_scan"
        res.details["live_count"] = len(live_positions)
        res.details["live_positions"] = [list(p) for p in live_positions[:20]]
        if not live_positions:
            res.stopped_reason = "no_live_positions"
            return self._finalize(res, sess, budget)

        candidates = self._select_candidates(live_positions)
        res.details["candidate_positions"] = [list(p) for p in candidates]
        self.log(f"selected {len(candidates)} candidates: {[(x, y, c) for x, y, c in candidates]}")
        if len(candidates) < self.MIN_CANDIDATES:
            res.stopped_reason = "not_enough_candidates"
            return self._finalize(res, sess, budget)

        attempts = 0
        max_len = 4 if len(candidates) <= 4 else 3
        res.details["max_sequence_length"] = max_len

        for length in range(2, max_len + 1):
            self.log(f"trying ordered sequences of length {length}")
            for seq in itertools.product(candidates, repeat=length):
                if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                    res.details["sequence_attempts"] = attempts
                    res.stopped_reason = f"budget_expired_{budget.why_expired(sess.steps_taken, sess.lives_used)}"
                    return self._finalize(res, sess, budget)

                attempts += 1
                rr = sess.reset()
                if rr.frame is None:
                    res.details["sequence_attempts"] = attempts
                    res.stopped_reason = "reset_failed"
                    return self._finalize(res, sess, budget)

                for idx, (x, y, _changed) in enumerate(seq, start=1):
                    if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                        res.details["sequence_attempts"] = attempts
                        res.stopped_reason = f"budget_expired_{budget.why_expired(sess.steps_taken, sess.lives_used)}"
                        return self._finalize(res, sess, budget)

                    step = sess.step_with_data(click_act, {"x": x, "y": y})
                    self._note_level(res, sess)
                    if res.max_levels_completed > 0:
                        winning_seq = [
                            {"act": click_act, "x": sx, "y": sy}
                            for sx, sy, _score in seq[:idx]
                        ]
                        self.log(f"WIN via sequence #{attempts}: {winning_seq}")
                        res.details["sequence_attempts"] = attempts
                        res.details["win_sequence"] = winning_seq
                        res.details["win_click_path"] = [[sx, sy] for sx, sy, _score in seq[:idx]]
                        res.details["win_sequence_length"] = idx
                        res.stopped_reason = "sequence_win"
                        return self._finalize(res, sess, budget)
                    if not step.ok or step.terminal:
                        break

        res.details["sequence_attempts"] = attempts
        res.stopped_reason = "search_exhausted"
        return self._finalize(res, sess, budget)

    def _scan_live_positions(self, sess: GameSession, budget: Budget, click_act: int) -> list[tuple[int, int, int]]:
        frame = sess.frame
        if frame is None:
            return []

        h, w = frame.shape
        live_positions: list[tuple[int, int, int]] = []
        self.log(f"scanning {w // self.GRID_STEP}x{h // self.GRID_STEP} grid for live positions")
        for y in range(self.GRID_STEP // 2, h, self.GRID_STEP):
            for x in range(self.GRID_STEP // 2, w, self.GRID_STEP):
                if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                    return sorted(live_positions, key=lambda p: (-p[2], p[1], p[0]))

                rr = sess.reset()
                if rr.frame is None:
                    return sorted(live_positions, key=lambda p: (-p[2], p[1], p[0]))

                before = rr.frame.copy()
                step = sess.step_with_data(click_act, {"x": x, "y": y})
                if step.frame is None or not step.ok or step.terminal:
                    continue

                diff = before != step.frame
                for nr in self.NOISE_ROWS:
                    if nr < diff.shape[0]:
                        diff[nr, :] = False
                changed = int(np.sum(diff))
                if changed >= self.LIVE_THRESHOLD:
                    live_positions.append((x, y, changed))

        return sorted(live_positions, key=lambda p: (-p[2], p[1], p[0]))

    def _select_candidates(self, live_positions: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        if not live_positions:
            return []

        ordered = sorted(live_positions, key=lambda p: (-p[2], p[1], p[0]))
        selected: list[tuple[int, int, int]] = []
        anchor = ordered[0]
        nearby = [anchor]
        nearby.extend(
            sorted(
                ordered[1:],
                key=lambda p: (
                    abs(p[0] - anchor[0]) + abs(p[1] - anchor[1]),
                    -p[2],
                    p[1],
                    p[0],
                ),
            )
        )

        for x, y, changed in nearby:
            if all(abs(x - sx) + abs(y - sy) >= self.MIN_SPACING for sx, sy, _ in selected):
                selected.append((x, y, changed))
            if len(selected) >= self.MAX_CANDIDATES:
                return selected

        for pos in ordered:
            if pos not in selected:
                selected.append(pos)
            if len(selected) >= self.MAX_CANDIDATES:
                break
        return selected
