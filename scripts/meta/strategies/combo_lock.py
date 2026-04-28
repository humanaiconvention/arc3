"""combo_lock — N-slot / K-cycle combination lock solver.

Mechanic family (tr87-style):
  - Cursor moves through N slots via two "cursor" actions
  - Each slot has K cyclic states; two "value" actions advance/retreat
  - Win when all slots simultaneously match a target combination
  - GAME_OVER when life depletes; RESET restarts a life (slots → 0, cursor → 0)

Auto-discovery decides N, K, and which actions are cursor vs value. We don't
hardcode tr87 specifics — the same code should beat any game in this family.

Confidence: profile.looks_like_combo_lock.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from meta.budget import Budget
from meta.common import GameSession
from meta.profile import GameProfile
from meta.result import StrategyResult
from meta.strategies.base import Strategy


class ComboLock(Strategy):
    name = "combo_lock"

    MAX_SLOTS = 9
    MAX_CYCLE = 12
    MAX_SEARCH_SPACE = 50_000  # K^N — skip if too large for brute force

    def confidence(self, profile: GameProfile) -> float:
        base = float(profile.looks_like_combo_lock)
        # Bump on the WASD inverse-pair signature: 1↔2 (value pair) and
        # 3↔4 (cursor pair) is the structural prerequisite tr87 satisfied.
        # All 5 games carrying this signature deserve a combo_lock attempt
        # even if surface heuristics don't fire (cd82/dc22/ka59/re86 may
        # look different superficially but share the action graph).
        pairs = [tuple(sorted(p)) for p in (profile.inverse_pairs or [])]
        if (1, 2) in pairs and (3, 4) in pairs:
            return max(base, 0.50)
        return base

    # ── Entry point ──────────────────────────────────────────────────────────
    def run(self, sess: GameSession, profile: GameProfile, budget: Budget) -> StrategyResult:
        res = self._start(sess, budget)
        res.details = {}

        r = sess.reset()
        if r.frame is None:
            res.stopped_reason = "reset_failed"
            return self._finalize(res, sess, budget)

        disc = self._discover(sess, budget)
        if disc is None:
            res.stopped_reason = "discovery_failed"
            return self._finalize(res, sess, budget)

        self.log(f"discovered: N={disc['n_slots']} at cols {disc['slot_cols']}, "
                 f"K={disc['cycle']}, fwd_cursor=act{disc['fwd_cursor']}, "
                 f"fwd_value=act{disc['fwd_value']}, "
                 f"search_space={disc['cycle']**disc['n_slots']}")
        res.details.update(disc)

        if disc["cycle"] ** disc["n_slots"] > self.MAX_SEARCH_SPACE:
            res.stopped_reason = f"search_space_too_large_{disc['cycle']**disc['n_slots']}"
            return self._finalize(res, sess, budget)

        # Fresh reset before brute force
        r = sess.reset()
        if r.frame is None:
            res.stopped_reason = "pre_brute_reset_failed"
            return self._finalize(res, sess, budget)

        won = self._brute_force(sess, res, budget, disc)
        if won is not None:
            res.details["win_combo"] = list(won)

        return self._finalize(res, sess, budget)

    # Rows that change on EVERY action (life timer, scorecard indicator).
    # Excluded when measuring the per-action "slot" footprint.
    NOISE_ROWS = frozenset([59, 60, 63])

    # ── Discovery ────────────────────────────────────────────────────────────
    def _discover(self, sess: GameSession, budget: Budget) -> Optional[dict]:
        """Infer N slots, K cycle, and action mapping."""
        # Action 1 vs action 3 comparison: value-cycling has narrower col footprint.
        def probe(act) -> Optional[tuple]:
            rr = sess.reset()
            if rr.frame is None: return None
            f0 = rr.frame.copy()
            r = sess.step(act)
            if not r.ok or r.frame is None: return None
            diff = f0 != r.frame
            # Zero out noise rows so they don't pollute row/col signatures
            for nr in self.NOISE_ROWS:
                if nr < diff.shape[0]:
                    diff[nr, :] = False
            rows = sorted(set(np.where(diff)[0].tolist()))
            cols = sorted(set(np.where(diff)[1].tolist()))
            return rows, cols

        p1 = probe(1); p3 = probe(3)
        if p1 is None or p3 is None:
            self.log("probe of actions 1/3 failed during discovery")
            return None

        # Narrower column footprint = value action
        if len(p1[1]) <= len(p3[1]):
            fwd_value, bwd_value = 1, 2
            fwd_cursor, bwd_cursor = 4, 3
            value_rows, value_cols = p1
        else:
            fwd_value, bwd_value = 3, 4
            fwd_cursor, bwd_cursor = 2, 1
            value_rows, value_cols = p3
        if not value_cols:
            return None

        # Infer cycle K by pressing value action and finding first repeat.
        rr = sess.reset()
        if rr.frame is None: return None
        r_min, r_max = min(value_rows), max(value_rows) + 1
        c_min, c_max = min(value_cols), max(value_cols) + 1
        def win(f): return f[r_min:r_max, c_min:c_max]

        history = [win(rr.frame).copy()]
        cycle = 0
        for i in range(self.MAX_CYCLE + 2):
            if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                return None
            r = sess.step(fwd_value)
            if not r.ok or r.frame is None: break
            cur = win(r.frame).copy()
            if any(np.array_equal(cur, h) for h in history):
                # Find which past frame it matches
                for idx, h in enumerate(history):
                    if np.array_equal(cur, h):
                        if idx == 0:
                            cycle = i + 1
                        break
                break
            history.append(cur)
        if cycle == 0:
            cycle = 7  # reasonable default

        # Helper: diff two frames with noise rows masked out
        def _clean_diff_cols(f0, f1):
            diff = f0 != f1
            for nr in self.NOISE_ROWS:
                if nr < diff.shape[0]:
                    diff[nr, :] = False
            return sorted(set(np.where(diff)[1].tolist()))

        # Infer N slots: walk cursor, locate distinct slot cols.
        rr = sess.reset()
        if rr.frame is None: return None
        # First slot: press value action once to see where change happens
        f0 = rr.frame.copy()
        r = sess.step(fwd_value)
        if not r.ok or r.frame is None: return None
        cols = _clean_diff_cols(f0, r.frame)
        if not cols: return None
        slot0 = min(cols)
        slot_width = max(cols) - slot0 + 1
        slot_cols = [slot0]

        # Now walk the cursor; after each cursor move, press value to locate current slot
        for _ in range(self.MAX_SLOTS):
            if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                return None
            rm = sess.step(fwd_cursor)
            if not rm.ok or rm.frame is None: break
            before_val = rm.frame.copy()
            rv = sess.step(fwd_value)
            if not rv.ok or rv.frame is None: break
            c = _clean_diff_cols(before_val, rv.frame)
            if not c: continue
            here = min(c)
            # We've wrapped around if we hit any previously-seen col, not just slot 0
            if here in slot_cols and len(slot_cols) > 1:
                break
            if here != slot_cols[-1]:
                slot_cols.append(here)

        n_slots = len(slot_cols)
        if n_slots < 2 or n_slots > self.MAX_SLOTS:
            self.log(f"slot discovery gave N={n_slots}; aborting")
            return None

        return dict(
            n_slots=n_slots, cycle=cycle,
            fwd_cursor=fwd_cursor, bwd_cursor=bwd_cursor,
            fwd_value=fwd_value, bwd_value=bwd_value,
            slot_cols=slot_cols, slot_width=slot_width,
        )

    # ── Brute force ──────────────────────────────────────────────────────────
    def _brute_force(self, sess: GameSession, res: StrategyResult, budget: Budget,
                     disc: dict) -> Optional[tuple]:
        """Nested-loop brute force with correct reset recovery.

        We track local cursor_pos and slot_idx[]. After any action error we
        soft-reset and sync state to all-zeros. If soft_reset itself fails, we
        give up with stopped_reason='unrecoverable'.

        Odometer: slot 0 cycles fastest (amortizes outer slot navigation).
        """
        N = disc["n_slots"]
        K = disc["cycle"]
        fc, bc = disc["fwd_cursor"], disc["bwd_cursor"]
        fv, bv = disc["fwd_value"], disc["bwd_value"]

        cursor = [0]
        slot_idx = [0] * N
        last_reset_at_tried = [-1]
        tried = [0]

        def note():
            self._note_level(res, sess)

        def do_step(act) -> bool:
            if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                return False
            r = sess.step(act)
            note()
            if res.max_levels_completed > 0:
                return True  # we're done; caller should check
            if not r.ok:
                # Handle recovery here — transparent to caller
                rr = sess.soft_reset()
                if rr.frame is None or rr.terminal:
                    return False
                cursor[0] = 0
                for i in range(N): slot_idx[i] = 0
                last_reset_at_tried[0] = tried[0]
                return False  # step itself didn't succeed, but state is clean
            return True

        def move_cursor(target: int) -> bool:
            """Move game cursor to slot target. If a step fails, state is synced
            to all-zeros via recovery inside do_step; caller should retry."""
            if cursor[0] == target: return True
            fwd = (target - cursor[0]) % N
            bwd = (cursor[0] - target) % N
            act = fc if fwd <= bwd else bc
            n = min(fwd, bwd)
            for _ in range(n):
                ok = do_step(act)
                if res.max_levels_completed > 0:
                    return True
                if not ok:
                    return False
            cursor[0] = target
            return True

        def set_slot(slot: int, target_idx: int) -> bool:
            c = slot_idx[slot]
            if c == target_idx: return True
            fwd = (target_idx - c) % K
            bwd = (c - target_idx) % K
            act = fv if fwd <= bwd else bv
            n = min(fwd, bwd)
            for _ in range(n):
                ok = do_step(act)
                if res.max_levels_completed > 0:
                    return True
                if not ok:
                    return False
            slot_idx[slot] = target_idx
            return True

        def enter_combo(combo: tuple) -> bool:
            """Drive the game to the given combo, restarting after GAME_OVER."""
            # Max 3 restart attempts per combo (guards against repeated failures)
            for attempt in range(3):
                restart = False
                for slot_i in range(N):
                    if not move_cursor(slot_i):
                        if res.max_levels_completed > 0: return True
                        if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                            return False
                        restart = True; break
                    if not set_slot(slot_i, combo[slot_i]):
                        if res.max_levels_completed > 0: return True
                        if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                            return False
                        restart = True; break
                    if res.max_levels_completed > 0: return True
                if not restart:
                    return True
                # Life lost; do_step already recovered. Next loop iteration starts over.
            return False

        total = K ** N
        # Odometer iteration: slot 0 fastest.
        combo = [0] * N
        prev_combo = [-1] * N  # force first iteration to set every slot
        last_log = 0.0
        import time as _time
        t0 = _time.time()

        def enter_combo_fast(combo_t: tuple, prev_t: list) -> bool:
            """Only touch slots that changed. If recovery happens, we refresh
            prev_t → [0]*N so the retry re-navigates everything correctly."""
            for attempt in range(3):
                all_set = True
                for slot_i in range(N):
                    if combo_t[slot_i] == prev_t[slot_i]:
                        continue  # unchanged, skip nav
                    if not move_cursor(slot_i):
                        if res.max_levels_completed > 0: return True
                        if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                            return False
                        # Recovery happened inside do_step; state is all-zero.
                        for i in range(N): prev_t[i] = 0
                        all_set = False; break
                    if not set_slot(slot_i, combo_t[slot_i]):
                        if res.max_levels_completed > 0: return True
                        if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                            return False
                        for i in range(N): prev_t[i] = 0
                        all_set = False; break
                    if res.max_levels_completed > 0: return True
                if all_set:
                    for i in range(N): prev_t[i] = combo_t[i]
                    return True
            return False

        won_combo = None
        prior_level = 0

        while True:
            if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                res.stopped_reason = (f"budget_expired: "
                                      f"{budget.why_expired(sess.steps_taken, sess.lives_used)}")
                return won_combo

            ok = enter_combo_fast(tuple(combo), prev_combo)

            # Did we win a new level on this combo?
            if res.max_levels_completed > prior_level:
                won_combo = tuple(combo) if won_combo is None else won_combo
                self.log(f"WIN L{res.max_levels_completed} at combo {combo}")
                prior_level = res.max_levels_completed
                # After a level win, game state has likely changed (cursor,
                # slot values, possibly mechanic). Reset local tracking so
                # we re-navigate from scratch on the next iteration.
                cursor[0] = 0
                for i in range(N): slot_idx[i] = 0
                for i in range(N): prev_combo[i] = -1
                # If fully beaten, return.
                if profile.win_levels > 0 and res.max_levels_completed >= profile.win_levels:
                    return won_combo

            if not ok:
                if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                    res.stopped_reason = "budget_expired_mid_combo"
                else:
                    res.stopped_reason = "unrecoverable"
                return won_combo

            tried[0] += 1

            now = _time.time()
            if now - last_log > 15:
                rate = tried[0] / max(now - t0, 1e-9)
                eta = (total - tried[0]) / rate if rate > 0 else 0
                self.log(f"tried {tried[0]}/{total}  lives={sess.lives_used}  "
                         f"rate={rate:.2f}/s  eta={eta:.0f}s")
                last_log = now

            # Increment odometer (slot 0 fastest)
            i = 0
            while i < N:
                combo[i] += 1
                if combo[i] < K:
                    break
                combo[i] = 0
                i += 1
            if i == N:
                break  # exhausted search

        res.stopped_reason = "search_exhausted"
        return None
