"""cluster_click_then_nav — generalised sc25 grammar.

Hypothesis: a tight cluster of click positions, bracketed by nav presses.
Specifically: nav_act × N_before  →  click each cluster point  →  nav_act × N_after.

This is the schema sc25 won with: act3×5 + 4 cluster clicks at (30,50)/(25,55)/
(35,55)/(30,60) + act3×4.

Why this strategy probes locally instead of relying on profile.click_heatmap:
  characterize's heat-map probe runs clicks from a *fresh reset*. For games
  like sc25 where clicks only become meaningful AFTER a nav prelude, the
  initial-state heat-map is uniformly inert and reveals nothing. So this
  strategy does its own scoped heat-map: for each (nav_act, N_before)
  candidate, probe a coarse grid AFTER applying the prelude. Clusters
  detected in those state-conditional heat-maps drive the win attempts.

Confidence: requires both a nav action and a click action in the available
set. Heat-map probing happens at run time, not confidence time.
"""
from __future__ import annotations

import itertools

import numpy as np

from meta.budget import Budget
from meta.common import GameSession
from meta.profile import GameProfile
from meta.result import StrategyResult
from meta.strategies.base import Strategy


# Match characterize's noise rows so probes ignore the life timer.
_NOISE_ROWS = (59, 60, 63)


class ClusterClickThenNav(Strategy):
    name = "cluster_click_then_nav"

    PROBE_GRID_STEP = 12        # ~5x5 grid (25 positions) on a 64x64 frame
    PROBE_LIVE_THRESHOLD = 1
    CLUSTER_RADIUS = 24         # max manhattan radius from anchor
    CLUSTER_MIN_PTS = 3
    CLUSTER_MAX_PTS = 5
    MAX_NAV_CANDIDATES = 4      # try every nav action — single-action effects
                                # don't reliably surface the right nav (sc25)
    N_BEFORE_CHOICES = (3, 5)
    N_AFTER_CHOICES = (3, 4)

    def confidence(self, profile: GameProfile) -> float:
        avail = set(profile.available_actions or [])
        if not (avail & {1, 2, 3, 4}) or not (avail & {5, 6}):
            return 0.0
        # Modest baseline; the strategy does its own probing rather than
        # gating on the initial-state heat-map.
        return 0.30

    def run(self, sess: GameSession, profile: GameProfile, budget: Budget) -> StrategyResult:
        res = self._start(sess, budget)
        res.details = {"probes": []}

        avail = list(profile.available_actions or [])
        nav_acts = self._rank_nav_acts(profile)[: self.MAX_NAV_CANDIDATES]
        click_act = 6 if 6 in avail else (5 if 5 in avail else None)
        if not nav_acts or click_act is None:
            res.stopped_reason = "missing_nav_or_click"
            return self._finalize(res, sess, budget)

        res.details.update({
            "click_act": click_act,
            "nav_candidates": nav_acts,
            "probe_grid_step": self.PROBE_GRID_STEP,
        })
        self.log(f"click_act={click_act} nav_candidates={nav_acts}")

        attempts = 0
        for nav_act in nav_acts:
            for n_before in self.N_BEFORE_CHOICES:
                if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                    res.details["attempts"] = attempts
                    res.stopped_reason = "budget_expired"
                    return self._finalize(res, sess, budget)

                # State-conditional heat-map: probe clicks AFTER nav_act × n_before.
                heatmap, probe_meta = self._probe_with_prelude(
                    sess, nav_act, n_before, click_act, budget, res
                )
                res.details["probes"].append({
                    "nav_act": nav_act, "n_before": n_before,
                    **probe_meta,
                })
                self.log(
                    f"  probe nav{nav_act}×{n_before}: live={probe_meta['n_live']} "
                    f"terminal={probe_meta['n_terminal']} probes={probe_meta['n_probes']}"
                )

                # If the probe accidentally won (click-after-prelude solved a level),
                # _probe_with_prelude updated res via _note_level. Surface it as a win.
                if res.max_levels_completed > 0:
                    win_seq = []
                    win_seq.extend([{"type": "nav", "act": nav_act}] * n_before)
                    win_seq.append({"type": "click", "act": click_act,
                                    "x": probe_meta.get("win_x"),
                                    "y": probe_meta.get("win_y")})
                    res.details.update({
                        "win_sequence": win_seq,
                        "win_during": "probe",
                        "win_nav_act": nav_act,
                        "win_n_before": n_before,
                        "attempts": attempts,
                    })
                    res.stopped_reason = "probe_win"
                    self.log(
                        f"WIN during probe: nav{nav_act}×{n_before} click "
                        f"({probe_meta.get('win_x')},{probe_meta.get('win_y')})"
                    )
                    return self._finalize(res, sess, budget)

                clusters = self._find_clusters(heatmap)
                if not clusters:
                    continue

                # Iterate clusters × N_after.
                for cluster_idx, cluster in enumerate(clusters):
                    for n_after in self.N_AFTER_CHOICES:
                        if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                            res.details["attempts"] = attempts
                            res.stopped_reason = "budget_expired"
                            return self._finalize(res, sess, budget)

                        attempts += 1
                        won, win_path = self._try_full_sequence(
                            sess, nav_act, n_before, cluster, n_after, click_act, res
                        )
                        if won:
                            win_seq = win_path
                            res.details.update({
                                "win_sequence": win_seq,
                                "win_during": "full_sequence",
                                "win_cluster_idx": cluster_idx,
                                "win_nav_act": nav_act,
                                "win_n_before": n_before,
                                "win_n_after": n_after,
                                "attempts": attempts,
                            })
                            res.stopped_reason = "cluster_nav_win"
                            self.log(
                                f"WIN attempt #{attempts}: cluster={cluster_idx} "
                                f"nav={nav_act}×{n_before} clicks={len(cluster)} "
                                f"nav={nav_act}×{n_after}"
                            )
                            return self._finalize(res, sess, budget)

        res.details["attempts"] = attempts
        res.stopped_reason = "exhausted_no_win"
        return self._finalize(res, sess, budget)

    # ── Probe helpers ───────────────────────────────────────────────────────
    def _probe_with_prelude(self, sess: GameSession, nav_act: int, n_before: int,
                            click_act: int, budget: Budget, res: StrategyResult
                            ) -> tuple[list, dict]:
        """Coarse heat-map probe of click responsivity after a nav prelude.
        For each grid position: reset → nav×N → click → diff. Updates res via
        _note_level if the probe accidentally lands a level.

        Returns (heatmap, meta). meta includes win_x / win_y if a win occurred.
        """
        rr = sess.reset()
        if rr.frame is None:
            return [], {"n_probes": 0, "n_terminal": 0, "n_live": 0,
                        "skipped_reason": "reset_failed"}

        h, w = rr.frame.shape
        heatmap: list = []
        n_probes = 0
        n_terminal = 0
        win_x = win_y = None

        for y in range(self.PROBE_GRID_STEP // 2, h, self.PROBE_GRID_STEP):
            for x in range(self.PROBE_GRID_STEP // 2, w, self.PROBE_GRID_STEP):
                if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                    return self._sort_heatmap(heatmap), {
                        "n_probes": n_probes, "n_terminal": n_terminal,
                        "n_live": len(heatmap),
                        "win_x": win_x, "win_y": win_y,
                        "skipped_reason": "budget_expired",
                    }

                rr = sess.reset()
                if rr.frame is None:
                    return self._sort_heatmap(heatmap), {
                        "n_probes": n_probes, "n_terminal": n_terminal,
                        "n_live": len(heatmap),
                        "win_x": win_x, "win_y": win_y,
                        "skipped_reason": "reset_failed",
                    }

                # Apply prelude.
                mid_frame = rr.frame
                aborted = False
                for _ in range(n_before):
                    if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                        aborted = True
                        break
                    r = sess.step(nav_act)
                    self._note_level(res, sess)
                    if res.max_levels_completed > 0:
                        # Prelude alone solved it — record nothing position-specific.
                        return self._sort_heatmap(heatmap), {
                            "n_probes": n_probes, "n_terminal": n_terminal,
                            "n_live": len(heatmap),
                            "win_x": None, "win_y": None,
                            "win_during_prelude": True,
                        }
                    if not r.ok or r.terminal:
                        aborted = True
                        n_terminal += 1
                        break
                    mid_frame = r.frame
                if aborted:
                    continue

                r = sess.step_with_data(click_act, {"x": x, "y": y})
                n_probes += 1
                self._note_level(res, sess)
                if res.max_levels_completed > 0:
                    win_x, win_y = x, y
                    return self._sort_heatmap(heatmap), {
                        "n_probes": n_probes, "n_terminal": n_terminal,
                        "n_live": len(heatmap),
                        "win_x": win_x, "win_y": win_y,
                    }
                if r.frame is None:
                    continue
                if r.terminal:
                    n_terminal += 1
                    continue

                diff = mid_frame != r.frame
                for nr in _NOISE_ROWS:
                    if nr < diff.shape[0]:
                        diff[nr, :] = False
                changed = int(np.sum(diff))
                if changed >= self.PROBE_LIVE_THRESHOLD:
                    heatmap.append({"x": int(x), "y": int(y),
                                    "cells_changed": changed, "act": click_act})

        return self._sort_heatmap(heatmap), {
            "n_probes": n_probes, "n_terminal": n_terminal,
            "n_live": len(heatmap),
            "win_x": win_x, "win_y": win_y,
        }

    def _try_full_sequence(self, sess: GameSession, nav_act: int, n_before: int,
                           cluster: list, n_after: int, click_act: int,
                           res: StrategyResult) -> tuple[bool, list]:
        """Run nav×n_before → click each cluster point → nav×n_after. Returns
        (won, win_path). win_path is the partial sequence up to the winning step."""
        path: list = []
        rr = sess.reset()
        if rr.frame is None:
            return False, path

        for _ in range(n_before):
            r = sess.step(nav_act)
            path.append({"type": "nav", "act": nav_act})
            self._note_level(res, sess)
            if res.max_levels_completed > 0:
                return True, path
            if not r.ok or r.terminal:
                return False, path

        for x, y in cluster:
            r = sess.step_with_data(click_act, {"x": x, "y": y})
            path.append({"type": "click", "act": click_act, "x": x, "y": y})
            self._note_level(res, sess)
            if res.max_levels_completed > 0:
                return True, path
            if not r.ok or r.terminal:
                return False, path

        for _ in range(n_after):
            r = sess.step(nav_act)
            path.append({"type": "nav", "act": nav_act})
            self._note_level(res, sess)
            if res.max_levels_completed > 0:
                return True, path
            if not r.ok or r.terminal:
                return False, path

        return False, path

    # ── Ranking + cluster detection ─────────────────────────────────────────
    def _rank_nav_acts(self, profile: GameProfile) -> list[int]:
        """Return available nav actions ordered by single-action effect size,
        descending. Ties broken by numeric order."""
        avail = profile.available_actions or []
        nav = [a for a in (1, 2, 3, 4) if a in avail]
        if not nav:
            return []
        def score(a: int) -> tuple:
            eff = profile.action_effects.get(str(a), {})
            return (-int(eff.get("cells_changed", 0)), a)
        return sorted(nav, key=score)

    def _sort_heatmap(self, heatmap: list) -> list:
        return sorted(heatmap, key=lambda e: (-int(e["cells_changed"]),
                                              int(e["y"]), int(e["x"])))

    def _find_clusters(self, heatmap: list) -> list[list[tuple[int, int]]]:
        """Return up to a few lists of (x, y) tuples. Each cluster has
        CLUSTER_MIN_PTS..CLUSTER_MAX_PTS positions within CLUSTER_RADIUS
        manhattan distance of an anchor (highest-score live position)."""
        if not heatmap:
            return []
        sorted_pts = self._sort_heatmap(heatmap)
        used = set()
        clusters: list[list[tuple[int, int]]] = []
        for anchor in sorted_pts:
            ax, ay = int(anchor["x"]), int(anchor["y"])
            if (ax, ay) in used:
                continue
            members: list[tuple[int, int]] = []
            for cand in sorted_pts:
                cx, cy = int(cand["x"]), int(cand["y"])
                if (cx, cy) in used:
                    continue
                if abs(cx - ax) + abs(cy - ay) <= self.CLUSTER_RADIUS:
                    members.append((cx, cy))
                if len(members) >= self.CLUSTER_MAX_PTS:
                    break
            if len(members) >= self.CLUSTER_MIN_PTS:
                clusters.append(members)
                used.update(members)
            if len(clusters) >= 3:
                break
        return clusters
