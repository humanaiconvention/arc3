"""
generic_solver.py — General-purpose game solver for ARC-AGI-3 games.

Approach:
1. Frame analysis: identify objects (connected regions by pixel value)
2. Action probing: learn what each action moves, from recordings or live
3. Objective inference: detect what positional relationship wins a level
4. Planning: greedy/BFS to move controlled object to goal

Works for any game: click (available_actions=[6]), keyboard ([1-4]),
or keyboard_click (mixed). No game-specific hardcoding.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Object representation ─────────────────────────────────────────────────────

@dataclass
class GameObject:
    """A connected region of same-valued pixels in the frame."""
    value: int
    pixels: List[Tuple[int, int]]   # list of (row, col)

    @property
    def centroid(self) -> Tuple[float, float]:
        if not self.pixels:
            return (0.0, 0.0)
        rows = [p[0] for p in self.pixels]
        cols = [p[1] for p in self.pixels]
        return (sum(rows) / len(rows), sum(cols) / len(cols))

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        rows = [p[0] for p in self.pixels]
        cols = [p[1] for p in self.pixels]
        return (min(rows), min(cols), max(rows), max(cols))

    @property
    def size(self) -> int:
        return len(self.pixels)


def find_objects(frame: np.ndarray,
                 min_size: int = 2,
                 ignore_values: Optional[set] = None) -> List[GameObject]:
    """BFS connected-component labelling for each unique non-background value."""
    if ignore_values is None:
        ignore_values = set()
    h, w = frame.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: List[GameObject] = []

    for val in np.unique(frame):
        if val in ignore_values:
            continue
        mask = (frame == val) & ~visited
        positions = list(zip(*np.where(mask)))
        if not positions:
            continue
        # BFS within the value mask
        not_visited = set(positions)
        while not_visited:
            start = next(iter(not_visited))
            component = []
            queue = deque([start])
            visited[start] = True
            not_visited.discard(start)
            while queue:
                r, c = queue.popleft()
                component.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in not_visited and not visited[nr, nc]:
                        visited[nr, nc] = True
                        not_visited.discard((nr, nc))
                        queue.append((nr, nc))
            if len(component) >= min_size:
                objects.append(GameObject(value=int(val), pixels=component))

    return objects


def detect_background(frame: np.ndarray) -> int:
    """The most common pixel value is assumed to be background."""
    vals, counts = np.unique(frame, return_counts=True)
    return int(vals[np.argmax(counts)])


def frame_to_state(frame: np.ndarray,
                   background: int,
                   ui_rows_top: int = 2,
                   ui_rows_bottom: int = 2) -> Dict[int, Tuple[float, float]]:
    """
    Reduce frame to a compact state: value → centroid, ignoring background
    and UI rows (timer, score bars at top/bottom).
    """
    h = frame.shape[0]
    inner = frame[ui_rows_top:h - ui_rows_bottom, :]
    bg = background
    state = {}
    for val in np.unique(inner):
        if val == bg:
            continue
        positions = list(zip(*np.where(inner == val)))
        if len(positions) >= 2:
            rows = [p[0] + ui_rows_top for p in positions]
            cols = [p[1] for p in positions]
            state[val] = (sum(rows) / len(rows), sum(cols) / len(cols))
    return state


def objects_adjacent(obj_a: GameObject, obj_b: GameObject, max_dist: int = 2) -> bool:
    """True if any pixel of obj_a is within max_dist of any pixel of obj_b."""
    set_a = set(obj_a.pixels)
    for r, c in obj_b.pixels:
        for dr in range(-max_dist, max_dist + 1):
            for dc in range(-max_dist, max_dist + 1):
                if (r + dr, c + dc) in set_a:
                    return True
    return False


# ── Action effect model ───────────────────────────────────────────────────────

@dataclass
class ActionEffect:
    """What action `action_id` did to the frame."""
    action_id: Any                          # GameAction or int
    click_data: Optional[Dict] = None       # {'x': px, 'y': py} for click actions
    moved_values: List[int] = field(default_factory=list)   # pixel values that moved
    delta_centroids: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    pixels_changed: int = 0
    reward: float = 0.0
    level_up: bool = False


def compute_action_effect(before: np.ndarray,
                          after: np.ndarray,
                          action_id: Any,
                          click_data: Optional[Dict],
                          reward: float = 0.0,
                          level_up: bool = False) -> ActionEffect:
    """Compare two frames to compute what the action actually did."""
    changed = np.where(before != after)
    pixels_changed = len(changed[0])

    if pixels_changed == 0:
        return ActionEffect(action_id=action_id, click_data=click_data,
                            pixels_changed=0, reward=reward, level_up=level_up)

    # Find centroids before/after for each value that changed
    changed_vals = set(before[changed].tolist()) | set(after[changed].tolist())
    delta_centroids: Dict[int, Tuple[float, float]] = {}
    moved_values: List[int] = []

    for val in changed_vals:
        pos_before = np.where(before == val)
        pos_after  = np.where(after == val)
        if len(pos_before[0]) == 0 or len(pos_after[0]) == 0:
            continue
        cen_before = (float(np.mean(pos_before[0])), float(np.mean(pos_before[1])))
        cen_after  = (float(np.mean(pos_after[0])),  float(np.mean(pos_after[1])))
        dr = cen_after[0] - cen_before[0]
        dc = cen_after[1] - cen_before[1]
        if abs(dr) > 0.5 or abs(dc) > 0.5:
            delta_centroids[val] = (dr, dc)
            moved_values.append(val)

    return ActionEffect(
        action_id=action_id,
        click_data=click_data,
        moved_values=moved_values,
        delta_centroids=delta_centroids,
        pixels_changed=pixels_changed,
        reward=reward,
        level_up=level_up,
    )


# ── Active probing ────────────────────────────────────────────────────────────

class GameProber:
    """
    Systematically probes each available action at the start of a level
    to learn game dynamics: which action moves which object by how much.

    Probe phase: take each action once (or twice for confirmation).
    After probing, identify the player value and per-action deltas.
    """

    def __init__(self) -> None:
        self.pending: List[int] = []
        self.current_probe: Optional[int] = None
        self.results: Dict[int, List[ActionEffect]] = defaultdict(list)
        self.done: bool = True

    def setup(self, available_actions: List[int], repeats: int = 1) -> None:
        self.pending = list(available_actions) * repeats
        self.current_probe = None
        self.done = not bool(self.pending)

    def next_probe(self) -> Optional[int]:
        """Return next action to probe and mark it as current.  None if done."""
        if not self.pending:
            self.done = True
            return None
        self.current_probe = self.pending[0]
        return self.current_probe

    def record_result(self, effect: ActionEffect) -> None:
        """Store effect for the current probe action and advance the queue."""
        if self.current_probe is not None:
            self.results[self.current_probe].append(effect)
            if self.pending and self.pending[0] == self.current_probe:
                self.pending.pop(0)
            self.current_probe = None
        if not self.pending:
            self.done = True

    def player_value(self) -> Optional[int]:
        """
        The pixel value that moves most consistently and by the largest amount
        across probed actions.

        Scoring: count × avg_magnitude — real players move 4+ px per action;
        boundary artifacts that pass the 0.5px delta filter move ≤2px.
        This ensures a fast-moving player beats a slowly-drifting artifact even
        when both appear in the same number of probe effects.
        """
        val_counts: Dict[int, int] = defaultdict(int)
        val_magnitudes: Dict[int, List[float]] = defaultdict(list)
        for act_id, effs in self.results.items():
            seen: set = set()
            for eff in effs:
                for v in eff.moved_values:
                    if v not in seen:
                        val_counts[v] += 1
                        seen.add(v)
                for v, (dr, dc) in eff.delta_centroids.items():
                    val_magnitudes[v].append((dr * dr + dc * dc) ** 0.5)
        if not val_counts:
            return None
        def _score(v: int) -> float:
            mags = val_magnitudes.get(v, [1.0])
            avg_mag = sum(mags) / len(mags)
            return val_counts[v] * avg_mag
        return max(val_counts, key=_score)

    def action_deltas(self, target_value: int) -> Dict[int, Tuple[float, float]]:
        """Mean (dr, dc) for target_value across all probes per action."""
        deltas: Dict[int, Tuple[float, float]] = {}
        for act_id, effs in self.results.items():
            drs, dcs = [], []
            for eff in effs:
                if target_value in eff.delta_centroids:
                    dr, dc = eff.delta_centroids[target_value]
                    drs.append(dr)
                    dcs.append(dc)
            if drs:
                deltas[act_id] = (sum(drs) / len(drs), sum(dcs) / len(dcs))
        return deltas

    def best_action_toward(self,
                           target_value: int,
                           desired_dr: float,
                           desired_dc: float) -> Optional[int]:
        """Return action_id that moves target_value most toward (desired_dr, desired_dc)."""
        deltas = self.action_deltas(target_value)
        mag = (desired_dr ** 2 + desired_dc ** 2) ** 0.5
        if mag < 0.001 or not deltas:
            return None
        best_act, best_dot = None, -1e9
        for act_id, (dr, dc) in deltas.items():
            dot = (dr * desired_dr + dc * desired_dc) / mag
            if dot > best_dot:
                best_dot = dot
                best_act = act_id
        return best_act


def bfs_plan(start: Tuple[float, float],
             goal: Tuple[float, float],
             action_deltas: Dict[int, Tuple[float, float]],
             max_steps: int = 60,
             tolerance: float = 5.0) -> List[int]:
    """
    BFS path from start centroid to goal centroid using action deltas.

    Returns list of integer action IDs.
    Quantises positions to 2-pixel cells to keep the search tractable.
    """
    if not action_deltas:
        return []

    def dist(a: Tuple, b: Tuple) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def to_key(pos: Tuple[int, int]) -> Tuple[int, int]:
        # 1-pixel resolution — game grid is 64×64 so state space is manageable
        return pos

    start_i = (int(round(start[0])), int(round(start[1])))
    goal_i  = (int(round(goal[0])),  int(round(goal[1])))

    # Only keep actions with meaningful movement
    int_deltas: Dict[int, Tuple[int, int]] = {
        act: (int(round(dr)), int(round(dc)))
        for act, (dr, dc) in action_deltas.items()
        if abs(dr) + abs(dc) > 0.3
    }
    if not int_deltas:
        return []

    queue: deque = deque([(start_i, [])])
    visited: set = {to_key(start_i)}

    while queue:
        pos, path = queue.popleft()
        if dist(pos, goal_i) <= tolerance or len(path) >= max_steps:
            return path
        for act_id, (dr, dc) in int_deltas.items():
            new_pos = (max(0, min(63, pos[0] + dr)),
                       max(0, min(63, pos[1] + dc)))
            key = to_key(new_pos)
            if key not in visited:
                visited.add(key)
                queue.append((new_pos, path + [act_id]))

    return []


def infer_goal_from_frame(frame: np.ndarray,
                           player_value: Optional[int],
                           background: Optional[int]) -> List[Tuple[int, float, float]]:
    """
    Infer candidate goal positions from frame pixel statistics alone.

    Goals are typically rare, visually distinct pixel values (not background,
    not the player, not common structural values like walls/floors).

    Returns [(value, centroid_row, centroid_col), ...] sorted rarest-first.
    """
    bg = background if background is not None else detect_background(frame)
    vals, counts = np.unique(frame, return_counts=True)
    total = frame.size

    # Top-3 most common values are likely structural (background, walls, floor)
    top_common = {int(v) for v in vals[np.argsort(counts)[-3:]]}
    exclude = top_common | ({player_value} if player_value is not None else set())

    candidates: List[Tuple[int, float, float]] = []
    for v, c in zip(vals.tolist(), counts.tolist()):
        v = int(v)
        if v in exclude:
            continue
        if c < 2 or c > total * 0.15:  # ignore single pixels and near-ubiquitous values
            continue
        pos = np.where(frame == v)
        cr = float(np.mean(pos[0]))
        cc = float(np.mean(pos[1]))
        candidates.append((v, cr, cc))

    # Sort by pixel count ascending (rarest first — most likely to be the goal)
    candidates.sort(key=lambda t: int(np.sum(frame == t[0])))
    return candidates[:8]


# ── Directional game decode ───────────────────────────────────────────────────

def _decode_directional_planned(
        frame: np.ndarray,
        available_actions: List[Any],
        model: "GameModel",
        action_int: int,
        goal_frame: Optional[np.ndarray]) -> Tuple[Any, Optional[Dict]]:
    """
    Probe → identify player → navigate toward goal for directional games.

    Phases:
    1. Probe: try each action once to learn which value is the player.
    2. Identify player value + goal (from winning frame or frame heuristic).
    3. Greedy navigation: each step pick the action that moves player toward goal.
    4. Fallback to MCTS-mapped random if no goal can be inferred.
    """
    from arcengine import GameAction

    bg = model.background or detect_background(frame)
    legal = sorted(int(a) for a in available_actions)

    # Phase 1: probe
    if model.prober is None:
        model.prober = GameProber()
        model.prober.setup(legal, repeats=1)

    if not model.prober.done:
        nxt = model.prober.next_probe()
        if nxt is not None:
            return GameAction.from_id(nxt), None

    # Phase 2a: identify player value
    if model.player_value_int is None and model.prober is not None:
        pv_candidate = model.prober.player_value()
        if pv_candidate is not None:
            pv_size = int(np.sum(frame == pv_candidate))
            # Require ≥0.15% of frame pixels — rejects phantom transition artifacts (val=0)
            # that accumulate during probing but aren't real game objects.
            # 0.15% of 4096px = ~6px, handles small players like ls20 (10px) and tu93 (8px).
            # Floor of 3px catches sub-pixel moves; timer noise is typically 1-3px (filtered by
            # magnitude heuristic upstream).
            min_pixels = max(3, int(frame.size * 0.0015))
            max_pixels = int(frame.size * 0.15)   # >15% = background/wall, not player
            bg_val = detect_background(frame)
            # Reject: too small, too large (background), value 0 (always black bg),
            # or matches detected background.  Timer contamination often selects val=0.
            if (min_pixels <= pv_size <= max_pixels
                    and pv_candidate != 0
                    and pv_candidate != bg_val):
                model.player_value_int = pv_candidate
            else:
                # Too few/many pixels or is background: likely artifact or timer noise.
                # Find the actual player: smallest consistently-moving non-background value.
                val_sizes = {int(v): int(np.sum(frame == v)) for v in np.unique(frame)}
                best_pv, best_score = None, frame.size + 1
                for act_effs in model.prober.results.values():
                    for eff in act_effs:
                        for mv in eff.moved_values:
                            mv_int = int(mv)
                            if mv_int == pv_candidate or mv_int == bg_val or mv_int == 0:
                                continue
                            sz = val_sizes.get(mv_int, 0)
                            if min_pixels <= sz <= frame.size * 0.20 and sz < best_score:
                                best_score = sz
                                best_pv = mv_int
                model.player_value_int = best_pv
                reason = "too_large" if pv_size > max_pixels else ("bg" if pv_candidate in (0, bg_val) else "too_small")
                print(f"  [player→] rejected val={pv_candidate}({pv_size}px,{reason}), using val={best_pv}({best_score}px)", flush=True)

    pv = model.player_value_int
    if pv is not None and model._last_dist is None and not getattr(model, '_probed_logged', False):
        # First time out of probe phase — print what we learned (once per level)
        model._probed_logged = True  # type: ignore[attr-defined]
        model._last_dist = -999.0    # sentinel: marks "past probe phase"
        deltas = model.prober.action_deltas(pv) if model.prober else {}
        print(f"  [probe→nav] player_value={pv} deltas={deltas}", flush=True)
        # Filter out weak deltas (timer noise / background drift) before deciding
        # on slider mode.  Threshold: magnitude must be ≥1.5px to count as a real
        # directional effect.  sc25 act2 gives 0.73px (timer contam) — filtered out,
        # leaving only acts 3+4 as effective → correctly enters slider mode [3,4].
        MIN_DELTA_PX = 1.5
        effective_deltas = {a: d for a, d in deltas.items()
                            if (d[0]**2 + d[1]**2)**0.5 >= MIN_DELTA_PX}
        if len(effective_deltas) != len(deltas):
            print(f"  [delta-filter] filtered weak deltas → effective={effective_deltas}", flush=True)
        # Detect slider game: ≤2 effective actions → systematic scan mode
        model._slider_mode = len(effective_deltas) <= 2 and len(effective_deltas) > 0
        if model._slider_mode:
            model._slider_actions = list(effective_deltas.keys())
            model._slider_phase = 0   # index into scan phases
            model._slider_phase_steps = 0
            print(f"  [slider mode] actions={model._slider_actions}", flush=True)

    # Detect dead-directional or auto-animation in mixed games:
    # • Dead-dir: all probe results yield 0px change (e.g. sc25 from diagnostic)
    # • Auto-animation: all probe results yield the SAME pixel count (±10%)
    #   This means dir actions don't control anything — the game animates itself
    #   (e.g. ar25, ka59, m0r0, wa30 all show identical px change per action).
    # In both cases, skip directional entirely → 100% non-dir/click exploration.
    if (model.prober is not None and model.prober.done
            and not getattr(model, '_all_dir_zero', False)
            and not getattr(model, '_all_dir_checked', False)):
        model._all_dir_checked = True  # type: ignore[attr-defined]
        if model.prober.results:
            per_action_px = [
                max((e.pixels_changed for e in effs), default=0)
                for effs in model.prober.results.values()
            ]
            if per_action_px:
                min_px = min(per_action_px)
                max_px = max(per_action_px)
                all_zero = max_px == 0
                # Auto-anim: game animates itself — directional keys have no control.
                # Two conditions must both hold:
                # 1. Pixel counts are similar across all actions (ratio < 1.15)
                # 2. The player centroid moves in the SAME direction regardless of key
                #    (opposite keys like up/down move player differently in real games)
                # This prevents misclassifying genuine directional games (sk48, ls20)
                # where all dirs move ~same px count but in opposite directions.
                # Real auto-anim: ar25(109px), ka59(19px), m0r0(100px), re86(61px), wa30(32px)
                # False positives without direction check: sk48(18px all dirs), ls20(52px all dirs)
                px_uniform = (max_px > 0 and len(per_action_px) >= 2
                              and min_px > 0 and max_px / min_px < 1.15)
                auto_anim = False
                if px_uniform and pv is not None and model.prober is not None:
                    # Check if opposite action pairs move player in same direction
                    # (auto-anim) vs opposite directions (real directional control)
                    deltas_all = model.prober.action_deltas(pv)
                    # Pair up: (act1↔act2) and (act3↔act4) — canonical opposites
                    legal_sorted = sorted(model.prober.results.keys())
                    if len(deltas_all) >= 2:
                        delta_vecs = [(dr, dc) for dr, dc in deltas_all.values()]
                        # Compute sum of all delta vectors — in auto-anim they nearly
                        # cancel to ~0 across all, but opposite pairs cancel each other
                        # too in directional games. Better test: check if any two actions
                        # have delta vectors pointing in *opposite* directions (dot < 0).
                        # If yes → real directional game. If all dots ≥ 0 → auto-anim.
                        has_opposite = False
                        dvs = list(delta_vecs)
                        for i in range(len(dvs)):
                            for j in range(i+1, len(dvs)):
                                dot = dvs[i][0]*dvs[j][0] + dvs[i][1]*dvs[j][1]
                                if dot < -0.1:   # opposite directions
                                    has_opposite = True
                                    break
                            if has_opposite:
                                break
                        auto_anim = not has_opposite
                elif px_uniform and pv is None:
                    # No player identified yet — fall back to px-only check but
                    # require larger minimum (≥15px) to reduce false positives
                    auto_anim = max_px >= 15

                if all_zero:
                    model._all_dir_zero = True  # type: ignore[attr-defined]
                    print(f"  [dead-dir] all probe actions yielded 0px change — switching to 100% non-dir mode", flush=True)
                elif auto_anim:
                    model._all_dir_zero = True  # type: ignore[attr-defined]
                    print(f"  [auto-anim] all probe actions yield similar px ({min_px}–{max_px}) — switching to 100% non-dir mode", flush=True)

    # Phase 2b: identify goal centroid (one-shot from winning frame)
    if model.goal_centroid is None and goal_frame is not None and pv is not None:
        gpos = np.where(goal_frame == pv)
        if len(gpos[0]) > 0:
            model.goal_centroid = (float(np.mean(gpos[0])),
                                   float(np.mean(gpos[1])))

    # Phase 2c: grid scan — navigate to 4×4 grid positions to cover the playfield.
    # Activated when no winning frame is available (covers all directional games without recordings).
    # This is more robust than guessing the goal from pixel statistics.
    if model.goal_centroid is None and pv is not None and model._last_dist == -999.0:
        if not getattr(model, '_grid_targets', None):
            model._grid_targets = [  # type: ignore[attr-defined]
                (float(r), float(c))
                for r in [8, 24, 40, 56]
                for c in [8, 24, 40, 56]
            ]
            model._grid_idx = 0  # type: ignore[attr-defined]
            print(f"  [grid-scan] starting 4×4 grid exploration, pv={pv}", flush=True)
        model.goal_centroid = model._grid_targets[model._grid_idx]

    # Phase 3a: slider scan (for games with ≤2 effective actions)
    # Strategy:
    #   Two-direction (true slider): sweep sa[0] SWEEP steps right, then step back
    #     with sa[-1] one increment at a time with HOLD pauses.
    #   One-direction (sa[0]==sa[-1] or both actions go the same way): hammer sa[0]
    #     every step — covers all positions if game wraps, or saturates if it doesn't.
    # HOLD pauses fire non-slider actions (including clicks) so slider+click games
    # can win while the player is parked at each scanned position.
    CLICK_IDS = {6, 7}
    if getattr(model, '_slider_mode', False) and model._slider_actions:
        sa = model._slider_actions
        # Fix 2: Sort no_ops so click actions (6,7) fire on earliest HOLD steps.
        # With HOLD=2, step_in_left cycles 1,2 — if clicks are at index ≥3 in
        # no_ops they never fire. Sorting clicks first guarantees they trigger.
        no_ops = sorted([a for a in legal if a not in sa],
                        key=lambda a: (0 if a in CLICK_IDS else 1, a))
        SWEEP = 16   # 16 steps covers ~16 discrete positions per sweep direction
        HOLD = 2     # 2-step pause: fire click/non-dir actions while parked at each pos
                     # Rationale: sc25 timer ≈ 52 steps/life.
                     # Two-dir cycle = 16 + 16*(2+1) = 64 steps → ~0.8 cycles/life.
                     # One-dir: no cycle — hammer sa[0] every step → max coverage.

        # Detect one-direction slider: sa[0]==sa[-1] OR both action deltas point
        # in the same half-plane (dot product ≥ 0) — i.e. no effective reversal.
        one_dir = (len(sa) == 1 or sa[0] == sa[-1])
        if not one_dir and model.prober is not None:
            d = model.prober.action_deltas(pv or 0) if pv is not None else {}
            if sa[0] in d and sa[-1] in d:
                v0 = d[sa[0]]; v1 = d[sa[-1]]
                dot = v0[0]*v1[0] + v0[1]*v1[1]
                if dot >= 0:   # same direction → cannot reverse → one_dir strategy
                    one_dir = True

        if one_dir:
            # Fix 3: Per-life position probing for one-dir sliders.
            # Instead of hammering sa[0] every step (saturates at extreme),
            # budget a different number of initial sa[0] steps per life:
            #   life 0: 0 steps (stay at start), then wait
            #   life 1: 5 steps, then wait
            #   life 2: 10 steps, then wait  ... etc.
            # "Wait" fires no_ops (including clicks) so slider+click games
            # can win at each probed position.
            phase = model._slider_phase
            model._slider_phase += 1
            life_idx = getattr(model, '_lives_used', 0)
            budget = life_idx * 5  # 0, 5, 10, 15, 20 steps of sa[0] per life

            if phase % 5 == 0 and pv is not None:
                cur_pos = np.where(frame == pv)
                if len(cur_pos[0]) > 0:
                    pr = float(np.mean(cur_pos[0])); pc = float(np.mean(cur_pos[1]))
                    print(f"  [slider-pos] phase={phase} life={life_idx} budget={budget} "
                          f"pv={pv} pos=({pr:.1f},{pc:.1f}) one_dir sa={sa}", flush=True)

            if phase < budget:
                # Still moving to target position
                return GameAction.from_id(sa[0]), None
            else:
                # At target position — fire no_ops (clicks, etc.) to trigger win condition
                if no_ops:
                    nop_id = no_ops[phase % len(no_ops)]
                    if nop_id in CLICK_IDS and pv is not None:
                        cur_pos = np.where(frame == pv)
                        if len(cur_pos[0]) > 0:
                            pr = int(round(float(np.mean(cur_pos[0]))))
                            pc = int(round(float(np.mean(cur_pos[1]))))
                        else:
                            pr, pc = 32, 32
                        return GameAction.from_id(nop_id), {"x": pc, "y": pr}
                    return GameAction.from_id(nop_id), None
                # No non-slider actions available — keep hammering
                return GameAction.from_id(sa[0]), None

        phase = model._slider_phase  # global step counter (not reset per life)
        model._slider_phase += 1

        # Log player position every 5 slider steps for diagnostics
        if phase % 5 == 0 and pv is not None:
            cur_pos = np.where(frame == pv)
            if len(cur_pos[0]) > 0:
                pr = float(np.mean(cur_pos[0])); pc = float(np.mean(cur_pos[1]))
                print(f"  [slider-pos] phase={phase} pv={pv} pos=({pr:.1f},{pc:.1f}) sa={sa}", flush=True)

        cycle = SWEEP + (SWEEP * (HOLD + 1))  # sweep right + sweep left with holds
        pos_in_cycle = phase % cycle

        if pos_in_cycle < SWEEP:
            # First half: sweep toward first extreme
            return GameAction.from_id(sa[0]), None
        else:
            # Second half: step toward second extreme one increment at a time,
            # pausing HOLD steps at each position.
            offset = pos_in_cycle - SWEEP
            step_in_left = offset % (HOLD + 1)
            if step_in_left == 0:
                # Take one step in the return direction
                return GameAction.from_id(sa[-1]), None
            else:
                # Hold: fire a non-slider action (click, fire, or directional no-op)
                if no_ops:
                    nop_id = no_ops[step_in_left % len(no_ops)]
                    if nop_id in CLICK_IDS and pv is not None:
                        # Click at current player centroid — covers slider+click games
                        cur_pos = np.where(frame == pv)
                        if len(cur_pos[0]) > 0:
                            pr = int(round(float(np.mean(cur_pos[0]))))
                            pc = int(round(float(np.mean(cur_pos[1]))))
                        else:
                            pr, pc = 32, 32
                        return GameAction.from_id(nop_id), {"x": pc, "y": pr}
                    return GameAction.from_id(nop_id), None
                else:
                    return GameAction.from_id(sa[-1]), None

    # Phase 3b: greedy navigation toward goal (for multi-direction games)
    if model.goal_centroid is not None and pv is not None:
        cur_pos = np.where(frame == pv)
        if len(cur_pos[0]) > 0:
            cr = float(np.mean(cur_pos[0]))
            cc = float(np.mean(cur_pos[1]))
            goal_r, goal_c = model.goal_centroid

            # Track progress
            dist = ((cr - goal_r) ** 2 + (cc - goal_c) ** 2) ** 0.5
            if model._last_dist is not None and dist >= model._last_dist - 0.5:
                model.steps_without_progress += 1
            else:
                model.steps_without_progress = 0
            model._last_dist = dist

            # Advance grid scan: when close enough OR stuck for too long at current target
            _advance_grid = (
                (dist < 8.0 and getattr(model, '_grid_targets', None)) or
                (model.steps_without_progress > 12 and getattr(model, '_grid_targets', None))
            )
            if _advance_grid:
                model._grid_idx = (model._grid_idx + 1) % len(model._grid_targets)
                model.goal_centroid = model._grid_targets[model._grid_idx]
                goal_r, goal_c = model.goal_centroid
                desired_dr = goal_r - cr
                desired_dc = goal_c - cc
                model.steps_without_progress = 0
            else:
                desired_dr = goal_r - cr
                desired_dc = goal_c - cc

            # Greedy: pick action whose delta best aligns with desired direction
            if model.prober is not None:
                best_act = model.prober.best_action_toward(pv, desired_dr, desired_dc)
                if best_act is not None:
                    return GameAction.from_id(best_act), None

    # Phase 4: MCTS fallback
    return GameAction.from_id(legal[action_int % len(legal)]), None


# ── Game model ────────────────────────────────────────────────────────────────

class GameModel:
    """
    Accumulates knowledge about a game from observations.

    Tracks:
    - background value
    - which actions move which objects, and by how much
    - what the winning condition looks like
    - how many clicks of each type have been used
    """

    def __init__(self, game_id: str):
        self.game_id = game_id
        self.background: Optional[int] = None

        # action_key → list of ActionEffect
        self._effects: Dict[str, List[ActionEffect]] = defaultdict(list)
        # action_key → mean (dr, dc) per moved value
        self._mean_effects: Dict[str, Dict[int, Tuple[float, float]]] = {}

        # objective: (moving_value, goal_value) inferred from win frames
        self.moving_values: List[int] = []   # values that can be moved by actions
        self.winning_frames: List[np.ndarray] = []

        # Click grid for click games: bin_id → effect summary
        self._click_bins: Dict[int, List[ActionEffect]] = defaultdict(list)

        # Exploration state: tracks which click position to try next
        self.explore_idx: int = 0               # index into candidate positions
        self.explore_candidates: List[Tuple[int, int]] = []   # (x, y) to probe
        self.last_click: Optional[Tuple[int, int]] = None
        self.steps_at_last_click: int = 0       # how many consecutive same-position clicks

        # Probe-then-navigate for directional games
        self.prober: Optional[GameProber] = None
        self.player_value_int: Optional[int] = None   # which pixel value is the player
        self.goal_centroid: Optional[Tuple[float, float]] = None
        self.goal_candidates: Optional[List[Tuple[int, float, float]]] = None
        self._last_dist: Optional[float] = None
        self.steps_without_progress: int = 0
        self.bfs_plan_queue: List[int] = []           # remaining BFS actions (unused if greedy)

        # Click games: track dead positions (bins that never cause any frame change)
        self.dead_clicks: set = set()   # set of (bx, by) grid-bin tuples

        # Slider game scan mode (set by _decode_directional_planned)
        self._slider_mode: bool = False
        self._slider_actions: List[int] = []
        self._slider_phase: int = 0       # which "stay at position" phase
        self._slider_phase_steps: int = 0 # steps in current phase

    def _action_key(self, action_id: Any, click_data: Optional[Dict]) -> str:
        if click_data is not None:
            x, y = click_data.get('x', 0), click_data.get('y', 0)
            n = 4
            bx, by = int(x * n // 64), int(y * n // 64)
            return f"click_bin_{bx}_{by}"   # col-first: bx=column bin, by=row bin
        return str(action_id)

    def record_effect(self, effect: ActionEffect) -> None:
        key = self._action_key(effect.action_id, effect.click_data)
        self._effects[key].append(effect)
        for val, (dr, dc) in effect.delta_centroids.items():
            if val not in self.moving_values:
                self.moving_values.append(val)
        if effect.level_up and effect.click_data is not None:
            x, y = effect.click_data.get('x', 0), effect.click_data.get('y', 0)
            n = 4  # n_bins
            bx, by = int(x * n // 64), int(y * n // 64)
            bin_id = bx + by * n   # consistent: bx is col-index (minor), by is row-index (major)
            self._click_bins[bin_id].append(effect)
        # Feed result into the active prober (directional game probe phase)
        if self.prober is not None and not self.prober.done:
            self.prober.record_result(effect)
        # Track dead click positions (bins that never cause any frame change at all)
        if effect.click_data is not None and effect.pixels_changed == 0:
            x = effect.click_data.get('x', 0)
            y = effect.click_data.get('y', 0)
            n = 4
            bx, by = int(x * n // 64), int(y * n // 64)
            self.dead_clicks.add((bx, by))

    def reset_for_new_level(self) -> None:
        """Call when a level completes to start fresh exploration for the next level."""
        self._click_bins = defaultdict(list)
        self.explore_idx = 0
        self.explore_candidates = []
        self.last_click = None
        self.steps_at_last_click = 0
        # Recompute mean effects
        self._mean_effects = {}
        for k, eff_list in self._effects.items():
            combined: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
            for e in eff_list:
                for val, delta in e.delta_centroids.items():
                    combined[val].append(delta)
            self._mean_effects[k] = {
                val: (
                    sum(d[0] for d in deltas) / len(deltas),
                    sum(d[1] for d in deltas) / len(deltas),
                )
                for val, deltas in combined.items()
            }
        # Reset directional game navigation state for new level
        self.prober = None
        self.player_value_int = None
        self.goal_centroid = None
        self.goal_candidates = None
        self._last_dist = None
        self.steps_without_progress = 0
        self.bfs_plan_queue = []
        self._slider_mode = False
        self._slider_actions = []
        self._slider_phase = 0
        self._slider_phase_steps = 0
        self.__dict__.pop('_probed_logged', None)  # reset one-shot flag
        self.__dict__.pop('_grid_targets', None)   # reset grid scan
        self.__dict__.pop('_grid_idx', None)
        self.__dict__.pop('_non_dir_ctr', None)    # reset non-dir cycling counter
        # Keep dead_clicks across levels (structural game property)

    def record_win(self, frame: np.ndarray) -> None:
        self.winning_frames.append(frame.copy())

    def best_action_for_direction(self,
                                  target_value: int,
                                  goal_centroid: Tuple[float, float],
                                  current_centroid: Tuple[float, float],
                                  available_action_keys: List[str]) -> Optional[str]:
        """
        Among available actions, pick the one that moves target_value
        most in the direction of (goal_centroid - current_centroid).
        """
        gr, gc = goal_centroid
        cr, cc = current_centroid
        desired_dr = gr - cr
        desired_dc = gc - cc
        desired_mag = (desired_dr ** 2 + desired_dc ** 2) ** 0.5
        if desired_mag < 0.001:
            return None

        best_key = None
        best_dot = -1e9
        for key in available_action_keys:
            effects = self._mean_effects.get(key, {})
            if target_value not in effects:
                continue
            dr, dc = effects[target_value]
            dot = (dr * desired_dr + dc * desired_dc) / desired_mag
            if dot > best_dot:
                best_dot = dot
                best_key = key
        return best_key

    def summary(self) -> str:
        lines = [f"GameModel({self.game_id})",
                 f"  background={self.background}",
                 f"  moving_values={self.moving_values}",
                 f"  known_actions={list(self._effects.keys())}",
                 f"  wins_observed={len(self.winning_frames)}"]
        return "\n".join(lines)


# ── Probing strategy for click games ─────────────────────────────────────────

def click_centers_for_bins(n_bins: int = 4) -> List[Tuple[int, int]]:
    """Return (x, y) centre pixels for each of the n_bins×n_bins click bins."""
    bin_size = 64 // n_bins
    centers = []
    for by in range(n_bins):
        for bx in range(n_bins):
            cx = int((bx + 0.5) * bin_size)
            cy = int((by + 0.5) * bin_size)
            centers.append((cx, cy))
    return centers


def pick_object_click_positions(frame: np.ndarray, background: int,
                                 max_per_value: int = 2) -> List[Tuple[int, int, int]]:
    """
    Find candidate click positions by detecting non-background objects.
    Returns list of (x, y, value) sorted by descending object size.
    Useful for exploration: click each distinct object to see what happens.
    """
    objs = find_objects(frame, ignore_values={background})
    # Group by value and pick centroids
    by_value: Dict[int, List[GameObject]] = defaultdict(list)
    for o in objs:
        by_value[o.value].append(o)
    candidates = []
    for val, obj_list in by_value.items():
        # Sort by size descending, take top N
        for obj in sorted(obj_list, key=lambda o: o.size, reverse=True)[:max_per_value]:
            cr, cc = obj.centroid
            candidates.append((int(cc), int(cr), val))  # (x=col, y=row, value)
    # Sort by object size overall (largest first)
    candidates.sort(key=lambda t: sum(
        o.size for o in by_value[t[2]]), reverse=True)
    return candidates


def infer_click_target(frame: np.ndarray,
                       model: GameModel,
                       goal_frame: Optional[np.ndarray] = None,
                       n_bins: int = 4) -> Tuple[int, int]:
    """
    Given current frame and what we know about the game, choose the
    best (x, y) click position.

    Strategy (in priority order):
    1. If we know which click positions caused level-ups previously, click there.
    2. If we have a goal frame, find moving objects and click toward the goal direction.
    3. If we know which values are moveable, click at the centroid of their mover.
    4. If we have recorded movement effects, pick the action that best advances toward goal.
    5. Otherwise, click at non-background object centroids (structured exploration).
    """
    bg = model.background or detect_background(frame)

    # Priority 1: replay known level-up click bins
    best_win_bin = None
    best_win_count = 0
    for bin_id, effs in model._click_bins.items():
        win_count = sum(1 for e in effs if e.level_up)
        if win_count > best_win_count:
            best_win_count = win_count
            best_win_bin = bin_id
    if best_win_bin is not None:
        bx = best_win_bin % n_bins      # bin_id = bx + by*n_bins → bx = bin_id % n
        by = best_win_bin // n_bins     # by = bin_id // n
        centers = click_centers_for_bins(n_bins)
        return centers[by * n_bins + bx]

    # Priority 2: goal-frame-guided click
    if goal_frame is not None:
        bg_goal = detect_background(goal_frame)
        state_now = frame_to_state(frame, bg)
        state_goal = frame_to_state(goal_frame, bg_goal)
        for val in model.moving_values:
            if val in state_now and val in state_goal:
                cn = state_now[val]
                cg = state_goal[val]
                if abs(cn[0] - cg[0]) > 2 or abs(cn[1] - cg[1]) > 2:
                    action_keys = [f"click_bin_{bx}_{by}"
                                   for by in range(n_bins) for bx in range(n_bins)]
                    best = model.best_action_for_direction(val, cg, cn, action_keys)
                    if best is not None:
                        parts = best.split('_')
                        bx, by = int(parts[-2]), int(parts[-1])
                        centers = click_centers_for_bins(n_bins)
                        return centers[by * n_bins + bx]

    # Priority 3: exploration — cycle through all distinct non-background objects.
    # Initialise once and cycle; never reset mid-run so the index keeps advancing.
    objs_all = find_objects(frame, ignore_values={bg})
    obj_centroids = []
    seen_vals = set()
    for o in sorted(objs_all, key=lambda x: x.size, reverse=True):
        if o.value not in seen_vals:
            cr, cc = o.centroid
            obj_centroids.append((int(cc), int(cr)))  # (x=col, y=row)
            seen_vals.add(o.value)

    if obj_centroids:
        # Initialise candidate list once; do not modify after that
        # (prevents explore_idx drift from minor centroid jitter each frame)
        if not model.explore_candidates:
            candidates = obj_centroids[:]
            # Always append a fine 6×6 grid (36 positions, 11px resolution) so
            # the entire 64×64 field is covered even if objects are missed.
            # Object centroids go first (most likely to be interactive).
            existing = set(obj_centroids)
            fine_grid = [
                (int((bx + 0.5) * 64 // 6), int((by + 0.5) * 64 // 6))
                for by in range(6) for bx in range(6)
            ]
            candidates += [p for p in fine_grid if p not in existing]
            model.explore_candidates = candidates
            model.explore_idx = 0

        # Advance to next candidate after max_tries_per_position attempts
        max_tries_per_position = 1
        if model.steps_at_last_click >= max_tries_per_position:
            model.explore_idx = (model.explore_idx + 1) % len(model.explore_candidates)
            model.steps_at_last_click = 0

        idx = model.explore_idx % len(model.explore_candidates)
        chosen = model.explore_candidates[idx]
        if chosen == model.last_click:
            model.steps_at_last_click += 1
        else:
            model.last_click = chosen
            model.steps_at_last_click = 1
        return chosen

    # Priority 4: click at most-effective recorded bin
    centers = click_centers_for_bins(n_bins)
    best_idx = 0
    best_score = -1
    for idx, (cx, cy) in enumerate(centers):
        bx_b = cx * n_bins // 64
        by_b = cy * n_bins // 64
        key = f"click_bin_{bx_b}_{by_b}"
        effs = model._effects.get(key, [])
        score = sum(e.pixels_changed for e in effs) + sum(
            1 for e in effs if e.level_up) * 100
        if score > best_score:
            best_score = score
            best_idx = idx

    # Priority 5: click at non-background object centroids (exploration)
    if best_score <= 0:
        candidates = pick_object_click_positions(frame, bg)
        if candidates:
            return (candidates[0][0], candidates[0][1])

    return centers[best_idx]


# ── Generic action decision ───────────────────────────────────────────────────

def generic_decode(frame: np.ndarray,
                   available_actions: List[Any],
                   model: GameModel,
                   action_int: int,           # from MuZero MCTS
                   n_bins: int = 4,
                   goal_frame: Optional[np.ndarray] = None
                   ) -> Tuple[Any, Optional[Dict]]:
    """
    Choose action and optional click data for the current game state.

    Returns (action_obj_id, data_dict_or_None).
    action_obj_id is the integer action ID to pass to GameAction.from_id().
    """
    from arcengine import GameAction

    CLICK_IDS = {6, 7}   # action IDs that require x,y click data

    click_ids_in_avail = [a for a in available_actions if a in CLICK_IDS]

    if list(available_actions) == [6]:
        # Pure click game (action 6 only) — use smart click exploration
        act = GameAction.from_id(6)
        px, py = infer_click_target(frame, model, goal_frame, n_bins)
        return act, {"x": px, "y": py}

    if not any(a in [1, 2, 3, 4] for a in available_actions) and click_ids_in_avail:
        # No directional actions at all (e.g. su15=[6,7], sb26=[5,6,7]).
        # Route all click actions to smart exploration, cycling through each click type.
        non_dir = [a for a in sorted(available_actions) if a not in [1, 2, 3, 4]]
        click_actions = [a for a in non_dir if a in CLICK_IDS]
        if click_actions:
            # Cycle through all click types using action_int; all positions via infer_click_target
            click_id = click_actions[action_int % len(click_actions)]
            act = GameAction.from_id(click_id)
            px, py = infer_click_target(frame, model, goal_frame, n_bins)
            return act, {"x": px, "y": py}
        # Only non-click non-dir actions (e.g. action 5 only) → cycle them
        chosen = non_dir[action_int % len(non_dir)] if non_dir else sorted(available_actions)[0]
        return GameAction.from_id(chosen), None

    elif all(a in [1, 2, 3, 4] for a in available_actions):
        # Pure directional / keyboard game — probe-then-navigate strategy
        return _decode_directional_planned(
            frame, available_actions, model, action_int, goal_frame)

    else:
        # Mixed keyboard_click or other — prefer directional navigation
        legal = sorted(available_actions)
        dir_actions = [a for a in legal if a in [1, 2, 3, 4]]
        non_dir_actions = [a for a in legal if a not in [1, 2, 3, 4]]

        # Slider mode shortcut: if probe phase already identified this as a slider game,
        # route directly to _decode_directional_planned which contains the SWEEP scan.
        # Pass ALL legal actions (not just dir_actions) so the HOLD phase can fire
        # click/fire actions (e.g. action 6) — needed for slider+click games like sc25
        # where the win condition is "slide to position X, then click."
        if getattr(model, '_slider_mode', False):
            return _decode_directional_planned(
                frame, legal, model, action_int, goal_frame)

        # Spawn detection: if any NON-DIRECTIONAL non-click action caused large pixel
        # changes (e.g. action 5 fires a projectile), bump to 50% fire rate.
        # Must exclude directional keys (1-4) — their movement also causes large px changes
        # but is NOT spawn behavior (e.g. sp80 left/right = 162px → falsely triggered).
        if non_dir_actions and not getattr(model, '_spawn_detected', False):
            dir_keys = {str(GameAction.from_id(a)) for a in dir_actions}
            for key, effs in model._effects.items():
                if key.startswith('click_bin'):
                    continue
                if key in dir_keys:
                    continue  # Skip directional action effects (e.g. sp80 move = 162px)
                if any(e.pixels_changed > 40 for e in effs):
                    model._spawn_detected = True  # type: ignore[attr-defined]
                    model._fire_every = 2          # type: ignore[attr-defined]
                    print(f"  [spawn] detected spawn action (key={key}), fire_every=2 (50%)", flush=True)
                    break

        # Dead-directional override: all probe actions yielded 0px change (e.g. sc25).
        # Route directly to smart click exploration (infer_click_target) for full coverage.
        # Also fire non-click non-dir actions (e.g. action 5 = spawn/fire) at 25% rate.
        if getattr(model, '_all_dir_zero', False):
            fire_actions = [a for a in non_dir_actions if a not in CLICK_IDS]
            if fire_actions and action_int % 4 == 0:
                chosen = fire_actions[(action_int // 4) % len(fire_actions)]
                return GameAction.from_id(chosen), None
            click_id = next((a for a in non_dir_actions if a in CLICK_IDS), None)
            if click_id is not None:
                act = GameAction.from_id(click_id)
                px, py = infer_click_target(frame, model, goal_frame, n_bins)
                return act, {"x": px, "y": py}
            # No click action available despite dead dirs — fall through to non-dir cycle
            use_dir = False
        else:
            fire_every = getattr(model, '_fire_every', 4)
            use_dir = dir_actions and not (action_int % fire_every == 0 and non_dir_actions)
        if use_dir:
            return _decode_directional_planned(
                frame, dir_actions, model, action_int, goal_frame)
        # Non-directional turn: cycle through non-directional actions safely.
        # Use a dedicated counter (not action_int) so all non-dir actions get equal
        # turns — action_int at multiples of fire_every always has the same remainder,
        # which would freeze the cycle on one action (e.g. always act 5, never act 6).
        ctr = getattr(model, '_non_dir_ctr', 0)
        model._non_dir_ctr = ctr + 1  # type: ignore[attr-defined]
        chosen_id = non_dir_actions[ctr % len(non_dir_actions)] if non_dir_actions else legal[0]
        act = GameAction.from_id(chosen_id)
        if chosen_id in CLICK_IDS:
            # Use smart click exploration (same as pure click games) rather than
            # fixed bin cycling — this respects win-bin history and object centroids.
            px, py = infer_click_target(frame, model, goal_frame, n_bins)
            return act, {"x": px, "y": py}
        return act, None


# ── Learning from recordings ──────────────────────────────────────────────────

def learn_from_recordings(game_id: str,
                           recordings_dir: Path,
                           model: GameModel) -> int:
    """
    Parse available recordings for game_id and populate model with
    action effects observed in those recordings.
    Returns number of episodes loaded.
    """
    pattern = f"{game_id}*.recording.jsonl"
    files = list(recordings_dir.glob(pattern))
    loaded = 0
    for fpath in files:
        try:
            _load_single_recording(fpath, model)
            loaded += 1
        except Exception:
            pass
    return loaded


def _load_single_recording(fpath: Path, model: GameModel) -> None:
    """
    Parse one .recording.jsonl into the game model.

    Recording format: each line is JSON with keys 'timestamp' and 'data'.
    data.frame        → (C,H,W) or (H,W) pixel array
    data.levels_completed → int
    data.action_input → {'id': int, 'data': {'x':..,'y':..}}  (may be all RESET=0)

    Since many recordings are "action-blind" (action_input always shows id=0),
    we learn purely from frame-to-frame differences: which values move, and
    what the final frame looks like when a level completes.
    """
    lines = fpath.read_text(encoding='utf-8').splitlines()
    if not lines:
        return

    prev_frame = None
    prev_lv: int = 0

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            outer = json.loads(raw)
        except json.JSONDecodeError:
            continue
        inner = outer.get('data', outer)  # handle both wrapped and flat formats

        frame_raw = inner.get('frame') or inner.get('observation')
        if frame_raw is None:
            continue
        arr = np.array(frame_raw, dtype=np.int32)
        if arr.ndim == 3:
            arr = arr[-1]   # take last channel (most recent frame)

        if model.background is None:
            model.background = detect_background(arr)

        lv = int(inner.get('levels_completed', 0) or 0)
        level_up = lv > prev_lv

        if level_up:
            model.record_win(arr)

        # Action-agnostic learning: observe frame differences.
        # Only track values that moved with meaningful displacement (>= 2px),
        # to filter out high-frequency noise (timer bars, flickering backgrounds).
        if prev_frame is not None:
            eff = compute_action_effect(
                prev_frame, arr,
                action_id=None, click_data=None,
                reward=1.0 if level_up else 0.0,
                level_up=level_up,
            )
            # Filter: only record effects where at least one value moved >= 2 pixels
            meaningful = {v: d for v, d in eff.delta_centroids.items()
                          if abs(d[0]) >= 2.0 or abs(d[1]) >= 2.0}
            if meaningful:
                eff.delta_centroids = meaningful
                eff.moved_values = list(meaningful.keys())
                model.record_effect(eff)

        # Try to extract explicit action if available
        ai = inner.get('action_input')
        if isinstance(ai, dict) and ai.get('id', 0) not in (0, None):
            act_id = ai.get('id')
            act_data = ai.get('data') or {}
            if act_id == 6 and 'x' in act_data:
                click_data = {'x': act_data['x'], 'y': act_data['y']}
                if prev_frame is not None:
                    eff2 = compute_action_effect(
                        prev_frame, arr, action_id=6, click_data=click_data,
                        reward=1.0 if level_up else 0.0, level_up=level_up,
                    )
                    model.record_effect(eff2)

        prev_frame = arr
        prev_lv = lv


# ── Goal inference ────────────────────────────────────────────────────────────

def infer_critical_values(initial_frame: np.ndarray,
                           winning_frame: np.ndarray,
                           background: Optional[int] = None,
                           ui_rows_top: int = 1) -> List[Tuple[int, float, float]]:
    """
    Compare initial and winning frames to find values that changed position.
    Returns [(value, delta_row, delta_col), ...] sorted by magnitude of movement.

    This is more reliable than action-pair learning because we directly compare
    the 'start' and 'goal' states without needing to know the intermediate actions.
    """
    bg = background or detect_background(initial_frame)
    # Ignore UI rows (timer etc.) at top/bottom
    h = initial_frame.shape[0]
    init = initial_frame[ui_rows_top:h - ui_rows_top, :]
    win  = winning_frame[ui_rows_top:h - ui_rows_top, :]

    results = []
    for val in np.unique(init):
        if val == bg:
            continue
        pos_init = np.where(init == val)
        pos_win  = np.where(win  == val)
        if len(pos_init[0]) == 0 or len(pos_win[0]) == 0:
            continue
        cr_init = float(np.mean(pos_init[0]))
        cc_init = float(np.mean(pos_init[1]))
        cr_win  = float(np.mean(pos_win[0]))
        cc_win  = float(np.mean(pos_win[1]))
        dr = cr_win - cr_init
        dc = cc_win - cc_init
        if abs(dr) >= 2.0 or abs(dc) >= 2.0:
            results.append((int(val), dr, dc))
    results.sort(key=lambda t: (t[1]**2 + t[2]**2)**0.5, reverse=True)
    return results


def infer_goal(model: GameModel) -> Optional[Tuple[int, int]]:
    """
    From winning frames, infer the (moving_value, goal_value) pair that
    was adjacent/overlapping at the moment of victory.
    """
    if not model.winning_frames or not model.moving_values:
        return None

    for wframe in model.winning_frames:
        bg = model.background or detect_background(wframe)
        objs = find_objects(wframe, ignore_values={bg})
        mover_objs = [o for o in objs if o.value in model.moving_values]
        fixed_objs  = [o for o in objs if o.value not in model.moving_values]
        for mov in mover_objs:
            for fix in fixed_objs:
                if objects_adjacent(mov, fix, max_dist=4):
                    return (mov.value, fix.value)
    return None


# ── Public factory ────────────────────────────────────────────────────────────

def make_game_model(game_id: str,
                    recordings_dir: Optional[Path] = None) -> GameModel:
    """
    Create and pre-populate a GameModel for the given game.

    Loads from recordings if available, then:
    1. Filters moving_values to those that show large consistent displacement
    2. Uses initial→goal frame comparison to identify the critical value
    3. Loads persisted winning frames from previous runs (results/win_frames/)
    """
    model = GameModel(game_id)
    if recordings_dir is None:
        return model

    # Load persisted winning frames from previous runs so future lives can
    # navigate to known good positions (e.g. sp80 player position at win).
    game_prefix = game_id.split('-')[0]
    # generic_solver.py lives in neurosym/, arc3 root is one level up
    _arc3_root = Path(__file__).resolve().parent.parent
    win_frames_dir = _arc3_root / "results" / "win_frames"
    if win_frames_dir.is_dir():
        wf_paths = sorted(win_frames_dir.glob(f"{game_id}_lv*.npy")) or \
                   sorted(win_frames_dir.glob(f"{game_prefix}_lv*.npy"))
        for wf_path in wf_paths:
            try:
                wf = np.load(str(wf_path)).astype(np.int32)
                model.winning_frames.append(wf)
                print(f"  [win-cache] loaded {wf_path.name}", flush=True)
            except Exception:
                pass

    # Find all recordings for this game (match game_id prefix)
    game_prefix = game_id.split('-')[0]   # e.g. "vc33" from "vc33-9851e02b"
    full_prefix = game_id                 # exact match first
    files = sorted(recordings_dir.glob(f"{full_prefix}*.recording.jsonl"))
    if not files:
        files = sorted(recordings_dir.glob(f"{game_prefix}*.recording.jsonl"))

    loaded = 0
    initial_frames: list = []
    for fpath in files[:10]:  # cap at 10 recordings
        try:
            # Collect initial frame from this recording before loading effects
            lines = fpath.read_text(encoding='utf-8').splitlines()
            for raw in lines[:2]:  # first frame = initial state
                if not raw.strip():
                    continue
                outer = json.loads(raw)
                inner = outer.get('data', outer)
                fr = inner.get('frame') or inner.get('observation')
                if fr is not None:
                    arr = np.array(fr, dtype=np.int32)
                    if arr.ndim == 3:
                        arr = arr[-1]
                    initial_frames.append(arr)
                    break
            _load_single_recording(fpath, model)
            loaded += 1
        except Exception:
            pass

    if loaded == 0:
        return model

    # Use initial→winning frame comparison to identify critical values
    if model.winning_frames and initial_frames:
        init_frame = initial_frames[0]
        win_frame  = model.winning_frames[0]
        critical = infer_critical_values(
            init_frame, win_frame, background=model.background)
        if critical:
            # Overwrite moving_values with the critically-moving values only
            model.moving_values = [val for val, dr, dc in critical]

    return model
