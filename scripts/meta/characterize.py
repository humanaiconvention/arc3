"""Generic game characterization.

Takes an open GameSession, observes initial state, probes each action once,
and produces a GameProfile with heuristic signals strategies use to rank.

Design goals:
- Cheap: bounded by len(ALL_ACTIONS) + 1 resets (<30s typical).
- Pure observation: no attempts at winning.
- Does not consume lives if avoidable (we reset fresh per action probe).

Heuristic signals (looks_like_*) are 0-1 confidence scores based on cheap
structural cues. Strategies use them to decide order/budget, but all
non-zero strategies get at least some budget.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from meta.common import GameSession, life_cells, ALL_ACTIONS
from meta.profile import ActionEffect, GameProfile


# Rows that change on nearly every action (life-timer tick, scorecard indicator).
# We strip them from the "cells changed" measurement so heuristics aren't fooled.
_NOISE_ROWS = (59, 60, 63)

# Click heat-map probe defaults. Sparse 8-pixel grid covers a 64x64 frame in 64
# probes; danger-abort kicks in fast if most clicks GAME_OVER (e.g. tu93).
_CLICK_HEATMAP_GRID_STEP = 8
_CLICK_HEATMAP_LIVE_THRESHOLD = 1
_CLICK_HEATMAP_DANGER_ABORT = 6   # if first 12 probes have >=6 terminals, give up
_CLICK_HEATMAP_DANGER_AFTER = 12

# Action-pair probe: tests every ordered pair (a, b) over the game's available
# actions in {1,2,3,4,5}. Self-pairs (a, a) included. ACTION6/7 are coord
# actions (no useful default-coord behaviour) so they're excluded. ACTION5 is
# included because in mixed nav+click games it has a default position. The
# probe reveals non-commutativity and conditional effects (b is a no-op alone
# but active after a — e.g. re86's "act5 preserves partial removals" hypothesis).
_PAIR_PROBE_ACTIONS = (1, 2, 3, 4, 5)

# 3-action triple probe: gated on len(inverse_pairs) >= 3 (triangular subgroup
# signature). Cost: O(n^3) — for n=4 that's 64 triples × 4 calls = ~80s. Used
# to detect true 3-cycle group structure on games like tu93 and g50t.
_TRIPLE_PROBE_MIN_PAIRS = 3

# State-conditional probe: for static games with nav actions, probe each
# action's post-prelude response. Reveals "unfreeze levers" (g50t acts 2/4)
# and dominant nav actions hidden by a static initial state. Cost: per game
# ~30s (4 navs × 5 probe-actions × ~5 API calls each). Gated to avoid
# wasting budget on responsive games where the standard pair probe already
# captures the same info.
_PRELUDE_PROBE_STATIC_THRESHOLD = 0.7
_PRELUDE_PROBE_N_BEFORE = 3


def _probe_action(sess: GameSession, base_frame: np.ndarray, act_id: int) -> ActionEffect:
    """Press one action once from a freshly reset state. Returns ActionEffect.
    Caller must ensure sess is at freshly-reset state before calling."""
    life_before = life_cells(base_frame)
    r = sess.step(act_id)

    if r.frame is None:
        return ActionEffect(
            action_id=act_id,
            cells_changed=0,
            rows_changed=[],
            cols_changed=[],
            n_level_gained=r.levels_completed,
            is_terminal=r.terminal,
            exception=r.exception or ("no_frame" if r.state is None else ""),
            delta_life=0,
        )

    diff = base_frame != r.frame
    # Mask noise rows so they don't count toward cell change
    for nr in _NOISE_ROWS:
        if nr < diff.shape[0]:
            diff[nr, :] = False
    n_changed = int(np.sum(diff))
    rows = sorted(set(np.where(diff)[0].tolist()))
    cols = sorted(set(np.where(diff)[1].tolist()))
    life_after = life_cells(r.frame)
    return ActionEffect(
        action_id=act_id,
        cells_changed=n_changed,
        rows_changed=rows,
        cols_changed=cols,
        n_level_gained=r.levels_completed,
        is_terminal=r.terminal,
        exception=r.exception,
        delta_life=life_after - life_before if life_before >= 0 and life_after >= 0 else 0,
    )


def _probe_click_heatmap(sess: GameSession, click_act: int,
                         grid_step: int = _CLICK_HEATMAP_GRID_STEP) -> tuple[list, dict]:
    """Sparse coordinate-click scan. Resets before each probe so every click is
    measured from the same baseline. Aborts early on a danger field (most clicks
    GAME_OVER). Returns (heatmap, meta).

    heatmap: [{x, y, cells_changed, act}, ...] sorted by cells_changed desc.
    meta: {grid_step, click_act, n_probes, n_terminal, n_live, danger_aborted}.
    """
    rr = sess.reset()
    if rr.frame is None:
        return [], {"grid_step": grid_step, "click_act": click_act,
                    "n_probes": 0, "n_terminal": 0, "n_live": 0,
                    "danger_aborted": False, "skipped_reason": "reset_failed"}

    h, w = rr.frame.shape
    heatmap: list = []
    n_probes = 0
    n_terminal = 0
    danger_aborted = False

    for y in range(grid_step // 2, h, grid_step):
        for x in range(grid_step // 2, w, grid_step):
            if n_probes >= _CLICK_HEATMAP_DANGER_AFTER and n_terminal >= _CLICK_HEATMAP_DANGER_ABORT:
                danger_aborted = True
                break

            rr = sess.reset()
            if rr.frame is None:
                break
            before = rr.frame.copy()
            r = sess.step_with_data(click_act, {"x": x, "y": y})
            n_probes += 1
            if r.frame is None:
                continue
            if r.terminal:
                n_terminal += 1
                continue
            diff = before != r.frame
            for nr in _NOISE_ROWS:
                if nr < diff.shape[0]:
                    diff[nr, :] = False
            changed = int(np.sum(diff))
            if changed >= _CLICK_HEATMAP_LIVE_THRESHOLD:
                heatmap.append({"x": int(x), "y": int(y),
                                "cells_changed": changed, "act": click_act})
        if danger_aborted:
            break

    heatmap.sort(key=lambda e: -e["cells_changed"])
    return heatmap, {
        "grid_step": grid_step,
        "click_act": click_act,
        "n_probes": n_probes,
        "n_terminal": n_terminal,
        "n_live": len(heatmap),
        "danger_aborted": danger_aborted,
    }


def _frame_hash(frame: np.ndarray) -> str:
    """Stable hash of a frame for commutativity comparisons. Masks noise rows
    so life-timer ticks don't make every pair look non-commutative."""
    import hashlib
    f = frame.copy()
    for nr in _NOISE_ROWS:
        if nr < f.shape[0]:
            f[nr, :] = 0
    return hashlib.sha1(f.tobytes()).hexdigest()[:16]


def _nchanged(a: np.ndarray, b: np.ndarray) -> int:
    """Count cells changed between two frames, ignoring noise rows."""
    diff = a != b
    for nr in _NOISE_ROWS:
        if nr < diff.shape[0]:
            diff[nr, :] = False
    return int(np.sum(diff))


def _probe_action_pairs(sess: GameSession, available_actions: list,
                        base_frame: np.ndarray) -> tuple[dict, dict]:
    """Probe every ordered pair (a, b) and self-pair (a, a) over the available
    actions in {1,2,3,4,5}. Coord-only games (5/6/7 only) are skipped.
    Returns (pairs, derived).

    pairs: {"a_b": {final_hash, n_changed_from_initial, delta_b_after_a, terminal}}.
    derived: {non_commutative_pairs, idempotent_actions, conditional_effect_pairs}.
    """
    coord_only = (any(a in available_actions for a in [5, 6, 7])
                  and not any(a in available_actions for a in [1, 2, 3, 4]))
    pairs: dict = {}
    if coord_only:
        return pairs, {"skipped_reason": "coord_only_game",
                       "actions_probed": []}

    actions = [a for a in _PAIR_PROBE_ACTIONS if a in available_actions]
    if len(actions) < 2:
        return pairs, {"skipped_reason": "fewer_than_2_pair_actions",
                       "actions_probed": actions}

    base_hash = _frame_hash(base_frame)

    for a in actions:
        for b in actions:
            rr = sess.reset()
            if rr.frame is None:
                pairs[f"{a}_{b}"] = {"skipped": "reset_failed"}
                continue
            r1 = sess.step(a)
            if r1.frame is None or r1.terminal:
                pairs[f"{a}_{b}"] = {"skipped": "first_action_failed",
                                     "terminal_after_first": bool(r1.terminal)}
                continue
            mid_frame = r1.frame.copy()
            r2 = sess.step(b)
            if r2.frame is None:
                pairs[f"{a}_{b}"] = {"skipped": "second_action_failed"}
                continue
            pairs[f"{a}_{b}"] = {
                "final_hash": _frame_hash(r2.frame),
                "n_changed_from_initial": _nchanged(base_frame, r2.frame),
                "delta_b_after_a": _nchanged(mid_frame, r2.frame),
                "terminal": bool(r2.terminal),
            }

    # Derived signals
    non_commutative = []
    for a in actions:
        for b in actions:
            if a >= b:
                continue
            ab = pairs.get(f"{a}_{b}", {}).get("final_hash")
            ba = pairs.get(f"{b}_{a}", {}).get("final_hash")
            if ab and ba and ab != ba:
                non_commutative.append([a, b])

    idempotent = []
    for a in actions:
        aa = pairs.get(f"{a}_{a}", {}).get("final_hash")
        if aa == base_hash:
            idempotent.append({"action": a, "kind": "self_inverse"})

    # Conditional effect: b after a produced cell changes (delta_b_after_a > 0).
    # Caller can join with action_effects to find b that's a no-op alone.
    conditional_effect_candidates = []
    for a in actions:
        for b in actions:
            if a == b:
                continue
            d = pairs.get(f"{a}_{b}", {}).get("delta_b_after_a", 0)
            if d > 0:
                conditional_effect_candidates.append([a, b, d])

    return pairs, {
        "actions_probed": actions,
        "non_commutative_pairs": non_commutative,
        "idempotent_actions": idempotent,
        "conditional_effect_candidates": conditional_effect_candidates,
        "n_pairs_probed": len(pairs),
    }


def _probe_action_triples(sess: GameSession, available_actions: list,
                          base_frame: np.ndarray) -> tuple[dict, list]:
    """Probe every ordered triple (a, b, c) over the available pair-probe
    actions. Returns (triples, three_cycles).

    triples: {"a_b_c": {final_hash, n_changed_from_initial, terminal}}
    three_cycles: [[a, b, c], ...] where the triple returns to base — these
        are the candidate 3-cycle group elements. Trivial triples involving
        repeated actions where the underlying pair is also a cycle (e.g.
        (1,1,1) with toggle 1) are *not* filtered — caller can decide.

    Cost: O(n^3) probes × ~4 API calls each. Caller should gate.
    """
    coord_only = (any(a in available_actions for a in [5, 6, 7])
                  and not any(a in available_actions for a in [1, 2, 3, 4]))
    if coord_only:
        return {}, []
    actions = [a for a in _PAIR_PROBE_ACTIONS if a in available_actions]
    if len(actions) < 3:
        return {}, []

    base_hash = _frame_hash(base_frame)
    triples: dict = {}

    for a in actions:
        for b in actions:
            for c in actions:
                rr = sess.reset()
                if rr.frame is None:
                    triples[f"{a}_{b}_{c}"] = {"skipped": "reset_failed"}
                    continue
                ra = sess.step(a)
                if ra.frame is None or ra.terminal:
                    triples[f"{a}_{b}_{c}"] = {"skipped": "first_action_failed",
                                               "terminal_after": 1 if ra.terminal else 0}
                    continue
                rb = sess.step(b)
                if rb.frame is None or rb.terminal:
                    triples[f"{a}_{b}_{c}"] = {"skipped": "second_action_failed",
                                               "terminal_after": 2 if rb.terminal else 0}
                    continue
                rc = sess.step(c)
                if rc.frame is None:
                    triples[f"{a}_{b}_{c}"] = {"skipped": "third_action_failed"}
                    continue
                triples[f"{a}_{b}_{c}"] = {
                    "final_hash": _frame_hash(rc.frame),
                    "n_changed_from_initial": _nchanged(base_frame, rc.frame),
                    "terminal": bool(rc.terminal),
                }

    three_cycles: list = []
    for key, entry in triples.items():
        if entry.get("final_hash") == base_hash and entry.get("n_changed_from_initial") == 0:
            a, b, c = (int(x) for x in key.split("_"))
            three_cycles.append([a, b, c])

    return triples, three_cycles


def _probe_post_prelude_response(sess: GameSession, available_actions: list,
                                 base_frame: np.ndarray,
                                 n_before: int = _PRELUDE_PROBE_N_BEFORE) -> dict:
    """For each available nav action as a 3-step prelude, probe each pair-probe
    action and record cells_changed vs the post-prelude frame. Used on
    static-init games (sc25, g50t, lf52, etc.) where the initial state is
    unresponsive but becomes active after a setup.

    Returns: {"act<P>_x<N>": {"unfreeze": bool, "responses": {act_id: cells_changed, ...}}}
    """
    out: dict = {}
    nav_options = [a for a in (1, 2, 3, 4) if a in available_actions]
    if not nav_options:
        return out
    pair_actions = [a for a in _PAIR_PROBE_ACTIONS if a in available_actions]
    if not pair_actions:
        return out
    base_hash = _frame_hash(base_frame)

    for prelude_act in nav_options:
        # First, capture the post-prelude frame.
        rr = sess.reset()
        if rr.frame is None:
            continue
        prelude_failed = False
        for _ in range(n_before):
            r = sess.step(prelude_act)
            if not r.ok or r.terminal:
                prelude_failed = True
                break
        if prelude_failed:
            continue
        post_prelude_frame = sess.frame.copy()
        unfrozen = _frame_hash(post_prelude_frame) != base_hash

        responses: dict = {}
        for a in pair_actions:
            rr = sess.reset()
            if rr.frame is None:
                continue
            ok = True
            for _ in range(n_before):
                r = sess.step(prelude_act)
                if not r.ok or r.terminal:
                    ok = False
                    break
            if not ok:
                continue
            r = sess.step(a)
            if r.frame is None or r.terminal:
                responses[str(a)] = -1  # sentinel: no usable response
                continue
            responses[str(a)] = _nchanged(post_prelude_frame, r.frame)
        out[f"act{prelude_act}_x{n_before}"] = {
            "unfreeze": unfrozen,
            "responses": responses,
        }
    return out


def characterize(sess: GameSession, probe_each_fresh: bool = True,
                 probe_click_heatmap: bool = True,
                 probe_action_pairs: bool = True,
                 probe_action_triples: str = "auto",
                 probe_post_prelude: str = "auto") -> tuple:
    """Run the full characterization on an already-opened session.

    Args:
        sess: an open GameSession (will be reset fresh; any prior state lost).
        probe_each_fresh: if True, env.reset() between each action probe so
            every effect is measured from the same baseline. If False, apply
            actions sequentially (cheaper but mixed baselines — only useful
            for very expensive games).

    Returns:
        (profile, base_frame) where base_frame is the initial frame (2D int32).
    """
    started = time.time()
    r = sess.reset()
    if r.frame is None:
        raise RuntimeError(f"Reset returned no frame for {sess.game_id}")

    base_frame = r.frame.copy()
    win_levels = sess.win_levels
    available_actions = list(sess.available_actions) or ALL_ACTIONS
    frame_shape = [int(base_frame.shape[0]), int(base_frame.shape[1])]
    coord_only_click_game = any(a in available_actions for a in [5, 6, 7]) and not any(
        a in available_actions for a in [1, 2, 3, 4]
    )

    effects: dict = {}
    for act_id in ALL_ACTIONS:
        if coord_only_click_game and act_id in (5, 6, 7):
            effects[str(act_id)] = ActionEffect(
                action_id=act_id,
                cells_changed=0,
                rows_changed=[],
                cols_changed=[],
                exception="coord_probe_skipped",
            ).__dict__
            continue
        if probe_each_fresh:
            rr = sess.reset()
            if rr.frame is None:
                # reset failed — record the rest as errors and stop
                for remaining in ALL_ACTIONS[ALL_ACTIONS.index(act_id):]:
                    effects[str(remaining)] = ActionEffect(
                        action_id=remaining, cells_changed=0, rows_changed=[],
                        cols_changed=[], exception="reset_failed",
                    ).__dict__
                break
            local_base = rr.frame.copy()
        else:
            local_base = sess.frame.copy() if sess.frame is not None else base_frame

        eff = _probe_action(sess, local_base, act_id)
        effects[str(act_id)] = eff.__dict__

    # Structural analysis
    dynamic_rows = set()
    for eff in effects.values():
        dynamic_rows.update(eff.get("rows_changed", []))
    n_dynamic = len(dynamic_rows)
    n_static = int(base_frame.shape[0]) - n_dynamic

    # Heuristic signals
    signals = _compute_signals(base_frame, effects, available_actions)

    # Coordinate-click heat-map (only if a click action is available).
    # Prefer ACTION5; fall back to ACTION6. Skipped if neither is available
    # or if probe_click_heatmap=False.
    click_heatmap: list = []
    click_heatmap_meta: dict = {}
    if probe_click_heatmap:
        click_act = 5 if 5 in available_actions else (6 if 6 in available_actions else None)
        if click_act is not None:
            click_heatmap, click_heatmap_meta = _probe_click_heatmap(sess, click_act)

    # Action-pair non-commutativity probe (only if 2+ nav actions available).
    action_pair_effects: dict = {}
    derived_pair_signals: dict = {}
    if probe_action_pairs:
        action_pair_effects, derived_pair_signals = _probe_action_pairs(
            sess, available_actions, base_frame
        )

    # Higher-level derived signals from probe data.
    inverse_pairs = _derive_inverse_pairs(action_pair_effects)
    toggle_actions = _derive_toggle_actions(action_pair_effects)
    looks_like_cursor_click = _derive_cursor_click_signal(click_heatmap)

    # 3-action cycle probe — gated on triangular subgroup signature.
    # "auto" mode: run only if 3+ inverse pairs detected (tu93/g50t pattern).
    # True = always run. False = never.
    action_triple_effects: dict = {}
    three_cycles: list = []
    do_triples = (probe_action_triples is True or
                  (probe_action_triples == "auto" and len(inverse_pairs) >= _TRIPLE_PROBE_MIN_PAIRS))
    if do_triples:
        action_triple_effects, three_cycles = _probe_action_triples(
            sess, available_actions, base_frame
        )

    # State-conditional probe — gated on static-init signal + nav presence.
    # "auto" mode: run only if looks_like_static >= threshold AND nav exists.
    post_prelude_responses: dict = {}
    do_prelude = (probe_post_prelude is True or
                  (probe_post_prelude == "auto"
                   and signals.get("looks_like_static", 0.0) >= _PRELUDE_PROBE_STATIC_THRESHOLD
                   and any(a in available_actions for a in (1, 2, 3, 4))))
    if do_prelude:
        post_prelude_responses = _probe_post_prelude_response(
            sess, available_actions, base_frame
        )

    # Derive dominant_actions: top-K actions by cells_changed.
    # For responsive games (post_prelude empty), use initial action_effects.
    # For static games (post_prelude populated), use the unfreezing prelude's
    # post-prelude responses to find which actions actually move state.
    dominant_actions = _derive_dominant_actions(
        effects, post_prelude_responses, available_actions
    )

    prof = GameProfile(
        game_id=sess.game_id,
        game_prefix=sess.game_id.split("-", 1)[0],
        win_levels=win_levels,
        available_actions=available_actions,
        frame_shape=frame_shape,
        action_effects=effects,
        life_row=63,
        life_cells_initial=life_cells(base_frame),
        n_static_rows=n_static,
        n_dynamic_rows=n_dynamic,
        high_entropy_regions=_find_busy_regions(base_frame),
        click_heatmap=click_heatmap,
        click_heatmap_meta=click_heatmap_meta,
        action_pair_effects=action_pair_effects,
        derived_pair_signals=derived_pair_signals,
        inverse_pairs=inverse_pairs,
        toggle_actions=toggle_actions,
        looks_like_cursor_click=looks_like_cursor_click,
        action_triple_effects=action_triple_effects,
        three_cycles=three_cycles,
        post_prelude_responses=post_prelude_responses,
        dominant_actions=dominant_actions,
        profile_elapsed=time.time() - started,
        **signals,
    )
    return prof, base_frame


# ── Higher-level derivations ────────────────────────────────────────────────
def _derive_inverse_pairs(pair_effects: dict) -> list:
    """Return [[a, b], ...] for distinct a < b where (a, b) and (b, a) both
    return to the base frame (delta-from-initial == 0). These are mutual
    inverses — the canonical signature of direction-pair nav structure."""
    pairs: list = []
    seen = set()
    for key, entry in pair_effects.items():
        try:
            a, b = (int(x) for x in key.split("_"))
        except (ValueError, AttributeError):
            continue
        if a >= b:
            continue
        if entry.get("n_changed_from_initial") != 0:
            continue
        rev = pair_effects.get(f"{b}_{a}", {})
        if rev.get("n_changed_from_initial") != 0:
            continue
        if (a, b) in seen:
            continue
        seen.add((a, b))
        pairs.append([a, b])
    return sorted(pairs)


def _derive_toggle_actions(pair_effects: dict) -> list:
    """Return [a, ...] for actions where (a, a) returns to the base frame.
    These are self-inverse — typically commit/select toggles."""
    actions: list = []
    for key, entry in pair_effects.items():
        try:
            a, b = (int(x) for x in key.split("_"))
        except (ValueError, AttributeError):
            continue
        if a != b:
            continue
        if entry.get("n_changed_from_initial") == 0:
            actions.append(a)
    return sorted(set(actions))


def _derive_dominant_actions(action_effects: dict,
                             post_prelude_responses: dict,
                             available_actions: list,
                             top_k: int = 2,
                             min_cells: int = 4) -> list:
    """Pick the actions that actually move state on this game.

    Decision tree:
      1. If post_prelude_responses exists (game is static-init), prefer
         state-conditional data. Rank by (is_unfreezing_prelude,
         cumulative_response). This correctly identifies g50t [2, 4] as
         the movers (unfreezing preludes) and sc25 [3, 4] as dominant
         (highest post-prelude response).
      2. Otherwise use single-action initial effects (responsive games).

    Returns up to top_k action ids in descending order of priority.
    """
    if post_prelude_responses:
        unfreezing: set = set()
        cumulative: dict = {}
        for prelude_key, payload in post_prelude_responses.items():
            if not payload.get("unfreeze"):
                continue
            try:
                # Key form: "act<P>_x<N>"
                p_act = int(prelude_key.split("_", 1)[0].replace("act", ""))
                if p_act in available_actions:
                    unfreezing.add(p_act)
            except (ValueError, AttributeError):
                pass
            for act_id_str, n in (payload.get("responses") or {}).items():
                try:
                    act_id = int(act_id_str)
                except ValueError:
                    continue
                n = int(n)
                if n >= min_cells:
                    cumulative[act_id] = max(cumulative.get(act_id, 0), n)

        if unfreezing or cumulative:
            # Score: unfreezing preludes get a strong positive; ties broken by
            # cumulative response then ascending id. Non-unfreezing actions
            # only qualify if cumulative >= min_cells.
            scored = []
            for a in available_actions:
                is_unfreeze = 1 if a in unfreezing else 0
                cum = cumulative.get(a, 0)
                if is_unfreeze == 0 and cum < min_cells:
                    continue
                scored.append((a, is_unfreeze, cum))
            scored.sort(key=lambda t: (-t[1], -t[2], t[0]))
            return [a for a, _, _ in scored[:top_k]]

    # Responsive game (or static with no useful prelude data) — use initials.
    initial_scores: dict = {}
    for act_id_str, eff in (action_effects or {}).items():
        try:
            act_id = int(act_id_str)
        except ValueError:
            continue
        if act_id not in available_actions:
            continue
        initial_scores[act_id] = int(eff.get("cells_changed", 0) or 0)

    if not initial_scores or max(initial_scores.values()) < min_cells:
        return []
    scored = sorted(initial_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [a for a, n in scored[:top_k] if n >= min_cells]


def _derive_cursor_click_signal(heatmap: list) -> float:
    """0-1 confidence the click action is just cursor placement.
    Signature: heatmap is densely populated with a single, low cells_changed
    value (e.g. uniformly 1 or 2 cells per click). Means clicking anywhere
    just moves a cursor — there's no "this position triggers a fill" map.
    """
    if not heatmap or len(heatmap) < 8:
        return 0.0
    counts = [int(e.get("cells_changed", 0)) for e in heatmap]
    distinct = sorted(set(counts))
    if not distinct:
        return 0.0
    # Confidence high if (a) only one distinct count seen and (b) it's low
    # and (c) heatmap is densely populated (most grid positions live).
    density = len(heatmap) / 64.0    # default grid is ~64 positions
    if len(distinct) == 1 and distinct[0] <= 4 and density >= 0.5:
        return min(1.0, 0.5 + 0.5 * density)
    if len(distinct) <= 2 and max(distinct) <= 4 and density >= 0.5:
        return min(0.7, 0.3 + 0.4 * density)
    return 0.0


# ── Heuristics ──────────────────────────────────────────────────────────────
def _compute_signals(base_frame: np.ndarray, effects: dict, available_actions: list) -> dict:
    """Produce 0-1 confidence scores for each family of strategy.

    These are CHEAP heuristics; they don't have to be perfect. If wrong, the
    orchestrator will still run lower-confidence strategies with smaller budgets.
    """
    notes = []

    def get_eff(a): return effects.get(str(a), {})

    e1 = get_eff(1); e2 = get_eff(2); e3 = get_eff(3); e4 = get_eff(4); e5 = get_eff(5)

    n1 = e1.get("cells_changed", 0)
    n2 = e2.get("cells_changed", 0)
    n3 = e3.get("cells_changed", 0)
    n4 = e4.get("cells_changed", 0)
    n5 = e5.get("cells_changed", 0)

    # combo-lock signal: small symmetric changes for actions 1/2, bounded region
    looks_like_combo_lock = 0.0
    if 1 <= n1 <= 40 and 1 <= n2 <= 40:
        # symmetry bonus
        if abs(n1 - n2) <= 10:
            looks_like_combo_lock += 0.4
        # cursor-like 3/4 with similar row footprint
        if 5 <= n3 <= 60 and 5 <= n4 <= 60:
            looks_like_combo_lock += 0.3
        rows1 = set(e1.get("rows_changed", []))
        # slot display tends to be concentrated near bottom
        if rows1 and max(rows1) >= 50:
            looks_like_combo_lock += 0.2

    # grid-click signal: ACTION5 or ACTION6 produces noticeable change
    looks_like_grid_click = 0.0
    e6 = get_eff(6)
    n6 = e6.get("cells_changed", 0)
    e6_excp = e6.get("exception", "")
    if n5 > 0 and 5 in available_actions:
        looks_like_grid_click = min(1.0, 0.4 + n5 / 200.0)
    # If ACTION5 isn't usable but ACTION6 is (e.g. lp85 only has 6)
    elif n6 > 0 and 6 in available_actions and not e6_excp:
        looks_like_grid_click = min(1.0, 0.4 + n6 / 200.0)
    # If the game only offers click actions (5 or 6), bump regardless
    if (5 in available_actions or 6 in available_actions) and not any(
        a in available_actions for a in [1, 2, 3, 4]
    ):
        looks_like_grid_click = max(looks_like_grid_click, 0.6)

    # cursor-game signal: 3/4 produce moderate changes spread across cols
    looks_like_cursor_game = 0.0
    if 10 <= n3 <= 80 and 10 <= n4 <= 80:
        cols = set(e3.get("cols_changed", [])) | set(e4.get("cols_changed", []))
        if len(cols) >= 8:
            looks_like_cursor_game = 0.5

    # static: nothing does anything
    looks_like_static = 0.0
    if max(n1, n2, n3, n4, n5) < 3:
        looks_like_static = 0.9
        notes.append("Very few cells change on any action — static or click-only?")

    return dict(
        looks_like_combo_lock=round(looks_like_combo_lock, 3),
        looks_like_grid_click=round(looks_like_grid_click, 3),
        looks_like_cursor_game=round(looks_like_cursor_game, 3),
        looks_like_static=round(looks_like_static, 3),
        notes=notes,
    )


def _find_busy_regions(frame: np.ndarray) -> list:
    """Return coarse bounding boxes of 'busy' areas (high non-background variance).

    We split the frame into a 4x4 grid and record cells whose unique-value count
    is in the top half. Cheap to compute, useful to orient strategies.
    """
    h, w = frame.shape
    rows_step = max(1, h // 4)
    cols_step = max(1, w // 4)
    regions = []
    scores = []
    for i in range(4):
        for j in range(4):
            r0, r1 = i * rows_step, min(h, (i + 1) * rows_step)
            c0, c1 = j * cols_step, min(w, (j + 1) * cols_step)
            sub = frame[r0:r1, c0:c1]
            u = len(np.unique(sub))
            regions.append([r0, r1, c0, c1, int(u)])
            scores.append(u)
    if not scores:
        return []
    threshold = sorted(scores, reverse=True)[min(8, len(scores) - 1)]
    return [r for r in regions if r[4] >= threshold]
