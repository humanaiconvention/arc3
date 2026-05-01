"""Shared feature extraction for the ARC3 probe-signature policy heuristic.

Used by:
  - D:/arc3/scripts/training/arc3/train_arc3_policy_heuristic.py (training)
  - D:/arc3/scripts/meta/strategies/policy_guided_walk.py (inference)

Both call build_features(ctx, profile) with the same shape:
  ctx     — per-step context dict with step_index, trace_len, prior_actions,
            and probe_signature fields (or empty/synthesized at inference).
  profile — full GameProfile dict (loaded from disk or live characterize).

Returns (features, names) where features is a list of floats and names is a
list of strings of equal length. Names are the canonical feature schema.
"""
from __future__ import annotations

from collections import Counter


PRIOR_ACTION_LOOKBACK = 3
ACTION_RANGE = (1, 2, 3, 4, 5, 6, 7)
PROBE_ACTION_RANGE = (1, 2, 3, 4, 5)


def build_features(ctx: dict, profile: dict) -> tuple[list[float], list[str]]:
    feats: list[float] = []
    names: list[str] = []

    def add(name: str, value) -> None:
        feats.append(float(value))
        names.append(name)

    # --- Trace position
    step_index = int(ctx.get("step_index", 0) or 0)
    trace_len = max(1, int(ctx.get("trace_len", 1) or 1))
    add("step_index_norm", step_index / trace_len)
    add("trace_len", trace_len)

    # --- Profile-level scalars
    add("win_levels", int(profile.get("win_levels", 0) or 0))
    add("dynamic_rows", int(profile.get("n_dynamic_rows", 0) or 0))
    add("looks_like_cursor_click", float(profile.get("looks_like_cursor_click", 0.0) or 0.0))
    add("looks_like_combo_lock", float(profile.get("looks_like_combo_lock", 0.0) or 0.0))
    add("looks_like_grid_click", float(profile.get("looks_like_grid_click", 0.0) or 0.0))
    add("looks_like_static", float(profile.get("looks_like_static", 0.0) or 0.0))

    # --- Probe data presence (sometimes from ctx.probe_signature, sometimes
    # synthesized at inference time from the profile directly)
    sig = ctx.get("probe_signature") or {}
    has_probe = sig.get("has_probe_data")
    if has_probe is None:
        has_probe = bool(profile.get("action_pair_effects") or profile.get("click_heatmap"))
    has_triple = sig.get("has_triple_data")
    if has_triple is None:
        has_triple = bool(profile.get("action_triple_effects"))
    has_prelude = sig.get("has_post_prelude_data")
    if has_prelude is None:
        has_prelude = bool(profile.get("post_prelude_responses"))
    add("has_probe_data", 1 if has_probe else 0)
    add("has_triple_data", 1 if has_triple else 0)
    add("has_post_prelude_data", 1 if has_prelude else 0)

    # --- Inverse-pair signatures
    inverse_pairs = profile.get("inverse_pairs") or []
    pairs_set = {tuple(sorted(p)) for p in inverse_pairs}
    add("ip_wasd", 1 if {(1, 2), (3, 4)}.issubset(pairs_set) else 0)
    add("ip_triangular_size_3", 1 if len(inverse_pairs) == 3 else 0)
    add("ip_count", len(inverse_pairs))

    # --- Toggle / dominant per action
    toggles = set(int(a) for a in (profile.get("toggle_actions") or []))
    dominants = set(int(a) for a in (profile.get("dominant_actions") or []))
    available = set(int(a) for a in (profile.get("available_actions") or []))
    for a in PROBE_ACTION_RANGE:
        add(f"toggle_act{a}", 1 if a in toggles else 0)
        add(f"dominant_act{a}", 1 if a in dominants else 0)

    # --- Per-action availability + initial effect size
    action_effects = profile.get("action_effects") or {}
    for a in ACTION_RANGE:
        add(f"avail_act{a}", 1 if a in available else 0)
        eff = action_effects.get(str(a)) or {}
        add(f"effect_act{a}_cells", int(eff.get("cells_changed", 0) or 0))

    # --- Prior actions
    prior = ctx.get("prior_actions") or []
    last_n = prior[-PRIOR_ACTION_LOOKBACK:]
    counts = Counter()
    for step in last_n:
        act = step.get("action", "") if isinstance(step, dict) else ""
        if act.startswith("ACTION"):
            try:
                counts[int(act.replace("ACTION", ""))] += 1
            except ValueError:
                pass
    for a in ACTION_RANGE:
        add(f"prior{PRIOR_ACTION_LOOKBACK}_count_act{a}", counts.get(a, 0))

    last_action_id = 0
    if prior:
        last = prior[-1]
        if isinstance(last, dict):
            try:
                last_action_id = int((last.get("action") or "").replace("ACTION", ""))
            except ValueError:
                last_action_id = 0
    for a in (0,) + ACTION_RANGE:
        add(f"last_action_{a}", 1 if last_action_id == a else 0)

    return feats, names
