"""Helpers for turning recorded winners into deterministic replay plans."""
from __future__ import annotations

from typing import Optional

from meta.registry import load_registry


def _normalize_step(step):
    if isinstance(step, dict):
        act_id = step.get("act", step.get("action"))
        step_type = str(step.get("type", "")).lower()
        if act_id is None and step_type == "click" and "x" in step and "y" in step:
            # Manual probe scripts consistently use ACTION6 for coordinate clicks.
            # Older seeded sequences recorded {"type":"click","x":...,"y":...}
            # without the action id, so default them to replayable ACTION6.
            act_id = 6
        if act_id is None:
            raise ValueError(f"Cannot normalize replay step without action id: {step!r}")
        act_id = int(act_id)
        data = step.get("data")
        if data is None and "x" in step and "y" in step:
            data = {"x": int(step["x"]), "y": int(step["y"])}
        out = {"act": act_id}
        if data:
            out["data"] = data
        return out
    return int(step)


def _combo_steps(details: dict) -> list:
    combo = [int(v) for v in details["win_combo"]]
    n_slots = int(details.get("n_slots") or len(combo))
    cycle = int(details.get("cycle") or 7)
    fwd_cursor = int(details.get("fwd_cursor_act") or 4)
    bwd_cursor = int(details.get("bwd_cursor_act") or 3)
    fwd_value = int(details.get("fwd_value_act") or 1)
    bwd_value = int(details.get("bwd_value_act") or 2)

    cursor = 0
    slot_values = [0] * n_slots
    steps: list = []

    for slot_i in range(n_slots):
        if cursor != slot_i:
            fwd = (slot_i - cursor) % n_slots
            bwd = (cursor - slot_i) % n_slots
            act = fwd_cursor if fwd <= bwd else bwd_cursor
            steps.extend([act] * min(fwd, bwd))
            cursor = slot_i

        current = slot_values[slot_i]
        target = combo[slot_i]
        if current != target:
            fwd = (target - current) % cycle
            bwd = (current - target) % cycle
            act = fwd_value if fwd <= bwd else bwd_value
            steps.extend([act] * min(fwd, bwd))
            slot_values[slot_i] = target

    return steps


def build_replay_plan(level: int, winner: dict) -> Optional[dict]:
    details = winner.get("details", {}) or {}
    strategy = winner.get("strategy", "?")

    if details.get("win_combo"):
        steps = _combo_steps(details)
        return {
            "kind": "combo",
            "target_level": int(level),
            "strategy": strategy,
            "steps": steps,
            "summary": f"combo replay ({len(steps)} actions)",
        }

    if details.get("win_click"):
        click = details["win_click"]
        if not isinstance(click, list) or len(click) != 2:
            return None
        cycle_presses = int(details.get("win_cycle_presses", 0))
        click_act = int(details.get("click_act", 5))
        cycle_act = int(details.get("cycle_act", 4))
        burst = int(details.get("win_burst_clicks", 1))
        steps = [cycle_act] * cycle_presses
        click_step = {
            "act": click_act,
            "data": {"x": int(click[0]), "y": int(click[1])},
        }
        steps.extend([click_step] * burst)
        return {
            "kind": "nav_click",
            "target_level": int(level),
            "strategy": strategy,
            "steps": steps,
            "summary": (
                f"nav/click replay ({cycle_presses} cycle, "
                f"{burst} click at {click[0]},{click[1]})"
            ),
        }

    if details.get("win_sequence"):
        steps = [_normalize_step(step) for step in details["win_sequence"]]
        return {
            "kind": "sequence",
            "target_level": int(level),
            "strategy": strategy,
            "steps": steps,
            "summary": f"sequence replay ({len(steps)} actions)",
        }

    return None


def iter_replay_plans(game_id: str) -> list[dict]:
    reg = load_registry()
    entry = reg.get(game_id, {})
    if not entry and "-" in game_id:
        prefix = game_id.split("-", 1)[0]
        entry = reg.get(prefix, {})
    if not entry:
        prefix = game_id.split("-", 1)[0]
        entry = next((v for k, v in reg.items() if k.startswith(prefix)), {})
    winners = entry.get("winners", {})
    plans: list[dict] = []
    for level_text, winner in sorted(winners.items(), key=lambda kv: int(kv[0])):
        plan = build_replay_plan(int(level_text), winner)
        if plan is not None:
            plans.append(plan)
    return plans


def lookup_replay_plan(game_id: str) -> Optional[dict]:
    plans = iter_replay_plans(game_id)
    return plans[0] if plans else None
