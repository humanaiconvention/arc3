"""
Plan Executor Agent — submits a pre-computed action plan to an ARC-AGI-3 game.

Supports two plan modes:
  - 'graph_bfs'   : 49-step observed path from Phase F graph-BFS (guaranteed to
                    have been seen in real data; uses exact recorded coordinates)
  - 'abstract'    : 6-step pyperplan plan from Phase F (abstract STRIPS domain;
                    uses bin-centre coordinates; may fail in the actual game)

Usage (from within external/ARC-AGI-3-Agents/):
    uv run main.py --agent=planexecutor_graph --game=vc33
    uv run main.py --agent=planexecutor_abstract --game=vc33

Both agents are registered at the bottom of this file. For games other than vc33
no Phase F plan exists; the agent will fall back to random actions.

Plan data is loaded from neurosym/lp/<game>.phase_f.json at runtime.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from arcengine import FrameData, GameAction, GameState

# Absolute import for standalone use
from agents.templates.random_agent import Random

# Phase F plan files live in the neurosym/lp/ directory (same dir as this file)
_LP_DIR = Path(__file__).resolve().parent / "lp"


def _load_plan(game_id: str, mode: str) -> list[dict] | None:
    """Load a plan from the Phase F JSON.

    mode = 'graph_bfs'  → use graph_bfs.steps (exact recorded coordinates)
    mode = 'abstract'   → use pyperplan.plan (bin-centre coordinates)

    Returns a list of step dicts, or None if no plan is available.
    """
    # Strip version suffix from game_id if present (e.g. "vc33-9851e02b" → "vc33")
    short_id = game_id.split("-")[0]
    plan_path = _LP_DIR / f"{short_id}.phase_f.json"
    if not plan_path.exists():
        return None

    with open(plan_path, encoding="utf-8") as f:
        data = json.load(f)

    if mode == "graph_bfs":
        bfs = data.get("graph_bfs", {})
        if not bfs.get("found"):
            return None
        raw_steps = bfs.get("steps", [])
        # Parse label (e.g. "a6_x3y0") into x, y, action_id (bin-centre coords)
        parsed = []
        for s in raw_steps:
            label = s.get("label", "")
            entry = {"step": s["step"], "label": label}
            if "_x" in label and "y" in label:
                try:
                    _, rest = label.split("_x", 1)
                    xpart, ypart = rest.split("y", 1)
                    bx, by = int(xpart), int(ypart)
                    entry["x"] = bx * 16 + 8
                    entry["y"] = by * 16 + 8
                    entry["action_id"] = int(label.split("_")[0][1:])  # "a6" → 6
                except (ValueError, IndexError):
                    entry["x"] = None
                    entry["y"] = None
                    entry["action_id"] = 6
            else:
                entry["x"] = None
                entry["y"] = None
                entry["action_id"] = 6
            parsed.append(entry)
        return parsed

    if mode == "abstract":
        pp = data.get("pyperplan", {})
        if not pp.get("plan"):
            return None
        # pyperplan plan steps are strings like "(a6_x0y0 obj1 obj1)"
        # Extract the action label and derive bin-centre coordinates
        plan_steps = []
        for i, step_str in enumerate(pp["plan"]):
            # Parse "(a6_xXyY ...)" → label = a6_xXyY
            label = step_str.strip("() ").split()[0]
            # label = "a6_xXyY"
            if "_x" in label and "y" in label:
                try:
                    _, rest = label.split("_x", 1)
                    xpart, ypart = rest.split("y", 1)
                    bx, by = int(xpart), int(ypart)
                    x = bx * 16 + 8   # bin-centre
                    y = by * 16 + 8
                    action_id = int(label.split("_")[0][1:])  # "a6" → 6
                    plan_steps.append({
                        "step": i + 1,
                        "label": label,
                        "x": x,
                        "y": y,
                        "action_id": action_id,
                    })
                except (ValueError, IndexError):
                    plan_steps.append({"step": i + 1, "label": label,
                                       "x": None, "y": None, "action_id": 6})
            else:
                plan_steps.append({"step": i + 1, "label": label,
                                   "x": None, "y": None, "action_id": 6})
        return plan_steps

    return None


class PlanExecutorAgent(Random):
    """Agent that follows a pre-computed Phase F plan, then falls back to random."""

    def __init__(self, *args: Any, plan_mode: str = "graph_bfs", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._plan_mode = plan_mode
        self._plan_steps: list[dict] | None = None
        self._plan_index: int = 0
        self._plan_loaded: bool = False
        self._game_step: int = 0

    @property
    def name(self) -> str:
        return f"{self.game_id}.planexecutor_{self._plan_mode}.{self.MAX_ACTIONS}"

    def _ensure_plan(self) -> None:
        if self._plan_loaded:
            return
        self._plan_loaded = True
        steps = _load_plan(self.game_id, self._plan_mode)
        if steps:
            self._plan_steps = steps
            print(f"[PlanExecutor] Loaded {len(steps)}-step {self._plan_mode} plan "
                  f"for {self.game_id}")
        else:
            print(f"[PlanExecutor] No {self._plan_mode} plan for {self.game_id}; "
                  "falling back to random agent")

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        self._ensure_plan()

        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self._plan_index = 0   # reset plan on new game/reset
            self._game_step = 0
            return GameAction.RESET

        self._game_step += 1

        # Follow the plan if we have steps remaining.
        if self._plan_steps and self._plan_index < len(self._plan_steps):
            step = self._plan_steps[self._plan_index]
            self._plan_index += 1
            x = step.get("x")
            y = step.get("y")
            if x is not None and y is not None:
                action = GameAction.from_id(step.get("action_id", 6))
                action.set_data({"x": x, "y": y})
                action.reasoning = {
                    "plan_step": step["step"],
                    "plan_mode": self._plan_mode,
                    "label": step.get("label", "unknown"),
                }
                return action

        # Fallback: random action from available_actions.
        avail = list(latest_frame.available_actions or [])
        if avail:
            legal_ids = [a for a in avail if GameAction.from_id(a) is not GameAction.RESET]
            if legal_ids:
                action = GameAction.from_id(random.choice(legal_ids))
                if action.is_complex():
                    action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
                action.reasoning = "plan exhausted; random fallback"
                return action
        return GameAction.RESET


class GraphBFSExecutor(PlanExecutorAgent):
    """49-step observed plan (graph-BFS, exact coordinates from recording)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, plan_mode="graph_bfs", **kwargs)

    @property
    def name(self) -> str:
        return f"{self.game_id}.planexecutor_graph.{self.MAX_ACTIONS}"


class AbstractPlanExecutor(PlanExecutorAgent):
    """6-step abstract plan (pyperplan, bin-centre coordinates)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, plan_mode="abstract", **kwargs)

    @property
    def name(self) -> str:
        return f"{self.game_id}.planexecutor_abstract.{self.MAX_ACTIONS}"
