"""
Standalone copy of RandomRecordActions for use outside the ARC-AGI-3-Agents
package (e.g. direct import in neurosym scripts). For production use inside
the agents package see:
    external/ARC-AGI-3-Agents/agents/templates/record_actions_agent.py

See that file for full documentation and design rationale.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from typing import Any

from arcengine import FrameData, GameAction, GameState

# Absolute import path for standalone use.
from agents.templates.random_agent import Random


class RandomRecordActions(Random):
    """Random agent that samples only legal actions and interleaves
    action_request entries into the recording JSONL."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_available: list[int] = []

    @property
    def name(self) -> str:
        return f"{self.game_id}.randomrecordactions.{self.MAX_ACTIONS}"

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        avail: list[int] = list(latest_frame.available_actions or [])
        self._last_available = avail

        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            return GameAction.RESET

        if avail:
            # Use from_id() — available_actions integers don't map directly to
            # GameAction enum values via the constructor.
            legal_ids = [
                aid for aid in avail
                if GameAction.from_id(aid) is not GameAction.RESET
            ]
            if not legal_ids:
                return GameAction.RESET
            action = GameAction.from_id(random.choice(legal_ids))
        else:
            action = random.choice(
                [a for a in GameAction if a is not GameAction.RESET]
            )

        if action.is_simple():
            action.reasoning = f"RNG told me to pick {action.value}"
        elif action.is_complex():
            action.set_data(
                {"x": random.randint(0, 63), "y": random.randint(0, 63)}
            )
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": "RNG said so!",
            }
        return action

    def _record_action(self, action: GameAction) -> None:
        if not hasattr(self, "recorder") or self.is_playback:
            return
        data = action.action_data.model_dump() if hasattr(action, "action_data") else {}
        was_legal: bool | None = (
            int(action.value) in self._last_available
            if self._last_available
            else None
        )
        event = {
            "event": "action_request",
            "action_id": int(action.value),
            "action_name": action.name if hasattr(action, "name") else str(action),
            "action_data": data,
            "is_complex": action.is_complex() if hasattr(action, "is_complex") else False,
            "available_actions": list(self._last_available),
            "legal": was_legal,
        }
        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": event,
        }
        with open(self.recorder.filename, "a", encoding="utf-8") as f:
            json.dump(out, f)
            f.write("\n")

    def take_action(self, action: GameAction) -> FrameData | None:
        self._record_action(action)
        return super().take_action(action)
