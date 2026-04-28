"""StrategyResult — output of every strategy attempt.

Results are the atoms the registry stores. One per (game, strategy, run).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StrategyResult:
    game_id: str
    strategy: str                         # strategy name (e.g. "combo_lock")
    started_at: float = field(default_factory=time.time)
    elapsed: float = 0.0

    # Core outcome
    max_levels_completed: int = 0         # highest level reached during this attempt
    new_levels: list = field(default_factory=list)  # levels completed for the first time
    won_any_level: bool = False           # completed >= 1 level
    stopped_reason: str = "not_run"       # why the strategy ended

    # Resource usage
    steps: int = 0
    lives_used: int = 0

    # Strategy-specific details (free-form JSON-safe dict)
    details: dict = field(default_factory=dict)

    # Error info
    exception: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)
