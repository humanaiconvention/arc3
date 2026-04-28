"""Budget management.

Every characterize/strategy invocation gets a Budget. It's the single source
of truth for 'can I do another thing?'. Time, steps, and lives are tracked.

A Strategy MUST check budget.expired before each action. Strategies that
ignore the budget will be killed by the orchestrator at the hard wall time
(2x budget.time_seconds).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Budget:
    time_seconds: float = 300.0       # soft wall clock budget (seconds)
    max_steps: int = 10_000           # absolute step cap
    max_lives: int = 50               # absolute life cap (we respect free retries)
    label: str = ""                   # human label for logs

    _started_at: float = field(default_factory=time.time)
    _steps_at_start: int = 0
    _lives_at_start: int = 0

    def bind(self, steps_at_start: int = 0, lives_at_start: int = 0):
        """Snapshot the counters at the moment we start using this budget."""
        self._started_at = time.time()
        self._steps_at_start = steps_at_start
        self._lives_at_start = lives_at_start
        return self

    @property
    def elapsed(self) -> float:
        return time.time() - self._started_at

    @property
    def time_left(self) -> float:
        return self.time_seconds - self.elapsed

    def steps_used(self, current: int) -> int:
        return max(0, current - self._steps_at_start)

    def lives_used(self, current: int) -> int:
        return max(0, current - self._lives_at_start)

    def expired(self, steps: int = 0, lives: int = 0) -> bool:
        """Return True if any constraint has been hit."""
        if self.elapsed >= self.time_seconds:
            return True
        if self.steps_used(steps) >= self.max_steps:
            return True
        if self.lives_used(lives) >= self.max_lives:
            return True
        return False

    def why_expired(self, steps: int = 0, lives: int = 0) -> str:
        if self.elapsed >= self.time_seconds:
            return f"time ({self.elapsed:.0f}s/{self.time_seconds:.0f}s)"
        if self.steps_used(steps) >= self.max_steps:
            return f"steps ({self.steps_used(steps)}/{self.max_steps})"
        if self.lives_used(lives) >= self.max_lives:
            return f"lives ({self.lives_used(lives)}/{self.max_lives})"
        return "not expired"
