"""Shared utilities for the meta-solver.

Wraps the ARC-AGI-3 environment with a consistent interface every strategy
uses. All I/O, timing, life tracking, and error handling lives here so
strategies can focus on logic.

Key invariant: every step() goes through GameSession.step(), so we have a
single choke point for delay, exception handling, and state updates.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Path/env setup (runs on import, idempotent) ─────────────────────────────
_ROOT = pathlib.Path(__file__).resolve().parents[2]
_AGENTS_ROOT = _ROOT / "external" / "ARC-AGI-3-Agents"


def _candidate_site_packages(venv_root: pathlib.Path) -> list[pathlib.Path]:
    """Return Windows + POSIX venv site-packages paths if they exist."""
    candidates = [venv_root / "Lib" / "site-packages"]
    lib_root = venv_root / "lib"
    if lib_root.exists():
        candidates.extend(sorted(lib_root.glob("python*/site-packages")))
    return [p for p in candidates if p.exists()]


for _p in [
    _ROOT / "neurosym",
    _AGENTS_ROOT,
    *_candidate_site_packages(_AGENTS_ROOT / ".venv"),
]:
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

_ENV_FILE = _AGENTS_ROOT / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# Deferred imports (need path setup first)
from arc_agi import Arcade  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

# ── Public constants ────────────────────────────────────────────────────────
ROOT = _ROOT
# Minimum time between API calls PER SESSION. With N parallel workers, total
# RPS is roughly N / STEP_DELAY. Keep this conservative — the server 500s if
# we push too hard.
STEP_DELAY = float(os.environ.get("META_STEP_DELAY", "0.15"))
ALL_ACTIONS = [1, 2, 3, 4, 5, 6, 7]  # ACTION5/6/7 often no-ops, but probe them
ACTION_RESET = "RESET"
STEP_RETRY_ATTEMPTS = 4
RESET_RETRY_ATTEMPTS = 5


# ── Result types ────────────────────────────────────────────────────────────
@dataclass
class StepResult:
    """Return value of GameSession.step(). Strategies should read .ok, .won_level, .frame."""
    ok: bool                # False = action failed (terminal state or exception)
    state: object = None    # Raw ARC state (may be None on exception)
    frame: np.ndarray = None  # 2D int32, or None if no frame
    levels_completed: int = 0
    game_state: object = None  # GameState enum value
    exception: str = ""     # stringified exception if step failed

    @property
    def terminal(self) -> bool:
        return self.game_state in (GameState.GAME_OVER, GameState.WIN)

    @property
    def won_any(self) -> bool:
        """True if levels_completed > 0 since game start."""
        return self.levels_completed > 0


# ── Game session ────────────────────────────────────────────────────────────
@dataclass
class GameSession:
    """One active connection to an ARC-AGI-3 game. Tracks basic state.

    Usage:
        with GameSession.open("tr87") as sess:
            r = sess.reset()
            r = sess.step(1)
            ...
    """
    game_id: str
    env: object = field(repr=False, default=None)
    _state: object = field(repr=False, default=None)
    _frame: np.ndarray = field(repr=False, default=None)
    lives_used: int = 0
    steps_taken: int = 0
    max_levels_completed: int = 0
    started_at: float = field(default_factory=time.time)
    _last_step_at: float = 0.0

    @classmethod
    def open(cls, game_prefix: str) -> "GameSession":
        """Open a session by game prefix (e.g. 'tr87'). Returns the first full id used."""
        arcade = Arcade()
        game_full = next(
            g for g in (e.game_id for e in arcade.available_environments)
            if g.startswith(game_prefix)
        )
        env = arcade.make(game_full)
        return cls(game_id=game_full, env=env)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    # ── Frame helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _gf(state) -> Optional[np.ndarray]:
        if state is None or getattr(state, "frame", None) is None:
            return None
        f = np.array(state.frame, dtype=np.int32)
        return f[0] if f.ndim == 3 else f

    @property
    def frame(self) -> Optional[np.ndarray]:
        return self._frame

    @property
    def state(self):
        return self._state

    @property
    def levels_completed(self) -> int:
        return int(getattr(self._state, "levels_completed", 0) or 0)

    @property
    def available_actions(self) -> list:
        return list(getattr(self._state, "available_actions", []) or [])

    @property
    def win_levels(self) -> int:
        return int(getattr(self._state, "win_levels", 0) or 0)

    @property
    def is_terminal(self) -> bool:
        if self._state is None: return True
        gs = getattr(self._state, "state", None)
        return gs in (GameState.GAME_OVER, GameState.WIN)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    # ── Step / reset ─────────────────────────────────────────────────────────
    def _pace(self):
        dt = time.time() - self._last_step_at
        if dt < STEP_DELAY:
            time.sleep(STEP_DELAY - dt)
        self._last_step_at = time.time()

    @staticmethod
    def _is_retryable_exception(exc: Exception, *, allow_server_errors: bool = False) -> bool:
        msg = repr(exc)
        markers = [
            "429 Client Error",
            "Too Many Requests",
            "ReadTimeout",
            "ConnectTimeout",
            "ConnectionError",
        ]
        if allow_server_errors:
            markers.extend([
                "500 Server Error",
                "502 Server Error",
                "503 Server Error",
                "504 Server Error",
            ])
        return any(marker in msg for marker in markers)

    @staticmethod
    def _retry_backoff(attempt: int) -> float:
        return min(4.0, 0.75 * (2 ** attempt))

    def reset(self) -> StepResult:
        """Fresh reset (scorecard start). Retries on transient/rate-limit failures."""
        for attempt in range(RESET_RETRY_ATTEMPTS):
            self._pace()
            try:
                s = self.env.reset()
            except Exception as e:
                if (
                    attempt == RESET_RETRY_ATTEMPTS - 1
                    or not self._is_retryable_exception(e, allow_server_errors=True)
                ):
                    return StepResult(ok=False, exception=repr(e))
                time.sleep(self._retry_backoff(attempt))
                continue
            if s is not None:
                return self._adopt(s, reset=True)
            if attempt < RESET_RETRY_ATTEMPTS - 1:
                time.sleep(self._retry_backoff(attempt))
        return StepResult(ok=False, exception=f"reset_returned_none_{RESET_RETRY_ATTEMPTS}x")

    def soft_reset(self) -> StepResult:
        """Send GameAction.RESET. Used after GAME_OVER to continue (costs a life)."""
        self._pace()
        try:
            s = self.env.step(GameAction.RESET)
        except Exception as e:
            return StepResult(ok=False, exception=repr(e))
        self.lives_used += 1
        return self._adopt(s, reset=False)

    def step(self, act_id) -> StepResult:
        """Send a numeric action (1-7). Returns StepResult with frame + metadata.
        Pass GameAction.RESET via soft_reset(), not here."""
        for attempt in range(STEP_RETRY_ATTEMPTS):
            self._pace()
            try:
                s = self.env.step(GameAction.from_id(act_id))
            except Exception as e:
                if (
                    attempt == STEP_RETRY_ATTEMPTS - 1
                    or not self._is_retryable_exception(e, allow_server_errors=False)
                ):
                    return StepResult(ok=False, exception=repr(e))
                time.sleep(self._retry_backoff(attempt))
                continue
            self.steps_taken += 1
            return self._adopt(s, reset=False)
        return StepResult(ok=False, exception="step_retry_exhausted")

    def step_with_data(self, act_id, data: dict) -> StepResult:
        """Send a numeric action with extra data (e.g., click coordinates for ACTION5/6).
        data must include anything the server expects (e.g., {'x': 25, 'y': 7}).

        IMPORTANT: remote_wrapper.step() takes 'data' as a separate keyword argument.
        We must pass it there — setting it on the action object alone is not enough.
        """
        for attempt in range(STEP_RETRY_ATTEMPTS):
            self._pace()
            try:
                act = GameAction.from_id(act_id)
                # Pass data kwarg directly to env.step — this is what remote_wrapper reads.
                s = self.env.step(act, data=data)
            except Exception as e:
                if (
                    attempt == STEP_RETRY_ATTEMPTS - 1
                    or not self._is_retryable_exception(e, allow_server_errors=False)
                ):
                    return StepResult(ok=False, exception=repr(e))
                time.sleep(self._retry_backoff(attempt))
                continue
            self.steps_taken += 1
            return self._adopt(s, reset=False)
        return StepResult(ok=False, exception="step_with_data_retry_exhausted")

    def _adopt(self, s, reset: bool) -> StepResult:
        """Internal: update session state from API response and build StepResult."""
        if s is None:
            return StepResult(ok=False)
        self._state = s
        f = self._gf(s)
        if f is not None:
            self._frame = f
        lv = int(getattr(s, "levels_completed", 0) or 0)
        if lv > self.max_levels_completed:
            self.max_levels_completed = lv
        gs = getattr(s, "state", None)
        terminal = gs in (GameState.GAME_OVER, GameState.WIN)
        return StepResult(
            ok=not terminal,
            state=s,
            frame=f,
            levels_completed=lv,
            game_state=gs,
        )


# ── Utility: read life row ──────────────────────────────────────────────────
def life_cells(frame: Optional[np.ndarray], row: int = 63) -> int:
    """Count non-zero cells on the given row (default: row 63 = life timer).
    Returns -1 if frame is unavailable or row out of bounds."""
    if frame is None or row >= frame.shape[0]:
        return -1
    return int(np.sum(frame[row] > 0))
