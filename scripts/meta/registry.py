"""Persistent registry of attempts and results.

Layout on disk:
  results/meta/registry.json              # top-level index
  results/meta/games/<prefix>/
      profile.json                        # latest GameProfile
      initial_frame.npy                   # initial frame from characterize
      attempts/<ts>_<strategy>.json       # one per StrategyResult
      attempts/<ts>_<strategy>.log        # free-form strategy log (optional)
      winners.json                        # compact summary: {L1: {strategy, combo, ts}, ...}

The registry itself (top-level) only stores the latest outcome per game:
  {
    "tr87-cd924810": {
        "prefix": "tr87",
        "last_profile_at": "2026-04-19T08:12:33",
        "max_levels_completed": 1,
        "winners": {"1": "combo_lock"},
        "attempts_tried": ["action_spam", "combo_lock"],
        "last_attempt_at": "2026-04-19T09:31:00"
    },
    ...
  }

Resume semantics: run_meta reads the registry on startup. Games whose
registry entry already records max_levels_completed >= win_levels are
skipped by default (use --replay to force).
"""
from __future__ import annotations

import datetime
import json
import pathlib
import shutil
import threading
from dataclasses import asdict
from typing import Optional

from meta.common import ROOT
from meta.profile import GameProfile
from meta.result import StrategyResult

REGISTRY_DIR = ROOT / "results" / "meta"
REGISTRY_FILE = REGISTRY_DIR / "registry.json"

# Serialize registry read-modify-write cycles across threads in the parallel
# runner. Single-threaded consumers pay a negligible cost.
_REGISTRY_LOCK = threading.RLock()


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _game_dir(game_id: str) -> pathlib.Path:
    # Use short prefix (everything before first '-') as folder name to
    # keep things tidy and align with how we already refer to games.
    prefix = game_id.split("-", 1)[0]
    return REGISTRY_DIR / "games" / prefix


def ensure_dirs(game_id: str) -> pathlib.Path:
    gd = _game_dir(game_id)
    (gd / "attempts").mkdir(parents=True, exist_ok=True)
    return gd


# ── Registry top-level ──────────────────────────────────────────────────────
def load_registry() -> dict:
    with _REGISTRY_LOCK:
        if REGISTRY_FILE.exists():
            try:
                return json.loads(REGISTRY_FILE.read_text())
            except Exception:
                # Backup and start fresh
                shutil.copy(REGISTRY_FILE, REGISTRY_FILE.with_suffix(".corrupt.json"))
                return {}
        return {}


def save_registry(reg: dict) -> None:
    with _REGISTRY_LOCK:
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        tmp = REGISTRY_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(reg, indent=2, default=str))
        tmp.replace(REGISTRY_FILE)


# ── Per-game record accessors ───────────────────────────────────────────────
def record_profile(prof: GameProfile, initial_frame=None) -> None:
    gd = ensure_dirs(prof.game_id)
    (gd / "profile.json").write_text(prof.to_json())
    if initial_frame is not None:
        import numpy as np
        np.save(str(gd / "initial_frame.npy"), initial_frame)

    with _REGISTRY_LOCK:
        reg = load_registry()
        entry = reg.setdefault(prof.game_id, {})
        entry["prefix"] = prof.game_prefix
        entry["last_profile_at"] = _now_iso()
        entry.setdefault("max_levels_completed", 0)
        entry.setdefault("winners", {})
        entry.setdefault("attempts_tried", [])
        save_registry(reg)


def record_attempt(res: StrategyResult, log_text: str = "") -> None:
    gd = ensure_dirs(res.game_id)
    stamp = _now_stamp()
    name = f"{stamp}_{res.strategy}"
    (gd / "attempts" / f"{name}.json").write_text(res.to_json())
    if log_text:
        (gd / "attempts" / f"{name}.log").write_text(log_text)

    def _replayable(details: dict) -> bool:
        """Does this result carry info a future replay could use?"""
        if not details: return False
        return any(k in details for k in ("win_combo", "win_sequence",
                                          "win_cycle_presses", "win_click"))

    # Atomic read-modify-write for the cross-game registry index.
    # save_registry also takes the lock; RLock makes the nesting safe.
    with _REGISTRY_LOCK:
        reg = load_registry()
        entry = reg.setdefault(res.game_id, {})
        entry["prefix"] = res.game_id.split("-", 1)[0]
        entry["last_attempt_at"] = _now_iso()
        prior_max = entry.get("max_levels_completed", 0)
        entry["max_levels_completed"] = max(prior_max, res.max_levels_completed)

        tried = entry.setdefault("attempts_tried", [])
        if res.strategy not in tried:
            tried.append(res.strategy)

        winners = entry.setdefault("winners", {})
        for lv in res.new_levels:
            key = str(lv)
            existing = winners.get(key)
            new_entry = {
                "strategy": res.strategy,
                "ts": _now_iso(),
                "elapsed": round(res.elapsed, 1),
                "details": res.details,
            }
            # Keep existing unless the new result is strictly better:
            #   - existing has no replayable details but new does, OR
            #   - both replayable and new is faster
            if existing is None:
                winners[key] = new_entry
            else:
                e_replay = _replayable(existing.get("details", {}))
                n_replay = _replayable(res.details)
                if n_replay and not e_replay:
                    winners[key] = new_entry  # upgrade: replayable > non
                elif n_replay and e_replay and res.elapsed < existing.get("elapsed", 1e9):
                    winners[key] = new_entry  # upgrade: faster replay
        save_registry(reg)


def load_profile(game_id: str) -> Optional[GameProfile]:
    gd = _game_dir(game_id)
    f = gd / "profile.json"
    if not f.exists():
        return None
    return GameProfile.from_json(f.read_text())


def already_solved(game_id: str, win_levels: int) -> bool:
    """Return True if registry says this game has been fully beaten."""
    reg = load_registry()
    entry = reg.get(game_id, {})
    if not entry:
        return False
    if win_levels <= 0:
        return False
    return entry.get("max_levels_completed", 0) >= win_levels
