"""ARC3Env — minimal gym-like wrapper around meta.common.GameSession.

Designed for RL self-play. Key choices:

  - reset() opens a fresh session and returns the initial frame.
  - step(action_id, click_xy) returns (next_frame, reward, done, info).
  - Reward shaping (sparse → semi-dense):
      * +1.0 per level completed (the only "real" reward)
      * +0.001 × cells_changed_from_initial (gentle exploration bonus)
      * -0.5 on terminal (game over)
      * 0.0 otherwise
  - done is True if game terminal OR a level was just completed (so trainers
    can choose whether to keep going past a level).

This wrapper is import-only here — the actual API calls happen via
GameSession. It assumes the ARC3 venv (with arc_agi installed) is active.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


REWARD_PER_LEVEL = 1.0
REWARD_PER_CHANGED_CELL = 0.001
REWARD_PER_TERMINAL = -0.5
REWARD_PER_NOVEL_HASH = 0.05      # added per never-before-seen state
REWARD_SIMILARITY_WEIGHT = 0.5    # × score_state(frame) per step
NOISE_ROWS = (59, 60, 63)
MAX_NOVEL_PER_EPISODE = 50         # cap novelty bonus to prevent runaway


def _nchanged_masked(a: np.ndarray, b: np.ndarray) -> int:
    diff = a != b
    for nr in NOISE_ROWS:
        if nr < diff.shape[0]:
            diff[nr, :] = False
    return int(np.sum(diff))


def _frame_hash(frame: np.ndarray) -> str:
    """Stable 16-char hash of a frame, masking noise rows."""
    import hashlib
    f = frame.copy()
    for nr in NOISE_ROWS:
        if nr < f.shape[0]:
            f[nr, :] = 0
    return hashlib.sha1(f.tobytes()).hexdigest()[:16]


def _frame_sha256(frame: Optional[np.ndarray]) -> Optional[str]:
    if frame is None:
        return None
    import hashlib
    arr = np.ascontiguousarray(frame, dtype=np.int32)
    h = hashlib.sha256()
    h.update(str(tuple(int(dim) for dim in arr.shape)).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _life_cells(frame: Optional[np.ndarray], row: int = 63) -> int:
    if frame is None or row >= frame.shape[0]:
        return -1
    return int(np.sum(frame[row] > 0))


@dataclass
class StepObservation:
    frame: Optional[np.ndarray]
    reward: float
    done: bool
    levels_completed: int
    terminal: bool
    info: dict


class ARC3Env:
    """Tiny env wrapper. Owns a GameSession instance; close on context exit.

    Reward shaping (in addition to the sparse level-win signal):
      - cells_changed_from_initial × REWARD_PER_CHANGED_CELL (exploration)
      - score_state(frame) × REWARD_SIMILARITY_WEIGHT (winning-typology proxy)
        — only added when use_similarity=True
      - novelty bonus per unseen state hash, capped at MAX_NOVEL_PER_EPISODE
        — only added when use_novelty=True

    Both extra signals default ON to make this the v3 "shaped reward + novelty"
    setup. Pass False to either to revert to v2 behavior.
    """

    def __init__(self, game_prefix: str, use_similarity: bool = True,
                 use_novelty: bool = True, persistent_novelty: bool = False):
        # Lazy import so the module stays importable from non-ARC contexts.
        from meta.common import GameSession  # noqa: E402
        self._GameSession = GameSession
        self.game_prefix = game_prefix
        self.sess = None
        self._initial_frame: Optional[np.ndarray] = None
        self._prior_levels = 0
        self.use_similarity = use_similarity
        self.use_novelty = use_novelty
        # When True, the visited-hashes set persists across episode boundaries
        # so the novelty bonus only fires on states unseen across the entire
        # training run. Drives the agent toward state regions that previous
        # episodes haven't covered, not just within one episode.
        self.persistent_novelty = persistent_novelty
        self._visited_hashes: set = set()
        self._novel_count_episode = 0
        self._novel_count_global = 0
        # Lazy-load similarity scorer to avoid import-time cost when unused.
        self._score_state = None
        self._initial_similarity = 0.0
        self._prior_life_cells = -1

    def __enter__(self):
        self.sess = self._GameSession.open(self.game_prefix)
        return self

    def __exit__(self, *_):
        return False

    @property
    def available_actions(self) -> list:
        if self.sess is None:
            return []
        return list(self.sess.available_actions or [])

    def reset(self) -> StepObservation:
        if self.sess is None:
            self.sess = self._GameSession.open(self.game_prefix)
        r = self.sess.reset()
        if r.frame is None:
            return StepObservation(None, 0.0, True, 0, True, {"reset_failed": True})
        self._initial_frame = r.frame.copy()
        self._prior_levels = 0
        # Novelty: per-episode set unless persistent_novelty=True, in which
        # case the global set carries over across episodes for this Env.
        if self.persistent_novelty:
            self._visited_hashes.add(_frame_hash(r.frame))
        else:
            self._visited_hashes = {_frame_hash(r.frame)}
        self._novel_count_episode = 0
        self._prior_life_cells = _life_cells(r.frame)
        # Episode-scoped similarity baseline: reward only the DELTA above
        # the initial frame's similarity. Otherwise games that start with
        # background-heavy frames (mostly val=10 etc.) get a constant high
        # similarity baseline that swamps any learning signal.
        self._initial_similarity = 0.0
        if self.use_similarity:
            if self._score_state is None:
                from diloco_lab.arc3_winning_states import score_state  # noqa: E402
                self._score_state = score_state
            self._initial_similarity = float(self._score_state(r.frame))
        return StepObservation(r.frame, 0.0, False, 0, False,
                               {"reset": True,
                                "frame_hash": _frame_sha256(r.frame),
                                "masked_frame_hash": _frame_hash(r.frame),
                                "frame_shape": [int(dim) for dim in r.frame.shape],
                                "initial_similarity": self._initial_similarity,
                                "life_cells": self._prior_life_cells})

    def step(self, act_id: int, click_xy: Optional[tuple] = None) -> StepObservation:
        if self.sess is None:
            raise RuntimeError("env not initialized; call reset() first")
        if act_id in (5, 6) and click_xy is not None:
            r = self.sess.step_with_data(int(act_id), {"x": int(click_xy[0]), "y": int(click_xy[1])})
        else:
            r = self.sess.step(int(act_id))

        levels = self.sess.levels_completed
        terminal = r.terminal
        reward = 0.0
        prior_levels = self._prior_levels
        components = {
            "level": 0.0,
            "cells_changed": 0.0,
            "similarity_delta": 0.0,
            "novelty": 0.0,
            "terminal": 0.0,
        }
        info: dict = {
            "action": int(act_id),
            "click": list(click_xy) if click_xy is not None else None,
            "levels_before": int(prior_levels),
            "levels_completed": int(levels),
            "terminal": bool(terminal),
            "exception": getattr(r, "exception", ""),
            "frame_hash": _frame_sha256(r.frame),
            "masked_frame_hash": _frame_hash(r.frame) if r.frame is not None else None,
            "frame_shape": [int(dim) for dim in r.frame.shape] if r.frame is not None else None,
        }
        if levels > self._prior_levels:
            components["level"] = REWARD_PER_LEVEL * (levels - self._prior_levels)
            reward += components["level"]
        if r.frame is not None and self._initial_frame is not None:
            n_changed = _nchanged_masked(self._initial_frame, r.frame)
            components["cells_changed"] = REWARD_PER_CHANGED_CELL * n_changed
            reward += components["cells_changed"]
            info["cells_changed_from_initial"] = n_changed

        if r.frame is not None and self.use_similarity:
            if self._score_state is None:
                from diloco_lab.arc3_winning_states import score_state  # noqa: E402
                self._score_state = score_state
            sim = float(self._score_state(r.frame))
            # Reward the IMPROVEMENT over the episode's initial similarity.
            # Pure absolute similarity is contaminated by background-heavy
            # frames that match a winner's signature without doing anything.
            sim_delta = sim - self._initial_similarity
            components["similarity_delta"] = REWARD_SIMILARITY_WEIGHT * sim_delta
            reward += components["similarity_delta"]
            info["state_similarity"] = sim
            info["state_similarity_delta"] = sim_delta

        if r.frame is not None and self.use_novelty:
            h = _frame_hash(r.frame)
            if h not in self._visited_hashes:
                self._visited_hashes.add(h)
                if self._novel_count_episode < MAX_NOVEL_PER_EPISODE:
                    components["novelty"] = REWARD_PER_NOVEL_HASH
                    reward += components["novelty"]
                    self._novel_count_episode += 1
                info["novel"] = True
            else:
                info["novel"] = False

        if terminal:
            components["terminal"] = REWARD_PER_TERMINAL
            reward += components["terminal"]
        life_cells_now = _life_cells(r.frame)
        info["life_cells"] = life_cells_now
        info["life_delta"] = (
            life_cells_now - self._prior_life_cells
            if life_cells_now >= 0 and self._prior_life_cells >= 0
            else 0
        )
        self._prior_life_cells = life_cells_now
        info["reward_components"] = components
        info["reward_total"] = reward
        info["verified_win"] = bool(levels > prior_levels)
        done = bool(terminal or (levels > self._prior_levels))
        self._prior_levels = levels
        return StepObservation(r.frame, reward, done, levels, terminal, info)
