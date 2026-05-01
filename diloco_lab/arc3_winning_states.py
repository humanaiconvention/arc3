"""Winning-state signature extractor + scorer.

For each of the 8 ARC3 winners we have tensor replays at
`D:/arc3/data/arc3_exports/<tensor_label>/tensor_replays/<prefix>/frames.npz`.
The LAST frame in each is the post-win frame. This module:

  - Loads those final frames once at import time.
  - Computes a value-distribution signature per winner (histogram of pixel
    values 0..N, normalized).
  - Exposes `score_state(frame) → similarity in [0, 1]` returning the
    similarity to the nearest winning frame's signature.

Why value-distribution signatures (not raw pixels): cross-game generalisation.
A winning frame for cn04 looks NOTHING like a winning frame for vc33 in raw
pixel terms (different sizes, different color palettes, different layouts).
But the value distribution captures coarse "structural shape": e.g.,
"mostly-uniform background + small concentrated cluster" maps roughly the
same across many puzzle types. With only 8 examples this is the most
defensible aggregation we have.

Honest caveat: this is a *very* rough proxy. The expectation is that it
provides a directional signal (above 0.5 mean similarity correlates loosely
with "puzzle-like state structure"), not a precise win-distance metric.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


_TENSOR_ROOT = Path(
    "D:/arc3/data/arc3_exports/2026-04-27-meta-registry-replays-tensors-v2/tensor_replays"
)
_NOISE_ROWS = (59, 60, 63)
_VALUE_RANGE = 12  # values 0..11 covers all observed palettes; safer to over-cover


def _frame_signature(frame: np.ndarray, value_range: int = _VALUE_RANGE) -> np.ndarray:
    """Return a length-`value_range` normalized histogram of frame values,
    after masking noise rows. The signature captures coarse value
    distribution and is invariant to spatial layout."""
    f = np.asarray(frame, dtype=np.int64)
    if f.ndim == 3 and f.shape[0] == 1:
        f = f[0]
    masked = f.copy()
    h = masked.shape[0]
    for nr in _NOISE_ROWS:
        if nr < h:
            masked[nr, :] = -1  # mark to skip
    flat = masked[masked >= 0]
    counts = np.bincount(np.clip(flat, 0, value_range - 1), minlength=value_range)
    total = counts.sum()
    if total == 0:
        return np.zeros(value_range, dtype=np.float32)
    return (counts / total).astype(np.float32)


def _load_winner_signatures(tensor_root: Path = _TENSOR_ROOT) -> tuple[list[str], np.ndarray]:
    """Load post-win signature for each game with a tensor replay."""
    prefixes: list[str] = []
    sigs: list[np.ndarray] = []
    if not tensor_root.exists():
        return prefixes, np.zeros((0, _VALUE_RANGE), dtype=np.float32)
    for game_dir in sorted(tensor_root.iterdir()):
        npz = game_dir / "frames.npz"
        if not npz.exists():
            continue
        try:
            arr = np.load(npz)["frames"]
        except Exception:
            continue
        if arr.shape[0] == 0:
            continue
        win_frame = arr[-1]
        prefixes.append(game_dir.name)
        sigs.append(_frame_signature(win_frame))
    if not sigs:
        return prefixes, np.zeros((0, _VALUE_RANGE), dtype=np.float32)
    return prefixes, np.stack(sigs, axis=0)


# Loaded once at import — refresh by re-importing module.
WINNER_PREFIXES, WINNER_SIGNATURES = _load_winner_signatures()


def score_state(frame: Optional[np.ndarray]) -> float:
    """Similarity in [0, 1] of `frame`'s value distribution to the closest
    winning-frame signature. Higher is more "winning-like." Uses 1 - L1/2
    distance (L1 between unit-norm histograms is in [0, 2]).

    Returns 0.0 if no winners loaded or `frame` is None.
    """
    if frame is None or WINNER_SIGNATURES.shape[0] == 0:
        return 0.0
    sig = _frame_signature(frame)
    diffs = np.abs(WINNER_SIGNATURES - sig).sum(axis=1)  # (n_winners,)
    nearest = float(diffs.min())
    return float(1.0 - nearest / 2.0)


def per_winner_similarity(frame: Optional[np.ndarray]) -> dict[str, float]:
    """For diagnostics: similarity to each individual winner."""
    if frame is None or WINNER_SIGNATURES.shape[0] == 0:
        return {}
    sig = _frame_signature(frame)
    diffs = np.abs(WINNER_SIGNATURES - sig).sum(axis=1)
    return {p: float(1.0 - d / 2.0) for p, d in zip(WINNER_PREFIXES, diffs)}
