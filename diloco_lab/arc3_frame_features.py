"""Frame loading + featurization for the ARC3 CNN policy.

Two surfaces:

  load_step_frame(tensor_dir, prefix, step_index) → np.ndarray | None
    For training: looks up frames.npz in the per-game tensor_replays dir and
    returns the *before-action* frame for step_index (1-based). Returns None
    if the tensor export doesn't exist for this game (e.g., excluded
    random_walk winners) or step_index is out of range.

  preprocess_frame(frame, n_channels=11) → np.ndarray (C, 64, 64) float32
    Used at both train and inference. Maps raw int32 frame values into a
    one-hot-ish channel stack. Default: 11 channels for ARC3 values 0..10
    (most games use a small palette). Out-of-range values clip to channel 10.
    Output is float32 in {0.0, 1.0}.

Frame shape across the corpus is consistently (64, 64) int32; this is enforced
by the ARC3 environment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


FRAME_HW = 64
DEFAULT_CHANNELS = 11           # values 0..10 covers most observed palettes


def load_step_frame(tensor_root: Path, prefix: str,
                    step_index: int) -> Optional[np.ndarray]:
    """Return the frame *before* applying action at step_index (1-based).
    None if tensor data unavailable for this game or step.
    """
    npz = tensor_root / prefix / "frames.npz"
    if not npz.exists():
        return None
    arr = np.load(npz)["frames"]
    # frames[i] is the state AFTER action i; frames[0] is initial.
    # The frame BEFORE action step_index is frames[step_index - 1].
    idx = max(0, int(step_index) - 1)
    if idx >= arr.shape[0]:
        return None
    return arr[idx]


def preprocess_frame(frame: np.ndarray, n_channels: int = DEFAULT_CHANNELS) -> np.ndarray:
    """Convert int32 (H, W) frame to float32 (C, H, W) one-hot channels.

    Values >= n_channels are folded into the last channel. NaN-safe by
    construction (raw ints).
    """
    if frame is None:
        return np.zeros((n_channels, FRAME_HW, FRAME_HW), dtype=np.float32)
    f = np.asarray(frame, dtype=np.int64)
    if f.ndim == 3 and f.shape[0] == 1:
        f = f[0]
    if f.shape != (FRAME_HW, FRAME_HW):
        # Pad or crop to canonical size (defensive — should not normally fire)
        out = np.zeros((FRAME_HW, FRAME_HW), dtype=np.int64)
        h, w = f.shape[-2:]
        out[:min(FRAME_HW, h), :min(FRAME_HW, w)] = f[:FRAME_HW, :FRAME_HW]
        f = out
    f = np.clip(f, 0, n_channels - 1)
    channels = np.zeros((n_channels, FRAME_HW, FRAME_HW), dtype=np.float32)
    for c in range(n_channels):
        channels[c] = (f == c).astype(np.float32)
    return channels


def empty_frame() -> np.ndarray:
    """Black 64x64 frame for fallbacks."""
    return np.zeros((FRAME_HW, FRAME_HW), dtype=np.int32)
