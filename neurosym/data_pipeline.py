"""
Pre-training data pipeline for MuZero on ARC-AGI-3.

Converts two sources into muzero-general GameHistory objects:

  1. ARC-AGI-3 recordings  (JSONL, neurosym/recordings/ or equivalent)
  2. Atari-HEAD            (gzip archives from zenodo.org/records/3451402)

Both are normalised to observation shape (1, 64, 64) float32 so the
same MuZero model handles both without architecture changes.

Domain mixing
-------------
The replay buffer receives GameHistory objects from both sources.
ARC-AGI-3 histories are flagged with `is_domain_anchor=True` so the
trainer can apply 3× loss upweighting to them.

The recommended mix is ~98% Atari + ~2% ARC-AGI-3 by sample count,
achieved by oversampling domain-anchor histories in the replay buffer
OR by loss-scaling (preferred — avoids distribution shift artifacts).

Usage
-----
    from neurosym.data_pipeline import ARC3Dataset, AtariHeadDataset

    arc3 = ARC3Dataset(recordings_dir="external/ARC-AGI-3-Agents/recordings")
    atari = AtariHeadDataset(root_dir="/kaggle/input/atari-head")

    all_histories = atari.load_all() + arc3.load_all()
    # shuffle and feed into muzero-general replay buffer
"""

from __future__ import annotations

import gzip
import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Lightweight GameHistory (mirrors muzero-general's self_play.GameHistory)
# No ray dependency — this module must be importable outside the muzero venv.
# ---------------------------------------------------------------------------

@dataclass
class GameHistory:
    """
    Portable GameHistory compatible with muzero-general's replay buffer.

    child_visits  : policy target — uniform for offline data (no MCTS tree)
    root_values   : value target — Monte Carlo returns for offline data
    is_domain_anchor : True for ARC-AGI-3 recordings; triggers 3× loss weight
    level_boundaries : step indices where a level completed (for TTT scope 2)
    """
    observation_history: list = field(default_factory=list)
    action_history: list = field(default_factory=list)
    reward_history: list = field(default_factory=list)
    to_play_history: list = field(default_factory=list)
    child_visits: list = field(default_factory=list)
    root_values: list = field(default_factory=list)
    priorities: Optional[np.ndarray] = None
    game_priority: Optional[float] = None
    is_domain_anchor: bool = False
    level_boundaries: list = field(default_factory=list)   # step indices


def compute_mc_returns(rewards: list[float], discount: float = 0.99) -> list[float]:
    """Backward-pass Monte Carlo returns for value targets."""
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + discount * G
        returns.append(G)
    returns.reverse()
    return returns


def uniform_policy(n_actions: int, length: int) -> list[list[float]]:
    """Uniform visit-count policy for offline data (no MCTS)."""
    p = [1.0 / n_actions] * n_actions
    return [p] * length


# ---------------------------------------------------------------------------
# ARC-AGI-3 recording loader
# ---------------------------------------------------------------------------

_BIN_SIZE = 16          # 64px / 4 bins
_N_BINS = 4
_MAX_COLOR = 12.0


def _arc3_action_to_int(action_id: int, action_data: dict, click_game: bool,
                         action_ids_sorted: list[int]) -> int:
    """Map an ARC-AGI-3 action record to a discrete integer."""
    if click_game:
        x = int(action_data.get("x", 0))
        y = int(action_data.get("y", 0))
        bx = min(x // _BIN_SIZE, _N_BINS - 1)
        by = min(y // _BIN_SIZE, _N_BINS - 1)
        return by * _N_BINS + bx
    else:
        try:
            return action_ids_sorted.index(action_id)
        except ValueError:
            return 0


class ARC3Dataset:
    """
    Loads ARC-AGI-3 randomrecordactions JSONL recordings into GameHistory.

    Each JSONL file becomes one GameHistory.  The `is_domain_anchor` flag
    is set True on every history so the trainer knows to upweight the loss.

    Parameters
    ----------
    recordings_dir : path to directory containing *.recording.jsonl files
    pattern        : glob filter (default: only randomrecordactions files)
    discount       : discount for Monte Carlo return computation
    """

    def __init__(
        self,
        recordings_dir: str | pathlib.Path,
        pattern: str = "*.randomrecordactions.*.jsonl",
        discount: float = 0.99,
    ):
        self.recordings_dir = pathlib.Path(recordings_dir)
        self.pattern = pattern
        self.discount = discount

    def iter_files(self) -> Iterator[pathlib.Path]:
        yield from sorted(self.recordings_dir.glob(self.pattern))

    def load_file(self, path: pathlib.Path) -> Optional[GameHistory]:
        """Parse one JSONL recording → GameHistory.  Returns None on error."""
        events = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    events.append(json.loads(line))

        # Pair action_request + frame events
        pending_action: Optional[dict] = None
        frames: list[dict] = []   # {frame, state, levels_completed, action, action_data}

        prev_levels = 0
        for ev in events:
            d = ev.get("data", {})
            ev_type = d.get("event", "")

            if ev_type == "action_request":
                pending_action = {
                    "action_id": d.get("action_id", 0),
                    "action_data": d.get("action_data") or {},
                    "legal": d.get("legal", True),
                }
                continue

            # Frame event (no "event" key in frame records)
            if "frame" in d and pending_action is not None:
                frames.append({
                    "frame": d["frame"],
                    "state": d.get("state", "NOT_FINISHED"),
                    "levels_completed": d.get("levels_completed", 0),
                    "available_actions": d.get("available_actions", [6]),
                    "action": pending_action,
                })
                pending_action = None

        if len(frames) < 2:
            return None

        # Detect game type from first frame
        avail = frames[0]["available_actions"]
        click_game = (avail == [6])
        n_actions = _N_BINS * _N_BINS if click_game else len(avail)
        action_ids_sorted = sorted(avail) if not click_game else []

        hist = GameHistory(is_domain_anchor=True)
        rewards: list[float] = []
        prev_levels = 0
        level_boundaries: list[int] = []

        for i, f in enumerate(frames):
            # Observation: (1, 64, 64) float32 normalised
            raw = np.array(f["frame"], dtype=np.float32)
            if raw.ndim == 2:
                raw = raw[np.newaxis]   # add channel dim
            obs = raw / _MAX_COLOR

            # Reward
            delta = f["levels_completed"] - prev_levels
            reward = -0.01 + float(delta)
            if f["state"] == "WIN":
                reward += 10.0
            if delta > 0:
                level_boundaries.append(i)
                prev_levels = f["levels_completed"]

            # Action (the action that ARRIVED at this frame)
            act_int = _arc3_action_to_int(
                f["action"]["action_id"],
                f["action"]["action_data"],
                click_game,
                action_ids_sorted,
            )

            hist.observation_history.append(obs)
            hist.action_history.append(act_int)
            hist.reward_history.append(reward)
            hist.to_play_history.append(0)
            rewards.append(reward)

        hist.child_visits = uniform_policy(n_actions, len(hist.observation_history))
        hist.root_values = compute_mc_returns(rewards, self.discount)
        hist.level_boundaries = level_boundaries
        return hist

    def load_all(self, verbose: bool = True) -> list[GameHistory]:
        histories = []
        files = list(self.iter_files())
        for i, path in enumerate(files):
            h = self.load_file(path)
            if h is not None:
                histories.append(h)
                if verbose:
                    print(f"  ARC3 [{i+1}/{len(files)}] {path.name}: "
                          f"{len(h.observation_history)} steps, "
                          f"levels_crossed={len(h.level_boundaries)}")
        print(f"ARC3Dataset: {len(histories)} histories loaded "
              f"(domain_anchor=True, loss_weight=3x)")
        return histories


# ---------------------------------------------------------------------------
# Atari-HEAD loader
# ---------------------------------------------------------------------------

_ATARI_FRAME_H = 210
_ATARI_FRAME_W = 160
_OUT_H = 64
_OUT_W = 64

# Atari-HEAD action space per game (from ALE documentation)
# Mapping: game_name → n_actions (used for uniform policy)
ATARI_HEAD_N_ACTIONS: dict[str, int] = {
    "asterix": 9, "bank_heist": 18, "bowling": 6, "centipede": 18,
    "demon_attack": 6, "enduro": 9, "freeway": 3, "frostbite": 18,
    "hero": 18, "kung_fu_master": 14, "montezuma_revenge": 18,
    "ms_pacman": 9, "pitfall": 18, "private_eye": 18, "riverraid": 18,
    "road_runner": 18, "seaquest": 18, "skiing": 3, "space_invaders": 6,
    "venture": 18,
}
ATARI_DEFAULT_N_ACTIONS = 18


def _load_png_bytes_as_gray64(png_bytes: bytes) -> np.ndarray:
    """Load PNG bytes → (1, 64, 64) float32 grayscale."""
    try:
        import PIL.Image as _Image
        import io as _io
        img = _Image.open(_io.BytesIO(png_bytes)).convert("L").resize((_OUT_W, _OUT_H))
        return (np.array(img, dtype=np.float32) / 255.0)[np.newaxis]
    except Exception:
        return np.zeros((1, _OUT_H, _OUT_W), dtype=np.float32)


class AtariHeadDataset:
    """
    Loads Atari-HEAD trajectories from the official Zenodo release.
    (zenodo.org/records/3451402)

    Actual Zenodo format — after extracting the outer game zip:
        <root_dir>/
            <game_name>/
                <trial_id>.txt        — CSV: frame_id, unclipped_reward, action, ...
                <trial_id>.tar.bz2   — PNG frames: <subject_id>_<frame_num>.png

    Each row in the .txt CSV = one frame.  The PNG filename stem matches the
    frame_id column exactly (e.g. frame_id "RZ_3866968_5" → "RZ_3866968_5.png").

    The tar.bz2 is scanned sequentially; only frames in the sampled target set
    are read into memory, keeping peak RAM low even for long trials.

    Parameters
    ----------
    root_dir             : directory containing extracted game subdirs
    max_trials_per_game  : cap per game; None = unlimited
    max_steps_per_trial  : randomly sample this many steps per trial (None = all)
    discount             : discount factor for MC returns
    """

    def __init__(
        self,
        root_dir: str | pathlib.Path,
        max_trials_per_game: Optional[int] = 2,
        max_steps_per_trial: Optional[int] = 500,
        discount: float = 0.99,
    ):
        self.root_dir = pathlib.Path(root_dir)
        self.max_trials = max_trials_per_game
        self.max_steps = max_steps_per_trial
        self.discount = discount

    def _find_data_dir(self, outer: pathlib.Path) -> pathlib.Path:
        """
        Kaggle fully extracts nested archives, producing:
          outer/outer/<trial>.txt  +  outer/outer/<trial>/<trial>/RZ_*.png
        Local layout has:
          outer/<trial>.txt  +  outer/<trial>.tar.bz2
        Return the directory that actually contains the .txt files.
        """
        inner = outer / outer.name
        if inner.is_dir() and any(inner.glob("*.txt")):
            return inner   # Kaggle double-nested layout
        return outer       # local layout

    def iter_trials(self):
        """Yield (game, txt_path, frames_src) where frames_src is either a
        pathlib.Path pointing to a tar.bz2 file (local) or to a directory
        containing pre-extracted PNGs (Kaggle)."""
        for outer in sorted(self.root_dir.iterdir()):
            if not outer.is_dir():
                continue
            game = outer.name.lower().replace("-", "_")
            data_dir = self._find_data_dir(outer)
            txt_files = sorted(data_dir.glob("*.txt"))
            count = 0
            for txt_path in txt_files:
                tbz = txt_path.with_suffix(".tar.bz2")
                if tbz.exists():
                    yield game, txt_path, tbz          # local: tar.bz2
                    count += 1
                else:
                    # Kaggle: PNGs in <trial>/<trial>/ subdir
                    stem = txt_path.stem
                    png_dir = data_dir / stem / stem
                    if not png_dir.is_dir():
                        png_dir = data_dir / stem      # fallback: single level
                    if png_dir.is_dir():
                        yield game, txt_path, png_dir  # Kaggle: pre-extracted dir
                        count += 1
                if self.max_trials and count >= self.max_trials:
                    break

    def load_trial(self, game, txt_path, frames_src) -> Optional[GameHistory]:
        """Parse one Atari-HEAD trial → GameHistory.
        frames_src is either a .tar.bz2 path or a directory of PNGs."""
        import csv, random

        rows: list[dict] = []
        with open(txt_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    action_str = (row.get("action") or "0").strip()
                    reward_str = (row.get("unclipped_reward") or "0").strip()
                    action = int(action_str) if action_str not in ("", "null") else 0
                    reward = float(reward_str) if reward_str not in ("", "null") else 0.0
                    frame_id = (row.get("frame_id") or "").strip()
                    if frame_id:
                        rows.append({"frame_id": frame_id, "action": action, "reward": reward})
                except (ValueError, TypeError):
                    continue

        if len(rows) < 2:
            return None

        if self.max_steps and len(rows) > self.max_steps:
            rows = random.sample(rows, self.max_steps)
            rows.sort(key=lambda r: r["frame_id"])

        target_ids: set[str] = {r["frame_id"] for r in rows}

        # Load frame bytes — two strategies
        frame_bytes: dict[str, bytes] = {}
        if frames_src.suffix == ".bz2":
            import tarfile
            with tarfile.open(frames_src, mode="r:bz2") as tf:
                for member in tf.getmembers():
                    if not member.name.endswith(".png"):
                        continue
                    stem = pathlib.Path(member.name).stem
                    if stem in target_ids:
                        f = tf.extractfile(member)
                        if f:
                            frame_bytes[stem] = f.read()
        else:
            # Pre-extracted PNG directory (Kaggle layout)
            for png_path in frames_src.glob("*.png"):
                stem = png_path.stem
                if stem in target_ids:
                    frame_bytes[stem] = png_path.read_bytes()

        n_actions = ATARI_HEAD_N_ACTIONS.get(game, ATARI_DEFAULT_N_ACTIONS)
        hist = GameHistory(is_domain_anchor=False)
        step_rewards: list[float] = []

        for row in rows:
            fid = row["frame_id"]
            if fid not in frame_bytes:
                continue
            obs = _load_png_bytes_as_gray64(frame_bytes[fid])
            hist.observation_history.append(obs)
            hist.action_history.append(row["action"])
            hist.reward_history.append(row["reward"])
            hist.to_play_history.append(0)
            step_rewards.append(row["reward"])

        if len(hist.observation_history) < 2:
            return None

        hist.child_visits = uniform_policy(n_actions, len(hist.observation_history))
        hist.root_values = compute_mc_returns(step_rewards, self.discount)
        return hist

    def load_all(self, verbose: bool = True) -> list[GameHistory]:
        histories = []
        trials = list(self.iter_trials())
        for i, (game, txt_path, frames_src) in enumerate(trials):
            h = self.load_trial(game, txt_path, frames_src)
            if h is not None:
                histories.append(h)
                if verbose:
                    print(f"  Atari [{i+1}/{len(trials)}] {game}/{txt_path.stem}: "
                          f"{len(h.observation_history)} steps")
        print(f"AtariHeadDataset: {len(histories)} histories loaded")
        return histories


def _load_png_as_gray64(path: pathlib.Path) -> np.ndarray:
    """Load PNG file → (1, 64, 64) float32 grayscale."""
    try:
        import PIL.Image as _Image
        img = _Image.open(path).convert("L").resize((_OUT_W, _OUT_H))
        return (np.array(img, dtype=np.float32) / 255.0)[np.newaxis]
    except ImportError:
        pass
    # Fallback: zero frame
    return np.zeros((1, _OUT_H, _OUT_W), dtype=np.float32)


# ---------------------------------------------------------------------------
# Loss-upweighting helper
# ---------------------------------------------------------------------------

def apply_domain_upweight(
    histories: list[GameHistory],
    upweight: float = 3.0,
) -> list[GameHistory]:
    """
    Duplicate domain-anchor histories `upweight` times in the list so the
    replay buffer samples them proportionally more often.

    Prefer loss-scaling over resampling (avoids distribution shift), but
    this helper provides the resampling fallback for frameworks that don't
    expose per-sample loss weights.
    """
    anchors = [h for h in histories if h.is_domain_anchor]
    others = [h for h in histories if not h.is_domain_anchor]

    # Integer part
    n_copies = max(1, int(upweight))
    upsampled = others + anchors * n_copies
    return upsampled


# ---------------------------------------------------------------------------
# GVGAI curated dataset loader
# ---------------------------------------------------------------------------

GVGAI_N_ACTIONS = 5   # GVGAI default: up/down/left/right/use


class GVGAIDataset:
    """
    Loads self-generated GVGAI curated rollouts into GameHistory.

    Expected directory layout:
        <root_dir>/
            frames/
                <episode_id>/
                    000000.png
                    000001.png  ...
            episodes.jsonl        — one JSON object per episode (label schema)
            transitions.parquet   — optional; falls back to per-episode JSONL
            levels/
                manifest.jsonl    — authoritative mechanic map

    The episode label schema follows gvgai_manifest.json:
        {episode_id, game, level_id, new_mechanic_id, prior_mechanics_active,
         success, seed, policy_type, level_start_frame, level_end_frame, ...}

    Parameters
    ----------
    root_dir        : root of the generated GVGAI corpus
    games           : which games to load; None = all
    max_per_level   : cap episodes per (game, level) pair
    discount        : discount for MC returns
    """

    def __init__(
        self,
        root_dir: str | pathlib.Path,
        games: Optional[list[str]] = None,
        max_per_level: Optional[int] = None,
        discount: float = 0.99,
    ):
        self.root_dir = pathlib.Path(root_dir)
        self.games = games
        self.max_per_level = max_per_level
        self.discount = discount

    def _load_manifest(self) -> dict:
        """Load levels/manifest.jsonl → {game: {level_id: mechanic_info}}."""
        manifest_path = self.root_dir / "levels" / "manifest.jsonl"
        if not manifest_path.exists():
            # Fall back to bundled gvgai_manifest.json
            bundled = pathlib.Path(__file__).parent / "gvgai_manifest.json"
            if bundled.exists():
                import json as _json
                data = _json.loads(bundled.read_text())
                manifest = {}
                for g in data.get("games", []):
                    manifest[g["game"]] = {
                        lv["level_id"]: lv for lv in g["levels"]
                    }
                return manifest
            return {}
        manifest = {}
        with open(manifest_path) as fh:
            for line in fh:
                row = json.loads(line.strip())
                g, lvl = row["game"], row["level_id"]
                manifest.setdefault(g, {})[lvl] = row
        return manifest

    def load_episode(
        self,
        ep_meta: dict,
        frames_dir: pathlib.Path,
        manifest: dict,
    ) -> Optional[GameHistory]:
        """Convert one episode metadata dict + frames → GameHistory."""
        ep_id = ep_meta["episode_id"]
        game = ep_meta["game"]
        level_id = ep_meta["level_id"]
        ep_frames_dir = frames_dir / ep_id

        actions_path = ep_frames_dir / "actions.txt"
        rewards_path = ep_frames_dir / "rewards.txt"
        if not actions_path.exists() or not rewards_path.exists():
            return None

        actions = [int(l.strip()) for l in actions_path.read_text().splitlines() if l.strip()]
        rewards = [float(l.strip()) for l in rewards_path.read_text().splitlines() if l.strip()]
        frame_files = sorted(ep_frames_dir.glob("*.png"))

        n = min(len(actions), len(rewards), len(frame_files))
        if n < 2:
            return None

        hist = GameHistory(is_domain_anchor=False)
        # Embed mechanic metadata for TTT scope 2
        level_info = manifest.get(game, {}).get(level_id, {})
        hist.level_boundaries = [0]   # level starts at frame 0 for single-level episodes
        step_rewards: list[float] = []

        for i in range(n):
            obs = _load_png_as_gray64(frame_files[i])
            hist.observation_history.append(obs)
            hist.action_history.append(int(actions[i]))
            hist.reward_history.append(float(rewards[i]))
            hist.to_play_history.append(0)
            step_rewards.append(float(rewards[i]))

        hist.child_visits = uniform_policy(GVGAI_N_ACTIONS, n)
        hist.root_values = compute_mc_returns(step_rewards, self.discount)
        return hist

    def load_all(self, verbose: bool = True) -> list[GameHistory]:
        episodes_path = self.root_dir / "episodes.jsonl"
        frames_dir = self.root_dir / "frames"
        if not episodes_path.exists():
            if verbose:
                print(f"GVGAIDataset: episodes.jsonl not found at {episodes_path}")
            return []

        manifest = self._load_manifest()
        episodes = []
        with open(episodes_path) as fh:
            for line in fh:
                ep = json.loads(line.strip())
                if self.games and ep.get("game") not in self.games:
                    continue
                episodes.append(ep)

        # Cap per (game, level)
        if self.max_per_level:
            from collections import Counter
            counts: Counter = Counter()
            filtered = []
            for ep in episodes:
                key = (ep["game"], ep["level_id"])
                if counts[key] < self.max_per_level:
                    filtered.append(ep)
                    counts[key] += 1
            episodes = filtered

        histories = []
        for i, ep in enumerate(episodes):
            h = self.load_episode(ep, frames_dir, manifest)
            if h is not None:
                histories.append(h)
                if verbose and i % 500 == 0:
                    print(f"  GVGAI [{i+1}/{len(episodes)}] "
                          f"{ep['game']}/L{ep['level_id']}: {len(h.observation_history)} steps")

        print(f"GVGAIDataset: {len(histories)} histories loaded from {len(episodes)} episodes")
        return histories


# ---------------------------------------------------------------------------
# Atari-HEAD score-proxy curriculum signal
# ---------------------------------------------------------------------------

def tag_atari_curriculum_boundaries(
    histories: list[GameHistory],
    score_jump_threshold: float = 100.0,
) -> list[GameHistory]:
    """
    Add weak curriculum boundary tags to Atari-HEAD histories.

    Two proxy signals (per Perplexity recommendation):
    1. Episode boundary (each GameHistory is already one episode — boundary at step 0)
    2. Score discontinuity within an episode: cumulative reward jump > threshold
       suggests a stage transition or room change.

    Tags are stored in `level_boundaries` list as step indices.
    This is a *weak* signal — treat as auxiliary, not ground truth.

    Recommended games for strongest curriculum signal:
        alien, seaquest, montezuma_revenge (room/subgoal structure)
    Avoid for curriculum: pure reflex-speed games (asterix, freeway)
    """
    CURRICULUM_PREFERRED = {"alien", "seaquest", "montezuma_revenge",
                            "hero", "pitfall", "venture", "private_eye"}

    for h in histories:
        if h.level_boundaries:
            continue   # already tagged
        h.level_boundaries = [0]   # episode start always a boundary
        cumulative = 0.0
        last_jump = 0.0
        for i, r in enumerate(h.reward_history):
            cumulative += r
            if cumulative - last_jump > score_jump_threshold:
                h.level_boundaries.append(i)
                last_jump = cumulative
    return histories


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings", default="external/ARC-AGI-3-Agents/recordings",
                        help="Path to ARC-AGI-3 recordings directory")
    parser.add_argument("--atari", default=None,
                        help="Path to Atari-HEAD root dir (skip if not downloaded)")
    args = parser.parse_args()

    print("=== ARC3Dataset ===")
    arc3 = ARC3Dataset(args.recordings)
    arc3_histories = arc3.load_all(verbose=True)
    if arc3_histories:
        h = arc3_histories[0]
        print(f"  First history: {len(h.observation_history)} steps, "
              f"obs[0].shape={h.observation_history[0].shape}, "
              f"domain_anchor={h.is_domain_anchor}")
        print(f"  action range: {min(h.action_history)}–{max(h.action_history)}")
        print(f"  reward range: {min(h.reward_history):.3f}–{max(h.reward_history):.3f}")

    if args.atari:
        print("\n=== AtariHeadDataset ===")
        atari = AtariHeadDataset(args.atari, max_trials_per_game=2)
        atari_histories = atari.load_all(verbose=True)
        if atari_histories:
            h = atari_histories[0]
            print(f"  First history: {len(h.observation_history)} steps, "
                  f"obs[0].shape={h.observation_history[0].shape}")

    print("\n=== Mix check ===")
    mixed = apply_domain_upweight(arc3_histories, upweight=3.0)
    n_anchor = sum(1 for h in mixed if h.is_domain_anchor)
    n_other = len(mixed) - n_anchor
    pct = 100 * n_anchor / max(len(mixed), 1)
    print(f"  Total: {len(mixed)} histories, "
          f"anchor={n_anchor} ({pct:.1f}%), other={n_other}")
