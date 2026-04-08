"""
ARC-AGI-3 → MuZero AbstractGame adapter.

Bridges the ARC-AGI-3 Arcade() API to muzero-general's AbstractGame interface so
MuZero self-play, training, and TTT can run against live game environments.

Action encoding
---------------
Click games (available_actions=[6]):
    N_BINS × N_BINS spatial grid (default 4×4 = 16 discrete actions).
    action_int  = bin_y * N_BINS + bin_x
    x_pixel     = int((bin_x + 0.5) * 64 / N_BINS)   # bin centre
    y_pixel     = int((bin_y + 0.5) * 64 / N_BINS)

Multi-action games (available_actions=[1,2,...,K]):
    action_int  = action_id - 1   (0-indexed contiguous range)

Observation
-----------
Raw frame is (1, 64, 64) int8 with pixel values 0–max_color (~12).
Returned as float32 normalised to [0, 1] — shape unchanged (1, 64, 64).
Color identity is preserved (no jitter augmentation).

Reward shaping
--------------
    +1.0   per level completed (levels_completed delta)
    +10.0  game won (all win_levels completed)
    -0.01  per step (efficiency nudge, configurable)
    0.0    otherwise

Level boundary tracking
-----------------------
self.current_level  — 1-indexed level (increments on completion)
self.level_history  — list of (step, level, reward) for TTT scope 2
"""

import datetime
import pathlib
import sys
from typing import Optional

import numpy as np

# Allow running from neurosym/ or arc3/ root
_HERE = pathlib.Path(__file__).resolve().parent
_AGENTS_ROOT = _HERE.parent / "external" / "ARC-AGI-3-Agents"
if str(_AGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTS_ROOT))

from arcengine import GameAction, GameState

# muzero-general AbstractGame — imported lazily so this file works without
# the muzero deps installed (e.g. on the ARC-AGI-3 side only).
try:
    _mz_root = _HERE.parent / "external" / "muzero-general"
    if str(_mz_root) not in sys.path:
        sys.path.insert(0, str(_mz_root))
    from games.abstract_game import AbstractGame as _AbstractGame
except ImportError:
    # Fallback stub so the file is importable for inspection / testing
    class _AbstractGame:  # type: ignore
        pass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class MuZeroConfig:
    """
    MuZero hyperparameters for ARC-AGI-3.

    Override game_id, n_bins, and action_space_size after construction for
    different games.  ls20 has 4 simple actions; vc33/ft09/lp85 use the
    click grid (default 4×4 = 16 actions).
    """

    def __init__(self, game_id: str = "ls20", n_bins: int = 4):
        # ── Game ─────────────────────────────────────────────────────────────
        self.game_id = game_id
        self.n_bins = n_bins

        # Frame is (1, 64, 64); normalised to float32 [0,1]
        self.observation_shape = (1, 64, 64)

        # Set per-game below
        if game_id in ("ls20",):
            self.action_space = list(range(4))    # ACTION1–ACTION4 → 0–3
        else:
            # Click games: n_bins × n_bins spatial grid
            self.action_space = list(range(n_bins * n_bins))

        self.players = list(range(1))
        self.stacked_observations = 0            # increase for partial-obs games

        self.muzero_player = 0
        self.opponent = None

        # ── Self-play ─────────────────────────────────────────────────────────
        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.max_moves = 80                      # ARC-AGI-3 budget
        self.num_simulations = 25                # keep low for TTT speed
        self.discount = 0.99
        self.temperature_threshold = None

        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # ── Network (small for fast TTT) ──────────────────────────────────────
        self.network = "resnet"
        self.support_size = 10

        self.downsample = "CNN"                  # lightweight CNN downsampler
        self.blocks = 3
        self.channels = 64
        self.reduced_channels_reward = 16
        self.reduced_channels_value = 16
        self.reduced_channels_policy = 16
        self.resnet_fc_reward_layers = [32]
        self.resnet_fc_value_layers = [32]
        self.resnet_fc_policy_layers = [32]

        self.encoding_size = 32                  # for fullyconnected fallback
        self.fc_representation_layers = [64]
        self.fc_dynamics_layers = [64]
        self.fc_reward_layers = [32]
        self.fc_value_layers = [32]
        self.fc_policy_layers = [32]

        # ── Training ──────────────────────────────────────────────────────────
        import torch
        self.results_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "results"
            / game_id
            / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        )
        self.save_model = True
        self.training_steps = 20_000
        self.batch_size = 64
        self.checkpoint_interval = 50
        self.value_loss_weight = 0.25
        self.train_on_gpu = torch.cuda.is_available()

        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.lr_init = 3e-4
        self.lr_decay_rate = 0.9
        self.lr_decay_steps = 5_000

        # ── Replay buffer ─────────────────────────────────────────────────────
        self.replay_buffer_size = 1_000
        self.num_unroll_steps = 5
        self.td_steps = 20
        self.PER = True
        self.PER_alpha = 0.5

        self.use_last_model_value = True
        self.reanalyse_on_gpu = False

        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = 1.5

    def visit_softmax_temperature_fn(self, trained_steps: int) -> float:
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        return 0.25


# ---------------------------------------------------------------------------
# Game adapter
# ---------------------------------------------------------------------------

_MAX_COLOR = 12.0   # empirical max pixel value across 4 starter games


class ARC3Game(_AbstractGame):
    """
    MuZero AbstractGame adapter for a single ARC-AGI-3 environment.

    Parameters
    ----------
    game_id : str
        ARC-AGI-3 game identifier, e.g. "ls20-9607627b" or short "ls20".
    n_bins : int
        Grid resolution for click games (ACTION6). Default 4 → 4×4 = 16 actions.
    step_penalty : float
        Per-step efficiency penalty. Default -0.01.
    seed : int | None
        Unused (ARC-AGI-3 env is server-side); kept for AbstractGame compat.
    """

    N_MAX_COLOR = _MAX_COLOR

    def __init__(
        self,
        game_id: str = "ls20",
        n_bins: int = 4,
        step_penalty: float = -0.01,
        seed: Optional[int] = None,
    ):
        import arc_agi

        self.game_id = game_id
        self.n_bins = n_bins
        self.step_penalty = step_penalty

        self._arc = arc_agi.Arcade()
        self._env = self._arc.make(game_id)
        self._obs = None

        # Determined from first observation in reset()
        self._click_game: bool = False
        self._n_actions: int = 0

        # Level tracking for TTT scope 2
        self.current_level: int = 1
        self.prev_levels_completed: int = 0
        self.level_history: list[dict] = []   # [{step, level, reward, obs_hash}]
        self._step_count: int = 0

    # ── AbstractGame interface ───────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Reset to a fresh game. Returns initial observation (1, 64, 64) float32."""
        self._obs = self._env.reset()
        self.prev_levels_completed = 0
        self.current_level = 1
        self.level_history = []
        self._step_count = 0

        # Detect game type from first obs
        avail = list(self._obs.available_actions)
        if avail == [6]:
            self._click_game = True
            self._n_actions = self.n_bins * self.n_bins
        else:
            self._click_game = False
            self._n_actions = len(avail)
            self._action_ids = sorted(avail)   # e.g. [1,2,3,4]

        return self._frame_to_obs(self._obs.frame)

    def step(self, action_int: int):
        """
        Apply a discrete action.

        Returns
        -------
        observation : np.ndarray  (1, 64, 64) float32
        reward      : float
        done        : bool
        """
        game_action, action_data = self._decode_action(action_int)
        self._obs = self._env.step(game_action, data=action_data)
        self._step_count += 1

        reward = self._compute_reward(self._obs)
        done = self._obs.state in (GameState.GAME_OVER, GameState.WIN)

        # Record level boundary
        if self._obs.levels_completed > self.prev_levels_completed:
            self.current_level = self._obs.levels_completed + 1
            self.level_history.append({
                "step": self._step_count,
                "level_completed": self._obs.levels_completed,
                "reward": reward,
            })
            self.prev_levels_completed = self._obs.levels_completed

        return self._frame_to_obs(self._obs.frame), reward, done

    def legal_actions(self) -> list[int]:
        """Return all discrete action indices (full grid or full action set)."""
        if self._n_actions == 0:
            # Before first reset — return full grid
            return list(range(self.n_bins * self.n_bins))
        return list(range(self._n_actions))

    def render(self):
        """Print a compact ASCII representation of the current frame."""
        if self._obs is None:
            print("[ARC3Game] not started")
            return
        frame = np.array(self._obs.frame)[0]   # (64, 64)
        # Downsample 4× for terminal display
        small = frame[::4, ::4]
        chars = " .+#@%&*OX123456789ABCDEF"
        for row in small:
            print("".join(chars[min(v, len(chars) - 1)] for v in row))

    def close(self):
        pass

    def action_to_string(self, action_int: int) -> str:
        if self._click_game:
            bx = action_int % self.n_bins
            by = action_int // self.n_bins
            px = int((bx + 0.5) * 64 / self.n_bins)
            py = int((by + 0.5) * 64 / self.n_bins)
            return f"click({px},{py}) [bin {bx},{by}]"
        else:
            aid = self._action_ids[action_int]
            return f"ACTION{aid}"

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _frame_to_obs(self, frame) -> np.ndarray:
        """Convert raw frame (1,64,64) int8 → float32 normalised [0,1]."""
        arr = np.array(frame, dtype=np.float32)
        arr /= self.N_MAX_COLOR
        return arr  # (1, 64, 64)

    def _decode_action(self, action_int: int) -> tuple:
        """Map discrete action_int → (GameAction, data_dict) ready to submit."""
        if self._click_game:
            bx = action_int % self.n_bins
            by = action_int // self.n_bins
            px = int((bx + 0.5) * 64 / self.n_bins)
            py = int((by + 0.5) * 64 / self.n_bins)
            action = GameAction.from_id(6)
            action.set_data({"x": px, "y": py})
            data = {"x": px, "y": py}
        else:
            aid = self._action_ids[action_int]
            action = GameAction.from_id(aid)
            data = {}
        return action, data

    def _compute_reward(self, obs) -> float:
        reward = self.step_penalty
        delta = obs.levels_completed - self.prev_levels_completed
        if delta > 0:
            reward += float(delta)              # +1.0 per level completed
        if obs.state == GameState.WIN or obs.levels_completed == obs.win_levels:
            reward += 10.0                      # game-win bonus
        return reward


# ---------------------------------------------------------------------------
# TTT wrappers — scope 1 (intra-level) and scope 2 (cross-level)
# ---------------------------------------------------------------------------

class TTTScope:
    """
    Base class for test-time training wrappers around ARC3Game.

    Tracks prediction errors per level and adjusts the gradient-step budget
    via surprise-driven decay: fewer steps when the model's prediction loss
    is already low (accumulated world model is a good fit for this level).
    """

    # TTT budget at level 1 (max).  Decays each level.
    BASE_TTT_STEPS = 32
    DECAY_FACTOR = 0.65     # per-level exponential decay

    def __init__(self, game: ARC3Game):
        self.game = game
        self._level_ttt_steps: dict[int, int] = {}

    def ttt_budget(self, level: int) -> int:
        """
        Gradient steps to allocate for TTT at the start of `level`.

        Level 1 gets BASE_TTT_STEPS.  Each subsequent level gets
        DECAY_FACTOR × prior level's budget (floored at 1).
        The decay is surprise-driven in the full implementation; here it is
        a fixed exponential schedule as a conservative baseline.
        """
        steps = max(1, int(self.BASE_TTT_STEPS * (self.DECAY_FACTOR ** (level - 1))))
        self._level_ttt_steps[level] = steps
        return steps

    def level_transitions(self) -> list[dict]:
        """Return recorded level boundary events from the underlying game."""
        return self.game.level_history


class TTTScope1(TTTScope):
    """
    Intra-level adaptation.

    On each level start, performs `ttt_budget(level)` gradient steps on the
    most recent observations to tune the dynamics/prediction heads to this
    level's specific mechanics.
    """
    pass   # gradient logic lives in the MuZero trainer; this provides budget


class TTTScope2(TTTScope):
    """
    Cross-level accumulation.

    Carries the model state learned from levels 1..L-1 into level L.
    Selectively re-weights prior level data based on feature similarity
    (levels with similar mechanics contribute more to the prior).

    The accumulated knowledge makes TTT at level L cheaper — budget decays
    naturally because the model's prediction error at level start is already
    lower when the world model is well-calibrated.
    """

    # Slightly stronger retention of prior levels vs. pure intra-level TTT
    BASE_TTT_STEPS = 16
    DECAY_FACTOR = 0.70


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20", help="game_id to test")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--bins", type=int, default=4)
    args = parser.parse_args()

    print(f"Smoke-testing ARC3Game(game_id='{args.game}', n_bins={args.bins})")
    g = ARC3Game(game_id=args.game, n_bins=args.bins)
    obs = g.reset()
    print(f"  reset OK — obs shape={obs.shape}, dtype={obs.dtype}, "
          f"range=[{obs.min():.3f}, {obs.max():.3f}]")
    print(f"  click_game={g._click_game}, n_actions={g._n_actions}")
    print(f"  legal_actions={g.legal_actions()}")

    scope1 = TTTScope1(g)
    for step in range(args.steps):
        action = g.legal_actions()[step % g._n_actions]
        obs, reward, done = g.step(action)
        budget = scope1.ttt_budget(g.current_level)
        print(f"  step {step+1}: action={g.action_to_string(action)}, "
              f"reward={reward:.3f}, done={done}, "
              f"level={g.current_level}, ttt_budget={budget}")
        if done:
            break

    print("Smoke test complete.")
