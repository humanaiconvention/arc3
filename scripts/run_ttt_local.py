"""
run_ttt_local.py — ARC-AGI-3 MuZero TTT local runner

Loads a Stage 2 checkpoint and runs test-time training (TTT) against the live
ARC-AGI-3 platform, writing per-game Stage 3 checkpoints and a summary JSON.

Usage
-----
    python D:/arc3/scripts/run_ttt_local.py
    python D:/arc3/scripts/run_ttt_local.py --checkpoint D:/arc3/results/v1/stage2_finetune_2026-04-09--07-24-15.checkpoint
    python D:/arc3/scripts/run_ttt_local.py --games vc33 --sims 50 --dry-run

Arguments
---------
    --checkpoint   Path to stage2_finetune.checkpoint  [default: latest in results/]
    --games        Comma-separated game IDs             [default: vc33]
    --max-moves    Action budget per game               [default: 80]
    --sims         MCTS simulations per step            [default: 25]
    --out-dir      Output directory                     [default: results/ttt/<ts>]
    --dry-run      Skip ARC API, just test model load   [default: False]
"""

import argparse
import copy
import datetime
import json
import os
import pathlib
import pickle
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = pathlib.Path(__file__).resolve().parent
_ARC3_ROOT   = _SCRIPT_DIR.parent
_MUZERO_DIR  = _ARC3_ROOT / "external" / "muzero-general"
_NEUROSYM    = _ARC3_ROOT / "neurosym"
_AGENTS_ROOT = _ARC3_ROOT / "external" / "ARC-AGI-3-Agents"
_AGENTS_VENV_SITE = _AGENTS_ROOT / ".venv" / "Lib" / "site-packages"
_ENV_FILE    = _AGENTS_ROOT / ".env"

for p in [str(_MUZERO_DIR), str(_NEUROSYM), str(_AGENTS_ROOT), str(_AGENTS_VENV_SITE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Load ARC API key from .env ─────────────────────────────────────────────────
def _load_dotenv(path: pathlib.Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_dotenv(_ENV_FILE)

# ── Imports that need sys.path set first ──────────────────────────────────────
import models                               # muzero-general/models.py (IMPALA-patched)
from self_play import MCTS                  # muzero-general/self_play.py
from arc3_game import MuZeroConfig, TTTScope1, TTTScope2

_arc_err_msg = ""
try:
    from arc_agi import Arcade
    from arcengine import GameAction, GameState
    _ARCENGINE_OK = True
except ImportError as _arc_err:
    _ARCENGINE_OK = False
    _arc_err_msg = str(_arc_err)
    Arcade = GameAction = GameState = None  # type: ignore

try:
    from generic_solver import (
        make_game_model, generic_decode, compute_action_effect,
        detect_background, find_objects,
    )
    _GENERIC_SOLVER_OK = True
except ImportError as _gs_err:
    _GENERIC_SOLVER_OK = False

# ── CLI ───────────────────────────────────────────────────────────────────────
def _find_latest_checkpoint(pattern="stage2_finetune*.checkpoint"):
    results = _ARC3_ROOT / "results"
    hits = sorted(results.rglob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None

p = argparse.ArgumentParser(description="ARC-AGI-3 MuZero TTT local runner")
p.add_argument("--checkpoint", default=None,
               help="Path to stage2_finetune.checkpoint (default: latest in results/)")
p.add_argument("--games",     default="vc33",
               help="Comma-separated game IDs (default: vc33)")
p.add_argument("--max-moves", type=int, default=80)
p.add_argument("--sims",      type=int, default=25,
               help="MCTS simulations per step (default: 25)")
p.add_argument("--out-dir",   default=None,
               help="Output directory (default: results/ttt/<timestamp>)")
p.add_argument("--dry-run",   action="store_true",
               help="Skip ARC API calls; validate model loading only")
p.add_argument("--n-lives",  type=int, default=1,
               help="Number of game lives to play per game (default: 1)")
args = p.parse_args()

_raw_games = args.games.strip()
if _raw_games.lower() == "all":
    # Fetch all available game IDs from the ARC API
    try:
        import sys as _sys
        for _p in [str(_AGENTS_ROOT), str(_AGENTS_VENV_SITE)]:
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
        from arc_agi import Arcade as _TmpArcade
        _tmp = _TmpArcade()
        _api_ids = [e.game_id for e in _tmp.available_environments]
        # Fix 1: Hardcode proven-solvers first to avoid shared-model corruption.
        # sp80's stochastic win only works before other games pollute the policy.
        _priority = ["vc33", "lp85", "sp80"]
        GAME_IDS = _priority + [g for g in _api_ids if g not in _priority]
        print(f"  [all] {len(GAME_IDS)} game IDs ({len(_api_ids)} from API, "
              f"priority: {','.join(_priority)})")
    except Exception as _e:
        print(f"  [all] Could not fetch environments: {_e}; falling back to vc33")
        GAME_IDS = ["vc33"]
else:
    GAME_IDS = [g.strip() for g in _raw_games.split(",") if g.strip()]
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
TS        = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
OUT_DIR   = pathlib.Path(args.out_dir) if args.out_dir else (_ARC3_ROOT / "results" / "ttt" / TS)
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARC_API_KEY = os.environ.get("ARC_API_KEY")

print(f"\n{'='*60}")
print(f"  ARC-AGI-3 MuZero TTT — local runner")
print(f"  device  : {DEVICE}")
if DEVICE == "cuda":
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  gpu     : {torch.cuda.get_device_name(0)} ({mem:.1f} GB)")
print(f"  games   : {GAME_IDS}")
print(f"  sims    : {args.sims}")
print(f"  out     : {OUT_DIR}")
print(f"  api key : {'set' if ARC_API_KEY else 'MISSING'}")
print(f"  arcengine: {'ok' if _ARCENGINE_OK else 'MISSING — ' + _arc_err_msg if not _ARCENGINE_OK else 'ok'}")
print(f"{'='*60}\n")

# ── Load Stage 2 checkpoint ───────────────────────────────────────────────────
ckpt_path = pathlib.Path(args.checkpoint) if args.checkpoint else _find_latest_checkpoint()
if ckpt_path is None or not ckpt_path.exists():
    print("ERROR: No Stage 2 checkpoint found. Pass --checkpoint <path>.")
    sys.exit(1)

print(f"Loading checkpoint: {ckpt_path}")
with open(ckpt_path, "rb") as f:
    payload = pickle.load(f)

S2_WEIGHTS = payload.get("weights")
S2_CONFIG  = payload.get("config")
print(f"  stage          : {payload.get('stage', '?')}")
print(f"  training_steps : {payload.get('training_steps', '?')}")
if S2_CONFIG:
    print(f"  network        : {getattr(S2_CONFIG, 'network', '?')}")
    print(f"  action_space   : {len(getattr(S2_CONFIG, 'action_space', []))} actions")
    print(f"  impala_out_dim : {getattr(S2_CONFIG, 'impala_out_dim', '?')}")

DRY_RUN = args.dry_run or (ARC_API_KEY is None) or (not _ARCENGINE_OK)
if not ARC_API_KEY:
    print("WARNING: ARC_API_KEY not set — forcing dry-run mode.")
if not _ARCENGINE_OK:
    print("WARNING: arcengine not importable — forcing dry-run mode.")
if DRY_RUN and not args.dry_run:
    print("Running in DRY-RUN mode (model loading only).")

# ── Build base config from checkpoint ────────────────────────────────────────
# Start from a fresh MuZeroConfig (defaults to IMPALA), then mirror everything
# from the Stage 2 checkpoint config so shapes match exactly.
BASE_CONFIG = MuZeroConfig(game_id="stage3", n_bins=4)
BASE_CONFIG.action_space = list(range(16))   # 4×4 click grid — universal default
BASE_CONFIG.max_moves    = args.max_moves
BASE_CONFIG.num_simulations = args.sims
BASE_CONFIG.results_path = OUT_DIR / "muzero_run"
BASE_CONFIG.results_path.mkdir(parents=True, exist_ok=True)
BASE_CONFIG.train_on_gpu = (DEVICE == "cuda")

# Mirror Stage 2 config attrs — covers network, impala_*, blocks, channels,
# reduced_channels_*, resnet_fc_*_layers, support_size, discount, etc.
_MIRROR_ATTRS = [
    "observation_shape", "stacked_observations", "support_size",
    "network", "downsample", "blocks", "channels",
    "impala_channels", "impala_out_dim",
    "reduced_channels_reward", "reduced_channels_value", "reduced_channels_policy",
    "resnet_fc_reward_layers", "resnet_fc_value_layers", "resnet_fc_policy_layers",
    "discount", "value_loss_weight", "weight_decay",
    "action_space",          # take action_space from checkpoint (should be 16)
]
if S2_CONFIG:
    for attr in _MIRROR_ATTRS:
        if hasattr(S2_CONFIG, attr):
            setattr(BASE_CONFIG, attr, getattr(S2_CONFIG, attr))

print(f"\nBase config ready:")
print(f"  network        : {BASE_CONFIG.network}")
print(f"  action_space   : {len(BASE_CONFIG.action_space)} actions")
print(f"  observation    : {BASE_CONFIG.observation_shape}")
print(f"  support_size   : {BASE_CONFIG.support_size}")
if BASE_CONFIG.network == "impala":
    print(f"  impala_channels: {BASE_CONFIG.impala_channels}")
    print(f"  impala_out_dim : {BASE_CONFIG.impala_out_dim}")

# ── Build model ───────────────────────────────────────────────────────────────
BASE_MODEL = models.MuZeroNetwork(BASE_CONFIG).to(DEVICE)
BASE_MODEL.ttt_config = BASE_CONFIG
if S2_WEIGHTS is not None:
    BASE_MODEL.set_weights(S2_WEIGHTS)
    print("Stage 2 weights loaded into model.")
else:
    print("WARNING: No weights in checkpoint — using random initialisation.")
BASE_WEIGHTS = copy.deepcopy(BASE_MODEL.get_weights())
n_params = sum(p.numel() for p in BASE_MODEL.parameters())
print(f"  params         : {n_params:,}")

# ── TTT helpers ───────────────────────────────────────────────────────────────
def _one_hot_policy(action_batch: torch.Tensor, policy_dim: int) -> torch.Tensor:
    target = torch.zeros(action_batch.shape[0], policy_dim, device=action_batch.device)
    target.scatter_(1, action_batch.long(), 1.0)
    return target


def run_ttt_update(model, optimizer, experience_buffer, n_steps: int, lr: float):
    """SGD-style TTT on (obs, action, reward, next_obs) tuples."""
    if n_steps <= 0 or not experience_buffer:
        return 0.0
    config = getattr(model, "ttt_config", BASE_CONFIG)
    support_size   = config.support_size
    discount       = config.discount
    val_w          = config.value_loss_weight
    batch_size     = min(32, len(experience_buffer))
    was_training   = model.training
    model.train()
    for g in optimizer.param_groups:
        g["lr"] = lr
    total_loss = 0.0
    for _ in range(n_steps):
        batch = (random.sample(experience_buffer, k=batch_size)
                 if len(experience_buffer) >= batch_size
                 else [random.choice(experience_buffer) for _ in range(batch_size)])
        obs  = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=DEVICE)
        act  = torch.tensor([b[1] for b in batch], dtype=torch.long, device=DEVICE).unsqueeze(-1)
        rew  = torch.tensor([[b[2]] for b in batch], dtype=torch.float32, device=DEVICE)
        nobs = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            nv_logits, _, np_logits, _ = model.initial_inference(nobs)
            nv_scalar   = models.support_to_scalar(nv_logits, support_size)
            np_target   = torch.softmax(np_logits, dim=1)
        rv_target  = models.scalar_to_support(rew + discount * nv_scalar, support_size).squeeze(1)
        rv1_target = models.scalar_to_support(nv_scalar, support_size).squeeze(1)
        rr_target  = models.scalar_to_support(rew, support_size).squeeze(1)
        v0, r0, p0, h = model.initial_inference(obs)
        if h.requires_grad:
            h.register_hook(lambda g: g * 0.5)
        v1, r1, p1, _ = model.recurrent_inference(h, act)
        p0_target = _one_hot_policy(act, p0.shape[1])
        lsm = torch.nn.LogSoftmax(dim=1)
        loss = (
            val_w * ((-rv_target  * lsm(v0)).sum(1) + (-rv1_target * lsm(v1)).sum(1))
            + (-rr_target * lsm(r1)).sum(1)
            + (-p0_target * lsm(p0)).sum(1)
            + (-np_target * lsm(p1)).sum(1)
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
    model.train(was_training)
    return total_loss / n_steps


def select_action(model, obs, config, temperature: float = 1.0) -> int:
    legal = list(getattr(config, "legal_actions", config.action_space))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        root, _ = MCTS(config).run(model, obs, legal, to_play=0,
                                   add_exploration_noise=False)
    visits = np.array(
        [root.children[a].visit_count if a in root.children else 0 for a in legal],
        dtype=np.float32)
    if temperature <= 1e-6 or np.all(visits == 0):
        action = legal[int(np.argmax(visits))]
    else:
        scaled = np.power(np.maximum(visits, 1e-8), 1.0 / temperature)
        s = scaled.sum()
        action = int(np.random.choice(legal, p=scaled / s)) if s > 0 else legal[int(np.argmax(visits))]
    model.train(was_training)
    return int(action)


def normalize_obs(frame) -> np.ndarray:
    arr = np.array(frame, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 3 and arr.shape[0] != 1:
        arr = arr[-1:]
    return arr / 12.0


def legal_actions_for_state(state, n_bins: int = 4):
    available = list(getattr(state, "available_actions", []) or [])
    if available == [6]:                         # click game
        return list(range(n_bins * n_bins)), True
    return [int(a) - 1 for a in available], False


_click_state: dict = {"left_block_idx": 0, "last_cross_col": None, "last_levels": -1}


def reset_click_state() -> None:
    """Reset per-game click state tracker."""
    _click_state["left_block_idx"] = 0
    _click_state["last_cross_col"] = None


def decode_action(action_int: int, available_actions, n_bins: int = 4,
                  obs_arr: "np.ndarray | None" = None,
                  game_model=None, goal_frame=None, game_id: str = ""):
    """
    Map a MuZero bin-action to a GameAction + pixel coordinates.

    Strategy (general):
    1. Use generic_solver.generic_decode with the per-game GameModel
       (learned from recordings + live play).
    2. For vc33 specifically, also try the maroon_solver heuristic as a
       cross-check — keeps Level 1 reliable while generic learning catches up.

    For directional/keyboard games (available_actions != [6]):
    - generic_decode maps action_int to the appropriate GameAction.
    """
    # ── Directional / keyboard games ──────────────────────────────────────────
    if list(available_actions) != [6]:
        if _GENERIC_SOLVER_OK and game_model is not None and obs_arr is not None:
            try:
                return generic_decode(obs_arr, available_actions, game_model,
                                      action_int, n_bins=n_bins, goal_frame=goal_frame)
            except Exception:
                pass
        # Fallback: map action_int directly to GameAction
        legal = sorted(int(a) for a in available_actions)
        chosen = legal[action_int % len(legal)] if legal else 1
        return GameAction.from_id(chosen), {}

    # ── Click games (available_actions == [6]) ────────────────────────────────
    act = GameAction.from_id(6)

    # Primary: generic solver with learned game model
    # Skip generic for vc33 — maroon_solver is verified accurate for it.
    # For all other click games, use generic_decode (learns from recordings + live play).
    if _GENERIC_SOLVER_OK and game_model is not None and obs_arr is not None \
            and not game_id.startswith("vc33"):
        try:
            _, data = generic_decode(obs_arr, available_actions, game_model,
                                     action_int, n_bins=n_bins, goal_frame=goal_frame)
            return act, data
        except Exception:
            pass

    # Fallback for vc33: maroon_solver heuristic (reliable Level 1 solution)
    if game_id.startswith("vc33") and obs_arr is not None:
        try:
            from maroon_solver import (
                _mode, _compute_clicks_right, _compute_clicks_left,
                _find_marker_col,
                RIGHT_MAROON_CLICK, LEFT_MAROON_BLOCKS, IDLE_CLICK,
                V_PINK14,
            )
            mode = _mode(obs_arr)
            if mode == "right":
                clicks_needed = _compute_clicks_right(obs_arr)
                if clicks_needed > 0:
                    px, py = RIGHT_MAROON_CLICK
                    _click_state["last_cross_col"] = None
                else:
                    px, py = LEFT_MAROON_BLOCKS[_click_state["left_block_idx"]]
            elif mode == "left":
                clicks_needed = _compute_clicks_left(obs_arr)
                cross_col = _find_marker_col(obs_arr, V_PINK14, (52, 58))
                last_col  = _click_state["last_cross_col"]
                if (cross_col is not None and last_col is not None
                        and cross_col == last_col and clicks_needed > 0):
                    _click_state["left_block_idx"] = (
                        (_click_state["left_block_idx"] + 1) % len(LEFT_MAROON_BLOCKS)
                    )
                _click_state["last_cross_col"] = cross_col
                px, py = (LEFT_MAROON_BLOCKS[_click_state["left_block_idx"]]
                          if clicks_needed > 0 else IDLE_CLICK)
            else:
                px, py = IDLE_CLICK
            return act, {"x": px, "y": py}
        except Exception:
            pass

    # Final fallback: bin-centre pixels
    bx = action_int % n_bins
    by = action_int // n_bins
    px = int((bx + 0.5) * 64 / n_bins)
    py = int((by + 0.5) * 64 / n_bins)
    return act, {"x": px, "y": py}


def shaped_reward(prev, nxt) -> float:
    r = -0.01
    delta = int(getattr(nxt, "levels_completed", 0) or 0) - \
            int(getattr(prev, "levels_completed", 0) or 0)
    if delta > 0:
        r += float(delta)
    win = int(getattr(nxt, "win_levels", 0) or 0)
    completed = int(getattr(nxt, "levels_completed", 0) or 0)
    if (getattr(nxt, "state", None) == GameState.WIN or
            (win > 0 and completed >= win)):
        r += 10.0
    return float(r)


# ── Per-game TTT loop ─────────────────────────────────────────────────────────
summary = {
    "checkpoint": str(ckpt_path),
    "timestamp": TS,
    "device": DEVICE,
    "dry_run": DRY_RUN,
    "games": {},
}

# ── Single Arcade instance shared across all games (one scorecard for submission) ──
_shared_arcade = None
if not DRY_RUN and _ARCENGINE_OK:
    try:
        _shared_arcade = Arcade()
        print(f"  Arcade ready — all games will share one scorecard\n")
    except Exception as _ae:
        print(f"  WARNING: Arcade init failed: {_ae}")

# ── Shared model — weights carry forward across games ────────────────────────
# We initialise ONE model and ONE optimizer for the entire run.
# Each game's TTT updates (scope1 + scope2) accumulate in the weights,
# so later games benefit from everything learned on earlier games.
# The optimizer is also shared so momentum/adaptive LR state carries forward,
# which helps the optimizer converge faster on each subsequent game.
# Per-game scope2_buffer is reset each game so that each game's scope2 pre-level
# update draws only from that game's own experience (avoids stale cross-game obs).
_shared_model = models.MuZeroNetwork(BASE_CONFIG).to(DEVICE)
_shared_model.set_weights(copy.deepcopy(BASE_WEIGHTS))
_shared_model.ttt_config = BASE_CONFIG
_shared_optimizer = torch.optim.Adam(
    _shared_model.parameters(), lr=3e-5,
    weight_decay=getattr(BASE_CONFIG, "weight_decay", 1e-4))
print(f"  Shared model initialised from stage2 checkpoint — weights carry forward across all {len(GAME_IDS)} games\n")

for game_id in GAME_IDS:
    print(f"\n{'='*60}")
    print(f"  GAME: {game_id}  (dry_run={DRY_RUN})")
    print(f"{'='*60}")

    # Reuse shared model/optimizer — do NOT reset weights
    model     = _shared_model
    optimizer = _shared_optimizer
    model.ttt_config = BASE_CONFIG

    cfg = copy.deepcopy(BASE_CONFIG)
    cfg.game_id = game_id

    # TTT scope trackers — reset per game so scope2 budget scales from level 1
    _scope_ns = SimpleNamespace(level_history=[])
    scope1    = TTTScope1(_scope_ns)
    scope2    = TTTScope2(_scope_ns)

    scope2_buffer: list = []
    level_reports: list = []
    game_start = time.time()

    if DRY_RUN:
        # ── Dry-run: one random-obs forward pass, no ARC API ─────────────────
        # Use eval() so BatchNorm uses running stats (works with batch_size=1)
        model.eval()
        dummy_obs = np.zeros((1, 64, 64), dtype=np.float32)
        obs_t = torch.tensor(dummy_obs[np.newaxis], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            v, r, pi, h = model.initial_inference(obs_t)
        print(f"  forward pass OK — value shape {v.shape}, policy shape {pi.shape}")
        print(f"  policy probs (first 8): {torch.softmax(pi[0], -1)[:8].cpu().numpy().round(3)}")
        level_reports.append({"dry_run": True, "forward_pass": "ok"})

    else:
        # ── Live TTT loop ─────────────────────────────────────────────────────
        try:
                # Reuse the shared Arcade instance so all games appear on the same scorecard.
            arcade = _shared_arcade if _shared_arcade is not None else Arcade()
            env    = arcade.make(game_id)
            state  = env.reset()
            # Retry up to 3× if env.reset() returns None (transient API hiccup)
            for _retry in range(3):
                if state is not None and getattr(state, 'frame', None) is not None:
                    break
                print(f"  [retry {_retry+1}/3] env.reset() returned None for {game_id}, retrying...")
                import time as _time; _time.sleep(2)
                env   = arcade.make(game_id)
                state = env.reset()
            if state is None or getattr(state, 'frame', None) is None:
                raise RuntimeError(f"env.reset() returned None for {game_id} after 3 retries — skipping")
            obs    = normalize_obs(state.frame)

            available_actions          = list(getattr(state, "available_actions", []) or [])
            initial_legal, _           = legal_actions_for_state(state)
            if initial_legal:
                cfg.action_space  = initial_legal[:]
                cfg.legal_actions = initial_legal[:]
            model.ttt_config = cfg

            # ── Generic game model (learns mechanics from recordings + live play) ──
            _game_model  = None
            _goal_frame  = None
            if _GENERIC_SOLVER_OK:
                try:
                    _recordings_dir = _AGENTS_ROOT / "recordings"
                    _game_model = make_game_model(game_id, _recordings_dir)
                    if _game_model.winning_frames:
                        _goal_frame = _game_model.winning_frames[0]
                    _bg = _game_model.background
                    print(f"  game_model: bg={_bg} moving={_game_model.moving_values} "
                          f"wins={len(_game_model.winning_frames)}")
                except Exception as _gm_err:
                    print(f"  game_model init failed: {_gm_err}")

            total_steps   = 0
            done          = False
            lives_used    = 0
            # Fix 3: Expose lives_used on model so one-dir slider can vary
            # position budget per life instead of saturating at the extreme.
            if game_model is not None:
                game_model._lives_used = 0
            current_level = max(1, int(getattr(state, "levels_completed", 0) or 0) + 1)
            reset_click_state()   # initialise click state tracker

            while not done and total_steps < args.max_moves:
                # Scope 2: pre-level update on all prior experience
                s2_budget = scope2.ttt_budget(current_level)
                s2_loss   = run_ttt_update(model, optimizer, scope2_buffer, s2_budget, lr=1e-5)

                exp_buf:  list = []
                lvl_steps = 0
                lvl_rew   = 0.0
                lvl_start = current_level

                # ── Play one level ────────────────────────────────────────────
                while not done and total_steps < args.max_moves:
                    available_actions = list(getattr(state, "available_actions", []) or [])
                    legal, _          = legal_actions_for_state(state)
                    if not legal:
                        print(f"  level {lvl_start:02d} | no legal actions — stopping")
                        done = True
                        break

                    cfg.action_space  = legal[:]
                    cfg.legal_actions = legal[:]

                    action_int        = select_action(model, obs, cfg, temperature=1.0)
                    # Denormalize observation for solver pixel logic
                    _obs_arr = (obs[0] * 12.0).round().astype(np.int32) if obs.ndim == 3 else (obs * 12.0).round().astype(np.int32)
                    action_obj, data  = decode_action(
                        action_int, available_actions, obs_arr=_obs_arr,
                        game_model=_game_model, goal_frame=_goal_frame, game_id=game_id)
                    # Wrap env.step in a thread with 60s timeout to prevent indefinite hangs
                    import threading as _threading
                    _step_result = [None]
                    _step_exc    = [None]
                    def _do_step():
                        try:
                            _step_result[0] = env.step(action_obj, data=data) if data else env.step(action_obj)
                        except Exception as _e:
                            _step_exc[0] = _e
                    _t = _threading.Thread(target=_do_step, daemon=True)
                    _t.start()
                    _t.join(timeout=60)
                    if _t.is_alive():
                        print(f"  [timeout] env.step hung for 60s at step {total_steps+1} — ending game")
                        done = True
                        break
                    if _step_exc[0] is not None:
                        raise _step_exc[0]
                    next_state        = _step_result[0]
                    if next_state is None or getattr(next_state, 'frame', None) is None:
                        print(f"  [null-state] env.step returned None at step {total_steps+1} — ending game")
                        done = True
                        break
                    next_obs          = normalize_obs(next_state.frame)
                    reward            = shaped_reward(state, next_state)

                    # Record action effect in game model (online learning)
                    if _GENERIC_SOLVER_OK and _game_model is not None:
                        try:
                            _next_arr = (next_obs[0] * 12.0).round().astype(np.int32)
                            _lv_up = int(getattr(next_state, "levels_completed", 0) or 0) > \
                                     int(getattr(state, "levels_completed", 0) or 0)
                            _eff = compute_action_effect(
                                _obs_arr, _next_arr,
                                action_id=action_obj, click_data=data,
                                reward=float(reward), level_up=_lv_up)
                            _game_model.record_effect(_eff)
                            if _lv_up:
                                _wf_lv = int(getattr(state, "levels_completed", 0) or 0) + 1
                                _game_model.record_win(_next_arr)
                                # Persist this level's win frame for future runs
                                try:
                                    _wf_dir = _ARC3_ROOT / "results" / "win_frames"
                                    _wf_dir.mkdir(parents=True, exist_ok=True)
                                    np.save(_wf_dir / f"{game_id}_lv{_wf_lv}.npy", _next_arr)
                                except Exception:
                                    pass
                                # Goal for the NEXT level: load from disk cache if available.
                                # Don't carry the just-won level's frame into the next level —
                                # that causes infer_click_target to fixate on the old win position
                                # rather than exploring the new level's distinct win condition.
                                _next_lv_path = (_ARC3_ROOT / "results" / "win_frames"
                                                 / f"{game_id}_lv{_wf_lv + 1}.npy")
                                if _next_lv_path.exists():
                                    try:
                                        _goal_frame = np.load(str(_next_lv_path)).astype(np.int32)
                                        print(f"  [win-cache] loaded goal for level {_wf_lv + 1}", flush=True)
                                    except Exception:
                                        _goal_frame = None
                                else:
                                    _goal_frame = None  # explore fresh — no stale cross-level bias
                                _game_model.reset_for_new_level()  # fresh exploration
                        except Exception:
                            pass

                    # Per-step progress (every 5 steps)
                    if total_steps % 5 == 0 or reward > 0.5:
                        elapsed_s = time.time() - game_start
                        print(f"    step {total_steps+1:3d}/{args.max_moves} | "
                              f"lv={lvl_start} | act={action_int:2d} "
                              f"{'(click '+str(data)+')' if data else ''} | "
                              f"r={reward:+.2f} | {elapsed_s:.0f}s elapsed", flush=True)
                        try:
                            (OUT_DIR / "progress.json").write_text(json.dumps({
                                "game": game_id, "step": total_steps + 1,
                                "max_moves": args.max_moves,
                                "level": lvl_start,
                                "last_action": action_int,
                                "last_reward": round(float(reward), 3),
                                "elapsed_s": round(elapsed_s, 1),
                                "levels_completed": len([r for r in level_reports
                                                         if isinstance(r, dict) and r.get("reward", -999) > 0.5]),
                                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                            }, indent=2))
                        except Exception:
                            pass

                    exp_buf.append((obs.astype(np.float32), int(action_int),
                                    float(reward), next_obs.astype(np.float32)))

                    prev_lv  = int(getattr(state,      "levels_completed", 0) or 0)
                    next_lv  = int(getattr(next_state, "levels_completed", 0) or 0)
                    win_lv   = int(getattr(next_state, "win_levels", 0) or 0)
                    lvl_done = next_lv > prev_lv
                    terminal = getattr(next_state, "state", None) in (
                        GameState.GAME_OVER, GameState.WIN)
                    won      = win_lv > 0 and next_lv >= win_lv

                    obs   = next_obs
                    state = next_state
                    total_steps += 1
                    lvl_steps   += 1
                    lvl_rew     += reward

                    if lvl_done:
                        _scope_ns.level_history.append({
                            "step": total_steps, "level": next_lv, "reward": float(lvl_rew)})
                        current_level = next_lv + 1
                        reset_click_state()   # reset block cycling for new level
                        break
                    if terminal or won:
                        lives_used += 1
                        # Fix 3: Keep model informed of life count for per-life position probing
                        if game_model is not None:
                            game_model._lives_used = lives_used
                        if lives_used >= args.n_lives:
                            done = True
                        else:
                            # New life — send RESET action (same as maroon_solver pattern)
                            # env.reset() may return the GAME_OVER frame; GameAction.RESET
                            # sends a server-side reset that returns a clean Level 1 state.
                            print(f"  [life {lives_used}/{args.n_lives}] Game over — sending RESET action")
                            reset_action = GameAction.RESET  # RESET = from_id(0)
                            state  = env.step(reset_action)
                            if state is None or getattr(state, 'frame', None) is None:
                                print(f"  [life-reset] env.step(RESET) returned None — ending game")
                                done = True
                                break
                            obs    = normalize_obs(state.frame)
                            _reset_lv = int(getattr(state, "levels_completed", 0) or 0)
                            current_level = max(1, _reset_lv + 1)
                            reset_click_state()
                            available_actions = list(getattr(state, "available_actions", []) or [])
                            fresh_legal, _ = legal_actions_for_state(state)
                            if fresh_legal:
                                cfg.action_space  = fresh_legal[:]
                                cfg.legal_actions = fresh_legal[:]
                            _mode_after = "_mode unavailable"
                            try:
                                from maroon_solver import _mode as _ms_mode
                                _obs_check = (obs[0] * 12.0).round().astype(np.int32)
                                _mode_after = _ms_mode(_obs_check)
                            except Exception:
                                pass
                            print(f"    post-reset: lv={getattr(state,'levels_completed','?')} "
                                  f"state={getattr(state,'state','?')} mode={_mode_after}")
                        break

                # Scope 1: post-level update on this level's experience
                s1_budget = scope1.ttt_budget(lvl_start)
                s1_loss   = run_ttt_update(model, optimizer, exp_buf, s1_budget, lr=3e-5)
                scope2_buffer.extend(exp_buf)

                # Fix 4: Track actual level transitions, not just reward threshold
                _actually_transitioned = (current_level > lvl_start)
                report = {
                    "level":              lvl_start,
                    "steps":              lvl_steps,
                    "reward":             round(float(lvl_rew), 3),
                    "level_transitioned": _actually_transitioned,
                    "scope2_budget":      s2_budget,
                    "scope2_loss":        round(float(s2_loss), 4),
                    "scope1_budget":      s1_budget,
                    "scope1_loss":        round(float(s1_loss), 4),
                    "buffer_size":        len(scope2_buffer),
                }
                level_reports.append(report)
                print(f"  level {lvl_start:02d} | steps={lvl_steps:2d} | "
                      f"reward={lvl_rew:6.2f} | "
                      f"s2={s2_budget}steps/loss={s2_loss:.3f} | "
                      f"s1={s1_budget}steps/loss={s1_loss:.3f}")

                # Incremental checkpoint after each life boundary so TTT learning
                # is never lost to a mid-game hang or API timeout.
                try:
                    _ckpt_tmp = OUT_DIR / f"{game_id}_stage3_ttt.checkpoint"
                    with open(_ckpt_tmp, "wb") as _cf:
                        pickle.dump({
                            "weights":    model.get_weights(),
                            "config":     cfg,
                            "stage":      3,
                            "game_id":    game_id,
                            "dry_run":    DRY_RUN,
                            "stage2_ckpt": str(ckpt_path),
                            "reports":    level_reports,
                            "timestamp":  TS,
                        }, _cf)
                except Exception as _ce:
                    print(f"  [ckpt-warn] incremental save failed: {_ce}")

                if done:
                    break

        except Exception as exc:
            import traceback
            print(f"\n  ERROR in {game_id}: {exc}")
            traceback.print_exc()
            level_reports.append({"error": repr(exc)})

    # ── Save per-game Stage 3 checkpoint ──────────────────────────────────────
    ckpt_out = OUT_DIR / f"{game_id}_stage3_ttt.checkpoint"
    with open(ckpt_out, "wb") as f:
        pickle.dump({
            "weights":    model.get_weights(),
            "config":     cfg,
            "stage":      3,
            "game_id":    game_id,
            "dry_run":    DRY_RUN,
            "stage2_ckpt": str(ckpt_path),
            "reports":    level_reports,
            "timestamp":  TS,
        }, f)
    print(f"\n  Checkpoint saved: {ckpt_out}")

    elapsed = time.time() - game_start
    # Fix 4: Count actual level transitions (next_lv > prev_lv events) instead of
    # relying on cumulative reward > 0.5, which misses levels where the step penalty
    # (-0.01 × N steps) outweighs the +0.99 completion reward.
    levels_by_reward = sum(1 for r in level_reports
                           if isinstance(r, dict) and r.get("reward", -999) > 0.5)
    levels_by_transition = sum(1 for r in level_reports
                               if isinstance(r, dict) and r.get("level_transitioned", False))
    levels_completed = max(levels_by_reward, levels_by_transition)
    summary["games"][game_id] = {
        "levels_completed": levels_completed,
        "levels_by_reward": levels_by_reward,
        "levels_by_transition": levels_by_transition,
        "total_steps":      sum(r.get("steps", 0) for r in level_reports if isinstance(r, dict)),
        "elapsed_s":        round(elapsed, 1),
        "reports":          level_reports,
        "checkpoint":       str(ckpt_out),
    }

# ── Final summary ─────────────────────────────────────────────────────────────
summary_path = OUT_DIR / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2))

print(f"\n{'='*60}")
print(f"  TTT COMPLETE")
print(f"{'='*60}")
for gid, info in summary["games"].items():
    print(f"  {gid}: {info['levels_completed']} levels completed "
          f"in {info['total_steps']} steps ({info['elapsed_s']}s)")
print(f"\n  Summary  : {summary_path}")
print(f"  Out dir  : {OUT_DIR}")
print(f"{'='*60}\n")
