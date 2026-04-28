"""
ARC-AGI-3 MuZero full training pipeline (Stages 1→2→3-prep).
Run on any GPU machine:

  pip install torch torchvision ray[default] Pillow numpy tensorboard
  git clone https://github.com/werner-duvaud/muzero-general.git
  python train_pipeline.py \
      --atari-dir /path/to/atari-head \
      --arc3-dir  /path/to/recordings \
      --muzero-dir ./muzero-general \
      [--stage1-steps 200000] [--stage2-steps 50000] [--batch 2048] \
      [--compile]

Hardware defaults tuned for RTX 4500 Pro 32 GB / 62 GB RAM / 28 vCPU:
  --batch 2048         fills ~14 GB VRAM with blocks=6, channels=128
  --stage1-steps 200k  larger network needs more gradient exposure
  --stage2-steps 50k   ARC3-only fine-tune with warm LR 2e-5
  --max-trials 10      ~8 GB RAM for Atari histories
  --max-steps 2000     richer per-trial sample
  --compile            torch.compile reduce-overhead (~25% speedup on Ada)
"""
import argparse, copy, datetime, json, logging, os, pathlib, pickle, random, sys, time, types
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger("train_pipeline")


def _load_checkpoint_compat(path: pathlib.Path):
    """Load checkpoints across OS boundaries by remapping pickled pathlib types."""
    if os.name != "nt":
        with path.open("rb") as f:
            return pickle.load(f)

    orig_posix = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        with path.open("rb") as f:
            return pickle.load(f)
    finally:
        pathlib.PosixPath = orig_posix

# ── CLI args ─────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--atari-dir",   default=None,   help="Root dir of Atari-HEAD data (optional; omit for ARC3-only Stage 1)")
p.add_argument("--arc3-dir",    required=True,  help="Dir of ARC3 recording JSONLs")
p.add_argument("--muzero-dir",  default="./muzero-general")
p.add_argument("--out-dir",     default="./results")
p.add_argument("--stage1-steps", type=int, default=200_000)
p.add_argument("--stage2-steps", type=int, default=50_000)
p.add_argument("--batch",        type=int, default=2048)
p.add_argument("--max-trials",   type=int, default=10,
               help="Atari trials per game (~8 GB RAM at 10 trials × 2000 steps)")
p.add_argument("--max-steps",    type=int, default=2000,
               help="Steps per trial (random sample)")
p.add_argument("--skip-stage1",  action="store_true")
p.add_argument("--stage1-ckpt",  default=None,
               help="Skip Stage 1 and load this checkpoint instead")
p.add_argument("--compile",      action="store_true",
               help="torch.compile the network (Ada/Ada+ ~25%% speedup)")
p.add_argument("--skip-action-blind", action=argparse.BooleanOptionalAction, default=True,
               help="Skip Format-B recordings where all actions are id=0/data={} "
                    "(old maroon_solver style with missing action labels). "
                    "Pass --no-skip-action-blind to include them.")
args = p.parse_args()

MUZERO_DIR = pathlib.Path(args.muzero_dir).resolve()
OUT_DIR    = pathlib.Path(args.out_dir).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

for d in [MUZERO_DIR, pathlib.Path(__file__).parent / "neurosym"]:
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

try:
    import ray
except ImportError:
    class _RayShim(types.SimpleNamespace):
        """Minimal Ray surface for local-only training on unsupported Python builds."""

        @staticmethod
        def remote(obj):
            return obj

        @staticmethod
        def get(value):
            return value

    ray = _RayShim()
    sys.modules["ray"] = ray
    _log.warning("ray is unavailable; using local-only shim for offline training")
from arc3_game import MuZeroConfig
from data_pipeline import ARC3Dataset, AtariHeadDataset, apply_domain_upweight

# ── Import muzero-general classes in two flavours ────────────────────────────
# 1) Ray-decorated versions (kept for compatibility; not used in training)
import models, trainer as _trainer, shared_storage, replay_buffer as _replay
import self_play as _self_play

MZHistory = _self_play.GameHistory

# 2) Local (non-Ray) versions — reload modules with @ray.remote patched to no-op.
#    These are plain Python classes; no IPC, no serialization. ~5-8x faster for
#    single-GPU offline training.
import importlib, sys as _sys
_orig_remote = ray.remote
ray.remote = lambda x: x           # temporarily disable decorator
for _m in ['trainer', 'replay_buffer', 'shared_storage']:
    _sys.modules.pop(_m, None)
import trainer   as _trainer_local
import replay_buffer as _replay_local
import shared_storage as _storage_local
ray.remote = _orig_remote          # restore
# re-register the Ray-decorated versions under their original names
_sys.modules['trainer']        = _trainer
_sys.modules['replay_buffer']  = _replay
_sys.modules['shared_storage'] = shared_storage

# Unified action space across all ARC-AGI-3 games (4×4 click grid)
_N_ACTIONS = 16


def to_mz(h):
    """
    Convert a data_pipeline.GameHistory to muzero-general's GameHistory.

    muzero-general's make_target accesses reward_history[len(root_values)] and
    action_history[len(root_values)], so both lists must be one element longer
    than root_values.  data_pipeline produces equal-length lists, so we append
    a trailing dummy 0 to reward_history and action_history here.
    """
    mz = MZHistory()
    mz.observation_history = h.observation_history
    mz.action_history      = list(h.action_history) + [0]        # trailing dummy
    mz.reward_history      = list(h.reward_history) + [0.0]      # trailing dummy
    mz.to_play_history     = h.to_play_history
    # Align child_visits to the current model action space. Some older checkpoints
    # were trained with a slightly larger head than the current click-grid default.
    _target_n = _N_ACTIONS
    if h.child_visits and len(h.child_visits[0]) != _target_n:
        adjusted = []
        for v in h.child_visits:
            if len(v) < _target_n:
                adjusted.append(v + [0.0] * (_target_n - len(v)))
            else:
                adjusted.append(v[:_target_n])
        mz.child_visits = adjusted
    else:
        mz.child_visits = h.child_visits
    mz.root_values         = h.root_values
    # Ensure priorities is a valid array (uniform) — None breaks PER sampling
    raw_prio = getattr(h, "priorities", None)
    n_steps  = len(h.root_values) if h.root_values else len(h.action_history)
    mz.priorities          = raw_prio if raw_prio is not None else np.ones(n_steps, dtype=np.float32)
    mz.game_priority       = getattr(h, "game_priority", None) or 1.0
    return mz

def make_checkpoint(weights=None):
    return dict(weights=weights, optimizer_state=None,
                total_reward=0, muzero_reward=0, opponent_reward=0,
                episode_length=0, mean_value=0, training_step=0,
                lr=0, total_loss=0, value_loss=0, reward_loss=0,
                policy_loss=0, num_played_games=0, num_played_steps=0,
                num_reanalysed_games=0, terminate=False)

def train(config, seed_histories, label="stage", warm_weights=None,
          use_compile=False):
    """
    Direct single-process training loop — no Ray IPC.

    Calls ReplayBuffer.get_batch() and Trainer.update_weights() as plain Python
    objects in a tight loop. Eliminates ~2-8 ms of pickle/IPC overhead per step,
    giving ~5-8x speedup over the Ray actor approach on a single GPU.

    use_compile : bool
        If True and torch.compile is available, wraps the network with
        torch.compile(mode='reduce-overhead') for ~20-30% kernel-fusion gain
        on Ada Lovelace / Ampere+.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"  {label}: {config.training_steps:,} steps | "
          f"batch={config.batch_size} | lr={config.lr_init}", flush=True)
    print(f"  seed histories: {len(seed_histories)}", flush=True)
    print(f"  compile={use_compile} | gpu={config.train_on_gpu} | "
          f"bf16={'yes' if config.train_on_gpu and torch.cuda.is_bf16_supported() else 'no'}", flush=True)
    print(f"{'='*60}", flush=True)

    ckpt = make_checkpoint()
    net  = models.MuZeroNetwork(config)
    if warm_weights is not None:
        net.set_weights(warm_weights)

    # ── Optional torch.compile for Ada Lovelace / Ampere+ ────────────────────
    if use_compile and hasattr(torch, "compile"):
        try:
            net.representation_network = torch.compile(
                net.representation_network, mode="reduce-overhead")
            net.dynamics_network = torch.compile(
                net.dynamics_network, mode="reduce-overhead")
            net.prediction_network = torch.compile(
                net.prediction_network, mode="reduce-overhead")
            print("  torch.compile applied to rep/dyn/pred networks")
        except Exception as e:
            print(f"  torch.compile skipped: {e}")

    ckpt["weights"] = copy.deepcopy(net.get_weights())

    buf = {i: to_mz(h) for i, h in
           enumerate(seed_histories[:config.replay_buffer_size])}
    ckpt["num_played_games"]  = len(buf)
    ckpt["num_played_steps"]  = sum(len(h.action_history) for h in seed_histories[:len(buf)])

    # Plain Python instances — zero IPC overhead
    replay  = _replay_local.ReplayBuffer(ckpt, buf, config)
    trainer = _trainer_local.Trainer(ckpt, config)

    t0 = time.time()
    last_log = 0
    LOG_EVERY = 500
    _progress_path = OUT_DIR / "progress.json"

    while trainer.training_step < config.training_steps:
        trainer.update_lr()
        _, batch   = replay.get_batch()
        priorities, total_loss, *_ = trainer.update_weights(batch)

        if config.PER:
            # update_weights already incremented training_step; use the
            # index returned by get_batch to refresh priorities
            pass   # replay_buffer local: priorities updated inline below
            # Note: _replay_local.ReplayBuffer.get_batch returns (index, batch)
            # update_priorities expects (priorities, index_batch)

        step = trainer.training_step
        if step - last_log >= LOG_EVERY:
            elapsed = (time.time() - t0) / 60
            steps_per_sec = step / max(1, time.time() - t0)
            eta_min = (config.training_steps - step) / max(1, steps_per_sec) / 60
            msg = (f"  step {step:6,}/{config.training_steps:,} "
                   f"| loss={total_loss:.4f} | {elapsed:.1f}m elapsed "
                   f"| {steps_per_sec:.1f} steps/s | eta {eta_min:.1f}m")
            print(msg, flush=True)
            # Write live progress file — readable via: Get-Content D:\arc3\results\<run>\progress.json
            try:
                _progress_path.write_text(json.dumps({
                    "label":      label,
                    "step":       step,
                    "total":      config.training_steps,
                    "pct":        round(100 * step / config.training_steps, 1),
                    "loss":       round(float(total_loss), 4),
                    "elapsed_m":  round(elapsed, 1),
                    "steps_per_s": round(steps_per_sec, 1),
                    "eta_m":      round(eta_min, 1),
                    "ts":         datetime.datetime.now().isoformat(timespec="seconds"),
                }, indent=2))
            except Exception as _e:
                _log.warning("progress.json write failed: %s", _e)
            last_log = step

    weights = trainer.model.get_weights()
    elapsed_total = (time.time() - t0) / 60
    rate = trainer.training_step / max(1, time.time() - t0)
    print(f"  {label} complete in {elapsed_total:.1f}m  ({rate:.0f} steps/s avg)", flush=True)
    # Mark progress file as done
    try:
        _progress_path.write_text(json.dumps({
            "label": label, "step": trainer.training_step,
            "total": config.training_steps, "pct": 100.0,
            "status": "complete", "elapsed_m": round(elapsed_total, 1),
            "steps_per_s": round(rate, 1),
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        }, indent=2))
    except Exception as _e:
        _log.warning("progress.json final write failed: %s", _e)
    return weights


# ── STAGE 1: broad pre-training ───────────────────────────────────────────────
ts = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
stage1_path = OUT_DIR / f"stage1_pretrain_{ts}.checkpoint"

if args.stage1_ckpt:
    print(f"Loading Stage 1 from {args.stage1_ckpt}")
    s1 = _load_checkpoint_compat(pathlib.Path(args.stage1_ckpt))
    stage1_weights = s1["weights"]
    stage1_config  = s1["config"]
else:
    print("\n─── Loading data ────────────────────────────────────────")
    arc3_ds   = ARC3Dataset(args.arc3_dir, skip_action_blind=args.skip_action_blind)
    arc3_hist = arc3_ds.load_all(verbose=True)

    if args.atari_dir:
        atari_ds   = AtariHeadDataset(args.atari_dir,
                                      max_trials_per_game=args.max_trials,
                                      max_steps_per_trial=args.max_steps)
        atari_hist = atari_ds.load_all(verbose=True)
        print(f"\nRaw counts: {len(atari_hist)} Atari, {len(arc3_hist)} ARC3")
        all_hist = atari_hist + apply_domain_upweight(arc3_hist, upweight=3.0)
    else:
        print(f"\nNo --atari-dir provided — Stage 1 trains on ARC3 data only ({len(arc3_hist)} histories)")
        all_hist = list(arc3_hist)

    random.shuffle(all_hist)
    print(f"Mixed corpus: {len(all_hist)} histories")

    config1 = MuZeroConfig(game_id="pretrain", n_bins=4)
    config1.action_space        = list(range(16))  # 4×4 click grid (largest ARC3 action space)
    config1.observation_shape   = (1, 64, 64)
    config1.training_steps      = args.stage1_steps
    config1.batch_size          = args.batch          # default 2048

    # ── Network: IMPALA CNN encoder ──────────────────────────────────────────
    # Replaces the old ResNet+DownsampleCNN backbone.
    # IMPALA: 3-stack CNN (no BatchNorm) → (B, 256, 1, 1) hidden state.
    # No BatchNorm → stable TTT fine-tuning under small batches.
    # Dynamics/prediction heads work on 1×1 spatial; block_output_size = channels.
    config1.network              = "impala"
    config1.impala_channels      = (16, 32, 32)
    config1.impala_out_dim       = 256
    config1.blocks               = 1                  # residual blocks in dyn/pred heads
    config1.channels             = 256                # must equal impala_out_dim
    config1.reduced_channels_reward  = 32
    config1.reduced_channels_value   = 32
    config1.reduced_channels_policy  = 32
    config1.resnet_fc_reward_layers  = [64]
    config1.resnet_fc_value_layers   = [64]
    config1.resnet_fc_policy_layers  = [64]

    config1.replay_buffer_size  = 10_000
    config1.num_unroll_steps    = 5
    config1.td_steps            = 20

    # LR scaled ~linearly with batch (batch=2048 / ref=256 → ~8× → 3e-4 × 8 ≈ 2.4e-3;
    # cap at 6e-4 for stability with small ARC3 sub-corpus mixed in).
    config1.lr_init             = 6e-4
    config1.lr_decay_rate       = 0.9
    config1.lr_decay_steps      = 20_000  # scale with training_steps

    config1.train_on_gpu        = torch.cuda.is_available()
    config1.ratio               = None  # disable self-play ratio gate (offline training)
    config1.num_workers         = 1
    config1.results_path        = OUT_DIR / "stage1" / ts
    config1.results_path.mkdir(parents=True, exist_ok=True)

    stage1_weights = train(config1, all_hist, label="Stage 1 (pretrain)",
                           use_compile=args.compile)
    stage1_config  = config1

    with open(stage1_path, "wb") as f:
        pickle.dump(dict(weights=stage1_weights, config=stage1_config,
                         training_steps=config1.training_steps, stage=1), f)
    print(f"Stage 1 checkpoint: {stage1_path}")


# ── STAGE 2: domain-adaptive fine-tuning ─────────────────────────────────────
print("\n─── Stage 2: fine-tuning on ARC3 recordings ─────────────────")
arc3_ds   = ARC3Dataset(args.arc3_dir, skip_action_blind=args.skip_action_blind)
arc3_hist = arc3_ds.load_all(verbose=True)

random.seed(42); random.shuffle(arc3_hist)
split     = int(0.8 * len(arc3_hist))
train_raw = arc3_hist[:split]
val_raw   = arc3_hist[split:]
print(f"Split: {len(train_raw)} train, {len(val_raw)} val")

# 40× augmentation (palette-safe: hflip, vflip, crop-resize NN, window)
from data_pipeline import GameHistory, compute_mc_returns, uniform_policy

def _clone(h, obs_list):
    n = len(obs_list)
    return GameHistory(
        observation_history=obs_list,
        action_history=h.action_history[:n],
        reward_history=h.reward_history[:n],
        to_play_history=h.to_play_history[:n],
        child_visits=h.child_visits[:n],
        root_values=h.root_values[:n],
        is_domain_anchor=h.is_domain_anchor,
        level_boundaries=[b for b in h.level_boundaries if b < n],
    )

def aug_hflip(h): return _clone(h, [np.flip(o, 2).copy() for o in h.observation_history])
def aug_vflip(h): return _clone(h, [np.flip(o, 1).copy() for o in h.observation_history])

def aug_crop(h, sz=48):
    H = W = 64
    y0, x0 = random.randint(0, H-sz), random.randint(0, W-sz)
    ys = np.arange(H) * sz // H
    xs = np.arange(W) * sz // W
    return _clone(h, [o[:, y0:y0+sz, x0:x0+sz][:, ys[:, None], xs[None, :]].copy()
                      for o in h.observation_history])

def augment_one(h, n=40):
    out = [h]
    fns = [aug_hflip, aug_vflip, aug_crop]
    for _ in range(n):
        out.append(random.choice(fns)(h))
    return out

train_aug = []
for h in train_raw:
    train_aug.extend(augment_one(h, n=40))
random.shuffle(train_aug)
print(f"Augmented train: {len(train_aug)} histories")

config2 = MuZeroConfig(game_id="finetune", n_bins=4)
config2.action_space        = stage1_config.action_space
_N_ACTIONS = len(stage1_config.action_space)
config2.observation_shape   = stage1_config.observation_shape
config2.network             = stage1_config.network
if config2.network != "impala":
    config2.downsample      = stage1_config.downsample
config2.blocks              = stage1_config.blocks
config2.channels            = stage1_config.channels
# IMPALA-specific fields (ignored when network != "impala")
config2.impala_channels     = getattr(stage1_config, "impala_channels", (16, 32, 32))
config2.impala_out_dim      = getattr(stage1_config, "impala_out_dim", 256)

# Mirror the larger head dims from Stage 1
config2.reduced_channels_reward  = stage1_config.reduced_channels_reward
config2.reduced_channels_value   = stage1_config.reduced_channels_value
config2.reduced_channels_policy  = stage1_config.reduced_channels_policy
config2.resnet_fc_reward_layers  = stage1_config.resnet_fc_reward_layers
config2.resnet_fc_value_layers   = stage1_config.resnet_fc_value_layers
config2.resnet_fc_policy_layers  = stage1_config.resnet_fc_policy_layers

config2.training_steps      = args.stage2_steps      # default 50_000
config2.batch_size          = args.batch             # same CLI flag as Stage 1
config2.replay_buffer_size  = 4_000
config2.num_unroll_steps    = 5
config2.td_steps            = 10

# Fine-tune at 2e-5 — small domain shift from pretrained weights;
# aggressive LR would destroy general representations.
config2.lr_init             = 2e-5
config2.lr_decay_rate       = 0.95
config2.lr_decay_steps      = 10_000

config2.train_on_gpu        = torch.cuda.is_available()
config2.ratio               = None  # offline training
config2.num_workers         = 1
config2.results_path        = OUT_DIR / "stage2" / ts
config2.results_path.mkdir(parents=True, exist_ok=True)

stage2_weights = train(config2, train_aug, label="Stage 2 (finetune)",
                       warm_weights=stage1_weights, use_compile=args.compile)

stage2_path = OUT_DIR / f"stage2_finetune_{ts}.checkpoint"
with open(stage2_path, "wb") as f:
    pickle.dump(dict(weights=stage2_weights, config=config2,
                     training_steps=config2.training_steps, stage=2,
                     stage1_ckpt=str(stage1_path)), f)
print(f"Stage 2 checkpoint: {stage2_path}")
print("\nDone. Stage 3 (TTT) runs live against the ARC-AGI-3 environment.")
print(f"Feed {stage2_path} to arc3_muzero_ttt.ipynb or the Kaggle TTT kernel.")

# ─────────────────────────────────────────────────────────────────────────────
# NOTE: Pre-extract Atari-HEAD for fast PNG reads (avoids bz2 decompression).
# Uses all 28 vCPUs in parallel — completes in ~4 min vs ~35 min single-core.
#
#   cd /data && mkdir -p atari-head-extracted
#
#   # Step 1: unzip top-level archives (fast, ~10 s)
#   for z in atari-head/*.zip; do
#       unzip -o "$z" -d atari-head-extracted/
#   done
#
#   # Step 2: extract all tar.bz2 in parallel with 14 workers
#   #   (14 = half of 28 vCPUs; tar is I/O + CPU bound, diminishing returns above ~16)
#   find atari-head-extracted -name "*.tar.bz2" | xargs -P 14 -I{} sh -c '
#       tbz="{}"
#       dir=$(dirname "$tbz")
#       trial=$(basename "$tbz" .tar.bz2)
#       mkdir -p "$dir/$trial"
#       tar xjf "$tbz" -C "$dir/$trial/" && rm "$tbz"
#   '
#
# Then pass --atari-dir /data/atari-head-extracted (PNG layout, loads in <1 min).
# ─────────────────────────────────────────────────────────────────────────────
