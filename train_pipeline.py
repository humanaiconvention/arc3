"""
ARC-AGI-3 MuZero full training pipeline (Stages 1→2→3-prep).
Run on any GPU machine:

  pip install torch torchvision ray[default] Pillow numpy tensorboard
  git clone https://github.com/werner-duvaud/muzero-general.git
  python train_pipeline.py \
      --atari-dir /path/to/atari-head \
      --arc3-dir  /path/to/recordings \
      --muzero-dir ./muzero-general \
      [--stage1-steps 50000] [--stage2-steps 20000] [--batch 128]
"""
import argparse, copy, datetime, pathlib, pickle, random, sys, time
import numpy as np
import torch

# ── CLI args ─────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--atari-dir",   required=True,  help="Root dir of Atari-HEAD data")
p.add_argument("--arc3-dir",    required=True,  help="Dir of ARC3 recording JSONLs")
p.add_argument("--muzero-dir",  default="./muzero-general")
p.add_argument("--out-dir",     default="./results")
p.add_argument("--stage1-steps", type=int, default=50_000)
p.add_argument("--stage2-steps", type=int, default=20_000)
p.add_argument("--batch",        type=int, default=128)
p.add_argument("--max-trials",   type=int, default=2,
               help="Atari trials per game (keep low to fit in RAM)")
p.add_argument("--max-steps",    type=int, default=500,
               help="Steps per trial (random sample)")
p.add_argument("--skip-stage1",  action="store_true")
p.add_argument("--stage1-ckpt",  default=None,
               help="Skip Stage 1 and load this checkpoint instead")
args = p.parse_args()

MUZERO_DIR = pathlib.Path(args.muzero_dir).resolve()
OUT_DIR    = pathlib.Path(args.out_dir).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

for d in [MUZERO_DIR, pathlib.Path(__file__).parent / "neurosym"]:
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

import ray
from arc3_game import MuZeroConfig
from data_pipeline import ARC3Dataset, AtariHeadDataset, apply_domain_upweight
import models, trainer as _trainer, shared_storage, replay_buffer as _replay
import self_play as _self_play

MZHistory = _self_play.GameHistory

def to_mz(h):
    mz = MZHistory()
    mz.observation_history = h.observation_history
    mz.action_history      = h.action_history
    mz.reward_history      = h.reward_history
    mz.to_play_history     = h.to_play_history
    mz.child_visits        = h.child_visits
    mz.root_values         = h.root_values
    mz.priorities          = getattr(h, "priorities", None)
    mz.game_priority       = getattr(h, "game_priority", None)
    return mz

def make_checkpoint(weights=None):
    return dict(weights=weights, optimizer_state=None,
                total_reward=0, muzero_reward=0, opponent_reward=0,
                episode_length=0, mean_value=0, training_step=0,
                lr=0, total_loss=0, value_loss=0, reward_loss=0,
                policy_loss=0, num_played_games=0, num_played_steps=0,
                num_reanalysed_games=0, terminate=False)

def train(config, seed_histories, label="stage"):
    """Run continuous_update_weights until training_steps, return final weights."""
    print(f"\n{'='*60}")
    print(f"  {label}: {config.training_steps:,} steps | "
          f"batch={config.batch_size} | lr={config.lr_init}")
    print(f"  seed histories: {len(seed_histories)}")
    print(f"{'='*60}")

    if not ray.is_initialized():
        ray.init(num_gpus=torch.cuda.device_count(), ignore_reinit_error=True)

    ckpt = make_checkpoint()
    net  = models.MuZeroNetwork(config)
    ckpt["weights"] = copy.deepcopy(net.get_weights())

    buf = {i: to_mz(h) for i, h in
           enumerate(seed_histories[:config.replay_buffer_size])}
    ckpt["num_played_games"] = len(buf)
    ckpt["num_played_steps"] = sum(len(h.action_history) for h in seed_histories[:len(buf)])

    storage_actor = shared_storage.SharedStorage.remote(ckpt, config)
    replay_actor  = _replay.ReplayBuffer.remote(ckpt, buf, config)
    trainer_actor = _trainer.Trainer.remote(ckpt, config)

    t0 = time.time()
    train_future = trainer_actor.continuous_update_weights.remote(
        replay_actor, storage_actor)

    last_step = -1
    while True:
        time.sleep(20)
        try:
            step = ray.get(storage_actor.get_info.remote("training_step"))
        except Exception:
            step = last_step
        if step != last_step:
            try:
                loss = ray.get(storage_actor.get_info.remote("total_loss"))
                elapsed = (time.time() - t0) / 60
                print(f"  step {step:6,}/{config.training_steps:,} "
                      f"| loss={loss:.4f} | {elapsed:.1f}m")
            except Exception:
                pass
            last_step = step
        if step >= config.training_steps:
            break
        try:
            if ray.get(storage_actor.get_info.remote("terminate")):
                break
        except Exception:
            pass

    ray.get(train_future)
    weights = ray.get(storage_actor.get_info.remote("weights"))
    print(f"  {label} complete. {(time.time()-t0)/60:.1f}m total.")
    return weights


# ── STAGE 1: broad pre-training ───────────────────────────────────────────────
ts = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
stage1_path = OUT_DIR / f"stage1_pretrain_{ts}.checkpoint"

if args.stage1_ckpt:
    print(f"Loading Stage 1 from {args.stage1_ckpt}")
    with open(args.stage1_ckpt, "rb") as f:
        s1 = pickle.load(f)
    stage1_weights = s1["weights"]
    stage1_config  = s1["config"]
else:
    print("\n─── Loading data ────────────────────────────────────────")
    arc3_ds   = ARC3Dataset(args.arc3_dir)
    arc3_hist = arc3_ds.load_all(verbose=True)

    atari_ds   = AtariHeadDataset(args.atari_dir,
                                  max_trials_per_game=args.max_trials,
                                  max_steps_per_trial=args.max_steps)
    atari_hist = atari_ds.load_all(verbose=True)

    print(f"\nRaw counts: {len(atari_hist)} Atari, {len(arc3_hist)} ARC3")

    all_hist = atari_hist + apply_domain_upweight(arc3_hist, upweight=3.0)
    random.shuffle(all_hist)
    print(f"Mixed corpus: {len(all_hist)} histories")

    config1 = MuZeroConfig(game_id="pretrain", n_bins=4)
    config1.action_space        = list(range(18))
    config1.observation_shape   = (1, 64, 64)
    config1.training_steps      = args.stage1_steps
    config1.batch_size          = args.batch
    config1.replay_buffer_size  = 5_000
    config1.num_unroll_steps    = 5
    config1.td_steps            = 20
    config1.lr_init             = 3e-4
    config1.lr_decay_rate       = 0.9
    config1.lr_decay_steps      = 10_000
    config1.blocks              = 3
    config1.channels            = 64
    config1.train_on_gpu        = torch.cuda.is_available()
    config1.ratio               = None  # disable self-play ratio gate (offline training)
    config1.num_workers         = 1
    config1.results_path        = OUT_DIR / "stage1" / ts
    config1.results_path.mkdir(parents=True, exist_ok=True)

    stage1_weights = train(config1, all_hist, label="Stage 1 (pretrain)")
    stage1_config  = config1

    with open(stage1_path, "wb") as f:
        pickle.dump(dict(weights=stage1_weights, config=stage1_config,
                         training_steps=config1.training_steps, stage=1), f)
    print(f"Stage 1 checkpoint: {stage1_path}")


# ── STAGE 2: domain-adaptive fine-tuning ─────────────────────────────────────
print("\n─── Stage 2: fine-tuning on ARC3 recordings ─────────────────")
arc3_ds   = ARC3Dataset(args.arc3_dir)
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
config2.observation_shape   = stage1_config.observation_shape
config2.network             = stage1_config.network
config2.downsample          = stage1_config.downsample
config2.blocks              = stage1_config.blocks
config2.channels            = stage1_config.channels
config2.training_steps      = args.stage2_steps
config2.batch_size          = 64
config2.replay_buffer_size  = 2_000
config2.num_unroll_steps    = 5
config2.td_steps            = 10
config2.lr_init             = 3e-5
config2.lr_decay_rate       = 0.95
config2.lr_decay_steps      = 5_000
config2.train_on_gpu        = torch.cuda.is_available()
config2.ratio               = None  # offline training
config2.num_workers         = 1
config2.results_path        = OUT_DIR / "stage2" / ts
config2.results_path.mkdir(parents=True, exist_ok=True)

stage2_weights = train(config2, train_aug, label="Stage 2 (finetune)")

stage2_path = OUT_DIR / f"stage2_finetune_{ts}.checkpoint"
with open(stage2_path, "wb") as f:
    pickle.dump(dict(weights=stage2_weights, config=config2,
                     training_steps=config2.training_steps, stage=2,
                     stage1_ckpt=str(stage1_path)), f)
print(f"Stage 2 checkpoint: {stage2_path}")
print("\nDone. Stage 3 (TTT) runs live against the ARC-AGI-3 environment.")
print(f"Feed {stage2_path} to arc3_muzero_ttt.ipynb or the Kaggle TTT kernel.")

# ─────────────────────────────────────────────────────────────────────────────
# NOTE: Pre-extract Atari-HEAD for fast PNG reads (avoids bz2 decompression):
#
#   cd /data && mkdir -p atari-head-extracted
#   for z in atari-head/*.zip; do
#       game=$(basename "$z" .zip)
#       unzip -o "$z" -d atari-head-extracted/
#       for tbz in atari-head-extracted/$game/*.tar.bz2; do
#           trial=$(basename "$tbz" .tar.bz2)
#           mkdir -p "atari-head-extracted/$game/$trial"
#           tar xjf "$tbz" -C "atari-head-extracted/$game/$trial/"
#       done
#   done
#
# Then pass --atari-dir /data/atari-head-extracted (PNG layout, loads in <1 min).
# ─────────────────────────────────────────────────────────────────────────────
