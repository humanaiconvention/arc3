"""cem_self_play — Cross-Entropy Method self-play on a single ARC3 game.

Loads the v1 CNN policy as initialization, runs N iterations of:
  1. Sample BATCH episodes by acting under current policy + temperature
  2. Score episodes by total shaped reward
  3. Pick top ELITE_FRAC episodes
  4. Fine-tune CNN to maximize log-prob of those (state, action) pairs
  5. Repeat

Usage:
  python -u scripts/meta/rl/cem_self_play.py --game re86 --iterations 4 --batch 6 --episode-steps 25

Cost: iterations × batch × episode_steps × ~0.3s/step + reset overhead.
For default 4 × 6 × 25 = 600 actions × 0.3 ≈ 3 min API.

Saves trained model to D:/diloco_lab/state/arc3_policy_v2_cem/<game>/ and a
metrics.json with per-iteration reward statistics + win flags.

This is a tractable test of "does RL self-play surface ANY learning signal in
reasonable budget?" — not an attempt to generally solve ARC3 by RL.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Path setup so this script finds both the meta package and diloco_lab.
_HERE = Path(__file__).resolve()
_ARC3_SCRIPTS = _HERE.parents[2]   # D:/arc3/scripts
sys.path.insert(0, str(_ARC3_SCRIPTS))
sys.path.insert(0, "D:/diloco_lab")

from meta.common import GameSession  # noqa: E402  (touch first to load env)
from meta.characterize import characterize  # noqa: E402
from diloco_lab.arc3_env import ARC3Env  # noqa: E402
from diloco_lab.arc3_policy_features import build_features  # noqa: E402
from diloco_lab.arc3_frame_features import (  # noqa: E402
    DEFAULT_CHANNELS, FRAME_HW, preprocess_frame,
)


_V1_DIR = Path("D:/diloco_lab/state/arc3_policy_v1_cnn")


class TinyPolicyCNN(nn.Module):
    """Mirror of the v1 architecture; must match exactly to load state."""

    def __init__(self, n_classes: int, n_struct: int, n_channels: int = DEFAULT_CHANNELS):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.dropout_conv = nn.Dropout2d(0.4)
        self.fc_frame = nn.Linear(32 * 8 * 8, 64)
        self.fc_struct = nn.Linear(n_struct, 32)
        self.dropout_fc = nn.Dropout(0.5)
        self.head = nn.Linear(64 + 32, n_classes)

    def forward(self, frame, struct):
        x = F.relu(self.conv1(frame)); x = self.pool(x)
        x = F.relu(self.conv2(x));     x = self.pool(x)
        x = F.relu(self.conv3(x));     x = self.pool(x)
        x = self.dropout_conv(x)
        x = x.flatten(1)
        x = F.relu(self.fc_frame(x))
        s = F.relu(self.fc_struct(struct))
        z = torch.cat([x, s], dim=1)
        z = self.dropout_fc(z)
        return self.head(z)


def load_v1_model() -> tuple[TinyPolicyCNN, dict, list[str], list[str]]:
    config = json.loads((_V1_DIR / "config.json").read_text(encoding="utf-8"))
    schema = json.loads((_V1_DIR / "feature_schema.json").read_text(encoding="utf-8"))
    label_map = json.loads((_V1_DIR / "label_map.json").read_text(encoding="utf-8"))
    classes = label_map["classes"]
    model = TinyPolicyCNN(
        n_classes=int(config["n_classes"]),
        n_struct=int(config["n_struct_features"]),
        n_channels=int(config.get("n_channels", DEFAULT_CHANNELS)),
    )
    state = torch.load(_V1_DIR / "model_state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model, config, classes, schema["feature_names"]


def play_episode(env: ARC3Env, model: TinyPolicyCNN, profile_dict: dict,
                 schema_names: list[str], classes: list[str], max_steps: int,
                 click_xy: tuple, temperature: float, rng: random.Random
                 ) -> dict:
    """Roll one episode under the current policy. Returns trajectory + metadata."""
    obs = env.reset()
    if obs.frame is None or obs.done:
        return {"reward": 0.0, "won": False, "steps": 0, "trajectory": [], "skipped": True}

    avail = set(int(a) for a in env.available_actions)
    avail_indices = []
    avail_ids = []
    class_to_id = {}
    for cls_name in classes:
        try:
            class_to_id[cls_name] = int(cls_name.replace("ACTION", ""))
        except ValueError:
            continue
    for idx, cls_name in enumerate(classes):
        cid = class_to_id.get(cls_name)
        if cid is not None and cid in avail:
            avail_indices.append(idx)
            avail_ids.append(cid)
    if not avail_ids:
        return {"reward": 0.0, "won": False, "steps": 0, "trajectory": [], "skipped": True}

    trajectory: list = []   # list of (frame_t, struct_t, label_idx_chosen)
    prior_actions: list = []
    total_reward = 0.0
    won = False
    model.eval()
    for step_i in range(max_steps):
        ctx = {
            "step_index": step_i + 1,
            "trace_len": max_steps,
            "prior_actions": prior_actions,
            "probe_signature": {"has_probe_data": True},
        }
        feats, names = build_features(ctx, profile_dict)
        if names != schema_names:
            return {"reward": 0.0, "won": False, "steps": step_i,
                    "trajectory": [], "skipped": True,
                    "skipped_reason": "schema_drift"}

        frame_pp = preprocess_frame(env.sess.frame)  # current frame
        ft = torch.from_numpy(frame_pp).unsqueeze(0)
        st = torch.from_numpy(np.asarray([feats], dtype=np.float32))
        with torch.no_grad():
            logits = model(ft, st)
            full_probs = F.softmax(logits, dim=1)[0].numpy()

        # Restrict + temperature sample
        ap = np.array([float(full_probs[i]) for i in avail_indices], dtype=np.float64)
        ap = np.clip(ap, 1e-9, 1.0)
        if temperature <= 0.0:
            chosen_local = int(np.argmax(ap))
        else:
            log_p = np.log(ap) / temperature
            log_p -= log_p.max()
            w = np.exp(log_p); w /= w.sum()
            chosen_local = rng.choices(range(len(avail_ids)), weights=w.tolist(), k=1)[0]
        global_idx = avail_indices[chosen_local]
        act_id = avail_ids[chosen_local]
        label = f"ACTION{act_id}"

        # Save (state, target action) for elite update
        trajectory.append({
            "frame_pp": frame_pp,             # ndarray (C, H, W)
            "struct": np.asarray(feats, dtype=np.float32),
            "target_idx": int(global_idx),
            "act_id": act_id,
        })

        if act_id in (5, 6):
            obs = env.step(act_id, click_xy=click_xy)
            prior_actions.append({"action": label, "x": click_xy[0], "y": click_xy[1]})
        else:
            obs = env.step(act_id)
            prior_actions.append({"action": label})

        total_reward += float(obs.reward)
        if obs.levels_completed > 0:
            won = True
            break
        if obs.done:
            break

    return {"reward": total_reward, "won": won, "steps": len(trajectory),
            "trajectory": trajectory, "skipped": False,
            "action_dist": Counter(t["act_id"] for t in trajectory)}


def update_on_elites(model: TinyPolicyCNN, elites: list, lr: float, epochs: int) -> dict:
    """Fine-tune model to maximize log-prob of (state, action) pairs from elite
    trajectories. Standard cross-entropy supervised update."""
    if not elites:
        return {"updated": False}
    Xf, Xs, y = [], [], []
    for ep in elites:
        for step in ep["trajectory"]:
            Xf.append(step["frame_pp"])
            Xs.append(step["struct"])
            y.append(step["target_idx"])
    if not y:
        return {"updated": False}
    Xf_t = torch.from_numpy(np.asarray(Xf, dtype=np.float32))
    Xs_t = torch.from_numpy(np.asarray(Xs, dtype=np.float32))
    y_t = torch.from_numpy(np.asarray(y, dtype=np.int64))

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    losses = []
    model.train()
    for _ in range(epochs):
        optim.zero_grad()
        logits = model(Xf_t, Xs_t)
        loss = F.cross_entropy(logits, y_t)
        loss.backward()
        optim.step()
        losses.append(float(loss))
    model.eval()
    return {"updated": True, "n_pairs": int(len(y)), "loss_first": losses[0],
            "loss_last": losses[-1]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--episode-steps", type=int, default=25)
    ap.add_argument("--elite-frac", type=float, default=0.34)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--update-epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=41923)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-similarity", action="store_true",
                    help="Disable winning-state-similarity reward shaping (default: enabled)")
    ap.add_argument("--no-novelty", action="store_true",
                    help="Disable novelty bonus (default: enabled)")
    ap.add_argument("--persistent-novelty", action="store_true",
                    help="Visited-hash set persists across episodes (option 2)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(f"D:/diloco_lab/state/arc3_policy_v2_cem/{args.game}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    print(f"Loading v1 CNN as initialization…", flush=True)
    model, config, classes, schema_names = load_v1_model()
    print(f"  classes={classes}  n_struct={config['n_struct_features']}")

    print(f"Characterizing {args.game} (live)…", flush=True)
    sess = GameSession.open(args.game)
    prof, _frame = characterize(sess)
    profile_dict = asdict(prof)
    profile_dict.setdefault("available_actions", prof.available_actions)
    print(f"  available_actions={prof.available_actions}  "
          f"dominant={prof.dominant_actions}  toggles={prof.toggle_actions}")
    click_xy = (32, 32)
    if prof.click_heatmap:
        top = prof.click_heatmap[0]
        click_xy = (int(top.get("x", 32)), int(top.get("y", 32)))
    print(f"  click_xy={click_xy}")

    history = []
    elite_n = max(1, int(round(args.batch * args.elite_frac)))
    print(f"\nSelf-play: iters={args.iterations} batch={args.batch} "
          f"episode_steps={args.episode_steps} elite_n={elite_n}", flush=True)

    use_similarity = not args.no_similarity
    use_novelty = not args.no_novelty
    print(f"Reward shaping: similarity={use_similarity} novelty={use_novelty} "
          f"persistent_novelty={args.persistent_novelty}")

    t_total = time.time()
    with ARC3Env(args.game, use_similarity=use_similarity,
                 use_novelty=use_novelty,
                 persistent_novelty=args.persistent_novelty) as env:
        # Reuse the already-opened session from characterize, not a new one,
        # if possible. ARC3Env.open does its own arcade lookup, so a separate
        # session is fine — characterize closed nothing.
        for it in range(1, args.iterations + 1):
            t_iter = time.time()
            episodes = []
            for ep_i in range(args.batch):
                ep = play_episode(
                    env, model, profile_dict, schema_names, classes,
                    args.episode_steps, click_xy, args.temperature, rng,
                )
                episodes.append(ep)
                marker = "WIN" if ep["won"] else f"r={ep['reward']:.3f}"
                print(f"  iter{it} ep{ep_i+1}/{args.batch}: steps={ep['steps']:2d}  {marker}", flush=True)

            non_skipped = [e for e in episodes if not e.get("skipped")]
            if not non_skipped:
                print(f"  iter{it}: all episodes skipped", flush=True)
                continue
            sorted_by_reward = sorted(non_skipped, key=lambda e: -e["reward"])
            elites = sorted_by_reward[:elite_n]

            update_info = update_on_elites(model, elites, args.lr, args.update_epochs)

            stats = {
                "iter": it,
                "elapsed_s": round(time.time() - t_iter, 1),
                "rewards": [round(e["reward"], 4) for e in episodes],
                "wins": int(sum(1 for e in episodes if e["won"])),
                "mean_reward": float(np.mean([e["reward"] for e in non_skipped])),
                "max_reward": float(max(e["reward"] for e in non_skipped)),
                "elite_threshold": float(elites[-1]["reward"]),
                "elite_size": len(elites),
                "elite_action_dist": dict(Counter(
                    a for e in elites for a in (e.get("action_dist") or {}).keys()
                )),
                "update": update_info,
            }
            history.append(stats)
            print(f"  iter{it}: mean_r={stats['mean_reward']:.3f}  "
                  f"max_r={stats['max_reward']:.3f}  "
                  f"wins={stats['wins']}  elapsed={stats['elapsed_s']}s", flush=True)

    total_elapsed = round(time.time() - t_total, 1)
    print(f"\nTotal: {total_elapsed}s")

    # Save final model + history
    torch.save(model.state_dict(), out_dir / "model_state.pt")
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "feature_schema.json").write_text(
        json.dumps({"feature_names": schema_names, "version": "v2_cem"}, indent=2))
    (out_dir / "label_map.json").write_text(json.dumps({
        "classes": classes,
        "label_to_idx": {c: i for i, c in enumerate(classes)},
    }, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps({
        "game": args.game,
        "iterations": args.iterations,
        "batch": args.batch,
        "episode_steps": args.episode_steps,
        "elite_frac": args.elite_frac,
        "temperature": args.temperature,
        "lr": args.lr,
        "update_epochs": args.update_epochs,
        "use_similarity": use_similarity,
        "use_novelty": use_novelty,
        "persistent_novelty": args.persistent_novelty,
        "total_elapsed_s": total_elapsed,
        "click_xy": list(click_xy),
        "history": history,
        "any_win": any(s["wins"] > 0 for s in history),
        "best_reward_overall": max((s["max_reward"] for s in history), default=0.0),
        "init_from": str(_V1_DIR),
    }, indent=2))
    print(f"Saved v2 CEM artefacts to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
