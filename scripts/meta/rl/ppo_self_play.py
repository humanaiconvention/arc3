"""ppo_self_play — PPO with value baseline on a single ARC3 game.

Replaces v2/v3/v3b CEM with a more stable on-policy actor-critic:

  - Actor: v1 CNN + softmax over actions (initialised from v1 weights).
  - Critic: parallel single-output value head (random init, learned from
    rollout returns).
  - Update: PPO clipped objective + value MSE + entropy bonus.
  - Advantage estimation: GAE(λ=0.95).
  - Multiple update epochs per rollout for sample efficiency.

CEM elite-collapsed because top-K distillation narrows the policy with
no constraint. PPO solves this two ways: (1) the clip ratio caps each
update's policy change, (2) the entropy bonus directly penalises
distribution narrowing.

Usage:
  python -u scripts/meta/rl/ppo_self_play.py --game re86 --iterations 6 --batch 8 --episode-steps 25
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve()
_ARC3_SCRIPTS = _HERE.parents[2]
sys.path.insert(0, str(_ARC3_SCRIPTS))
sys.path.insert(0, "D:/diloco_lab")

from meta.common import GameSession  # noqa: E402
from meta.characterize import characterize  # noqa: E402
from diloco_lab.arc3_env import ARC3Env  # noqa: E402
from diloco_lab.arc3_policy_features import build_features  # noqa: E402
from diloco_lab.arc3_frame_features import (  # noqa: E402
    DEFAULT_CHANNELS, FRAME_HW, preprocess_frame,
)


_V1_DIR = Path("D:/diloco_lab/state/arc3_policy_v1_cnn")


class TinyPolicyValueCNN(nn.Module):
    """v1 CNN + parallel value head. Same conv backbone, two heads:
    `head` (action logits) and `value_head` (scalar V(s))."""

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
        # PPO addition: value baseline
        self.value_head = nn.Linear(64 + 32, 1)

    def _trunk(self, frame, struct):
        x = F.relu(self.conv1(frame)); x = self.pool(x)
        x = F.relu(self.conv2(x));     x = self.pool(x)
        x = F.relu(self.conv3(x));     x = self.pool(x)
        x = self.dropout_conv(x)
        x = x.flatten(1)
        x = F.relu(self.fc_frame(x))
        s = F.relu(self.fc_struct(struct))
        return torch.cat([x, s], dim=1)

    def forward(self, frame, struct):
        z = self._trunk(frame, struct)
        z_drop = self.dropout_fc(z)
        logits = self.head(z_drop)
        value = self.value_head(z_drop).squeeze(-1)
        return logits, value


def load_v1_into_actor_critic() -> tuple[TinyPolicyValueCNN, dict, list[str], list[str]]:
    config = json.loads((_V1_DIR / "config.json").read_text(encoding="utf-8"))
    schema = json.loads((_V1_DIR / "feature_schema.json").read_text(encoding="utf-8"))
    label_map = json.loads((_V1_DIR / "label_map.json").read_text(encoding="utf-8"))
    classes = label_map["classes"]
    model = TinyPolicyValueCNN(
        n_classes=int(config["n_classes"]),
        n_struct=int(config["n_struct_features"]),
        n_channels=int(config.get("n_channels", DEFAULT_CHANNELS)),
    )
    v1_state = torch.load(_V1_DIR / "model_state.pt", map_location="cpu", weights_only=True)
    # value_head keys won't be in v1; load with strict=False.
    missing, unexpected = model.load_state_dict(v1_state, strict=False)
    print(f"  loaded v1 actor; new keys (random init): {missing}")
    return model, config, classes, schema["feature_names"]


def collect_rollout(env: ARC3Env, model: TinyPolicyValueCNN, profile_dict: dict,
                    schema_names: list[str], classes: list[str], max_steps: int,
                    click_xy: tuple, avail: set, class_to_id: dict,
                    avail_indices: list, avail_ids: list) -> dict:
    """Roll one episode under the current policy. Returns trajectory + rewards."""
    obs = env.reset()
    if obs.frame is None or obs.done:
        return {"reward": 0.0, "won": False, "steps": 0,
                "frames": [], "structs": [], "actions": [], "log_probs": [],
                "values": [], "rewards": [], "dones": [], "skipped": True}

    frames, structs, action_idxs, log_probs, values, rewards, dones = [], [], [], [], [], [], []
    prior_actions: list = []
    won = False
    total_reward = 0.0
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
            return {"reward": total_reward, "won": won, "steps": step_i,
                    "frames": frames, "structs": structs, "actions": action_idxs,
                    "log_probs": log_probs, "values": values,
                    "rewards": rewards, "dones": dones,
                    "skipped": True, "skipped_reason": "schema_drift"}

        frame_pp = preprocess_frame(env.sess.frame)
        ft = torch.from_numpy(frame_pp).unsqueeze(0)
        st = torch.from_numpy(np.asarray([feats], dtype=np.float32))
        with torch.no_grad():
            logits, value = model(ft, st)
            # Restrict to available actions BEFORE softmax to avoid sampling
            # an unavailable action.
            mask = torch.full_like(logits, -1e9)
            mask[0, avail_indices] = 0.0
            masked_logits = logits + mask
            probs = F.softmax(masked_logits, dim=1)
            dist = torch.distributions.Categorical(probs=probs)
            action_global = int(dist.sample().item())
            log_prob = dist.log_prob(torch.tensor(action_global)).item()

        if action_global not in avail_indices:
            # Safety net: if masking failed (e.g., all-zero), pick uniform random
            action_global = int(np.random.choice(avail_indices))
            log_prob = float(np.log(1.0 / len(avail_indices)))

        # Map model class index → game action id
        cls_name = classes[action_global]
        act_id = class_to_id.get(cls_name)
        if act_id is None or act_id not in avail:
            break
        label = f"ACTION{act_id}"

        frames.append(frame_pp)
        structs.append(np.asarray(feats, dtype=np.float32))
        action_idxs.append(action_global)
        log_probs.append(log_prob)
        values.append(float(value.item()))

        if act_id in (5, 6):
            obs = env.step(act_id, click_xy=click_xy)
            prior_actions.append({"action": label, "x": click_xy[0], "y": click_xy[1]})
        else:
            obs = env.step(act_id)
            prior_actions.append({"action": label})

        rewards.append(float(obs.reward))
        total_reward += float(obs.reward)
        dones.append(bool(obs.done))

        if obs.levels_completed > 0:
            won = True
            break
        if obs.done:
            break

    return {"reward": total_reward, "won": won, "steps": len(action_idxs),
            "frames": frames, "structs": structs, "actions": action_idxs,
            "log_probs": log_probs, "values": values,
            "rewards": rewards, "dones": dones, "skipped": False,
            "action_dist": Counter(int(classes[i].replace("ACTION", "")) for i in action_idxs)}


def gae_advantages(rewards: list, values: list, dones: list, gamma: float, lam: float
                   ) -> tuple[list, list]:
    """Compute GAE advantages and returns. values must be one longer than
    rewards (next-state value at end), or we bootstrap with 0 if terminal."""
    T = len(rewards)
    advantages = [0.0] * T
    last_adv = 0.0
    # Bootstrap: append a zero next-value (assume episode ended or no info)
    next_value = 0.0
    for t in reversed(range(T)):
        non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_adv = delta + gamma * lam * non_terminal * last_adv
        advantages[t] = last_adv
        next_value = values[t]
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def ppo_update(model: TinyPolicyValueCNN, batch: dict, lr: float,
               clip_eps: float, entropy_coef: float, value_coef: float,
               update_epochs: int) -> dict:
    """Standard PPO clipped objective with value MSE and entropy bonus."""
    Xf = torch.from_numpy(np.asarray(batch["frames"], dtype=np.float32))
    Xs = torch.from_numpy(np.asarray(batch["structs"], dtype=np.float32))
    actions = torch.from_numpy(np.asarray(batch["actions"], dtype=np.int64))
    old_log_probs = torch.from_numpy(np.asarray(batch["log_probs"], dtype=np.float32))
    advantages = torch.from_numpy(np.asarray(batch["advantages"], dtype=np.float32))
    returns = torch.from_numpy(np.asarray(batch["returns"], dtype=np.float32))
    # Normalise advantages
    if advantages.numel() > 1 and advantages.std() > 0:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    losses = []
    for _ in range(update_epochs):
        model.train()
        logits, values = model(Xf, Xs)
        # Cross-entropy log-prob of the taken action
        log_probs_all = F.log_softmax(logits, dim=1)
        new_log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        actor_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * log_probs_all).sum(dim=1).mean()
        loss = actor_loss + value_coef * value_loss - entropy_coef * entropy
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append({"actor": float(actor_loss), "value": float(value_loss),
                       "entropy": float(entropy), "total": float(loss)})
    return {"loss_first": losses[0], "loss_last": losses[-1],
            "n_pairs": int(actions.numel())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--episode-steps", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--entropy-coef", type=float, default=0.02)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--update-epochs", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=41923)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-similarity", action="store_true")
    ap.add_argument("--no-novelty", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(f"D:/diloco_lab/state/arc3_policy_v4_ppo/{args.game}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading v1 CNN as actor (value head random init)…", flush=True)
    model, config, classes, schema_names = load_v1_into_actor_critic()
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

    avail = set(int(a) for a in prof.available_actions)
    class_to_id: dict = {}
    for c in classes:
        try:
            class_to_id[c] = int(c.replace("ACTION", ""))
        except ValueError:
            continue
    avail_indices = []
    avail_ids = []
    for idx, c in enumerate(classes):
        cid = class_to_id.get(c)
        if cid is not None and cid in avail:
            avail_indices.append(idx)
            avail_ids.append(cid)
    if not avail_ids:
        print("No model class is in the game's available actions; abort.")
        return 1

    use_similarity = not args.no_similarity
    use_novelty = not args.no_novelty
    print(f"Reward shaping: similarity={use_similarity} novelty={use_novelty}")
    print(f"PPO: iters={args.iterations} batch={args.batch} ep_steps={args.episode_steps} "
          f"lr={args.lr} clip={args.clip_eps} ent={args.entropy_coef} epochs={args.update_epochs}")

    history = []
    t_total = time.time()
    with ARC3Env(args.game, use_similarity=use_similarity,
                 use_novelty=use_novelty) as env:
        for it in range(1, args.iterations + 1):
            t_iter = time.time()
            episodes = []
            for ep_i in range(args.batch):
                ep = collect_rollout(env, model, profile_dict, schema_names, classes,
                                     args.episode_steps, click_xy, avail, class_to_id,
                                     avail_indices, avail_ids)
                episodes.append(ep)
                marker = "WIN" if ep["won"] else f"r={ep['reward']:.3f}"
                print(f"  iter{it} ep{ep_i+1}/{args.batch}: steps={ep['steps']:2d}  {marker}",
                      flush=True)

            non_skipped = [e for e in episodes if not e.get("skipped")]
            if not non_skipped:
                print(f"  iter{it}: all episodes skipped"); continue

            # Compute GAE per episode then concatenate
            all_frames, all_structs, all_actions, all_log_probs = [], [], [], []
            all_advantages, all_returns = [], []
            for ep in non_skipped:
                advs, rets = gae_advantages(ep["rewards"], ep["values"], ep["dones"],
                                            args.gamma, args.gae_lambda)
                all_frames.extend(ep["frames"])
                all_structs.extend(ep["structs"])
                all_actions.extend(ep["actions"])
                all_log_probs.extend(ep["log_probs"])
                all_advantages.extend(advs)
                all_returns.extend(rets)

            update_info = ppo_update(model, {
                "frames": all_frames, "structs": all_structs,
                "actions": all_actions, "log_probs": all_log_probs,
                "advantages": all_advantages, "returns": all_returns,
            }, args.lr, args.clip_eps, args.entropy_coef, args.value_coef,
                args.update_epochs)

            stats = {
                "iter": it,
                "elapsed_s": round(time.time() - t_iter, 1),
                "rewards": [round(e["reward"], 4) for e in episodes],
                "wins": int(sum(1 for e in episodes if e["won"])),
                "mean_reward": float(np.mean([e["reward"] for e in non_skipped])),
                "max_reward": float(max(e["reward"] for e in non_skipped)),
                "min_reward": float(min(e["reward"] for e in non_skipped)),
                "n_pairs_updated": int(update_info["n_pairs"]),
                "update_loss_first": update_info["loss_first"],
                "update_loss_last": update_info["loss_last"],
            }
            history.append(stats)
            print(f"  iter{it}: mean_r={stats['mean_reward']:.3f}  "
                  f"max_r={stats['max_reward']:.3f}  min_r={stats['min_reward']:.3f}  "
                  f"wins={stats['wins']}  loss[a/v/e]= "
                  f"{update_info['loss_last']['actor']:.3f}/{update_info['loss_last']['value']:.3f}/"
                  f"{update_info['loss_last']['entropy']:.3f}  elapsed={stats['elapsed_s']}s",
                  flush=True)

    total_elapsed = round(time.time() - t_total, 1)
    print(f"\nTotal: {total_elapsed}s")

    torch.save(model.state_dict(), out_dir / "model_state.pt")
    (out_dir / "config.json").write_text(json.dumps({
        **config, "model": "TinyPolicyValueCNN", "trainer": "PPO",
    }, indent=2))
    (out_dir / "feature_schema.json").write_text(
        json.dumps({"feature_names": schema_names, "version": "v4_ppo"}, indent=2))
    (out_dir / "label_map.json").write_text(json.dumps({
        "classes": classes,
        "label_to_idx": {c: i for i, c in enumerate(classes)},
    }, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps({
        "game": args.game, "iterations": args.iterations,
        "batch": args.batch, "episode_steps": args.episode_steps,
        "lr": args.lr, "clip_eps": args.clip_eps,
        "entropy_coef": args.entropy_coef, "value_coef": args.value_coef,
        "update_epochs": args.update_epochs,
        "gamma": args.gamma, "gae_lambda": args.gae_lambda,
        "use_similarity": use_similarity, "use_novelty": use_novelty,
        "total_elapsed_s": total_elapsed, "click_xy": list(click_xy),
        "history": history,
        "any_win": any(s["wins"] > 0 for s in history),
        "best_reward_overall": max((s["max_reward"] for s in history), default=0.0),
        "init_from": str(_V1_DIR),
    }, indent=2))
    print(f"Saved PPO v4 artefacts to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
