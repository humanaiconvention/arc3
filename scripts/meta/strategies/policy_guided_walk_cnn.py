"""policy_guided_walk_cnn — same shape as PolicyGuidedWalk but uses the v1 CNN.

Loads the TinyPolicyCNN trained at D:/diloco_lab/state/arc3_policy_v1_cnn/.
At each step, pulls the live frame from the GameSession and feeds it through
the CNN alongside the same 53-dim structured feature vector v0 used.

Inference path: torch CPU only. 148K params runs comfortably <10ms/step.

Confidence: 0.35 if model artefact present and at least one model class is
in the game's available actions. Lazy-loaded class state.
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from meta.budget import Budget
from meta.common import GameSession
from meta.profile import GameProfile
from meta.result import StrategyResult
from meta.strategies.base import Strategy

_DILOCO_ROOT = Path("D:/diloco_lab")
sys.path.insert(0, str(_DILOCO_ROOT))
from diloco_lab.arc3_policy_features import build_features  # noqa: E402
from diloco_lab.arc3_frame_features import (  # noqa: E402
    DEFAULT_CHANNELS, FRAME_HW, preprocess_frame,
)

_MODEL_DIR = _DILOCO_ROOT / "state" / "arc3_policy_v1_cnn"


class _TinyPolicyCNN(nn.Module):
    """Inference-side mirror of TinyPolicyCNN from train_arc3_policy_cnn.py.
    Architecture must match exactly so state_dict loads."""

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

    def forward(self, frame: torch.Tensor, struct: torch.Tensor) -> torch.Tensor:
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


class PolicyGuidedWalkCNN(Strategy):
    name = "policy_guided_walk_cnn"

    MAX_STEPS = 200
    SOFT_RESET_AFTER = 30
    DEFAULT_TRACE_LEN_GUESS = 10
    FALLBACK_CLICK_X = 32
    FALLBACK_CLICK_Y = 32
    SAMPLE_TEMPERATURE = 0.7
    SAMPLE_SEED = 41923

    _model = None
    _config = None
    _schema = None
    _label_classes = None

    @classmethod
    def _load(cls) -> bool:
        if cls._model is not None:
            return True
        sd_path = _MODEL_DIR / "model_state.pt"
        if not sd_path.exists():
            return False
        try:
            cls._config = json.loads((_MODEL_DIR / "config.json").read_text(encoding="utf-8"))
            cls._schema = json.loads((_MODEL_DIR / "feature_schema.json").read_text(encoding="utf-8"))
            label_map = json.loads((_MODEL_DIR / "label_map.json").read_text(encoding="utf-8"))
            cls._label_classes = label_map["classes"]
            model = _TinyPolicyCNN(
                n_classes=int(cls._config["n_classes"]),
                n_struct=int(cls._config["n_struct_features"]),
                n_channels=int(cls._config.get("n_channels", DEFAULT_CHANNELS)),
            )
            state = torch.load(sd_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            cls._model = model
        except Exception:
            cls._model = None
            return False
        return True

    def confidence(self, profile: GameProfile) -> float:
        if not self._load():
            return 0.0
        avail = set(int(a) for a in (profile.available_actions or []))
        possible = set()
        for label in self._label_classes:
            try:
                possible.add(int(label.replace("ACTION", "")))
            except ValueError:
                continue
        if not (avail & possible):
            return 0.0
        return 0.35

    def run(self, sess: GameSession, profile: GameProfile, budget: Budget) -> StrategyResult:
        res = self._start(sess, budget)
        res.details = {}
        if not self._load():
            res.stopped_reason = "model_not_found"
            return self._finalize(res, sess, budget)

        avail = set(int(a) for a in (profile.available_actions or []))
        if not avail:
            res.stopped_reason = "no_available_actions"
            return self._finalize(res, sess, budget)

        profile_dict = asdict(profile)
        profile_dict.setdefault("available_actions", profile.available_actions)

        click_xy = (self.FALLBACK_CLICK_X, self.FALLBACK_CLICK_Y)
        if profile.click_heatmap:
            top = profile.click_heatmap[0]
            click_xy = (int(top.get("x", click_xy[0])), int(top.get("y", click_xy[1])))
        res.details["click_xy"] = list(click_xy)

        rr = sess.reset()
        if rr.frame is None:
            res.stopped_reason = "reset_failed"
            return self._finalize(res, sess, budget)

        prior_actions: list = []
        steps_since_progress = 0
        steps_taken_logical = 0
        prior_max = res.max_levels_completed
        chosen_actions: list = []
        rng = random.Random(self.SAMPLE_SEED)
        class_to_id: dict = {}
        for cls_name in self._label_classes:
            try:
                class_to_id[str(cls_name)] = int(str(cls_name).replace("ACTION", ""))
            except ValueError:
                continue
        res.details["sample_temperature"] = self.SAMPLE_TEMPERATURE

        for _ in range(self.MAX_STEPS):
            if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                res.stopped_reason = "budget_expired"
                break

            ctx = {
                "step_index": steps_taken_logical + 1,
                "trace_len": self.DEFAULT_TRACE_LEN_GUESS,
                "prior_actions": prior_actions,
                "probe_signature": {
                    "has_probe_data": bool(profile.action_pair_effects or profile.click_heatmap),
                    "has_triple_data": bool(profile.action_triple_effects),
                    "has_post_prelude_data": bool(profile.post_prelude_responses),
                },
            }
            feats, names = build_features(ctx, profile_dict)
            schema_names = self._schema.get("feature_names", [])
            if names != schema_names:
                res.stopped_reason = "feature_schema_mismatch"
                break

            # Live frame from session
            live_frame = sess.frame
            if live_frame is None:
                live_frame = np.zeros((FRAME_HW, FRAME_HW), dtype=np.int32)
            frame_tensor = torch.from_numpy(preprocess_frame(live_frame)).unsqueeze(0)
            struct_tensor = torch.from_numpy(np.asarray([feats], dtype=np.float32))

            with torch.no_grad():
                logits = self._model(frame_tensor, struct_tensor)
                probs = torch.softmax(logits, dim=1)[0].numpy()

            avail_indices = []
            avail_probs = []
            avail_ids = []
            for idx, cls_name in enumerate(self._label_classes):
                cid = class_to_id.get(str(cls_name))
                if cid is not None and cid in avail:
                    avail_indices.append(idx)
                    avail_probs.append(float(probs[idx]))
                    avail_ids.append(cid)
            if not avail_ids:
                res.stopped_reason = "no_available_prediction"
                break

            arr = np.asarray(avail_probs, dtype=np.float64)
            arr = np.clip(arr, 1e-9, 1.0)
            if self.SAMPLE_TEMPERATURE <= 0.0:
                chosen_idx = int(np.argmax(arr))
            else:
                log_p = np.log(arr) / self.SAMPLE_TEMPERATURE
                log_p -= log_p.max()
                weights = np.exp(log_p)
                weights /= weights.sum()
                chosen_idx = rng.choices(range(len(avail_ids)),
                                         weights=weights.tolist(), k=1)[0]
            act_id = avail_ids[chosen_idx]
            label = f"ACTION{act_id}"

            chosen_actions.append(act_id)
            if act_id in (5, 6):
                r = sess.step_with_data(act_id, {"x": click_xy[0], "y": click_xy[1]})
                prior_actions.append({"action": label, "x": click_xy[0], "y": click_xy[1]})
            else:
                r = sess.step(act_id)
                prior_actions.append({"action": label})
            steps_taken_logical += 1
            self._note_level(res, sess)

            if res.max_levels_completed > prior_max:
                prior_max = res.max_levels_completed
                steps_since_progress = 0
                res.details.setdefault("win_step", steps_taken_logical)
                res.stopped_reason = "level_won"
                self.log(f"WIN at logical step {steps_taken_logical}: act={label}")
                break

            steps_since_progress += 1
            if r.terminal:
                rr = sess.soft_reset()
                if rr.frame is None or rr.terminal:
                    res.stopped_reason = "terminal_unrecoverable"
                    break
                prior_actions = []
                steps_taken_logical = 0
                continue

            if steps_since_progress >= self.SOFT_RESET_AFTER:
                rr = sess.reset()
                if rr.frame is None:
                    res.stopped_reason = "reset_failed_midwalk"
                    break
                prior_actions = []
                steps_taken_logical = 0
                steps_since_progress = 0

        res.details["chosen_actions"] = chosen_actions
        res.details["model_dir"] = str(_MODEL_DIR)
        res.details["model_classes"] = list(self._label_classes)
        return self._finalize(res, sess, budget)
