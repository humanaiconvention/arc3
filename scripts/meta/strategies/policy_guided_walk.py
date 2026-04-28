"""policy_guided_walk — actions chosen by a learned heuristic over probe data.

Loads the GradientBoosting classifier trained at
`D:/diloco_lab/state/arc3_policy_v0/` and uses it to pick the next action at
each step, from features derived from the live GameProfile + trace state.

This is the proof-of-concept "learned policy uses probe data" strategy.
v0 trained on 65 rows from 8 solved games; ~57% top-1 on the 7 held-out
rows. Real value: see whether routing decisions over probe-derived features
generalise to unsolved games.

Confidence: 0.35 if model artefact is present AND the model's classes are
all in the game's available actions (i.e., the model can produce a usable
recommendation). 0 otherwise — degenerate case.

Click coordinate handling: the model predicts action_id only. For ACTION5/6
the strategy falls back to the top entry from `profile.click_heatmap_top`,
or the centre of the frame as a last resort.
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

from meta.budget import Budget
from meta.common import GameSession
from meta.profile import GameProfile
from meta.result import StrategyResult
from meta.strategies.base import Strategy

# Reuse the canonical feature builder from diloco_lab.
_DILOCO_ROOT = Path("D:/diloco_lab")
sys.path.insert(0, str(_DILOCO_ROOT))
from diloco_lab.arc3_policy_features import build_features  # noqa: E402

_MODEL_DIR = _DILOCO_ROOT / "state" / "arc3_policy_v0"


class PolicyGuidedWalk(Strategy):
    name = "policy_guided_walk"

    MAX_STEPS = 200             # learned policy doesn't benefit from random walks of 1000
    SOFT_RESET_AFTER = 30
    DEFAULT_TRACE_LEN_GUESS = 10
    FALLBACK_CLICK_X = 32        # frame centre
    FALLBACK_CLICK_Y = 32
    # Stochastic sampling: temperature applied to log-probs before sampling.
    # T=0 → argmax (deterministic). T=1 → sample at native model probs.
    # T>1 → flatter distribution (more exploration). 0.7 is a balanced default.
    SAMPLE_TEMPERATURE = 0.7
    SAMPLE_SEED = 41923

    _model = None
    _label_encoder = None
    _schema = None

    @classmethod
    def _load(cls) -> bool:
        if cls._model is not None:
            return True
        if not (_MODEL_DIR / "model.joblib").exists():
            return False
        try:
            cls._model = joblib.load(_MODEL_DIR / "model.joblib")
            cls._label_encoder = joblib.load(_MODEL_DIR / "label_encoder.joblib")
            cls._schema = json.loads((_MODEL_DIR / "feature_schema.json").read_text(encoding="utf-8"))
        except Exception:
            cls._model = None
            return False
        return True

    def confidence(self, profile: GameProfile) -> float:
        if not self._load():
            return 0.0
        # Need at least one label the model can emit to be in the available
        # set, otherwise the prediction can't be executed.
        avail = set(int(a) for a in (profile.available_actions or []))
        possible_act_ids = set()
        for label in self._label_encoder.classes_:
            try:
                possible_act_ids.add(int(label.replace("ACTION", "")))
            except ValueError:
                continue
        if not (avail & possible_act_ids):
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

        # Convert GameProfile to dict for feature builder
        profile_dict = asdict(profile)
        profile_dict.setdefault("available_actions", profile.available_actions)

        # Pick a click coord from heatmap top (or fallback)
        click_xy = (self.FALLBACK_CLICK_X, self.FALLBACK_CLICK_Y)
        if profile.click_heatmap:
            top = profile.click_heatmap[0]
            click_xy = (int(top.get("x", click_xy[0])), int(top.get("y", click_xy[1])))
        res.details["click_xy"] = list(click_xy)
        res.details["max_steps"] = self.MAX_STEPS

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
        # Map model classes (e.g., "ACTION3") to integer ids, once.
        class_to_id: dict = {}
        for cls_name in self._label_encoder.classes_:
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
                self.log(f"feature schema drift: model={len(schema_names)} live={len(names)}")
                break

            # Sample from the model's distribution over ONLY available actions.
            # Temperature controls exploration: T=0 → argmax, T>0 → softmax.
            if not hasattr(self._model, "predict_proba"):
                pred = self._model.predict([feats])[0]
                label = str(self._label_encoder.inverse_transform([pred])[0])
                act_id = class_to_id.get(label)
                if act_id is None or act_id not in avail:
                    res.stopped_reason = "predicted_unavailable_no_proba"
                    break
            else:
                probs = self._model.predict_proba([feats])[0]
                # Restrict to available actions and apply temperature.
                # Use log-probs + temperature scaling = softmax(logp / T).
                avail_indices = []
                avail_probs = []
                avail_ids = []
                for idx, cls_name in enumerate(self._label_encoder.classes_):
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
                    log_p -= log_p.max()  # numerical stability
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
        res.details["model_classes"] = list(self._label_encoder.classes_)
        return self._finalize(res, sess, budget)
