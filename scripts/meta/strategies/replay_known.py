"""replay_known — re-execute a previously-discovered winning combo.

When combo_lock (or another strategy) wins a level, it records the winning
combo in the registry. On subsequent runs we don't want to re-search — we
want to replay the known solution to fast-forward past solved levels,
giving other strategies budget to work on L2+.

This strategy checks the registry for a prior win on this game. If found,
it re-runs discovery (to sync action mapping & cursor direction) then drives
the combo directly. Win takes ~15-30 seconds for a 5-slot, 7-cycle puzzle.

Confidence: high (0.95) if a winning combo exists in the registry, else 0.
"""
from __future__ import annotations

from meta.budget import Budget
from meta.common import GameSession
from meta.profile import GameProfile
from meta.result import StrategyResult
from meta.replay_utils import lookup_replay_plan
from meta.strategies.base import Strategy


class ReplayKnown(Strategy):
    name = "replay_known"

    def confidence(self, profile: GameProfile) -> float:
        return 0.95 if lookup_replay_plan(profile.game_id) else 0.0

    def run(self, sess: GameSession, profile: GameProfile, budget: Budget) -> StrategyResult:
        res = self._start(sess, budget)
        res.details = {}

        plan = lookup_replay_plan(sess.game_id)
        if plan is None:
            res.stopped_reason = "no_prior_win"
            return self._finalize(res, sess, budget)

        r = sess.reset()
        if r.frame is None:
            res.stopped_reason = "reset_failed"
            return self._finalize(res, sess, budget)

        self.log(
            f"replaying L{plan['target_level']} via {plan['summary']} "
            f"(source={plan['strategy']})"
        )
        res.details.update({
            "replay_kind": plan["kind"],
            "target_level": plan["target_level"],
            "replay_len": len(plan["steps"]),
        })

        for i, act in enumerate(plan["steps"]):
            if budget.expired(steps=sess.steps_taken, lives=sess.lives_used):
                res.stopped_reason = "budget_expired"
                return self._finalize(res, sess, budget)
            # Plan steps are already normalized by replay_utils.build_replay_plan:
            # either a bare int (combo) or {"act": int, "data"?: {"x","y"}}.
            if isinstance(act, dict):
                act_id = int(act["act"])
                data = act.get("data")
                rr = sess.step_with_data(act_id, data) if data else sess.step(act_id)
            else:
                rr = sess.step(act)
            self._note_level(res, sess)
            if res.max_levels_completed >= plan["target_level"]:
                res.stopped_reason = "replay_complete"
                self.log(f"  replay matched at step {i+1}/{len(plan['steps'])}")
                return self._finalize(res, sess, budget)
            if rr.terminal:
                res.stopped_reason = "terminal_before_win"
                return self._finalize(res, sess, budget)
        res.stopped_reason = "plan_exhausted_no_win"
        return self._finalize(res, sess, budget)
