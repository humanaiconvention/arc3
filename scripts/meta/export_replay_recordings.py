"""Replay registry winners into clean Format-B recordings for MuZero training."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Optional

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_AGENTS_VENV = _ROOT / "external" / "ARC-AGI-3-Agents" / ".venv"


def _maybe_reexec_into_agents_venv() -> None:
    if os.environ.get("ARC3_META_BOOTSTRAPPED") == "1":
        return
    candidates = [
        _AGENTS_VENV / "Scripts" / "python.exe",
        _AGENTS_VENV / "bin" / "python",
    ]
    current = pathlib.Path(sys.executable).resolve()
    for candidate in candidates:
        if candidate.exists() and candidate.resolve() != current:
            env = dict(os.environ)
            env["ARC3_META_BOOTSTRAPPED"] = "1"
            completed = subprocess.run(
                [str(candidate), __file__, *sys.argv[1:]],
                env=env,
                check=False,
            )
            raise SystemExit(completed.returncode)


_maybe_reexec_into_agents_venv()

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from meta.common import ROOT, GameSession
import meta.common as common
from meta.registry import load_registry
from meta.replay_utils import iter_replay_plans

try:
    from agents.recorder import Recorder
except Exception as e:  # pragma: no cover - import path depends on local env
    raise SystemExit(
        "Failed to import Recorder. Ensure ARC-AGI-3-Agents dependencies are available."
    ) from e


def _serializable_state_name(value) -> str:
    if hasattr(value, "name"):
        return value.name
    return str(value)


def _frame_payload(sess: GameSession, step_result, action_id: int, action_data: Optional[dict]) -> dict:
    state = step_result.state
    frame = np.asarray(getattr(state, "frame", []))
    return {
        "game_id": getattr(state, "game_id", sess.game_id),
        "frame": frame.tolist(),
        "state": _serializable_state_name(getattr(state, "state", "UNKNOWN")),
        "levels_completed": int(getattr(state, "levels_completed", 0) or 0),
        "win_levels": int(getattr(state, "win_levels", 0) or 0),
        "action_input": {
            "id": int(action_id),
            "data": action_data or {},
            "reasoning": None,
        },
        "guid": getattr(state, "guid", None),
        "full_reset": bool(getattr(state, "full_reset", False)),
        "available_actions": list(getattr(state, "available_actions", []) or []),
    }


def _select_games(registry: dict, requested: list[str]) -> list[str]:
    if not requested:
        return [game_id for game_id in sorted(registry) if iter_replay_plans(game_id)]
    selected = []
    for game in requested:
        if game in registry:
            selected.append(game)
            continue
        match = next((gid for gid in registry if gid.startswith(f"{game}-")), None)
        if match is not None:
            selected.append(match)
    return selected


def export_one(game_id: str, plan: dict, out_dir: pathlib.Path, keep_failed: bool) -> dict:
    prefix = f"{game_id}.meta_replay.L{plan['target_level']}"
    recorder = Recorder(prefix=prefix)
    out_path = pathlib.Path(recorder.filename)

    summary = {
        "game_id": game_id,
        "level": plan["target_level"],
        "kind": plan["kind"],
        "steps": len(plan["steps"]),
        "recording": str(out_path),
        "success": False,
        "reason": "",
    }

    with GameSession.open(game_id.split("-", 1)[0]) as sess:
        reset = sess.reset()
        if reset.frame is None:
            summary["reason"] = f"reset_failed: {reset.exception or 'no_frame'}"
        else:
            recorder.record(_frame_payload(sess, reset, action_id=0, action_data={}))
            for index, step in enumerate(plan["steps"], start=1):
                if isinstance(step, dict):
                    act_id = int(step["act"])
                    data = step.get("data") or {}
                    rr = sess.step_with_data(act_id, data) if data else sess.step(act_id)
                else:
                    act_id = int(step)
                    data = {}
                    rr = sess.step(act_id)
                if rr.frame is None:
                    summary["reason"] = rr.exception or f"missing_frame_at_step_{index}"
                    break
                recorder.record(_frame_payload(sess, rr, action_id=act_id, action_data=data))
                if sess.max_levels_completed >= plan["target_level"]:
                    summary["success"] = True
                    summary["reason"] = f"won_level_at_step_{index}"
                    break
                if rr.terminal:
                    summary["reason"] = f"terminal_before_target_at_step_{index}"
                    break
            else:
                summary["reason"] = "plan_exhausted_no_win"

    if not summary["success"] and out_path.exists() and not keep_failed:
        out_path.unlink()
        summary["recording"] = ""
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", action="append", default=[],
                    help="Game full id or short prefix. Repeatable. Default: all replayable wins.")
    ap.add_argument("--out-dir", default=str(ROOT / "recordings" / "meta_replays"),
                    help="Directory for exported *.recording.jsonl files.")
    ap.add_argument("--step-delay", type=float, default=None,
                    help="Override META_STEP_DELAY for this export process.")
    ap.add_argument("--keep-failed", action="store_true",
                    help="Keep partial recordings even when replay does not reach the target level.")
    args = ap.parse_args()

    if args.step_delay is not None:
        common.STEP_DELAY = args.step_delay

    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["RECORDINGS_DIR"] = str(out_dir)

    registry = load_registry()
    games = _select_games(registry, args.game)
    if not games:
        raise SystemExit("No matching replayable games found.")

    exported = []
    for game_id in games:
        for plan in iter_replay_plans(game_id):
            print(f"[{game_id}] exporting {plan['summary']}...")
            summary = export_one(game_id, plan, out_dir, keep_failed=args.keep_failed)
            exported.append(summary)
            print(f"  -> success={summary['success']}  reason={summary['reason']}")

    manifest = out_dir / "meta_replay_manifest.json"
    manifest.write_text(json.dumps(exported, indent=2))
    succeeded = sum(1 for row in exported if row["success"])
    print(f"\nExported {succeeded}/{len(exported)} replay recordings to {out_dir}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
