"""upgrade_winners — retroactively promote the fastest replayable winner in attempts.

After code changes to record_attempt (which now prefers faster replayable wins),
existing registry entries may still point to non-replayable or slow winners.
This script walks each game's attempts/, finds the fastest replayable win
for each level, and updates the winners dict.

Usage:
  python scripts/meta/upgrade_winners.py             # apply
  python scripts/meta/upgrade_winners.py --dry-run
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from meta.registry import REGISTRY_DIR, load_registry, save_registry


def _replayable(details: dict) -> bool:
    if not details: return False
    return any(k in details for k in ("win_combo", "win_sequence",
                                      "win_cycle_presses", "win_click"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    reg = load_registry()
    changes = 0

    for gid, entry in reg.items():
        prefix = entry.get("prefix", gid.split("-", 1)[0])
        adir = REGISTRY_DIR / "games" / prefix / "attempts"
        if not adir.exists(): continue
        attempts = []
        for f in adir.glob("*.json"):
            try:
                a = json.loads(f.read_text())
                if a.get("max_levels_completed", 0) > 0 and a.get("new_levels"):
                    attempts.append(a)
            except Exception:
                continue
        if not attempts: continue

        winners = entry.setdefault("winners", {})
        # For each level, find best attempt (replayable first, then fastest)
        by_level: dict = {}
        for a in attempts:
            for lv in a.get("new_levels", []):
                by_level.setdefault(lv, []).append(a)
        for lv, cands in by_level.items():
            replay_cands = [c for c in cands if _replayable(c.get("details", {}))]
            pool = replay_cands or cands
            best = min(pool, key=lambda c: c.get("elapsed", 1e9))
            best_detail = best.get("details", {})
            existing = winners.get(str(lv))
            e_replay = _replayable((existing or {}).get("details", {}))
            n_replay = _replayable(best_detail)
            if existing is None:
                action = "set"
            elif n_replay and not e_replay:
                action = "upgrade-replayable"
            elif n_replay and e_replay and best.get("elapsed", 1e9) < existing.get("elapsed", 1e9):
                action = "upgrade-faster"
            else:
                continue
            print(f"[{prefix}] L{lv}: {action}  "
                  f"{existing.get('strategy','<none>') if existing else '<none>'} "
                  f"→ {best.get('strategy')} (elapsed {best.get('elapsed',0):.1f}s)")
            if not args.dry_run:
                winners[str(lv)] = {
                    "strategy": best.get("strategy"),
                    "ts": datetime.datetime.now().isoformat(timespec="seconds") + "  (upgraded)",
                    "elapsed": round(best.get("elapsed", 0), 1),
                    "details": best_detail,
                }
                changes += 1

    if args.dry_run:
        print("(dry-run) no changes written")
    else:
        save_registry(reg)
        print(f"Wrote {changes} updates.")


if __name__ == "__main__":
    main()
