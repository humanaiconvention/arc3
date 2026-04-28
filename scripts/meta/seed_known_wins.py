"""seed_known_wins — populate the registry with known wins from prior work.

We've already solved some L1s outside the meta-solver (tr87, vc33, lp85 L1,
sp80). Seeding the registry lets `replay_known` skip re-searching on future runs.

Usage:
  python scripts/meta/seed_known_wins.py           # seed all known wins
  python scripts/meta/seed_known_wins.py --dry-run

This is pure registry manipulation — no network calls.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from meta.common import ROOT
from meta.registry import REGISTRY_DIR, REGISTRY_FILE, ensure_dirs, load_registry, save_registry


# Known wins: keyed by short prefix. Must include win_combo (+ discovery hints for
# combo_lock-family games). For non-combo wins we just record the level.
KNOWN_WINS = {
    "tr87": {
        "level": 1,
        "strategy": "combo_lock",
        "details": {
            "win_combo": [5, 5, 3, 6, 5],
            "n_slots": 5,
            "cycle": 7,
            "fwd_cursor_act": 4,
            "bwd_cursor_act": 3,
            "fwd_value_act": 1,
            "bwd_value_act": 2,
            "source": "tr87_solver_v7b.py 2026-04-18 brute force",
        },
    },
    "vc33": {
        "level": 1,
        "strategy": "grid_click_fine",
        "details": {
            "win_burst_clicks": 3,
            "win_click": [62, 33],
            "click_act": 6,
            "source": "manual confirmation 2026-04-20: 3x ACTION6 at (62,33)",
        },
    },
    # Add more as we confirm them. Keeping this list narrow for now.
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    reg = load_registry()
    changed = 0

    for prefix, win in KNOWN_WINS.items():
        # Find the full game_id matching this prefix in registry; if not there,
        # use prefix as a standalone key (registry may not exist yet for a game).
        gid = next((k for k in reg if k.startswith(prefix)), None)
        if gid is None:
            # Best effort: use prefix as id; registry entry will be created.
            gid = prefix
            print(f"[{prefix}] no registry entry yet — seeding with prefix only")

        entry = reg.setdefault(gid, {})
        entry.setdefault("prefix", prefix)
        winners = entry.setdefault("winners", {})
        key = str(win["level"])
        if key in winners:
            print(f"[{prefix}] L{win['level']} already has a winner "
                  f"({winners[key].get('strategy','?')}); not overwriting.")
            continue
        winners[key] = {
            "strategy": win["strategy"],
            "ts": datetime.datetime.now().isoformat(timespec="seconds") + "  (seeded)",
            "details": win["details"],
        }
        entry["max_levels_completed"] = max(entry.get("max_levels_completed", 0), win["level"])
        changed += 1
        print(f"[{prefix}] seeded L{win['level']} win via {win['strategy']}")

    if changed == 0:
        print("No changes.")
        return

    if args.dry_run:
        print(f"\n(dry-run) Would write {changed} updates to {REGISTRY_FILE}")
    else:
        save_registry(reg)
        print(f"\nSaved {changed} updates to {REGISTRY_FILE}")


if __name__ == "__main__":
    main()
