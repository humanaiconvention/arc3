"""iterate — re-run with escalated budgets on games that showed partial progress.

Strategy:
  - Read registry
  - For each game that has some attempts but hasn't reached win_levels:
      - If max_levels_completed > 0: it's winning some levels → escalate budget
      - If max_levels_completed == 0 and signals are strong: retry with bigger budget
      - If no progress and signals are weak: skip (need new strategy)

Usage:
  python scripts/meta/iterate.py --escalate-factor 3 --budget 600
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from meta.common import ROOT
from meta.registry import load_registry, REGISTRY_DIR


def games_needing_escalation() -> list:
    """Return list of (prefix, reason, priority) for games to retry."""
    reg = load_registry()
    out = []
    for gid, entry in reg.items():
        prefix = entry.get("prefix", gid.split("-", 1)[0])
        max_lv = entry.get("max_levels_completed", 0)
        # Read profile for win_levels + signals
        pf = REGISTRY_DIR / "games" / prefix / "profile.json"
        if not pf.exists(): continue
        import json
        p = json.loads(pf.read_text())
        win_levels = p.get("win_levels", 0)
        if win_levels == 0: continue
        if max_lv >= win_levels: continue  # fully solved

        # Priority: partial progress > strong-signal no-progress > weak-signal no-progress
        combo = p.get("looks_like_combo_lock", 0)
        grid = p.get("looks_like_grid_click", 0)
        cursor = p.get("looks_like_cursor_game", 0)
        top_sig = max(combo, grid, cursor)

        if max_lv > 0:
            priority = 100 + max_lv * 10          # partial win, high priority
            reason = f"partial-win lv{max_lv}/{win_levels}"
        elif top_sig > 0.5:
            priority = 50 + int(top_sig * 10)     # strong signal, medium
            reason = f"strong-signal {top_sig:.2f}"
        else:
            priority = int(top_sig * 10)          # weak signal, low
            reason = f"weak-signal {top_sig:.2f}"
        out.append((prefix, reason, priority))
    out.sort(key=lambda x: -x[2])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=600.0,
                    help="Per-strategy budget (seconds).")
    ap.add_argument("--top-n", type=int, default=10,
                    help="How many games to retry this iteration.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    todo = games_needing_escalation()
    print(f"{len(todo)} games need escalation")
    for prefix, reason, pri in todo[:20]:
        print(f"  pri={pri:4d}  {prefix:8}  {reason}")

    if args.dry_run: return

    for prefix, reason, pri in todo[:args.top_n]:
        print(f"\n=== Escalating {prefix} ({reason}, pri={pri}) ===")
        cmd = [
            str(ROOT / "external" / "ARC-AGI-3-Agents" / ".venv" / "Scripts" / "python.exe"),
            "-u",
            str(pathlib.Path(__file__).resolve().parent / "run_meta.py"),
            "--game", prefix,
            "--budget", str(args.budget),
            "--replay",
        ]
        try:
            subprocess.run(cmd, check=True, cwd=str(ROOT))
        except subprocess.CalledProcessError as e:
            print(f"  run_meta failed for {prefix}: {e}")
        except KeyboardInterrupt:
            print("Interrupted.")
            break


if __name__ == "__main__":
    main()
