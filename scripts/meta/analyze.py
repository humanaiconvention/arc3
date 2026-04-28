"""analyze — summarize the registry.

Usage:
  python scripts/meta/analyze.py                # table across all games
  python scripts/meta/analyze.py --game tr87    # attempt details for one game
  python scripts/meta/analyze.py --wins         # only games with L1+ wins
  python scripts/meta/analyze.py --stuck        # games that showed no progress
  python scripts/meta/analyze.py --csv out.csv  # export

Reads only results/meta/. Does not touch the network.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from meta.common import ROOT
from meta.registry import REGISTRY_DIR, REGISTRY_FILE


def load_attempts(prefix: str) -> list:
    d = REGISTRY_DIR / "games" / prefix / "attempts"
    out = []
    if not d.exists(): return out
    for f in sorted(d.glob("*.json")):
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", help="Short prefix — show attempt details for one game")
    ap.add_argument("--wins", action="store_true", help="Only list games with L1+ wins")
    ap.add_argument("--stuck", action="store_true", help="Only list games with no progress")
    ap.add_argument("--csv", help="Export table to CSV")
    args = ap.parse_args()

    if not REGISTRY_FILE.exists():
        print(f"No registry yet: {REGISTRY_FILE}")
        return
    reg = json.loads(REGISTRY_FILE.read_text())

    if args.game:
        # Find full game_id starting with this prefix
        gid = next((k for k in reg if k.startswith(args.game)), None)
        if not gid:
            print(f"No registry entry for prefix '{args.game}'")
            return
        _show_game_detail(args.game, gid, reg[gid])
        return

    # Table: game prefix, attempts, max_lv, winning_strategy
    rows = []
    for gid, entry in reg.items():
        prefix = entry.get("prefix", gid.split("-", 1)[0])
        max_lv = entry.get("max_levels_completed", 0)
        if args.wins and max_lv < 1: continue
        if args.stuck and max_lv >= 1: continue
        winners = entry.get("winners", {})
        tried = entry.get("attempts_tried", [])

        # Get win_levels from profile.json if available
        pf = REGISTRY_DIR / "games" / prefix / "profile.json"
        win_levels = 0
        signals = {}
        if pf.exists():
            try:
                p = json.loads(pf.read_text())
                win_levels = p.get("win_levels", 0)
                signals = {
                    "combo": p.get("looks_like_combo_lock", 0),
                    "grid": p.get("looks_like_grid_click", 0),
                    "cursor": p.get("looks_like_cursor_game", 0),
                    "static": p.get("looks_like_static", 0),
                }
            except Exception:
                pass

        rows.append({
            "prefix": prefix,
            "max_lv": max_lv,
            "win_levels": win_levels,
            "tried_count": len(tried),
            "tried": tried,
            "winners": winners,
            "signals": signals,
        })
    rows.sort(key=lambda r: (-r["max_lv"], r["prefix"]))

    # Pretty print
    print(f"{'prefix':7} {'lv':>3}/{'tot':<3} {'tried':>5}  winners                    signals")
    print("-" * 92)
    for r in rows:
        w = r["winners"]
        wstr = ", ".join(f"L{k}:{v.get('strategy','?')}" for k, v in sorted(w.items()))
        s = r["signals"]
        sig = (f"C{s.get('combo',0):.1f} G{s.get('grid',0):.1f} "
               f"U{s.get('cursor',0):.1f} S{s.get('static',0):.1f}") if s else ""
        print(f"{r['prefix']:7} {r['max_lv']:>3}/{r['win_levels']:<3} {r['tried_count']:>5}  "
              f"{wstr:<26} {sig}")

    # Summary stats
    total = len(rows)
    solved = sum(1 for r in rows if r["max_lv"] >= 1)
    fully = sum(1 for r in rows if r["win_levels"] > 0 and r["max_lv"] >= r["win_levels"])
    print(f"\n  {total} games total | {solved} with L1+ | {fully} fully beaten")

    # Strategy win counts
    win_counter = Counter()
    for r in rows:
        for lv, w in r["winners"].items():
            win_counter[w.get("strategy", "?")] += 1
    if win_counter:
        print("\n  Strategy wins:")
        for strat, count in win_counter.most_common():
            print(f"    {strat:15} {count}")

    if args.csv:
        import csv
        out = pathlib.Path(args.csv)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["prefix", "max_lv", "win_levels", "tried_count", "tried",
                        "winners", "combo", "grid", "cursor", "static"])
            for r in rows:
                w.writerow([
                    r["prefix"], r["max_lv"], r["win_levels"], r["tried_count"],
                    ";".join(r["tried"]),
                    ";".join(f"L{k}:{v.get('strategy','?')}" for k, v in sorted(r["winners"].items())),
                    r["signals"].get("combo", 0),
                    r["signals"].get("grid", 0),
                    r["signals"].get("cursor", 0),
                    r["signals"].get("static", 0),
                ])
        print(f"\n  CSV → {out}")


def _show_game_detail(prefix: str, gid: str, entry: dict):
    print(f"=== {gid} ({prefix}) ===")
    print(f"max_levels_completed: {entry.get('max_levels_completed', 0)}")
    print(f"attempts tried: {entry.get('attempts_tried', [])}")
    print(f"winners: {entry.get('winners', {})}")
    attempts = load_attempts(prefix)
    print(f"\n{len(attempts)} recorded attempts:")
    for a in attempts:
        details_snippet = ""
        if a.get("details"):
            keys = list(a["details"].keys())[:3]
            details_snippet = "  " + "  ".join(f"{k}={a['details'][k]}" for k in keys)
        print(f"  {a.get('strategy','?'):15}  max_lv={a.get('max_levels_completed',0)}  "
              f"elapsed={a.get('elapsed',0):.1f}s  "
              f"stopped={a.get('stopped_reason','?')}{details_snippet}")


if __name__ == "__main__":
    main()
