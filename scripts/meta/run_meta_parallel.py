"""run_meta_parallel — orchestrator that processes multiple games concurrently.

Uses concurrent.futures.ThreadPoolExecutor. Each worker runs one game at a
time with its own GameSession; sessions are independent so they don't fight
over state, but they DO share the ARC-AGI-3 API server.

Total API RPS ≈ workers / STEP_DELAY. Example:
  workers=3, STEP_DELAY=0.15 → 20 RPS. The server has tolerated this so far.

Usage:
  python -u scripts/meta/run_meta_parallel.py --workers 3 --budget 240
  python -u scripts/meta/run_meta_parallel.py --workers 3 --game tr87   # single
  python -u scripts/meta/run_meta_parallel.py --workers 3 --replay

If the server starts 500-ing (rate limited), reduce --workers or raise
META_STEP_DELAY env var.
"""
from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from meta.common import ROOT, STEP_DELAY
from meta.run_meta import (
    CONFIDENCE_FLOOR,
    PER_STRATEGY_MAX_STEPS,
    PER_STRATEGY_MAX_LIVES,
    PER_STRATEGY_SECONDS,
    STRATEGY_BUDGET_MULT,
    list_games,
    run_game,
)
from meta.registry import load_registry


# Global print lock so worker logs don't interleave mid-line.
_PRINT_LOCK = threading.Lock()


def _log(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, flush=True)


def _run_one_game(prefix: str, strategies_to_run, per_strategy_seconds: float,
                  replay: bool) -> dict:
    """Wraps run_game for use in a worker. Returns the summary dict (never raises)."""
    start = datetime.datetime.now()
    _log(f"[worker] START  {prefix}  (step_delay={STEP_DELAY}s)")
    try:
        summary = run_game(
            prefix=prefix,
            strategies_to_run=strategies_to_run,
            per_strategy_seconds=per_strategy_seconds,
            replay=replay,
            verbose=True,
        )
    except Exception as e:
        traceback.print_exc()
        summary = {"prefix": prefix, "error": f"{e!r}"}
    elapsed = (datetime.datetime.now() - start).total_seconds()
    _log(f"[worker] DONE   {prefix}  ({elapsed:.0f}s)  "
         f"max_lv={summary.get('max_lv', '?')}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3,
                    help="Concurrent games (default 3). Server 500s if too high.")
    ap.add_argument("--budget", type=float, default=PER_STRATEGY_SECONDS,
                    help=f"Per-strategy time budget in seconds (default {PER_STRATEGY_SECONDS}).")
    ap.add_argument("--game", help="Single prefix. If omitted, all games.")
    ap.add_argument("--strategy", action="append", default=None)
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for prefix, gid in list_games():
            print(f"{prefix:8}  {gid}")
        return

    targets = list_games() if not args.game else [(args.game, args.game)]
    prefixes = [p for p, _ in targets]

    print(f"Parallel runner: workers={args.workers}  games={len(prefixes)}  "
          f"step_delay={STEP_DELAY}s  per_strategy_seconds={args.budget}")
    print(f"Estimated RPS ceiling: {args.workers / STEP_DELAY:.1f}")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = ROOT / "results" / "meta" / "runs"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_log = run_log_dir / f"{stamp}_parallel.log"
    _log(f"Run log: {run_log}")

    results: list = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="game") as ex:
            futures = {
                ex.submit(
                    _run_one_game,
                    prefix,
                    args.strategy,
                    args.budget,
                    args.replay,
                ): prefix for prefix in prefixes
            }
            for fut in as_completed(futures):
                prefix = futures[fut]
                try:
                    s = fut.result()
                except Exception as e:
                    s = {"prefix": prefix, "error": f"{e!r}"}
                results.append(s)
                with run_log.open("a", encoding="utf-8") as f:
                    f.write(f"{datetime.datetime.now().isoformat()}  {s}\n")
    except KeyboardInterrupt:
        print("\nInterrupted — waiting for running workers to finish cleanly…")

    # Final summary
    print("\n" + "=" * 70)
    print("PARALLEL RUN SUMMARY")
    print("=" * 70)
    reg = load_registry()
    solved = 0
    fully = 0
    for prefix, gid in targets:
        entry = reg.get(gid, {})
        if not entry:
            entry = next((v for k, v in reg.items() if k.startswith(prefix)), {})
        max_lv = entry.get("max_levels_completed", 0)
        wins = entry.get("winners", {})
        wl = 0
        pf = ROOT / "results" / "meta" / "games" / prefix / "profile.json"
        if pf.exists():
            import json
            try:
                wl = json.loads(pf.read_text()).get("win_levels", 0)
            except Exception:
                pass
        if max_lv >= 1:
            solved += 1
        if wl > 0 and max_lv >= wl:
            fully += 1
        ws = ",".join(f"L{k}:{v.get('strategy','?')}" for k, v in sorted(wins.items()))
        print(f"  {prefix:8}  lv={max_lv:>2}/{wl:<2}  {ws}")
    print(f"\n{solved}/{len(targets)} with L1+, {fully} fully beaten.")


if __name__ == "__main__":
    main()
