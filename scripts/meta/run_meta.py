"""run_meta — orchestrator.

Usage:
  python -u scripts/meta/run_meta.py                       # run all games, all strategies
  python -u scripts/meta/run_meta.py --game tr87           # single game
  python -u scripts/meta/run_meta.py --strategy action_spam --game tr87
  python -u scripts/meta/run_meta.py --replay              # re-run even fully-solved games

For each game:
  1. Open session, call characterize() → GameProfile (saved to registry)
  2. Rank strategies by strategy.confidence(profile) desc
  3. For each strategy above confidence_floor:
        - allocate budget from per_strategy_seconds
        - run it, record StrategyResult
        - if the game is fully beaten, stop
  4. Close session, move to next game

Outputs:
  results/meta/registry.json           (top-level index)
  results/meta/games/<prefix>/...      (profiles, attempts, winners)
  results/meta/runs/<ts>.log           (console log for this run)
"""
from __future__ import annotations

import argparse
import datetime
import pathlib
import sys
import traceback

# Ensure meta package imports work
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from meta.budget import Budget
from meta.characterize import characterize
from meta.common import GameSession, ROOT
from meta.profile import GameProfile
from meta.registry import (
    record_attempt,
    record_profile,
    load_registry,
    already_solved,
)
from meta.strategies import STRATEGIES


# ── Config ──────────────────────────────────────────────────────────────────
CONFIDENCE_FLOOR = 0.05          # skip strategies below this confidence
PER_STRATEGY_SECONDS = 300.0     # 5 min default per strategy per game
PER_STRATEGY_MAX_STEPS = 3000
PER_STRATEGY_MAX_LIVES = 30

# Strategies that assume click_act produces a state effect (cell fills, group
# removals, etc.). On games where the click is just cursor placement
# (looks_like_cursor_click >= 0.7) AND structured nav exists (inverse_pairs is
# non-empty), these strategies waste budget: their phase-1 scans see uniform
# "live" cells everywhere because every click moves the cursor by 1 cell, so
# they can't distinguish meaningful from meaningless click positions.
# They're still kept for click-only games (no nav) where cursor positioning
# IS the gameplay (e.g. vc33 burst at one spot).
CURSOR_CLICK_INCOMPATIBLE_STRATEGIES = {
    "grid_click",
    "grid_click_fine",
    "sequence_search",
    "cluster_click_then_nav",
}

# Per-strategy budget multiplier (× base budget). Lets heavy strategies get
# more time and cheap ones exit fast even with a big default.
#   combo_lock: big search space if valid
#   grid_click_fine: 4x more probes than coarse grid_click
#   replay_known: tiny, usually instant
#   random_walk: caps at 1000 steps (~250s); no benefit past that
STRATEGY_BUDGET_MULT = {
    "action_spam": 0.2,      # 60s default
    "mimic_target": 0.05,    # 15s default (reconnaissance)
    "cursor_walk": 0.2,      # 60s
    "replay_known": 0.1,     # 30s — ample for a replay
    "random_walk": 1.2,      # 288s at base=240 — enough to hit MAX_STEPS=1000
    "grid_click": 1.0,
    "grid_click_fine": 1.5,
    "sequence_search": 1.0,
    "nav_and_click": 1.0,
    "combo_lock": 2.0,       # combo_lock gets double budget
    "cluster_click_then_nav": 2.5,  # state-conditional probe + iteration
    "inverse_aware_walk": 1.2,      # parity with random_walk
    "mover_toggle_walk": 1.5,       # directed schema iteration
    "policy_guided_walk": 0.5,      # learned policy — small step budget
    "policy_guided_walk_cnn": 0.5,  # CNN policy — small step budget
}


# ── Helpers ─────────────────────────────────────────────────────────────────
def list_games() -> list:
    """Return [(prefix, full_id), ...] for all scorecard games."""
    from arc_agi import Arcade
    arcade = Arcade()
    out = []
    seen_prefixes = set()
    for e in arcade.available_environments:
        gid = e.game_id
        prefix = gid.split("-", 1)[0]
        if prefix in seen_prefixes:
            continue  # dedupe — there's sometimes multiple variants
        seen_prefixes.add(prefix)
        out.append((prefix, gid))
    return sorted(out)


def run_game(prefix: str, strategies_to_run=None, per_strategy_seconds: float = PER_STRATEGY_SECONDS,
             replay: bool = False, verbose: bool = True) -> dict:
    """Run characterize + strategies on a single game. Returns summary dict."""
    try:
        sess = GameSession.open(prefix)
    except StopIteration:
        return {"prefix": prefix, "error": "no game with that prefix"}
    except Exception as e:
        return {"prefix": prefix, "error": f"open failed: {e!r}"}

    log = lambda m: print(f"[{prefix}] {m}", flush=True) if verbose else None

    # 1. Characterize
    try:
        log("characterizing…")
        profile, initial_frame = characterize(sess)
        record_profile(profile, initial_frame=initial_frame)
        log(f"  profile: win_levels={profile.win_levels}  "
            f"actions={profile.available_actions}  "
            f"combo={profile.looks_like_combo_lock:.2f}  "
            f"grid={profile.looks_like_grid_click:.2f}  "
            f"cursor={profile.looks_like_cursor_game:.2f}  "
            f"static={profile.looks_like_static:.2f}")
    except Exception as e:
        traceback.print_exc()
        return {"prefix": prefix, "error": f"characterize failed: {e!r}"}

    # 2. Skip if fully solved
    if already_solved(sess.game_id, profile.win_levels) and not replay:
        log("already fully solved — skipping strategies (use --replay to force)")
        return {"prefix": prefix, "game_id": sess.game_id, "skipped": "already_solved"}

    # 3. Rank strategies
    cursor_click_gate_active = (
        getattr(profile, "looks_like_cursor_click", 0.0) >= 0.7
        and bool(getattr(profile, "inverse_pairs", None))
    )
    candidates = []
    gated: list[str] = []
    for cls in STRATEGIES:
        if strategies_to_run and cls.name not in strategies_to_run:
            continue
        strat = cls()
        conf = strat.confidence(profile)
        if conf < CONFIDENCE_FLOOR:
            continue
        if cursor_click_gate_active and strat.name in CURSOR_CLICK_INCOMPATIBLE_STRATEGIES:
            gated.append(f"{strat.name}({conf:.2f})")
            continue
        candidates.append((conf, strat))
    candidates.sort(key=lambda x: -x[0])
    if gated:
        log(f"cursor-click gate skipped: {gated}  "
            f"(looks_like_cursor_click={profile.looks_like_cursor_click:.2f}, "
            f"inverse_pairs={profile.inverse_pairs})")

    if not candidates:
        log("no strategies above confidence floor")
        return {"prefix": prefix, "game_id": sess.game_id, "strategies_run": []}

    log(f"strategies to try: {[(f'{c:.2f}', s.name) for c, s in candidates]}")

    # 4. Run each
    summary = {"prefix": prefix, "game_id": sess.game_id, "strategies_run": []}
    for conf, strat in candidates:
        mult = STRATEGY_BUDGET_MULT.get(strat.name, 1.0)
        strat_budget = per_strategy_seconds * mult
        log(f"running {strat.name} (confidence {conf:.2f}, budget {strat_budget:.0f}s)…")
        budget = Budget(
            time_seconds=strat_budget,
            max_steps=PER_STRATEGY_MAX_STEPS,
            max_lives=PER_STRATEGY_MAX_LIVES,
            label=strat.name,
        )
        try:
            res = strat.run(sess, profile, budget)
        except Exception as e:
            traceback.print_exc()
            from meta.result import StrategyResult
            res = StrategyResult(
                game_id=sess.game_id,
                strategy=strat.name,
                exception=f"{e!r}",
                stopped_reason="crashed",
            )
            res.elapsed = 0

        record_attempt(res, log_text=strat.log_text())
        summary["strategies_run"].append({
            "strategy": strat.name,
            "max_levels": res.max_levels_completed,
            "new_levels": res.new_levels,
            "elapsed": round(res.elapsed, 1),
            "stopped": res.stopped_reason,
        })
        log(f"  → max_lv={res.max_levels_completed}  "
            f"new={res.new_levels}  "
            f"elapsed={res.elapsed:.1f}s  "
            f"stopped={res.stopped_reason}")

        # If fully beaten, stop
        if res.max_levels_completed >= profile.win_levels > 0:
            log(f"FULLY BEATEN (lv{res.max_levels_completed}/{profile.win_levels})")
            break

    return summary


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", help="Short prefix of a single game (e.g. tr87). If omitted, all games.")
    ap.add_argument("--strategy", action="append", default=None,
                    help="Restrict to specific strategy names (repeatable). Default: all.")
    ap.add_argument("--replay", action="store_true",
                    help="Re-run strategies even for games already fully solved.")
    ap.add_argument("--budget", type=float, default=PER_STRATEGY_SECONDS,
                    help=f"Per-strategy time budget in seconds (default {PER_STRATEGY_SECONDS}).")
    ap.add_argument("--list", action="store_true", help="List all discoverable games and exit.")
    args = ap.parse_args()

    if args.list:
        for prefix, gid in list_games():
            print(f"{prefix:8}  {gid}")
        return

    targets = list_games() if not args.game else [(args.game, args.game)]

    # Per-run log
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log = ROOT / "results" / "meta" / "runs" / f"{stamp}.log"
    run_log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Run log: {run_log}")

    all_summaries = []
    for prefix, _gid in targets:
        try:
            s = run_game(
                prefix,
                strategies_to_run=args.strategy,
                per_strategy_seconds=args.budget,
                replay=args.replay,
            )
        except KeyboardInterrupt:
            print(f"\n[{prefix}] interrupted — aborting remaining games")
            break
        except Exception as e:
            traceback.print_exc()
            s = {"prefix": prefix, "error": f"{e!r}"}
        all_summaries.append(s)
        # Append a line to the run log without clobbering earlier ones
        line = f"{datetime.datetime.now().isoformat()}  {s}\n"
        with run_log.open("a", encoding="utf-8") as f:
            f.write(line)

    # Final summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    reg = load_registry()
    solved = 0
    total = 0
    for prefix, gid in targets:
        entry = reg.get(gid, {})
        if not entry:
            entry = next((v for k, v in reg.items() if k.startswith(prefix)), {})
        max_lv = entry.get("max_levels_completed", 0)
        wins = entry.get("winners", {})
        total += 1
        if max_lv >= 1:
            solved += 1
        print(f"  {prefix:8}  max_lv={max_lv}  winners={list(wins.keys())}")
    print(f"\n{solved}/{total} games with at least L1.")


if __name__ == "__main__":
    main()
