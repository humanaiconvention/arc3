# arc3 meta-solver

Strategy-tier system for ARC-AGI-3 games. Instead of hand-crafting a solver
per game, we characterize each game and run a ranked sequence of generic
strategies. Insights become new strategies (applicable to all games), not
new scripts (applicable to one game).

## Quick commands

```bash
# Full run across all games (~2-3h per iteration)
python -u scripts/meta/run_meta.py --budget 900

# Slower parallel run when the API is touchy
META_STEP_DELAY=0.35 python -u scripts/meta/run_meta_parallel.py --workers 2 --budget 240

# Single game
python -u scripts/meta/run_meta.py --game tr87 --budget 600 --replay

# Results table
python scripts/meta/analyze.py
python scripts/meta/analyze.py --game tr87          # details
python scripts/meta/analyze.py --wins               # only winners
python scripts/meta/analyze.py --csv out.csv        # export

# Re-run with bigger budgets on stuck games
python scripts/meta/iterate.py --budget 1800 --top-n 5

# Seed registry with known wins (skips re-search)
python scripts/meta/seed_known_wins.py

# Export replayable wins into clean training recordings
python scripts/meta/export_replay_recordings.py --out-dir recordings/meta_replays

# Submit the slower parallel run to Hugging Face Jobs
python scripts/meta/launch_hf_job.py --dry-run

# List discoverable games
python scripts/meta/run_meta.py --list
```

## Layout

```
scripts/meta/
├── common.py              GameSession, step pacing, frame/life helpers
├── budget.py              Time/step/life budget + expiry
├── profile.py             GameProfile dataclass (characterization output)
├── result.py              StrategyResult dataclass
├── registry.py            Persistent per-game results (results/meta/)
├── characterize.py        Probe each action → heuristic signals
├── run_meta.py            Orchestrator: rank strategies, run by confidence
├── analyze.py             Registry → summary table
├── iterate.py             Re-run stuck games with escalated budgets
├── seed_known_wins.py     Inject prior wins into registry
└── strategies/
    ├── base.py            Strategy base class
    ├── action_spam.py     Press each action N times (baseline)
    ├── random_walk.py     Random actions (baseline)
    ├── combo_lock.py      Auto-discover N×K slot puzzle + brute force
    ├── cursor_walk.py     Enumerate cursor positions, sample values
    ├── grid_click.py      ACTION5/6 coarse grid (step=8)
    ├── grid_click_fine.py step=4 variant for non-aligned clicks
    ├── nav_and_click.py   Cursor cycle + grid click combinations
    ├── mimic_target.py    Reconnaissance: region detection
    └── replay_known.py    Re-execute registry-recorded wins
```

## Design rules

1. **No bespoke per-game logic.** If we need new behavior, it becomes a
   generic strategy that applies to any matching game.
2. **Every strategy respects the Budget.** No infinite loops.
3. **Every attempt writes to the registry.** Never silent failures.
4. **Confidence is a hint, not a gate.** Low-confidence strategies still run
   (above the floor of 0.05) — baseline strategies catch easy wins even when
   heuristics miss them (m0r0 was won by random_walk despite 0.0 signals).

## Output layout

```
results/meta/
├── registry.json                     top-level per-game index
├── console/                          per-run console logs
├── runs/                             compact per-run summaries
└── games/<prefix>/
    ├── profile.json                  GameProfile from last characterize
    ├── initial_frame.npy             first frame snapshot
    └── attempts/
        ├── <ts>_<strategy>.json      StrategyResult
        └── <ts>_<strategy>.log       strategy-specific log buffer
```

## Progress so far

See `STATUS.md` for the live scoreboard. Meta-solver iteration 1 (2026-04-19)
delivered 3/25 L1+ wins from organic strategy application.

## Adding a new strategy

1. Create `strategies/my_thing.py` with a class inheriting `Strategy`.
2. Implement `confidence(profile)` (0.0–1.0) and `run(sess, profile, budget)`.
3. Add it to `STRATEGIES` in `strategies/__init__.py`.
4. Test on a single game: `run_meta.py --game tr87 --strategy my_thing`.
5. Let the orchestrator decide when to use it across all games.

Strategies must:
- Return a complete StrategyResult (use `self._start()` / `self._finalize()`).
- Call `self._note_level(res, sess)` after every action that might win.
- Check `budget.expired()` before each action.
- Handle GAME_OVER via `sess.soft_reset()` (costs a life).

## What still needs new strategies

(updated as we learn)

- **Target↔state region match** (tr87 L1-style box decoding, vc33)
- **Sequence-of-clicks** where one click doesn't win but a specific sequence does (lp85's 187-step natural solve)
- **State-dependent clicks** (damage-aware grid sweep that learns which positions are lethal)
- **Multi-level mechanics** (tr87 L2 is entirely different from L1 — need re-characterize-on-level-up)
