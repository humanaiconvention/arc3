# arc3 random-agent baselines

**Run date:** 2026-04-07
**Agent:** `random` (from `external/ARC-AGI-3-Agents/agents/`)
**Tags:** `baseline`
**Sweep script:** `D:\arc3\scripts\run_random_baseline.py`

## Results

| game | score | levels | actions | scorecard URL |
|---|---:|---:|---:|---|
| ls20 | 0.00 | 0/7 | 81 | https://three.arcprize.org/scorecards/19865133-7eed-4455-b2d1-c1e4520bc1ad |
| lp85 | 0.00 | 0/8 | 81 | https://three.arcprize.org/scorecards/da77bfd7-080d-4cc3-967c-25243605799b |
| ft09 | 0.00 | 0/6 | 81 | https://three.arcprize.org/scorecards/5c13119a-b4b8-4ecc-8f41-82f60a2f7513 |
| vc33 | 0.00 | 0/7 | 81 | https://three.arcprize.org/scorecards/38ab15e9-7581-416d-b3f0-53f5bb4dec99 |
| **total** | **0.00** | **0/28** | **324** | |

## Interpretation

This is the expected floor. ARC-AGI-3 games require intelligent exploration, world-modeling, goal inference, and planning — none of which the random agent does. Random achieves 0% on every starter game across 28 levels and 324 total actions (81 per game). Any future agent we ship must beat this baseline; the meaningful question is **how many levels** and **how many actions per level** versus random's 0/28 and 81/0.

For context, the ARC Prize 2026 reports frontier AI at 0.26% on this benchmark and humans at 100%. The baseline-to-top gap is 0% → 100%, with the prize structure rewarding any non-trivial progress along that curve.

## Per-game level counts

- ls20: 7 levels
- lp85: 8 levels
- ft09: 6 levels
- vc33: 7 levels
- **Total**: 28 levels across 4 starter games

These four are the public starter games, not the full evaluation set. The actual prize evaluation is hidden / served via the platform.

## Reproduction

```bash
# Requires ARC_API_KEY in D:\arc3\external\ARC-AGI-3-Agents\.env
python D:\arc3\scripts\run_random_baseline.py
```

Per-game logs in this directory: `<timestamp>_<game>_random.log`.

## Known cosmetic bug

The runner reports `exit: 2` for successful runs. Root cause: the official `main.py` ends with `os.kill(os.getpid(), signal.SIGINT)` and a SIGINT handler that calls `sys.exit(0)`. On Windows, this signal-handler exit path returns 2 instead of 0 to the parent. The scorecards are still successfully posted to the ARC platform — the exit code is misleading. The runner should be updated to detect a scorecard URL in the log and treat that as success regardless of exit code. Tracking item, not blocking.
