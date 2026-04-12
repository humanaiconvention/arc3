# arc3 — Entry Status & Architecture

**Last updated:** 2026-04-12 (session 8 analysis + 4 bug fixes applied)
**Authors:** Benjamin Haslam (Bazzer) + Guilherme Ferrari Brescia
**Target:** ARC Prize 2026 / ARC-AGI-3, Milestone #1 deadline 2026-06-30

---

## Current state at a glance

| Layer | Status |
|---|---|
| Random baseline | ✅ 0% on all 4 starter games (floor established) |
| Expert recordings | ✅ 135 total files; 47 clean loaded (88 skipped, action-blind) |
| IMPALA CNN | ✅ 3-stack ConvSequence, 256-dim, 3.61M params |
| MuZero world model | ✅ `MuZeroImpalaCNNNetwork` — representation, dynamics, prediction heads |
| Data pipeline | ✅ Hardened — dual-format, `skip_action_blind`, 6 crash bugs fixed |
| Stage 1 pretrain (v4) | ✅ 50k steps, batch=256, loss≈17.4, 5.5 steps/s |
| Stage 2 fine-tune (v4) | ✅ `results/v4/stage2_finetune_2026-04-10--00-28-55.checkpoint` (20k steps) |
| Generic solver | ✅ `neurosym/generic_solver.py` — magnitude player-detect, spawn-detect, grid-scan+timeout, 6×6 click, win-cache |
| Multiple game lives | ✅ `--n-lives N` flag; stochastic coverage via repeated attempts |
| All-25-games runner | ✅ `--games all` flag; single scorecard for all games |
| vc33 Level 1 | ✅ **3 steps** via maroon_solver hybrid |
| lp85 Level 1 | ✅ **~43 steps** via generic click exploration |
| sp80 Level 1 | ✅ **~15 steps** stochastic (needs n-lives ≥ 3) |
| lp85 Level 2 | ⬜ Fixed L2 cross-level goal pollution (session 8) |
| Slider games (tu93/sc25/g50t) | ⬜ 4 fixes applied: game order, HOLD click priority, per-life position probe, level tracking |
| Other games | ❌ 22/25 games: 0 levels — mechanics unknown; zero positive reward signal in 135 recordings |
| VLA training (Gemma 4) | ❌ ABANDONED — 630 visual tokens × 8B params infeasible on RTX 2080 |
| Symbolic→MCTS wiring | ⬜ Planned — pyperplan plan as policy prior |

---

## BREAKTHROUGH: First non-zero score achieved (2026-04-10)

**Game:** vc33 | **Levels completed:** 1 | **Steps:** 53 | **Checkpoint:** v4 Stage 2

Level 1 mechanics decoded:
- Right maroon button at screen position **(x=62, y=33)** — 3 clicks needed
- Each click moves the cross marker 4 columns left
- Win when cross white marker (cols ~50) reaches slider white marker (col 38)
- Our 16-bin MuZero correctly identified **bin 11** as the target region
- Bug: bin-centre coordinates (56, 40) missed the button; fixed to exact pixel (62, 33)

**Scorecard:** 4c288924-1f30-49fc-92d0-51a787b0c646 (TTT v4e run, 04:35 AM)

---

## Overnight run summary (2026-04-09 → 2026-04-10)

### MuZero v4 training
- Stage 1: 50,000 steps in 152 min at 5.5 steps/s, loss=17.4
- Stage 2: 20,000 steps in 64 min at 5.2 steps/s, loss=17.3
- Checkpoint: `results/v4/stage2_finetune_2026-04-10--00-28-55.checkpoint`

### VLA training (ABANDONED)
- Root cause: Gemma 4 pan-and-scan multi-crop produces 630 visual tokens (not 70, despite `max_soft_tokens=70`)
- At 630 tokens × 8B params in 8-bit on RTX 2080: 944–2534 seconds per optimizer step
- 47 steps (Stage 3, 62 examples) = 12–30 hours — infeasible in any overnight window
- Decision: abandon VLA training on BEAST; consider Kaggle 2×T4 if needed

### Zero-shot VLA evaluation (`test_vla_zeroshot.py`)
- Bug found: raw `<image>` token in prompt → "tokens: 0, features: 256" mismatch
- Fix: use `processor.apply_chat_template()` with `{"type": "image"}` content block
- Results (5 frames, natural-language prompt, Gemma 4 E4B base model, 8-bit):
  - Parse success: **5/5 = 100%**
  - All frames output: `(37, 57)` → bin 14 (mode-collapsed)
  - Bin 11 hit rate: 0/5 = 0%
  - Per-frame time: ~19 minutes (630 visual tokens on split CPU/GPU)
- Conclusion: VLA base model mode-collapses to a fixed coordinate; unusable as a game-state-specific prior

### Action encoding bug discovered and fixed
- ARC remote API endpoint: `POST /api/cmd/ACTION6` requires `{x, y}` coordinates in payload
- Without coordinates → 500 Server Error
- With bin-centre coordinates (56, 40) for bin 11 → API accepts but 0 levels (misses maroon button)
- With exact maroon_solver coordinates (62, 33) → level completes in 3 steps
- Fix applied in `scripts/run_ttt_local.py`: `decode_action()` now uses maroon_solver pixel logic from observation frame

---

## TTT result history

| Run | Checkpoint | Levels | Steps | Notes |
|---|---|---|---|---|
| Random baseline | — | 0/7 | 81 | Floor |
| Hierarchical TTT v3 | v3 (10k+10k) | 0/7 | — | Wrong action mapping |
| TTT v4a | v4 (50k+20k) | 0 in 50 | 50 | bin-centre (56,40) misses button |
| TTT v4b | v4 | 0 in 0 | 0 | empty data → 500 API error |
| TTT v4c | v4 | 0 in 50 | 50 | x=0,y=0 → still misses button |
| **TTT v4d** | v4 | **1 level** | 53 | maroon_solver pixel logic; Level 2 stuck |
| **TTT v4e** | v4 | **1 level** | 53 | + block cycling; Level 2 stuck |
| **TTT v4f** | v4 | **1 level** | 3+50 | + mode-transition fix; multiple lives |
| **bj5iocggi** | v4 | **2 levels** (sp80+vc33 L1) | 400×25 | 400-step 5-lives all-25-games run |
| **bg9bsgxvv** (session 3 fixes) | v4 | **2 levels** (lp85+vc33 L1) | 80×25 | All improvements active; sp80 stochastic miss |
| **session 7** (2026-04-11--18-51-59) | v4 | **2 wins**: sp80+vc33 L1; lp85 L1→L2 stuck | 400×25 | Slider shortcut fix active; SWEEP=16 HOLD=2; lp85 L2 stuck clicking (8,40) 200 steps — goal-frame cross-level bug |
| **session 8** (ttt_session8.log) | v4 | IN PROGRESS | 400×25 | 3 fixes: one-dir slider, click-during-hold, lp85 L2 fresh exploration |

**Level 2 status — CONFIRMED HARD CAP (2026-04-10):**
- Block 0 (1,46): advances cross RIGHT by +4 per click — but ONLY TWICE: 12→16→20, then stops
- Block 1 (1,37): moves cross LEFT -4
- Blocks 2 (1,25) and 3 (1,17): no effect on cross OR slider (tested exhaustively)
- Slider: stays fixed at col 28 — never moves
- Cross maximum: col 20 (needs col 28 to win, 8 columns short)
- Tests: 25 click positions from fresh cross=20 state, sequential block sequences, ratchet patterns — ALL negative
- Conclusion: vc33 Level 2 is unsolvable with current game mechanics. Server-side cap at col 20.

---

## Entry stack (current architecture)

```
ARC-AGI-3 submission pipeline
│
├── PERCEPTION
│   └── IMPALA CNN  (external/muzero-general/models.py)
│       • 3-stack ConvSequence (channels: 16→32→32)
│       • Output: (B, 256, 1, 1) — no BatchNorm
│       • 3.61M parameters
│
├── PIXEL DECODER  (decode_action in scripts/run_ttt_local.py)
│   └── maroon_solver pixel logic (neurosym/maroon_solver.py)
│       • Reads denormalised frame to detect game mode (right/left maroon)
│       • Right mode (Level 1): click (62, 33)
│       • Left mode (Level 2+): adaptive LEFT_MAROON_BLOCKS cycling
│       • MuZero MCTS still runs for action selection + TTT learning
│
├── WORLD MODEL  (MuZeroImpalaCNNNetwork)
│   • representation()  → IMPALA encode pixel obs
│   • dynamics()        → action broadcast + residual block
│   • prediction()      → policy head (16 bins) + value head
│
├── PLANNING
│   ├── MCTS (muzero-general UCB, 50 simulations/step)
│   └── TTT (scripts/run_ttt_local.py)
│       • Scope 1: post-level TTT on that level's experience buffer (lr=3e-5)
│       • Scope 2: pre-level TTT on all prior same-game experience (lr=1e-5)
│       • Cross-game transfer: ONE shared model+optimizer for entire run —
│         weights accumulate across all 25 games (not reset per game).
│         scope2_buffer reset per game so scope2 draws from current game only.
│       • Incremental checkpoint saved after each life boundary (not just end-of-game)
│       • 60s per-step timeout prevents indefinite API hangs
│       • TTT + maroon_solver hybrid achieves Level 1 completion
│
├── SYMBOLIC LAYER  (neurosym/)
│   • Phases B→F complete on vc33
│   • STRIPS domain induced; 6-step abstract plan via pyperplan
│   • [gap] Not yet wired as MCTS policy prior
│
├── TRAINING DATA  (external/ARC-AGI-3-Agents/recordings/)
│   • 135 total recording files; 47 clean loaded
│   • 31 histories have level crossings (all bin 11, all Level 1)
│   • No Level 2 crossings in any recording
│
├── TRAINING PIPELINE  (train_pipeline.py)
│   • Stage 1: pretrain on corpus (50k steps, batch=256)
│   • Stage 2: fine-tune warm-start (20k steps, batch=256)
│   • v4 checkpoint (2026-04-10) — best to date
│
└── KAGGLE DEPLOYMENT  (kaggle/)
    • arc3_muzero_finetune.ipynb — Stage 1+2 training on Kaggle 2×T4
    • arc3_muzero_ttt.ipynb     — TTT inference + live ARC platform submission
```

---

## Checkpoint history

| Checkpoint | Steps | Corpus | TTT result | Notes |
|---|---|---|---|---|
| `v1/stage2_*_07-24-15.checkpoint` | 50k S1 + 10k S2 | all 104 recordings | 0/7 | Action labels wrong (maroon_solver Format B) |
| `v2/stage2_*_15-57-*.checkpoint` | 10k S1 + 10k S2 | 47 clean | 0/7 | Hierarchical TTT |
| `v3/stage2_*_16-27-20.checkpoint` | 10k S1 + 10k S2 | 47 clean | 0/7 | Clean labels; Level 1 still missed (x,y bug) |
| **`v4/stage2_*_00-28-55.checkpoint`** | **50k S1 + 20k S2** | **47 clean** | **1 level** | **Current best — Level 1 achieved** |

---

## Key engineering discoveries (2026-04-10)

### ARC API action encoding
- ACTION6 (click) requires explicit `x,y` in POST body: `{game_id, guid, x, y}`
- Empty data → 500 error; wrong coordinates → 0 completions
- Correct coordinates for vc33 Level 1: `x=62, y=33` (right maroon button)
- These are frame-space coordinates (0–63 range) matching arcengine ComplexAction

### vc33 game mechanics (fully reverse-engineered)
- Level 1 (right maroon): click (62, 33) three times; cross moves left 4px/click; win when col ≤ 38
- Level 2+ (left maroon): 4 blocks at y≈17/25/37/46, x=1; adaptive cycling needed
- Timer: orange pixels in row 0 count down
- **Level 2 hard cap**: Block 0 only advances cross 2 steps (+8 cols: 12→16→20). Slider fixed at 28. Gap of 8 cols is impassable. Blocks 2/3 have no effect. Level 2 is provably unsolvable with observed mechanics.

### Multiple lives fix (2026-04-10)
- `GameAction.RESET` = `from_id(0)`, NOT `from_id(1)` (ACTION1 → 400 Bad Request)
- After GAME_OVER: `env.step(GameAction.RESET)` and `env.reset()` both return `lv=1 mode=left` — server preserves Level 1 completion
- Lives 2+ correctly start in Level 2 state (not fresh Level 1) — this is expected server behavior
- Fix: after life reset, `current_level = max(1, state.levels_completed + 1)` from server, not hardcoded 1
- `--n-lives` flag added to `run_ttt_local.py`; each life generates additional Level 2 TTT training data

### Mode-transition fix (right-idle → left blocks)
- Root cause of Level 2 never being entered: after Level 1 complete, `clicks_needed=0` in right-mode → was clicking IDLE=(32,32) instead of a left-maroon block
- Fix: right-mode idle now clicks `LEFT_MAROON_BLOCKS[left_block_idx]` to trigger Level 2 transition

### VLA training infeasibility on RTX 2080
- Gemma 4 E4B: pan-and-scan creates multiple image crops regardless of `max_soft_tokens`
- Actual visual token count: 630 (not 70) due to multi-crop architecture
- Time per optimizer step: 944–2534 seconds → infeasible for any training run
- Alternative: Kaggle 2×T4 has enough VRAM to potentially train Gemma 4 VLA

### Zero-shot VLA findings
- 100% parse rate using chat-template format (was 0% with raw `<image>` token)
- Mode-collapsed: always outputs `(37, 57)` regardless of frame content
- Per-frame inference: ~19 minutes on split CPU/GPU (630 visual tokens, 8-bit Gemma 4)
- VLA prior is NOT useful in current zero-shot form; would need fine-tuning to be game-aware

---

## Known gaps / next steps

1. **vc33 Level 2 — CLOSED** — Confirmed hard cap at col 20; Level 2 unsolvable. Max score from vc33 = 1 level per game.
2. **400-step 5-lives run — IN PROGRESS** — Run `bj5iocggi` launched. Uses all improvements below + `--max-moves 400 --n-lives 5 --games all`. Expected 3-6 wins.
3. **max_moves=80 was too low** — With 64-step timers, `--n-lives 5 --max-moves 80` only gave ~1.25 lives. Fixed: now 400 steps = 6+ lives per game.
4. **Bare game IDs are 70-80× slower** — Always use `--games all` to get UUID-suffix IDs from the API (0.1-0.2s/step vs 10-12s/step for bare IDs).
5. **generic_solver improvements (session 3, bj5iocggi run uses these):**
   - `player_value()` now uses count×avg_magnitude scoring — real players move 4px, boundary artifacts move <1px, no more val=0 confusion
   - Spawn detection for mixed games: non-dir action fired at 25% rate after observing >40px frame change (was always 12.5%)
   - Slider scan HOLD=3 (was 2) for more dwell time at each position
   - Grid scan stuck timeout: advance after 12 steps without progress (prevents blocking at walls)
   - Click exploration: always-on 6×6 fine grid (36 positions) appended after object centroids
   - Win frame cache: winning frames saved to `results/win_frames/`, loaded by next run to set navigation goals
6. **generic_solver improvements (session 3, post-bj5iocggi — next run):**
   - Dead-dir detection: all probe actions = 0px → 100% smart click mode
   - Auto-animation detection: all probe actions similar px count (within 15%) and >0 → 100% non-dir mode. Catches ar25(109px), ka59(19px), re86(61px), m0r0(100px), wa30(32px), lf52(1px), bp35(47px). These games had auto-animating frames where dir actions all do the same thing.
   - Default non-dir fire rate 25% (was 12.5%). Spawn detection → 50% (was 25%).
   - No-directional branch: when no dir actions in avail (e.g. su15=[6,7], sb26=[5,6,7]) → route to smart click with action cycling through all CLICK_IDS
   - min_pixels threshold lowered from 1% (40px) to 0.25% (10px). Helps ls20 where player is 10px.
   - Dead-dir/auto-anim path fires non-click non-dir actions (e.g. action 5) at 25% rate alongside click exploration.
   - **Spawn detection bug fix**: Effect keys are `GameAction.ACTION1` not `"1"`. Fixed dir key set to `{str(GameAction.from_id(a)) for a in dir_actions}` to correctly exclude directional moves (sp80 left/right = 162px was falsely triggering spawn detection → fire_every=2 regression → fixed, sp80 now solves in 29 steps).
   - **player_value misidentification fix**: Added max_pixels guard (≤15% of frame = ~614px) and explicit exclusion of value 0 (always background/black). Previously sc25/dc22 timer contamination caused value 0 (background, thousands of pixels) to be selected as the player. Now sc25 correctly detects player_value=9 (maroon slider) with actions [3,4]. Fallback path also excludes value 0.
   - Game ordering: `--games all` fetches 25 games in API registration order. **Order varies per scorecard.** bg9bsgxvv order: tu93→g50t→sp80→ls20→lp85→dc22→m0r0→ka59→tn36→cd82→r11l→vc33→sc25→sk48→wa30→tr87→re86→lf52→bp35→ar25→sb26→su15→s5i5→ft09→cn04. bj5iocggi order: lf52→su15→m0r0→sk48→sp80→vc33→…
   - Diagnostic data collected for all 25 games (isolated probe with resets):
     Pure click: ft09, r11l, s5i5, tn36. No-dir: sb26[5,6,7], su15[6,7]. 
     Auto-anim (all dir same px): ar25, bp35, ka59, lf52, m0r0, re86, wa30.
     Genuine directional: 
     - cd82 (act 3,4: 201px; acts 1,2: 1px dead) — mixed [1,2,3,4,5,6]
     - g50t (act 2,4: 48px; acts 1,3: 0px dead) — avail=[1,2,3,4,5], slider mode [2,4] horizontal nav
     - ls20 (acts 1,3,4: 52px; act 2: 2px dead) — avail=[1,2,3,4] pure dir, **player=value 12 (10px)**, 3 effective acts → normal nav
     - sk48 (act 1: 96px, act 2: 0px wall, acts 3,4: 12px) — avail=[1,2,3,4,6,7], **player=value 1 (16px)**, slider mode [1,3]
     - tr87 (act 3,4: 28px; acts 1,2: 13-15px) — avail=[1,2,3,4] pure dir
     - tu93 (act 4: 136px; acts 1-3: 1px timer) — avail=[1,2,3,4] pure dir
     - sp80 (acts 1,2: 162px; acts 3,4: 34px) — avail=[1,2,3,4,5,6], acts 3,4 also move player
     sc25: player=value 9 (maroon slider, 71px), actions [3,4] move it (not dead-dir — slider game!). Previous misdiagnosis was due to player_value=0 (background) from timer contamination.
     re86: auto-anim (all 61px), avail=[1,2,3,4,5] no click → 100% action 5 in next run.
     m0r0: auto-anim (all 100px), avail=[1,2,3,4,5,6] → 25% act5 + 75% act6 click.
     sb26: no-dir [5,6,7] → click cycling (acts 6,7). s5i5: pure click [6].
     Unknown (timer-like): dc22 (all 9px), sc25 (all 0px isolated but timer contam in-game).
   - Timer contamination issue: in-game consecutive probes include timer ticks; sc25 and dc22 show false motion. Isolated diagnostic (with reset) is more accurate.
   - **Smart click in mixed non-dir turns**: Mixed branch 25% non-dir click now uses `infer_click_target` (smart exploration) instead of fixed bin cycling. Applies to sc25, cd82, sp80, and any mixed game where non-dir action is a click.
   - **Non-dir cycling fix**: `action_int % len(non_dir_actions)` was always 0 for non-dir turns (they occur at multiples of fire_every). Fixed with dedicated `_non_dir_ctr` that increments each non-dir turn. Now properly cycles act5↔act6 for games like sp80 and cd82.
7. **Cross-game TTT transfer fixed (session 4, 2026-04-11)** — Previously each game reset to stage2 weights, discarding all cross-game learning. Now a single shared model+optimizer runs the entire 25-game session; weights accumulate across games. scope2_buffer still resets per game (so pre-level updates use same-game experience only). Incremental checkpoints saved after each life boundary. 60s per-step timeout prevents hangs.
7. **22 games remain 0 even with all improvements (bg9bsgxvv result)** — Pure-click games (ft09, r11l, s5i5, tn36) also got 0 despite click exploration, suggesting they need game-specific click sequences. Auto-anim games (lf52, ar25, m0r0, etc.) got 0 in 80 steps — may solve in 400-step run or need game-specific analysis. Next priority: watch bj5iocggi results for any new 400-step wins.
8. **Kaggle notebook update** — Backport all solver improvements + 400-step run config.
8. **Symbolic→MCTS wiring** — Low priority.
9. **VM (RTX 4500 Pro 24GB)** — Would enable VLA Stage 3 training.

---

## Environment

| Item | Value |
|---|---|
| Python | 3.12 (via uv), venv at `D:\arc3\.venv` |
| CUDA | 13.1 / torch 2.11.0+cu126 |
| GPU | RTX 2080, 8 GB VRAM |
| UV cache | `D:\arc3\.cache\uv\cache` (env: `UV_CACHE_DIR`) |
| ARC API key | `D:\arc3\external\ARC-AGI-3-Agents\.env` |

---

## Baseline (floor) and current best

| Game | Baseline | Current best | Notes |
|---|---|---|---|
| vc33 | 0/7 levels | **1/7 levels** | TTT v4e, maroon_solver hybrid; Level 2 hard cap |
| lp85 | 0/8 levels | **1/8 levels** | generic_solver exploration, Level 1 at step ~38 |
| ls20 | 0/7 levels | 0/7 | directional game, random MCTS doesn't find solution |
| ft09 | 0/6 levels | 0/6 | complex click puzzle, needs game-specific understanding |
| sp80 | 0 | **1/? levels** | stochastic win at step ~15 (action 5=fire); needs n-lives≥3 |
| Other 21 games | 0 | 0 | first attempts 2026-04-10; timer varies 50-64 steps; 400-step run underway |

**Submission scorecard** (2026-04-10): `d4fce856-5254-4aeb-86f3-d074aab00a84` — 2/25 games solved Level 1
**Best confirmed** (session 2): 3/25 games (vc33 + lp85 + sp80 stochastic)

ARC Prize 2026 context: frontier AI ~0.26%, humans ~100%. Any non-zero score is meaningful progress.
Scorecard URL (baseline): https://three.arcprize.org/scorecards/38ab15e9-7581-416d-b3f0-53f5bb4dec99
Scorecard URL (v4e best): 4c288924-1f30-49fc-92d0-51a787b0c646 (need to look up full URL)
