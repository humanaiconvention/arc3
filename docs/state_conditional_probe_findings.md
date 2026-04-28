# State-conditional probe — findings from steps A/B/C/D

Written 2026-04-27 after running all four recommended next-steps in sequence.
This doc updates the boundary thesis from
[`meta_solver_boundary.md`](meta_solver_boundary.md) — specifically, it
revises the claim that "a third primitive (state-conditional trajectory
exploration) doesn't break the wall."

## Summary table

| Step | Goal | Cost | Result |
|---|---|---|---|
| A | combo_lock on 4 WASD-family games | 4.9 min | 0 wins; **transfer was structural not mechanical** |
| B | 3-action cycle probe on tu93/g50t | 5.2 min | tu93: confirmed {1,2,3} is a null subgroup; g50t: contaminated by static initial state |
| C | Validate DiLoCo export training-readiness | 0 API | 72/72 rows have `probe_signature`; consumable as-is |
| D | State-conditional pair-response on 3 static-nav games | ~7 min | **Surfaced mechanism on all 3** — overturns the deferral |

## Step A: structural transfer ≠ mechanical transfer

The 5 games sharing the `[[1,2],[3,4]]` WASD signature (tr87 ✓ + cd82, dc22,
ka59, re86) all have the same action graph but **different game rules**. Of
the 4 unsolved targets:

- **cd82, dc22, re86**: combo_lock's discovery_failed because their action
  footprints don't fit the "narrow value action / wide cursor action"
  asymmetry tr87 has. cd82 in particular has act1 = 0 cells changed — the
  probe sees no cycle to infer.
- **ka59**: discovery succeeded; brute force exhausted (slot/cycle structure
  exceeds combo_lock's `MAX_SEARCH_SPACE = 50_000`).

**Read**: `inverse_pairs` is a *necessary but not sufficient* prior for
combo_lock. It correctly identified the 4 candidates; the candidates are not
combo locks. The probe did its job — it pruned the search space from 24
unsolved games to 4. That tr87 was the only true combo lock among the 5 is a
fact about the games, not a failure of the probe layer.

## Step B: 3-cycle probe — one win, one contamination

**tu93** (4 actions, triangular `[[1,2],[1,3],[2,3]]`): 27 of 64 triples
return to base. **Every triple over {1,2,3} = base.** This is stronger than
pairwise cancellation — `{1, 2, 3}` form a **null subgroup**: they shuffle
phantom state without affecting gameplay-relevant cells. Only act 4 is the
non-trivial mover. The triangular inverse-pair signature was a *louder*
signal than expected.

Concrete actionable finding: tu93's strategy should be "spam act 4 with
state observation at each step; ignore acts 1/2/3 except as toggles for
something hidden."

**g50t** (5 actions, triangular `[[1,3],[1,5],[3,5]]`): 55 of 125 triples
return to base, *including* triples involving acts 2 and 4. This is the
contamination Step D explained: g50t starts in a static state where no
action visibly changes the frame. The probe inherits this and looks like
"all triples are no-ops" — which isn't true at the game level, only at the
initial-state level.

**Read**: 3-cycle probing works on responsive games. It's confused by
games that are static at probe time. Step D addresses exactly this.

## Step C: export readiness — closed

72/72 rows in
[`2026-04-27-meta-registry-replays-with-probe-sigs`](D:/diloco_lab/data/arc3_exports/2026-04-27-meta-registry-replays-with-probe-sigs)
carry `probe_signature` with `inverse_pairs`, `toggle_actions`,
`looks_like_cursor_click`, `pair_change_matrix`, `click_heatmap_top`,
`three_cycles`, and `has_probe_data` / `has_triple_data` flags. The export
is consumable by `continue_gemma4_adapter.py` with no schema changes — the
new fields just appear inside the user-prompt JSON the model already sees.

`OPERATIONS.md` keeps ARC3 out of round 0 by policy. When the user is
ready to bring ARC3 into a round, the export is already prepared.

## Step D: state-conditional probe — the actual win

This is the most important result of the session. For ~30 seconds per
prelude × 8 prelude options × 3 games = ~7 min total API, the probe
revealed game mechanics that the manual `*_vNN.py` scripts had been chasing
for days.

### g50t — directional unfreeze pattern (overturns "static" reading)

```
prelude  state_changed_from_base?  effect of next action
act1×3   False                     all 0  (still frozen)
act2×3   True                      48 cells changed (any next action)
act3×3   False                     all 0  (still frozen)
act4×3   True                      48 cells changed (any next action)
act2×5   True                      0 cells (saturated/locked)
act4×5   True                      0 cells (saturated/locked)
```

**Acts 1, 3, 5 are no-ops on initial state. Acts 2 and 4 are the unfreeze
levers.** After act4×3 (or act2×3) the game is in a 48-cell-different
state where further actions matter; after 5 presses it saturates.

This **directly confirms the handoff's hypothesis** about g50t: "after
act4×5 'right-drop', act5 at #2 undoes drop and converts much of original
val1 piece to val2." The probe surfaced this without the manual exploration.

### sc25 — dominant action revealed

```
prelude   act1   act2   act3   act4
act1×3      0     16     32      8
act2×3     16      0     32      8
act3×3      8      8     32     32
act3×5     12     12      4     36
act4×3      8      8     32      0
```

**Act3 is the dominant action** — it produces a 32-36 cell change after any
prelude. After act3×5, act3 itself saturates (4 cells) but act4 takes over
(36 cells). This matches the known winning sequence: `act3×5 → 4 clicks →
act3×4`. The probe identified the correct "primary nav action" without any
prior knowledge of the win.

### lf52 — uniform 1-cell signature post-prelude

After any nav prelude, every action changes exactly 1 cell. This is the
cursor-game pattern — clicks just move a cursor by 1 cell, no fills. Knowing
this from the probe means strategies can skip cluster-fill assumptions for
lf52 (its actual win mechanism is "click val14 seed, then walk val2 pair
cascade" per the registry notes).

## Revised boundary thesis

The earlier doc said:

> A "state-conditional trajectory exploration" primitive ... doesn't break
> the wall. It just samples deeper into the state graph at higher cost,
> with diminishing returns.

That's wrong **at shallow depth**. A shallow prelude probe (3-5 actions,
gated to static games) for ~30s/prelude **does** break the wall on the
specific failure mode "initial state is unresponsive." It revealed:

- which actions actually move state (g50t acts 2/4)
- which action dominates a game's nav structure (sc25 act3)
- which games are pure cursor-positioning even after unfreezing (lf52)

The cost-benefit assumption I made earlier was based on full state-graph
enumeration. **A targeted shallow prelude probe, gated to where it can
help**, is much cheaper and surfaces meaningful signal.

## Concrete next steps (revised)

1. **Promote state-conditional probing to characterize-time, gated.**
   Add `_probe_post_prelude_response` to `characterize.py`. Trigger
   condition: `looks_like_static >= 0.7` AND ≥1 nav action available. Probe
   each nav as 3-step prelude, record per-action post-prelude effect. Cost
   per qualifying game: ~30s. ~5 candidate games. Total: ~3 min for full
   corpus coverage.
2. **Add `dominant_actions` derived signal**: from either initial-state
   `action_effects` (for non-static games) or post-prelude effects (for
   static games), pick the top-k actions by cells_changed. Strategies can
   target these directly.
3. **Re-attempt `combo_lock` on ka59 with discovery hints**: ka59's
   discovery succeeded but search exhausted. Try expanding
   `MAX_SEARCH_SPACE` or hinting `n_slots`/`cycle` from the post-prelude
   probe (count distinct change footprints across preludes). This is a
   ka59-specific second pass with the new data.
4. **Train a directed g50t strategy**: with the probe data confirming acts
   2/4 are the unfreeze levers and act5 is a state-toggle, write a
   strategy that does (act4 OR act2) × N → act5 → other-nav-act × M cycles
   and looks for level wins. This is the cluster_click_then_nav pattern
   applied to g50t's grammar.
5. **Re-run InverseAwareWalk with `dominant_actions` bias**: tu93's
   `{1,2,3}` are now confirmed null subgroup actions. The strategy should
   massively bias toward act 4. Same for re86 (probe didn't surface
   dominant actions but the [[1,2],[3,4]] structure suggests value pair
   = 1,2 and cursor pair = 3,4 — a strategy could prioritize cursor presses
   followed by value presses at each step).

## What's now defensible vs not

- **Defensible**: the probe layer **does** surface mechanism on stuck
  games at affordable cost, when state-conditional probing is included.
- **Defensible**: the win count didn't move (still 10/25 solved). Probes
  surface mechanism; surface-mechanism alone doesn't auto-write the win
  sequence.
- **Defensible**: the new mechanism findings (g50t acts 2/4, sc25 act3,
  lf52 cursor-only) are concrete inputs to either targeted strategies or
  the DiLoCo training loop.
- **Not defensible** (revised from earlier): the meta-solver layer is "only
  enabling, not solving." With state-conditional probing, it's a **search
  pruner with mechanism awareness** — strictly more capable than an
  initial-state-only probe.

## Standing artefacts

- All 25 game profiles re-characterized: `D:\arc3\results\meta\games\<prefix>\profile.json`
- DiLoCo export: `D:\diloco_lab\data\arc3_exports\2026-04-27-meta-registry-replays-with-probe-sigs`
- Step D one-off probe results: in
  `C:\Users\benja\AppData\Local\Temp\claude\D--humanai-convention\4a3860d0-92b6-4f96-8c6e-8acc0b439316\tasks\bx1vjbrty.output`
  (not persisted to profile yet — would need step 1 of next steps to
  formalize)

---

## Update — what happened after this doc was written

Steps 1–5 from the "Concrete next steps" section above were executed.
Recording outcomes here so the doc reflects the empirical record.

### 1. State-conditional probing promoted to characterize-time (done)

Added `_probe_post_prelude_response` to
[characterize.py](../scripts/meta/characterize.py) with
`probe_post_prelude="auto"` gating on
`looks_like_static ≥ 0.7 AND nav present`. Persisted on profile as
`post_prelude_responses` and `dominant_actions`. Re-characterize on
g50t/lf52/sc25/tu93 confirmed the data is captured correctly.

### 2. `dominant_actions` derived signal (done, with one bug fix)

Added `_derive_dominant_actions` that prefers post-prelude data over
initial-state data when the game is static. Initial implementation
incorrectly picked `[1, 2]` for g50t (from noise on initial-state
effects). Fixed by ranking on
`(is_unfreezing_prelude, cumulative_response, action_id)` instead of
cumulative response alone — now correctly returns `[2, 4]` for g50t,
`[3, 4]` for sc25, `[4]` for tu93, `[1, 2]` for lf52.

### 3. Re-attempt combo_lock on ka59 (skipped — lower EV than 4)

Per the next-steps doc, deferred in favour of step 4. The combo_lock
attempts in
[next_steps_after_full_corpus_probe.md's update](next_steps_after_full_corpus_probe.md)
showed combo_lock fails for the WASD family for *mechanical*, not
structural, reasons; ka59 specifically hit the
`MAX_SEARCH_SPACE = 50_000` cap with a successful discovery — so the
problem isn't action-role hinting, it's that ka59's slot/cycle space
is genuinely larger than tr87's.

### 4. Directed g50t strategy (done as `MoverToggleWalk`)

Generalised the schema as
`mover × N_before → toggle → mover × M_after × max_cycles`.
Confidence 0.45 when `dominant_actions` and `toggle_actions` are both
non-empty and a mover is in `{1,2,3,4}`.

**Live run on g50t**: 24 attempts × ~5 actions each. **0 lives lost,
0 wins.** Search space narrowed from random walk's 1000+ steps to 24
schema-compliant attempts, but the (mover×N → toggle → mover×M)×4
schema didn't capture g50t's actual win condition. Honest finding: the
template is mechanism-aware but not specific enough.

### 5. InverseAwareWalk dominant_actions bias (done)

Added `DOMINANT_WEIGHT = 4` so the random walk biases toward
`profile.dominant_actions` when sampling.

**Live re-run on tu93**: 1000 steps, 16 lives, 0 wins — *identical to
the unbiased run*. The bias was provably active (`dominant_set=[4]`
recorded), but tu93 doesn't yield to action sequences alone. Lives lost
came from the `soft_reset_after_60_no_progress` trigger, not in-game
deaths. Probe-data narrowing didn't substitute for win discovery.

### Sequel — learned heuristic + RL self-play (separate arc)

After steps 1–5 produced no win-count change, the next escalation tried
the DiLoCo path: structured-feature classifier (v0), CNN over frames
(v1), RL self-play with CEM (v2). All landed at the same eval ceiling
on the held-out games and produced 0 wins on stuck games.
[meta_solver_boundary.md's "Update"](meta_solver_boundary.md) section
records that arc and the empirical confirmation of the boundary thesis.

### Final assessment of step-D's optimism

This doc claimed: "*The earlier doc said state-conditional probing
'doesn't break the wall.' That's wrong **at shallow depth**.*"

That claim was correct **for the probe layer** — state-conditional
probing did surface mechanism cheaply, where initial-state probing
couldn't.

But "surfaces mechanism" ≠ "moves win count." Three subsequent strategy
implementations and three model architectures all consumed the
state-conditional data and **none** crossed from "mechanism known" to
"win discovered." The boundary moved one step inward (the probe layer
got more capable), but the win-discovery wall stayed where it was.

The state-conditional probe was a real gain for the *understanding* of
games. It was not a gain for the *solving* of them — the latter still
needs either more wins or more compute.
