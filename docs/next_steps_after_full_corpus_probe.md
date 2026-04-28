# Next steps after full-corpus probe characterization

Written 2026-04-27 after re-characterizing all 25 ARC3 games with the new
probe layer (click heatmap, action-pair non-commutativity, derived signals).

## What the corpus looks like now

25/25 games carry `has_probe_data=True`. Aggregated taxonomy:

| inverse_pairs | count | games | solved? |
|---|---|---|---|
| `[[1,2],[3,4]]` | **5** | tr87 ✓, cd82, dc22, ka59, re86 | 1 of 5 |
| `[[1,2],[1,3],[2,3]]` | 1 | tu93 | no |
| `[[1,3],[1,5],[3,5]]` | 1 | g50t | no |
| `[]` | 18 | everything else | 9 of 18 |

| cursor_click ≥ 0.9 | count | games |
|---|---|---|
| 1.00 | 4 | sb26 ✓, vc33 ✓, re86, ar25 |
| 0.00 | 21 | rest |

| toggle_actions | count | games |
|---|---|---|
| `[5]` | 2 | re86, wa30 |
| `[1,2,3]` | 1 | tu93 |
| `[1,2]` | 1 | cd82 |
| `[1,3,5]` | 1 | g50t |
| `[2]` | 1 | sk48 |
| `[]` | 19 | rest |

## The headline finding

**5 games share the exact same WASD action graph (`[[1,2],[3,4]]`).** Of those,
tr87 is the only one solved — it yielded to `combo_lock` with action roles
`fwd_cursor=4, bwd_cursor=3, fwd_value=1, bwd_value=2`. Those role assignments
*directly correspond* to the inverse-pair structure: 3↔4 is the cursor pair,
1↔2 is the value pair. The probe didn't infer the role assignments, but it
*did* identify which 4 games have the structural prerequisite for tr87's
solution to transfer.

**This is the first concrete cross-game prior the meta-solver layer has
produced.** It's also the cheapest test of the meta-solver direction: try
tr87's combo_lock parameters on cd82, dc22, ka59, re86. Each is one
deterministic strategy run.

## Priority-ordered next steps

### A. Cheapest + highest expected value: combo_lock on the WASD family

**Action**: run `combo_lock` on cd82, dc22, ka59, re86 with tr87's known
parameters (`fwd_cursor=4, bwd_cursor=3, fwd_value=1, bwd_value=2`,
varying n_slots and cycle).

**Cost**: 4 games × ~5 min strategy budget = ~20 min API.

**Expected outcomes**:
- If any of the 4 wins: confirms cross-game transfer via inverse-pair
  signature. Goes from 10 → 11–14 wins. Validates the entire probe layer
  as discovery infrastructure, not just gating.
- If none win: confirms the games share *structure* but differ in *slot
  arrangement / cycle length*. Still useful — narrows the remaining work
  to "find the right (n_slots, cycle) for each."

**Why first**: the only experiment in this list whose outcome could
materially change the win count, at the lowest cost.

### B. Cheap structural extension: 3-action cycle probe for triangular games

**Action**: extend the action-pair probe to also test ordered triples
`(a, b, c)` for tu93 and g50t. Detect 3-cycles (`a∘b∘c == identity`).

**Cost**: O(n³) per game; for n=4 that's 64 triples × ~4 calls = ~80s. Two
games = ~3 min.

**Expected outcome**: distinguishes "true 3-cycle group structure" from
"happens-to-share-pairwise-cancellation." Either way, sharper prior for
the next strategy iteration. Doesn't directly produce a win — feeds the
discovery layer (DiLoCo) better features.

### C. Wire signatures into DiLoCo training (no API)

**Action**: kick off an adapter training run in `D:\diloco_lab` against
the latest export
(`data/arc3_exports/2026-04-27-meta-registry-replays-with-probe-sigs`).
Compare loss/perplexity vs the baseline export without `probe_signature`
(`2026-04-27-meta-registry-replays`).

**Cost**: zero live API; pure ML. Time depends on adapter training
infrastructure already in place.

**Expected outcome**: if probe_signature improves training dynamics, that
validates the export wiring. If it doesn't, the signature representation
needs work (more features? different encoding?) before it's adapter-ready.
This is the **closure** of the original handoff loop: probe layer → tensor
exports → adapter training.

### D. State-conditional cluster discovery (the option I deferred earlier)

**Action**: probe-with-prelude in `characterize` to find cluster cells in
games like sc25 where clicks-from-reset are inert. Use the registry
winning sequences as known-good preludes for validation.

**Cost**: high. ~10 min per game with full prelude × grid sweep.

**Expected outcome**: lets `cluster_click_then_nav` (or future grammar
templates) actually find clusters in coord-puzzle games. The discovery
boundary moves slightly but doesn't solve the core problem.

**Why last**: most expensive, lowest expected value. Don't pursue until A
and B are done and we know whether structure-based transfer alone is
enough.

## What I'd do, if it were my call

In order:

1. **(A) combo_lock on 4 WASD-family games** — 20 min, falsifiable,
   directly tests the meta-solver thesis.
2. **(C) wire signatures into a real DiLoCo training run** — 0 API,
   closes the loop.
3. **(B) 3-action cycle probe** — only if (C) wants more features.
4. **(D)** — only if (A) hits the floor on what action-graph priors alone
   can do.

## Standing artefacts

- Full corpus profiles: `D:\arc3\results\meta\games\<prefix>\profile.json` (25/25)
- Re-characterize log: `D:\arc3\results\meta\recharacterize_progress.json`
- DiLoCo export with probe sigs:
  `D:\diloco_lab\data\arc3_exports\2026-04-27-meta-registry-replays-with-probe-sigs`
- Boundary doc: [`docs/meta_solver_boundary.md`](meta_solver_boundary.md)

---

## Update — outcomes of all 4 next-steps (later same session)

The four steps were executed in the recommended order. Outcomes:

### A. combo_lock on the 4 unsolved WASD-family games
Bumped `combo_lock.confidence` to 0.50 when `inverse_pairs` contains both
`(1,2)` and `(3,4)`, then ran on cd82, dc22, ka59, re86.
**Result: 0 wins.** Honest finding: `inverse_pairs` is a necessary but not
sufficient condition for combo_lock. Three of the four had
`discovery_failed` because their action footprints didn't match tr87's
"narrow value action / wide cursor action" asymmetry. ka59's discovery
succeeded but its cycle/slot count exceeded combo_lock's
`MAX_SEARCH_SPACE = 50_000`. The 5 WASD games share an action graph but
different game rules.

### C. DiLoCo export training-readiness
Smoke-checked: 72/72 rows carry `probe_signature`, manifest validates,
no parse errors. Consumable by `continue_gemma4_adapter.py` with no
schema changes. `OPERATIONS.md` keeps ARC3 out of round 0 by policy, so
"kick off training" is a Kaggle-upload + user-decision step.
**Result: export ready. Training not launched (policy-gated).**

### B. 3-action cycle probe
Added `_probe_action_triples` to `characterize`, gated on
`len(inverse_pairs) ≥ 3`. Live run on tu93 + g50t.
**Results:**
- **tu93** — 27 of 64 triples return to base. Every triple over `{1,2,3}`
  = base. Confirms `{1,2,3}` form a *null subgroup* on tu93 — they shuffle
  phantom state without affecting gameplay. Only act 4 is the lever. The
  triangular inverse-pair signature was a *louder* signal than expected.
- **g50t** — 55 of 125 triples return to base, *including* triples
  involving acts 2 and 4. Contamination from the static initial state
  (no action visibly changes the frame from initial). Surfaced the
  motivation for step D.

### D. State-conditional cluster discovery
Originally deferred per this doc as "most expensive, lowest expected
value." Empirically that was wrong — see
[state_conditional_probe_findings.md](state_conditional_probe_findings.md)
for the full results. Short version: **for ~30s API per prelude × 8
preludes × 3 candidate games (~7 min total), the probe surfaced
mechanism on all three:**
- **g50t**: acts 2 and 4 unfreeze the puzzle (3-step prelude → 48-cell
  state shift). Acts 1, 3, 5 are inert on initial state. Confirms the
  manual handoff hypothesis structurally.
- **sc25**: act 3 is the dominant nav (32–36 cell change after any
  prelude), matching the known winning sequence's act3×5 prelude.
- **lf52**: uniform 1-cell post-prelude — cursor-game even after
  unfreezing.

This overturned the original deferral. Step D was promoted to a
first-class characterize-time probe (gated on
`looks_like_static ≥ 0.7 + nav present`) and a `dominant_actions` derived
signal was added that prefers post-prelude data when available.

### Net win-count change after A/B/C/D

**0.** All four executed. Probe layer surfaced new mechanism for
several games. `dominant_actions` started feeding strategy gating and
biasing. `MoverToggleWalk` strategy was written to consume the new
signals — 24 directed attempts on g50t, 0 wins. The discovery problem
was not moved. The empirical conclusion fed into the boundary-thesis
update in [meta_solver_boundary.md](meta_solver_boundary.md).
