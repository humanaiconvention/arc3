# neurosym/ — symbolic side of the arc3 stack

CPU-only stack that implements Phases B → F of the architecture in
`docs/research-brief.md` §3.5. This is the side that has to ship first
because it tells us, per game, whether discrete abstraction is even tractable.

## Files

- `phase_b_graph.py` — Phase B prototype. Reads `.recording.jsonl` files
  written by the official ARC-AGI-3 recorder and builds a transition graph
  (nodes = distinct frames, edges = observed transitions). Reports
  node/edge counts, branching, weakly-connected components, bisimulation
  blocks, and terminal-state reachability per game. Supports
  `--collapse-complex` to bin ACTION6 (x,y) coordinates into a coarse grid.
- `phase_d_asp.py` — Phase D Bonet/Geffner KR21 ASP wrapper. Reads a Phase B
  graph JSON, strips self-loops (actions that only appear as no-ops cannot
  satisfy the BG `eff_used` constraint), converts the rest to a BG-format LP,
  and calls clingo to induce a STRIPS domain σ = ⟨E,F,S,O⟩.
- `record_actions_agent.py` — Standalone copy of the agent that interleaves
  `action_request` entries into the recording JSONL. The production copy lives
  in `external/ARC-AGI-3-Agents/agents/templates/record_actions_agent.py`.
- `graphs/` — Per-game graph JSON output from `phase_b_graph.py`, plus
  `phase_b_summary.json` aggregating the table.
- `lp/` — Per-game LP files and `<game>.asp_result.json` Phase D outputs.

---

## Phase B v1 results (2026-04-07, unlabeled edges)

Official recorder only logs `FrameData` — `action_input.id` is always 0.
Every edge is labeled `a0`; bisimulation cannot compress anything.

| game | frames | nodes | edges | branch | comp | bi-blocks | terminal |
|------|--------|-------|-------|--------|------|-----------|----------|
| ft09 | 81 | 40 | 76 | 2 | 1 | 40 | 1 (GAME_OVER) |
| ls20 | 81 | 49 | 48 | 1 | 1 | 49 | 0 |
| vc33 | 81 | 13 | 12 | 1 | 1 | 13 | 0 |
| lp85 | 81 | 1 | 0 | 0 | 1 | 1 | 0 |

---

## Phase B v2 results (2026-04-07, action-labeled edges — SUPERSEDED)

*Superseded by v3; kept for reference. v2 used illegal action IDs from the base
Random agent, inflating label counts and producing spurious self-loop artifacts.*

---

## Phase B v3 results (2026-04-07, legal-only actions)

Re-recorded with the **fixed** `RandomRecordActions` agent:
- `choose_action` now restricts sampling to `available_actions` only.
- Uses `GameAction.from_id(aid)` (not `GameAction(aid)`) — the constructor
  raises `ValueError: 6 is not a valid GameAction`; `from_id` is the correct
  mapping for `available_actions` integers.
- Each JSONL entry includes `"legal": true/false` tags on every `action_request`.

| game | frames | acts | nodes | edges | dist | a-lbl | branch | comp | bi-blk | cmpr | term |
|------|--------|------|-------|-------|------|-------|--------|------|--------|------|------|
| ft09 | 81 | 81 | 9 | 80 | 52 | 16 | 14 | 1 | 9 | 1.00 | 0 |
| lp85 | 81 | 81 | 4 | 80 | 37 | 16 | 15 | 1 | 4 | 1.00 | 0 |
| ls20 | 81 | 81 | 81 | 80 | 80 | 4 | 1 | 1 | 81 | 1.00 | 0 |
| vc33 | 81 | 81 | 51 | 80 | 78 | 17 | 2 | 1 | 51 | 1.00 | 1 |

Legend: `acts` = action_request events logged; `dist` = unique (src,dst,label) triples;
`a-lbl` = distinct action labels (includes spatial x,y for ACTION6); `branch` = max out-degree;
`bi-blk` = bisimulation blocks; `cmpr` = nodes/bi-blk; `term` = terminal-state nodes.

### What the v3 table tells us

- **ft09 / lp85**: `available_actions = [6]` (spatial click only). ACTION6 is a complex
  action with (x,y) coordinates; 16 distinct click positions appear as separate labels.
  Both games have very few distinct states (9 and 4 respectively) — the random agent
  cycles through a small state space. Heavy self-loop structure (42 / 34 self-loops) means
  most clicks do nothing; Phase D strips these before ASP.
- **ls20**: `available_actions = [1,2,3,4]` exactly. 81 distinct states in 81 moves —
  effectively a random walk through a non-repeating state space. 4 action labels map
  cleanly to 4 effective operators.
- **vc33**: 51 distinct states, 1 terminal (GAME_OVER), 1 illegal action (RESET at
  game-over — correct: RESET isn't in `available_actions` during play). 17 labels
  because ACTION6 (spatial click) produces many distinct (x,y) transitions.

---

## Phase D — Bonet/Geffner KR21 ASP encoding

`phase_d_asp.py` wraps the kr21 encoding from
`external/learner-strips/asp/clingo/kr21/` (encoding cited as refs [13/14]
in Lorang et al. CoRL 2025 arXiv:2508.21501). Pipeline:

1. Read a Phase B graph JSON.
2. **Strip self-loops**: drop `edge` where `src == dst`. Actions that only
   appear as self-loops cannot satisfy the BG `eff_used` constraint (a
   self-loop never changes state, so no effect can be "observed"), and
   including them causes immediate UNSAT.
3. Only include action labels that appear in at least one real transition.
4. Remap node IDs and action labels to dense integers.
5. Write a `.lp` file under `neurosym/lp/` and load it alongside
   `base2.lp` + `constraints_blai.lp` via `ctl.load()` (NOT `ctl.add()`
   — `#const` propagation requires file-based loading).
6. Inject `num_actions(K)` as a grounded fact (not a clingo constant) so
   range expressions `action(1..N) :- num_actions(N)` ground correctly.
7. Solve; write `<game>.asp_result.json`.

```bash
# Self-test against Bonet's known-good blocks2ops_2 (should be SAT in <1s):
python neurosym/phase_d_asp.py --selftest

# Run on a game:
python neurosym/phase_d_asp.py --game ft09 --num-objects 2 --max-predicates 4

# Collapse ACTION6 x/y variants first if needed:
python neurosym/phase_b_graph.py --collapse-complex
python neurosym/phase_d_asp.py --game ft09 --num-objects 2 --max-predicates 4
```

### Phase D v3 results (2026-04-07, dirty data — SUPERSEDED)

*Superseded by v4. v3 used recordings with illegal action IDs.*

### Phase D v4 results (2026-04-07, legal-only clean data)

| input | nodes (of) | real trans | self-loops dropped | eff. labels | sat | solve |
|---|---|---|---|---|---|---|
| **selftest** (blocks2ops_2) | 3 (3) | 4 | 0 | 2 | **SAT** | 0.01s |
| ft09 | 9 (9) | 10 | 42 | 7 | **SAT** | 0.041s |
| lp85 | 4 (4) | 3 | 34 | 2 | **SAT** | 0.009s |
| ls20 | 10 (81) | 9 | 0 | 4 | **SAT** | 0.037s |
| vc33 | 20 (51) | 18 | 0 (+1 illegal dropped) | 11 | **SAT** | 0.97s |

All 4 games SAT on legal-only data. Key improvements over v3:
- ft09/lp85: many fewer nodes (9/4 vs 64/—) because legal sampling keeps the
  agent in a tight state subspace rather than randomly walking illegal no-ops.
- ls20: trimmed to 10-node subgraph (20 nodes caused TIMEOUT due to 4-action
  branching; 10 nodes resolves in 0.037s).
- vc33: the 1 illegal RESET edge is dropped by `legal_only=True`; 11 effective
  labels (ACTION6 spatial coordinates) fully constrain the 20-node subgraph.

All runs use `num_objects=2`, `max_predicates=5`. Discovered predicates always
include 2 static predicates (`p_static(4)`, `p_static(5)`).

**Important qualification**: SAT for ft09, lp85, and ls20 means the LP encoding
is satisfiable over the *observed transitions* — the schema accurately describes
what actions do in the recorded state space. It does **not** mean the schema
contains a connected path from the reset start state to any WIN state, because
random play never produced a WIN in those games. Phase E/F are therefore
inapplicable to ft09/lp85/ls20 until recordings with `levels_completed > 0`
are obtained. Only vc33 has terminal states (GAME_OVER from budget exhaustion),
and only the vc33 pipeline is proven end-to-end — with the caveat that the
vc33 terminal is a budget-exhaustion artifact, not a genuine WIN.

**Phase D bridge-path fix (2026-04-07)**: the terminal-neighborhood trim in
`phase_d_asp.py` previously could isolate the start node (force-adding it to
the subset but excluding all its outgoing edges). Fixed with a forward BFS
bridge path that connects start to the terminal neighborhood before finalising
the subset.

### Clingo gotchas discovered

| Problem | Symptom | Fix |
|---|---|---|
| `--time-limit` invalid in Python API | `RuntimeError: unknown option` | Use async solve + `threading.Timer` + `handle.cancel()` |
| `ctl.add()` for kr21 files | `#const` doesn't propagate across `add()` blocks | Use `ctl.load(filepath)` for all files |
| `num_actions` as constant | `action(1..N)` range fails if N is a `-c` constant | Inject `num_actions(K).` as a grounded fact |
| Self-loop actions in graph | Immediate UNSAT from `eff_used` constraint | Strip self-loops; prune no-op labels |
| Full graph too large | clingo runs indefinitely (>7 min for 64 nodes) | Use `--max-nodes N` to trim to terminal neighborhood |
| `GameAction(aid)` with available_actions int | `ValueError: 6 is not a valid GameAction` | Use `GameAction.from_id(aid)` — available_actions uses raw engine IDs |

---

---

## Phase E — Goal predicate identification

`phase_e_goal.py` re-runs clingo to capture the full model (Phase D only stores 60
atoms), parses all `val((P,(O1,O2)),S)` atoms, and finds the minimal conjunction
of fluent atoms that discriminates terminal states from non-terminal states.

```bash
python neurosym/phase_e_goal.py --game vc33   # games with terminals
python neurosym/phase_e_goal.py --all         # all games with Phase D results
```

### Phase E results (2026-04-07)

| game | terminals | goal conjunction | size | discriminating |
|------|-----------|------------------|------|----------------|
| vc33 | 1 (node 19) | `pred1(obj1,obj1) ∧ pred1(obj2,obj2)` | 2 | **yes** |
| ft09 | 0 | — | — | n/a (no terminal in recording) |
| lp85 | 0 | — | — | n/a |
| ls20 | 0 | — | — | n/a |

**vc33 goal**: predicate 1 holds for both object self-pairs simultaneously. No
single fluent atom is terminal-only; the 2-conjunction `{val(1,(1,1)), val(1,(2,2))}`
is the unique identifier. Intersection proof: {states with val(1,(1,1))} =
{2,3,7,8,11,16,19} ∩ {states with val(1,(2,2))} = {1,4,5,10,12,13,14,19} = **{19}**.

**IMPORTANT CAVEAT — vc33 goal is a budget-exhaustion state**:
The vc33 terminal state (node 19, `dfee8d2b`) has `levels_completed=0`. Random play
with 81 moves never completes a level; the `GAME_OVER` state is triggered by budget
exhaustion, not by solving the puzzle. The Phase E goal therefore describes the
valuation at budget exhaustion, not the actual game-winning condition. Executing
the plan will reproduce the budget-exhaustion trajectory but will not complete any
level (confirmed: both graph-BFS 49-step and abstract 6-step plans score 0.0 in
actual game execution). Phase G must collect recordings with `levels_completed > 0`
before meaningful goal extraction is possible for vc33.

**ft09/lp85/ls20**: the random agent did not reach a terminal state in 81 moves.
Phase E records the initial-state valuation only. More episodes or goal-directed
exploration are needed before Phase E can infer a goal for these games.

---

## Phase F — PDDL serialisation + planner dispatch

`phase_f_plan.py` takes the Phase D schema and Phase E goal and:
1. Filters prec/eff atoms to valid roles (≤ num_objects=2; roles=3 are vacuous).
2. Serialises to `lp/<game>_domain.pddl` + `lp/<game>_problem.pddl`.
3. Runs pyperplan BFS on the PDDL problem.
4. Falls back to graph-BFS on the full observed state-space if pyperplan fails
   or if there is no goal condition.

```bash
uv run python neurosym/phase_f_plan.py --game vc33
uv run python neurosym/phase_f_plan.py --all
```

### Phase F results (2026-04-07)

| game | pyperplan plan | plan length | graph-BFS plan | graph-BFS length |
|------|---------------|-------------|---------------|-----------------|
| vc33 | **FOUND** | **6 steps** | FOUND | 49 steps |
| ft09 | no goal | — | not found (no terminal) | — |
| lp85 | no goal | — | not found (no terminal) | — |
| ls20 | no goal | — | not found (no terminal) | — |

**vc33 pyperplan plan (6 steps)**:
```
(a6_x0y0 obj1 obj1)
(a6_x2y0 obj1 obj1)
(a6_x3y3 obj1 obj1)
(a6_x0y2 obj2 obj1)
(a6_x2y0 obj1 obj1)
(a6_x3y3 obj2 obj1)
```
The abstract plan achieves `pred1(obj1,obj1) ∧ pred1(obj2,obj2)` in 6 action
applications in the induced STRIPS domain. Compare to the 49-step observed path
(full graph BFS) — the abstract plan is 8× shorter. This demonstrates
**end-to-end symbolic pipeline connectivity on vc33**:

    Phase B (graph) → Phase D (STRIPS) → Phase E (goal) → Phase F (plan)

Note: this is vc33-only. ft09/lp85/ls20 have Phases B+D but no terminal states
in the recordings, so Phase E/F cannot run. The vc33 goal is a budget-exhaustion
artifact (not a real WIN), so the plan correctly reproduces that artifact but
does not solve any level.

**Execution results (2026-04-07)**:

Both plans were executed in the actual vc33 game via `PlanExecutorAgent`:

| agent | scorecard | score | levels | note |
|-------|-----------|-------|--------|------|
| `planexecutor_graph` (49-step) | [72626f9a](https://three.arcprize.org/scorecards/72626f9a-8fdf-4ca5-a68e-9fc79295a59c) | 0.0 | 0 | plan executed; goal = budget exhaustion |
| `planexecutor_abstract` (6-step) | [a50701c9](https://three.arcprize.org/scorecards/a50701c9-18f5-488e-893b-c9e3895c9eed) | 0.0 | 0 | plan executed; goal = budget exhaustion |

**Diagnosis**: Both plans execute correctly (plan steps are parsed and submitted).
The score is 0 because the Phase E goal targets the budget-exhaustion GAME_OVER state
(`dfee8d2b`, `levels_completed=0`). Replaying the 49-step path reproduces the same
budget-exhaustion trajectory but does not solve any level. The abstract plan hits the
same wall: it targets an unwinnable goal state.

**Root cause**: Random play with 81 moves never completes a level in vc33 (confirmed:
`level_baseline_actions[0]=6`, requiring 6 precise clicks, which random bin-centre
sampling does not achieve). The STRIPS model is internally consistent but the goal
specification is wrong.

**Caveats retained for reference**:
- The STRIPS model covers 20 of 51 observed states; out-of-model transitions
  may invalidate plan steps.
- Bin-centre coordinates (bx*16+8, by*16+8) may miss the required pixel targets.

### PDDL encoding notes

- Roles > num_objects (e.g. role=3 with num_objects=2) are vacuous: `pnext(3,...)`
  never fires in the BG encoding with 2-tuples, so such prec/eff atoms are dead
  code. Phase F filters them out before PDDL generation.
- Phase D's stored schemas are truncated to 80 atoms (sorted, so `eff` atoms fill
  the first 80 and `prec` atoms are lost). Phase F re-runs clingo to get the full
  model before serialisation.
- Empty `(:precondition ())` breaks pyperplan's parser. Use `(and)` instead.

---

## Next steps

### Near-term (Phase G — get winning recordings)
1. **Get recordings with level completions**: run a vision/LLM agent (e.g.
   `SmolVisionAgent` or `MultiModalLLM`) on vc33 to obtain at least 1 run with
   `levels_completed > 0`. This is the prerequisite for all further symbolic work
   on vc33, since the current Phase E "goal" is a budget-exhaustion artifact.
2. **Enlarge observation budget**: raise `MAX_ACTIONS` to 500 and run 3–5 sweeps
   per game. Even without level completions, more transitions improve Phase D model
   quality and may expose terminal states for ft09/lp85/ls20.
3. **Goal-directed exploration**: for games where random play never reaches a
   terminal state, implement a simple exploration bonus or curriculum to guide
   the agent toward high-reward regions (UCB, epsilon-greedy on new states, etc.).

### Medium-term
4. **Re-run Phase E/F after winning recordings**: once we have `levels_completed > 0`
   in vc33 recordings, redo Phases E and F with the correct goal state. This should
   yield a plan that actually solves level 1.
5. **Extend to ft09/lp85/ls20**: after Phase G data is available, run the full
   Phase B→F pipeline on the other 3 games.

### Done (2026-04-07)
6. **MuZero env adapter** ✓ — `arc3_game.py`: `ARC3Game(AbstractGame)` fully
   implemented. Click-game (ls20/vc33) uses 4×4 bin grid (16 actions); multi-action
   (ft09/lp85) uses `len(available_actions)` discrete actions. `TTTScope1`
   (base=32, decay=0.65) and `TTTScope2` (base=16, decay=0.70) implemented.
7. **TTT adaptation layer** ✓ (design) — two-scope decaying budget in `arc3_game.py`.
   Stage 3 notebook `kaggle/arc3_muzero_ttt.ipynb` pending.
8. **Data pipeline** ✓ — `data_pipeline.py`: `ARC3Dataset`, `AtariHeadDataset`,
   `GVGAIDataset`. All three convert to muzero-general `GameHistory` format.
9. **GVGAI corpus generator** ✓ — `gvgai_rollout_gen.py` running in background.
   Target: 12,000 episodes (sokoban L1-5 + zelda L1-5). Status: ~1,200/12,000 done.
   Output: `D:\arc3\data\gvgai\`.
10. **Stage 1 notebook** ✓ — `kaggle/arc3_muzero_pretrain.ipynb`.
11. **Stage 2 notebook** ✓ — `kaggle/arc3_muzero_finetune.ipynb`.
