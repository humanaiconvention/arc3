# ARC Prize 2026 — ARC-AGI-3 Research Brief

**Team:** Ben (Bazzer) + Guilherme Ferrari Brescia (guiferrarib)
**Competition:** https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview
**Milestone #1 Deadline:** June 30, 2026
**Prize Pool:** $850K ($700K grand prize for 100% score)

---

## 1. What ARC-AGI-3 Is

Hundreds of turn-based game environments. No instructions, no rules, no stated goals. Agent must explore, model, infer goals, and act. Humans score 100%. Frontier AI scores 0.26%.

Four tested capabilities:
- **Exploration** — actively probe to gather information
- **Modeling** — build generalizable world model from observations
- **Goal-setting** — discover what winning looks like with zero instructions
- **Planning & Execution** — sequence actions toward inferred goals, course-correct

Competition mode constraints:
- No internet during eval
- One `make()` per environment, one scorecard
- Level resets only (no game resets)
- Scoring: level scores are squared, weighted by level index (later levels worth more)
- Must open-source under CC0 or MIT-0

## 2. SDK

```bash
pip install arc-agi
```

Agents repo:
```bash
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
```

Minimal agent loop:
```python
import arc_agi
from arcengine import GameAction, GameState

arc = arc_agi.Arcade()
env = arc.make("ls20", render_mode="terminal")

for step in range(100):
    action = GameAction.ACTION1
    obs = env.step(action)
    if obs and obs.state == GameState.WIN:
        break
    elif obs and obs.state == GameState.GAME_OVER:
        env.reset()

print(arc.get_scorecard())
```

Action space: `RESET`, `ACTION1`–`ACTION6`. Some actions are "complex" and take `x,y` coordinates (0–63). `env.action_space` returns available actions per frame.

Competition mode:
```python
from arc_agi import Arcade, OperationMode
arc = Arcade(operation_mode=OperationMode.COMPETITION)
```

Docs: https://docs.arcprize.org/
Games list: https://arcprize.org/tasks
API keys: https://docs.arcprize.org/api-keys

## 3. Proposed Architecture: MuZero + Neuro-Symbolic + TTT

### 3.1 Why This Stack

The 0.26% frontier score means LLM agents fail. Brute-force exploration fails. The agent needs:
- A learned world model (what happens when I take action X in state S?)
- Symbolic rule extraction (what are the game's rules?)
- Goal inference (what does winning look like?)
- Fast adaptation (learn the current game from few interactions)

### 3.2 MuZero (World Model + Planning)

MuZero learns environment dynamics without knowing rules. Uses MCTS for planning with a learned model. Proven on Atari, board games.

Reference implementation: https://github.com/werner-duvaud/muzero-general
- Pure PyTorch, runs on single GPU
- Trains on Atari with consumer hardware (2080 is sufficient)
- Network is small (few ResNet blocks)

Key MuZero components:
- **Representation function** h: observation → hidden state
- **Dynamics function** g: (hidden state, action) → next hidden state + reward
- **Prediction function** f: hidden state → policy + value
- **MCTS**: uses dynamics function to simulate ahead, guided by policy/value

Integration task: wire ARC-AGI-3 `env.step()` / `env.observation_space` to MuZero's game interface.

### 3.3 Neuro-Symbolic Layer (Rule Extraction + Planning)

**Two papers, one method.** The ICRA 2026 paper [Duggan et al.](https://arxiv.org/abs/2602.19260) is the head-to-head benchmark — it shows the neuro-symbolic stack beats fine-tuned VLAs (π₀) at **95% vs 34%** on 3-block Tower of Hanoi while using **80× less training energy** (0.85 MJ vs 68.5 MJ) and roughly **10× less inference energy** per episode. Critically, the NSM is trained on **50 simple stacking demos** and never sees a Hanoi solve, yet zero-shot solves both the 3-block (95%) and the unseen 4-block variant (78%) — both VLAs score 0% on 4-block. Training time: NSM 34 minutes vs VLA 1d 16h on the same RTX 4090.

The actual method is the CoRL 2025 paper [Lorang et al.](https://arxiv.org/abs/2508.21501). It is the first framework to **jointly** learn both the symbolic planning domain *and* the low-level controllers from a few raw demonstrations, with **no predefined predicates, types, or symbolic vocabulary**.

**Core algorithm (Lorang Alg. 1):**

1. **Graph construction.** Each demonstration is mapped to a node transition `(n_start, label, n_end)`. Nodes are abstract high-level states (initially black-box), edges are skills. Two human inputs only: (a) assign a transition label `l`, (b) match `n` to existing nodes by *visual comparison* of paired snapshots `(v, v')`.
2. **Bisimulation.** Compute minimal bisimulation `Ḡ` to collapse redundant nodes and shrink the search space the solver has to chew on.
3. **ASP-based abstraction.** Use the Bonet/Geffner ASP solver (refs 13/14) to find the simplest first-order PDDL domain `σ = ⟨E, F, S, O⟩` whose induced graph `G(σ)` is isomorphic to `Ḡ`. **This invents predicates and operators** — no schema is provided.
4. **Skill decomposition.** Each operator `o_i` is split into action-step sub-policies `π_{i,j}` (e.g. MOVE → reach-pick → pick → reach-drop → drop) with learned termination conditions.
5. **Oracle filter.** For each operator, compute the relevant object set `E_oi` (objects whose symbolic state changes under `o_i`). The oracle `μ(s̃, o_i) = τ(φ(s̃, o_i))` filters observations down to those objects in end-effector-relative coordinates. **This is symbolic attention** — the controller only sees what the operator says matters.
6. **Diffusion policies.** Train one diffusion policy per action step from the filtered demonstrations (multi-modal action distribution, robust to expert variation).

**Execution.** At test time the user picks initial+goal states (paired snapshots), the method emits MetricFF instance `T = ⟨E, F, O, s_0, s_g⟩`, MetricFF returns plan `P = [o_1, …, o_|P|]`, and each operator dispatches its skill, which sequences action-step controllers until learned termination.

The method also closes Silver et al.'s two desiderata for known/unknown settings: **KD1** subgoal-conditioned policies reach diverse low-level states fulfilling the same abstract goal; **KD2** the bilevel planner falls back to alternative skill sequences when a plan fails, supporting replanning under abstraction-induced constraints.

#### 3.3.1 Why this is the right shape for ARC-AGI-3

The same bilevel structure maps onto ARC-AGI-3 almost component-for-component, with two critical differences: (a) ARC's action space is **discrete**, so diffusion policies are overkill, and (b) ARC has **no human in the loop** during eval, so the two human inputs Lorang's method assumes have to be automated.

| Lorang component | ARC-AGI-3 mapping | Automation needed? |
|---|---|---|
| Demo trajectory `τ` | `(frame_t, action_t, frame_t+1)` from exploration | Free — comes from `env.step()` |
| Visual snapshot `v` | Rendered game grid | Free — comes from `env.observation` |
| Transition label `l` | The action ID we executed | Free — we know what we did |
| Node match (human visual compare) | Perceptual / structural grid equivalence | **Need an equivalence function**: perceptual hash + grid-cell diff + symmetry-group canonicalization |
| Bisimulation `Ḡ` | Collapse symmetric/redundant grid states | Same algorithm, parameterized by our equivalence relation |
| ASP solver → PDDL `σ` | Per-game rule discovery | Same — `clingo` + Bonet/Geffner inference rules |
| Operator-relevant object set `E_oi` | Grid cells/objects whose state changes under each action | Frame-differencing + connected-component analysis |
| Oracle filter `μ` | Attention over relevant grid regions per discovered operator | Derived automatically from `σ` |
| Diffusion policy `π_i` | Per-operator controller | **Replaced with categorical/Q-head** — ARC actions are discrete (`RESET`, `ACTION1-6`, `+xy ∈ [0,63]²`) |
| Action-step decomposition | Mostly N/A for ACTION1-5 (atomic) | Keep only for ACTION6 to choose the `xy` sub-target |
| MetricFF planner | Per-game plan synthesis | Substitute `pyperplan` (Python, BSD) or wrap `fast-downward` |

**The two human inputs that become automatic for ARC-AGI-3:**
- *Label assignment* — automatic; we know which `GameAction` we executed.
- *Node matching* — must be automated. Candidate equivalence relations to try in order: (i) exact grid equality, (ii) bit-vector hash of the canonicalized grid, (iii) structural equivalence under the symmetry group (rotation/reflection), (iv) embedding distance over a small frozen CNN. Start with (i)+(ii); promote to (iii) for symmetric games only when needed.

#### 3.3.2 Honest caveats (these will bite if ignored)

1. **Lorang assumes complete, noise-free graphs.** ARC-AGI-3 exploration is partial — ~100 interactions per game in competition mode means our `Ḡ` will be a fragment of the true graph. Bonet/Geffner show ASP is "robust to noisy or missing nodes and edges" (Lorang §7.1–§7.2), but we should *measure* this on our actual graphs as soon as we have them, not assume.
2. **ASP-discovered predicates may not exist for every game.** If a game's rules are continuous physics or the relevant "predicates" aren't expressible as discrete relations, the symbolic side will find no clean abstraction. That is a *signal*, not a failure: the agent falls back on the MuZero side alone for that game and we record the energy/score delta.
3. **The Markov assumption inside the oracle.** Lorang's oracle assumes the filtered observation preserves enough information for the policy to act Markov-optimally. ARC games with hidden state (turn parity, score, hidden inventory) need that state lifted into the predicate set explicitly, or the oracle will throw away the very thing we need.
4. **Diffusion policies are overkill for discrete ARC actions.** Replace with a softmax / Q-head; keep the action-step decomposition only for ACTION6's continuous `xy` sub-target.
5. **The original method requires human snapshot-matching during data collection.** We are removing the human; that puts more weight on the equivalence relation being right. Get that wrong and the graph fragments into too many nodes, the ASP solver can't find a compact domain, and the whole symbolic side degrades to nothing.

#### 3.3.3 Concrete dependencies (CPU-side, zero GPU cost)

Add to `arc3/neurosym/requirements.txt`:
- `clingo` — Answer Set Programming solver for the Bonet/Geffner inference rules
- `pyperplan` — Python PDDL planner, BSD-licensed; replaces MetricFF for portability. Wrap `fast-downward` if we need numeric fluents.
- `imagehash` — perceptual hash for the (ii) node-matching tier
- `networkx` — graph storage + bisimulation

This entire stack runs **on CPU**, leaving the 2080 free for MuZero rollouts (Phase G) and TTT updates (Phase H). That preservation of GPU headroom is the same energy story Duggan et al. measured: most of the reasoning is symbolic, and the GPU only fires when symbolic resolution is insufficient.

#### 3.3.4 Figure references from the source papers

These captions are quoted verbatim from the HTML versions of the two papers; they're recorded here so anyone reading the brief can locate the exact figure being referenced without leaving this document.

**Lorang et al. CoRL 2025 ([arXiv:2508.21501](https://arxiv.org/html/2508.21501v1)):**

- **Algorithm 1** — `Neuro-Symbolic Imitation Learning (𝒟, ζ, ℰ)`. No separate caption text in source; the algorithm box has 17 numbered lines split into a Learning Phase (lines 1–7: build graph, bisimulation, ASP abstract, oracle filter, train diffusion policies) and an Execution Phase (lines 8–17: query task, plan with MetricFF, sequential execution of skills and skill-steps). This is the spine of the method we are mapping onto Phases A–F.
- **Figure 1** — *"Our neuro-symbolic framework integrates graph construction, symbolic abstraction, planning, action decomposition, imitation learning and space filtering. Starting with just a few skill demonstrations (left), we construct a transition graph capturing transitions (edges), which are skill transitions between two high-level states (black-box nodes)."*
- **Figure 2** — *"Shown here is an example demonstration of the MOVE operator in the Towers of Hanoi domain, where block 1 is moved off block 2 and placed onto a platform. The agent partitions skill demonstrations into action steps, with an oracle ϕ filtering observations to simplify learning."* This is the figure that motivates our Phase G use of action-step decomposition for ACTION6's `xy` sub-target only.
- **Figure 3** — *"Illustrations of some of the simulation domains used for evaluation."* (Robosuite Stacking & Hanoi, Robosuite Kitchen, ROS2/Gazebo Pallets Storage.)
- **Figure 4** — *"Performance comparison between our Neuro-Symbolic (N-S) framework and baseline methods. Our approach achieves high success rates on short — Stacking & Forklift Pallet Loading/Unloading — and long-horizon tasks — including Towers of Hanoi, Multiple Pallets Storage, Nut Assembly & Kitchen — even with as few as 5 demonstrations. Our approach is domain agnostic, and works in very different scenarios."*
- **Figure 5** — *"Zero- and few-shot generalization results on Different Hanoi Towers configurations."* This is the curriculum-learning figure that shows 5 demos + 5 expert reach-place corrections beating 30 demos from scratch on 3×3, with scaling all the way to 7×5. The result we are betting on for ARC-AGI-3's "100 interactions per game in competition mode" budget.
- **Figure 8** — *"Task graphs for different environments. Stacking and Forklift Load/Unload Pallets graphs consist only of two nodes and a connecting edge, as they consider one-skill tasks and do not require planning."*
  - **8(a)** — *"Hanoi Towers Graph"*
  - **8(b)** — *"Forklift Multiple Pallets Storage Graph"*
  - **8(c)** — *"Nut Assembly Graph"*
  - **8(d)** — *"Kitchen Graph (two clusters connected via a unidirectional WAIT (or COOK) edge)."*

(Note on bisimulation: the source paper does not include a separate before/after figure of `G → Ḡ`. The bisimulation step is described in §4.1 of the paper and is the second line of Algorithm 1. The Hanoi-towers graph in Fig 8a is the *post-bisimulation* `Ḡ` and is what we should aim our Phase B output to look like for ls20 / lp85 / ft09 / vc33 — small, label-respecting, and connected.)

**Duggan et al. ICRA 2026 ([arXiv:2602.19260](https://arxiv.org/html/2602.19260v1)):**

- **Figure 1** — *"Overview of the experimental comparison between VLA models and the NSM. Both receive identical sensory inputs from simulation."* The figure shows that both architectures get the same `(image, wrist_image, state)` tuples; the difference is purely architectural. Same principle applies to ARC-AGI-3: the symbolic and neural sides will see the same `env.observation` and the architectural split lives entirely in how that input is processed.
- **Figure 2** — *"Example observations from the dataset. Left: Agent-view RGB image. Right: Wrist-mounted camera RGB image."*
- The Hanoi / Forklift / Stacking *result* numbers in Duggan are reported in **Tables I–IV** rather than separate figures. The most relevant numbers for our ARC-AGI-3 architecture decision:
  - **Table I (training)**: NSM 34 min / 0.85 MJ vs E2E-VLA 1d 16h / 68.5 MJ — ~80× less total energy.
  - **Table II (evaluation)**: 3-block Hanoi success NSM 95% vs E2E-VLA 34% vs PG-VLA 0%; episode energy NSM 0.83 kJ vs E2E-VLA 7.96 kJ. 4-block Hanoi (unseen): NSM 78% vs both VLAs 0%.
  - **Table III (VLM-as-planner)**: GPT-5 84% optimal / 16% invalid; Qwen-7B 100% invalid; PaliGemma-3B 100% invalid. *Receipts for "VLMs are unreliable as planners"* — the symbolic side is doing the planning, not an LLM.

### 3.4 TTT (Test-Time Training — Fast Adaptation)

From Guilherme's Genesis-152M architecture:
- https://huggingface.co/guiferrarib/genesis-152m-instruct
- TTT layer adapts model weights during inference
- Low-rank fast weights update from input sequence
- Dual form enables parallel gradient computation

Paper: "Learning to (Learn at Test Time): RNNs with Expressive Hidden States"
- Sun et al. — ICML 2024
- https://arxiv.org/abs/2407.04620

Application to ARC-AGI-3:
- Base network pretrained on public games via MuZero self-play
- TTT fast weights adapt during eval on unseen games
- Enables rapid model updating from few observations per game

### 3.5 Combined Architecture (Pipeline A → H)

```
                 ┌──────────────────────────────────────────┐
                 │      ARC-AGI-3 env (per game)            │
                 │  env.step(action) → (obs, reward, done)  │
                 └────────────┬─────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────────┐
  │ Phase A — Exploration (CPU)                                 │
  │ Random / curiosity-driven actions; log every transition.    │
  └────────────┬───────────────────────────────────────────────┘
               │
               ▼
  ┌────────────────────────────────────────────────────────────┐
  │ Phase B — Build transition graph G (CPU)                    │
  │ Nodes = grid frames clustered by perceptual hash +          │
  │ structural equivalence; edges = (frame_i, action, frame_j). │
  └────────────┬───────────────────────────────────────────────┘
               │
               ▼
  ┌────────────────────────────────────────────────────────────┐
  │ Phase C — Bisimulation reduction → Ḡ (CPU)                  │
  │ Collapse equivalent nodes; preserve label-respecting edges. │
  └────────────┬───────────────────────────────────────────────┘
               │
               ▼
  ┌────────────────────────────────────────────────────────────┐
  │ Phase D — ASP domain induction (CPU)                        │
  │ clingo + Bonet/Geffner inference rules → PDDL σ = ⟨E,F,S,O⟩│
  │ Predicates and operators are *invented*, not specified.     │
  └────────────┬───────────────────────────────────────────────┘
               │
               ▼
  ┌────────────────────────────────────────────────────────────┐
  │ Phase E — Goal inference (CPU)                              │
  │ When env emits WIN, backward-chain the predicate(s) that    │
  │ hold in the WIN frame and not its predecessor → s_g.        │
  └────────────┬───────────────────────────────────────────────┘
               │
               ▼
  ┌────────────────────────────────────────────────────────────┐
  │ Phase F — Plan (CPU)                                        │
  │ pyperplan on σ from current s_0 to inferred s_g             │
  │ → operator sequence P = [o_1, …, o_|P|]                     │
  └────────────┬───────────────────────────────────────────────┘
               │
               ▼
  ┌────────────────────────────────────────────────────────────┐
  │ Phase G — MuZero refinement (GPU, optional)                 │
  │ For operators with continuous parameters (e.g. ACTION6 xy   │
  │ target) or where σ is too coarse, run MCTS using MuZero     │
  │ h/g/f to choose the concrete action. Otherwise dispatch     │
  │ the symbolic skill directly.                                │
  └────────────┬───────────────────────────────────────────────┘
               │
               ▼
  ┌────────────────────────────────────────────────────────────┐
  │ Phase H — TTT adaptation (GPU, between levels)              │
  │ Fast-weight update on h() using the level's transition log; │
  │ symbolic σ is preserved across levels of the same game.     │
  └────────────┬───────────────────────────────────────────────┘
               │
               ▼
                  action → env.step()
```

The symbolic side (A → F) does the heavy reasoning on CPU. The neural side (G, H) only runs when the symbolic plan needs concretization or when we have spare compute between levels. This is the same architectural split that gave Duggan et al. their 80×/10× energy advantages, and it keeps the 2080 from being the bottleneck.

### 3.6 HAIC / Viability Convention Connection

The Viability Convention framework (M = C(t) − E(t) ≥ 0, [DOI 10.5281/zenodo.18144681](https://doi.org/10.5281/zenodo.18144681)) maps onto the agent loop:
- **E(t)** — environmental complexity / drift rate per game; each new game is a novel environment, so E(t) is high at game start and decays as σ stabilizes.
- **C(t)** — agent's corrective capacity; the symbolic σ + ASP solver + planner is the *durable* corrective capacity (cheap, interpretable, rerunnable), the MuZero rollouts + TTT updates are the *adaptive* corrective capacity (expensive, fast).
- **Semantic drift detection**: Prism geometry probes on `h()` (the MuZero representation function) measure when the learned representation diverges from the symbolic abstraction; that divergence is the operational E(t) signal.
- **The neuro-symbolic split is itself an implementation of corrective capacity** — the symbolic rules constrain the neural model's drift, exactly the C(t) > E(t) constraint Duggan et al. operationalize via the energy differential. The energy advantage measured in the ICRA paper is, in HAIC terms, the gain from concentrating C(t) in the side of the architecture where it stays cheap.

## 4. Immediate Action Items

### This Week (Remote)
- [ ] Play ls20, lp85, ft09, vc33 in browser at https://arcprize.org/tasks
- [ ] Guilherme: run random baseline agent on ls20 with terminal render
- [ ] Both: get ARC API keys from https://arcprize.org/platform
- [ ] Read Lorang et al. CoRL 2025: https://arxiv.org/abs/2508.21501
- [ ] Read Duggan et al. ICRA 2026: https://arxiv.org/abs/2602.19260

### Phase B + Phase D results (2026-04-07) — **symbolic side alive on ARC-AGI-3, all 4 games SAT**

**Phase B v3** (legal-only actions; `GameAction.from_id()` fix):

| game | frames | acts | nodes | dist edges | a-lbl | branch | bi-blk | term |
|------|--------|------|-------|------------|-------|--------|--------|------|
| ft09 | 81 | 81 | 9 | 52 | 16 | 14 | 9 | 0 |
| lp85 | 81 | 81 | 4 | 37 | 16 | 15 | 4 | 0 |
| ls20 | 81 | 81 | 81 | 80 | 4 | 1 | 81 | 0 |
| vc33 | 81 | 81 | 51 | 78 | 17 | 2 | 51 | 1 |

*a-lbl counts include ACTION6 spatial (x,y) coordinates as separate labels.*

**Phase D v4** (legal-only data; self-loops stripped; `legal_only=True` drops tagged-illegal edges):

| game | nodes (of) | real trans | eff. labels | **sat** | solve |
|------|-----------|-----------|-------------|---------|-------|
| selftest (blocks2ops_2) | 3 (3) | 4 | 2 | **SAT** | 0.01s |
| ft09 | 9 (9) | 10 | 7 | **SAT** | 0.041s |
| lp85 | 4 (4) | 3 | 2 | **SAT** | 0.009s |
| ls20 | 10 (81) | 9 | 4 | **SAT** | 0.037s |
| vc33 | 20 (51) | 18 | 11 | **SAT** | 0.97s |

**All 4 games SAT** after fixing the illegal-action data quality issue.

Highlights:

1. **Data quality was the bottleneck**: the base Random agent submitted illegal
   actions causing spurious self-loops and inflated label counts. Restricting to
   `available_actions` (ft09/lp85: [6], ls20: [1,2,3,4]) and using
   `GameAction.from_id()` gave clean graphs and fast SAT on all games.
2. **ft09/lp85** — both spatial-click games, `available_actions=[6]`. Tight
   state spaces (9 and 4 nodes) with heavy self-loop structure; Phase D strips
   42/34 self-loops respectively, leaving 10/3 real transitions.
3. **ls20** — 81 distinct states in 81 moves; trimmed to 10-node subgraph
   (20 nodes caused TIMEOUT). 4 clean operators from `available_actions=[1,2,3,4]`.
4. **vc33** — 1 terminal node (GAME_OVER), 20-node subgraph, 1 illegal RESET
   edge dropped. 11 effective labels from ACTION6 spatial positions; SAT in 0.97s.

Bugs fixed in Phase D pipeline (see `neurosym/README.md` for full table):
- Self-loop edges cause immediate UNSAT (BG `eff_used` constraint) → strip before LP
- `time_limit_s` silently unused → async + `threading.Timer` + `handle.cancel()`
- `GameAction(aid)` with available_actions ints → `ValueError` → use `GameAction.from_id(aid)`
- `num_edges_total` metadata stored trimmed count → capture before trimming
- Start node missing from terminal neighborhood trim → reserve 1 slot for earliest `first_seen_at`

### Back at BEAST
- [ ] Clone repos:
  ```bash
  git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
  git clone https://github.com/werner-duvaud/muzero-general.git
  ```
- [x] Run random agent baseline on all public games, capture scorecards (`scorecards/BASELINE_SUMMARY.md`)
- [x] Install clingo Python module (`pip install clingo` → 5.8.0)
- [x] Phase B v1 prototype (`neurosym/phase_b_graph.py`) — frames-only, unlabeled edges; 4-game summary table generated
- [x] Phase D v1 prototype (`neurosym/phase_d_asp.py`) — wraps `learner-strips/asp/clingo/kr21/`; selftest SAT on `blocks2ops_2.lp`; ARC graphs UNSAT-as-expected (1 edge label only)
- [x] Diagnose recorder gap: `agents/recorder.py` only writes FrameData, not the action that produced it (`action_input.id == 0` in every recording)
- [x] Write `neurosym/record_actions_agent.py` (Random subclass that interleaves `action_request` events into the JSONL)
- [x] Re-record ls20/lp85/ft09/vc33 with `RandomRecordActions`; confirmed `action_request` entries present
- [x] Update `phase_b_graph.py` to thread `action_request.action_id` onto the surrounding edge label
- [x] Re-run Phase B: confirmed distinct action labels in all re-recorded games (8 for ft09, 7 for ls20/vc33)
- [x] Re-run Phase D on ft09 → **SAT in 2.35s** (v3, dirty data); v4 with clean data → **SAT in 0.041s**
- [x] Scaled Phase D to ls20 → **SAT in 15.95s** (v3); v4 clean → **SAT in 0.037s** (10-node subgraph)
- [x] Diagnosed vc33 UNSAT (v3, dirty); v4 clean → **SAT in 0.97s** (20-node subgraph, legal_only=True)
- [x] Diagnosed GameAction.from_id bug: `GameAction(aid)` raises ValueError; fixed both record_actions_agent.py copies
- [x] All 4 games SAT on legal-only Phase D v4 data (ft09/lp85/ls20/vc33)
- [x] **Phase E**: Goal predicate identified for vc33: `pred1(obj1,obj1) ∧ pred1(obj2,obj2)` (size-2 conjunction, fully discriminating) — `neurosym/phase_e_goal.py`
- [x] **Phase F**: PDDL serialiser + pyperplan BFS + graph-BFS fallback — `neurosym/phase_f_plan.py`
  - vc33 abstract plan: **6 steps** (pyperplan on induced STRIPS domain)
  - vc33 graph-BFS plan: **49 steps** (shortest observed path in Phase B graph)
  - ft09/lp85/ls20: no terminal states in 81 random moves → Phase E/F deferred
- [x] **Plan executor agent**: `neurosym/plan_executor_agent.py` — `GraphBFSExecutor` and `AbstractPlanExecutor` registered in agents framework
- [x] **Executed both vc33 plans** — both score 0.0 (diagnosis: Phase E goal is a budget-exhaustion artifact, not a level-completion state; random play never completes a level)
  - 49-step graph-BFS: https://three.arcprize.org/scorecards/72626f9a-8fdf-4ca5-a68e-9fc79295a59c
  - 6-step abstract: https://three.arcprize.org/scorecards/a50701c9-18f5-488e-893b-c9e3895c9eed
- [ ] **Phase G**: get recordings with `levels_completed > 0` (use SmolVisionAgent or MultiModalLLM); prerequisite for correct Phase E goal on vc33
- [ ] Phase B/C: implement node-equivalence relation (start with exact-grid + perceptual-hash; canonicalize-by-symmetry second tier)
- [x] **Wire MuZero game interface to ARC-AGI-3 env**: `neurosym/arc3_game.py` — `ARC3Game(AbstractGame)`, click-game + multi-action modes, `TTTScope1`/`TTTScope2` decay schedules
- [ ] Profile GPU memory / inference time on 2080
- [x] **Architecture pivot (2026-04-07)**: symbolic B→F pipeline blocked on winning recordings; pivot to MuZero + TTT as primary path (see §3.6 below)
- [x] **Domain anchor recordings**: 5 sweeps × 4 games = 20 files, 1,620 episodes at `recordings/*.randomrecordactions.80.*.jsonl`
- [x] **Data pipeline**: `neurosym/data_pipeline.py` — `ARC3Dataset`, `AtariHeadDataset`, `GVGAIDataset`, `apply_domain_upweight(3×)`
- [x] **GVGAI rollout generator**: `neurosym/gvgai_rollout_gen.py` — sokoban L1-5 + zelda (MiniGrid) L1-5, 12,000-episode target; generation in progress at `data/gvgai/` (~1,200/12,000 done)
- [x] **External dataset placeholders**: Atari-HEAD (8.7 GB, zenodo) + Boxoban (~50 MB, Google DeepMind) — placeholder READMEs at `D:\articles\organized\rl-datasets\`

### Architecture Milestones
- [x] **Symbolic side end-to-end on vc33**: Phases B→F running + plans executed on vc33. Caveat: (1) vc33 terminal state is budget-exhaustion (not a WIN), so the plan targets the wrong goal; (2) ft09/lp85/ls20 have Phases B+D only — no terminal states in recordings means Phase E/F cannot run; (3) Phase D SAT for ft09/ls20/lp85 proves the LP encoding is satisfiable over observed data, not that start-to-goal planning is possible
- [x] **MuZero AbstractGame adapter**: `neurosym/arc3_game.py` complete; `TTTScope1` (base=32, decay=0.65) + `TTTScope2` (base=16, decay=0.70) implemented
- [x] **Data pipeline + general corpus**: `data_pipeline.py` loaders for all three sources; GVGAI generator running; Atari-HEAD placeholder in D:\articles
- [x] **Stage 1 pre-training notebook**: `kaggle/arc3_muzero_pretrain.ipynb` — 50k steps on T4×2, 3× domain-anchor resampling, checkpoint output
- [x] **Stage 2 fine-tuning notebook**: `kaggle/arc3_muzero_finetune.ipynb` — 40× NN-augmentation (no color jitter), boundary-step PER inflation, early stopping (patience=5)
- [ ] **Stage 1 + 2 training runs**: notebooks ready; blocked on (a) Atari-HEAD download, (b) GVGAI corpus completion (~4-6h), (c) Kaggle dataset upload
- [x] **Stage 3 TTT notebook**: `kaggle/arc3_muzero_ttt.ipynb` — Scope1 (intra-level, after each level) + Scope2 (cross-level, before each level), decaying budgets, dry-run mode, smoke-tested by Codex
- [ ] **TTT scope 1 + 2 evaluation**: verify on ls20 (cleanest game); ablation table
- [ ] Full pipeline on all public games + ablation table (symbolic-only / muzero-only / hybrid / hybrid+TTT)
- [ ] Competition mode submission (before June 30)

---

## 3.6 MuZero + TTT Architecture (pivot 2026-04-07)

### Motivation

The symbolic pipeline (Phases B→F) is structurally correct but requires winning recordings for Phase E goal extraction. Random play never completes a level in vc33 in 81 moves. The fundamental blocker is the goal specification problem — without labeled terminal states from actual level completions, planning targets budget-exhaustion states.

The MuZero + TTT path sidesteps this by learning game primitives from data rather than deriving them symbolically.

### Training pipeline (three stages)

**Stage 1 — Broad pre-training**
- Data: ~27 GB diverse game corpus — **identified and partially downloaded**:
  - Atari-HEAD (8.7 GB, zenodo.org/records/3451402, CC BY 4.0) — NOT YET DOWNLOADED; placeholder at `D:\articles\organized\rl-datasets\atari-head\`
  - GVGAI self-generated corpus (16-18 GB, `D:\arc3\data\gvgai\`) — generation running, ~1,200/12,000 episodes done
  - ARC-AGI-3 domain anchors (~532 MB augmented, `recordings/*.randomrecordactions.80.*.jsonl`)
- Mix: 98% general + 2% domain-anchor recordings at natural frequency
- Domain recordings: loss-upweighted 3× (proxy: 3× resampling in Stage 1; real PER inflation in Stage 2)
- Model: ResNet 3-block 64-channel MuZero (~5-10M params), action_space=18, obs=(1,64,64)
- Compute: Kaggle T4×2 (32GB VRAM total)
- Notebook: `kaggle/arc3_muzero_pretrain.ipynb` ✓

**Stage 2 — Domain-adaptive fine-tuning**
- Data: augmented 4-game ARC-AGI-3 recordings exclusively
- Augmentation (40× multiplier): hflip 50%, vflip 50%, crop-resize 60% (48×48→64×64 NN only), window subsampling always
- **Forbidden augmentation**: color jitter / hue shifts — ARC colors are semantic identifiers
- **Nearest-neighbor resize only**: bilinear would produce invalid non-palette values in the single-channel int8 frames
- Target: ~820 raw histories × 41 = ~33,620 augmented; 80/20 train/val split
- LR = 3e-5 (1/10th of Stage 1 LR 3e-4); early stopping patience=5 × 500 steps
- Loss injection: boundary-step PER priority inflation (2× within ±3 steps of level transition)
- Notebook: `kaggle/arc3_muzero_finetune.ipynb` ✓

**Stage 3 — TTT at deployment (two scopes)**
- **Scope 1 (intra-level)**: gradient updates on the first N interactions within level L; model learns this level's specific dynamics and reward structure within the action budget
- **Scope 2 (cross-level)**: carries knowledge from levels 1..L-1 into level L+1; each level is a mini-game with potentially different rules; selective inheritance (keep what transfers, update what changed)
- These are two separate gradient passes with different frozen/unfrozen layers and different context windows

### Data anchor sizing
- Raw domain data: 1,620 episodes × 81 frames = ~131,220 frames ≈ 800 MB uncompressed (1,64,64 float32)
- Post-augmentation (40×): ~33,000 effective episodes
- General corpus: Atari-HEAD 8.7 GB + GVGAI 16-18 GB = ~27 GB total — maintains ~2% domain anchor ratio
- TTT budgets: Scope1 base=32 steps × 0.65^(L-1); Scope2 base=16 × 0.70^(L-1). Level 7+: ≤2 steps

The reordering matters: the **symbolic side ships first** because it's the cheapest to run, has the highest data efficiency, and reveals which games are amenable to discrete abstraction at all. That answer then drives how aggressively we need MuZero on each game.

## 5. Key References

| Paper | Link | Relevance |
|-------|------|-----------|
| ARC-AGI-3 Docs | https://docs.arcprize.org/ | SDK, agent templates, scoring |
| ARC-AGI-3 Competition | https://arcprize.org/competitions/2026/arc-agi-3 | Rules, prizes, deadlines |
| ARC-AGI-3 Agents Repo | https://github.com/arcprize/ARC-AGI-3-Agents | Agent templates, random/LLM baselines |
| MuZero General | https://github.com/werner-duvaud/muzero-general | World model + MCTS planning |
| Duggan et al. ICRA 2026 | https://arxiv.org/abs/2602.19260 | Neuro-symbolic vs VLA benchmark |
| Lorang et al. CoRL 2025 | https://arxiv.org/abs/2508.21501 | Few-shot neuro-symbolic method |
| Sun et al. ICML 2024 (TTT) | https://arxiv.org/abs/2407.04620 | Test-time training |
| Genesis-152M | https://huggingface.co/guiferrarib/genesis-152m-instruct | Guilherme's hybrid attention + TTT model |
| Gated DeltaNet ICLR 2025 | https://arxiv.org/abs/2412.06464 | Linear attention mechanism in Genesis |
| FoX ICLR 2025 | https://arxiv.org/abs/2503.02130 | Forgetting attention in Genesis |
| AlphaZero.jl | https://jonathan-laurent.github.io/AlphaZero.jl/stable/ | Reference (Julia, not directly usable) |
| ARC-AGI-3 Launch Blog | https://arcprize.org/blog/arc-agi-3-launch | Benchmark design rationale |
| ARC Prize 2026 Rules | https://arcprize.org/competitions/2026 | Open-source requirements, eligibility |

## 6. Hardware

- **BEAST**: RTX 2080, local development and training on public games
- **Guilherme**: A100 80GB (Genesis was trained on this)
- **Cloud**: RunPod or similar for heavier training runs if needed
- **Competition eval**: hardware limits TBA by ARC Prize team

## 7. Open Questions

1. What are the eval compute/time limits? (Not yet announced)
2. Can MuZero learn fast enough from ~100 interactions per game in competition mode?
3. How to represent the observation space — raw pixels vs structured grid?
4. What's the right balance between exploration and exploitation per game?
5. How does the symbolic planner handle games with no clear discrete rules (e.g., continuous physics)?
6. Division of labor: who builds what?
7. **Node-equivalence relation per game family** — exact-grid is fast but fragments under symmetry; canonicalize-by-symmetry is correct but expensive; perceptual hash needs a tuned threshold per game. Is one tier enough, or do we adapt per game?
8. **ASP solver behavior on partial graphs** — Bonet/Geffner claim robustness, but how *partial* is too partial? What fraction of the true transition graph do we need from ~100 interactions before σ becomes non-trivial?
9. **Cross-game σ transfer** — if two games share predicates (e.g., "blocked", "above"), can we warm-start σ on game B from the σ learned on game A? Lorang's curriculum result on harder Hanoi configs (3×3 → 7×5) suggests yes, but ARC-AGI-3 games are more heterogeneous than Hanoi variants.
10. **Goal-inference on first WIN** — we only see WIN by accident during exploration. Can we get reliable goal predicates from a single WIN frame, or do we need multiple wins to disambiguate (which we may not get)?
11. **Energy budget for the symbolic side at eval time** — clingo can be slow on large encodings. What's our worst-case wall-clock per ASP solve, and does it fit inside the per-step time limit?
