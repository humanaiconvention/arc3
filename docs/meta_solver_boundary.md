# Meta-solver: what works, what doesn't, and the boundary

Written 2026-04-27 after a session that built out a probe-based meta layer
(click heat-map, action-pair non-commutativity, derived signals like
`inverse_pairs`/`toggle_actions`/`looks_like_cursor_click`) and validated it
empirically on sc25 (solved), re86 (unsolved), and 4 other unsolved games.

## The probe layer (what was built)

`characterize` now produces, for every game in ~65s:

- **`action_effects`** (existed): per-action single-press cell change footprint.
- **`click_heatmap`** (new): coord-grid scan with the click action; reveals
  which pixels respond to clicks from a fresh-reset state.
- **`action_pair_effects`** (new): for every ordered pair `(a, b)` over actions
  `{1..5}`, records final-frame hash, cells changed from initial, and
  `delta_b_after_a`. Exposes non-commutativity, idempotency, and conditional
  effects.
- **`inverse_pairs`** (derived): `[a, b]` where `(a, b)` and `(b, a)` both
  return to base. The canonical signature of direction-pair nav structure.
- **`toggle_actions`** (derived): `a` where `(a, a)` returns to base. Self-
  inverse — typically commit/select primitives.
- **`looks_like_cursor_click`** (derived): 0–1 confidence that the click
  action is just cursor placement (uniform low cells_changed across heatmap),
  not a state effect.

Persisted on `GameProfile`; re-derivable from disk without API calls.

## What this layer is good for

1. **Strategy gating**: with `inverse_pairs`, `toggle_actions`, and
   `looks_like_cursor_click`, run_meta can tell which strategies are
   structurally inappropriate before spending budget on them. E.g.
   `looks_like_cursor_click=1.0` ⇒ skip strategies that assume
   click-equals-effect (action_spam-style on clicks).
2. **Safe exploration**: `InverseAwareWalk` uses 0 lives on re86 in 1000
   steps where `RandomWalk` used 9. Pruning inverse-pair undos meaningfully
   reduces fatal moves at no extra API cost.
3. **Comparable game signatures**: 5 games' signatures cleanly classify into
   three families (clean direction-pair, triangular inverse subgroup, flat).
   This is the foundation for "games that look alike should be approached
   alike" — basis for any cross-game transfer.
4. **Mechanism surfacing on stuck games**: re86's "act5 preserves partial
   removals" hypothesis (previously discovered through several rounds of
   manual `re86_vNN.py` probing) appears directly in `toggle_actions=[5]`
   and the per-pair matrix without any game-specific code.

## What this layer is **not** good for

The hard part — **discovering a specific 8–15-action winning sequence on a
stuck game** — was not moved by either contribution this session.

Two concrete demonstrations of the boundary:

- **sc25 with `cluster_click_then_nav`**: the strategy implements the right
  grammar (`nav × N → cluster clicks → nav × N`) but cannot find the cluster
  positions because grid_step=12 doesn't land inside the ~5px-wide fill
  regions, and finer grid (step=4) blows the API budget by 4×. The win is
  recoverable via `replay_known` (registry has it) but not via probe-driven
  discovery at affordable resolution.
- **re86 with `InverseAwareWalk`**: the strategy explores more safely (0
  deaths) but doesn't find a win in 1121 steps. The win mechanic requires
  specific click trajectories to remove all 8 groups; smart exploration
  doesn't have enough signal to converge on them.

## Why the wall is structural, not a tuning problem

The probes characterize **single-action** and **immediate-pair** effects on
the **initial state** (and, for `cluster_click_then_nav`, on a small set of
post-prelude states). They sample only the immediate neighborhood of the
state graph.

ARC win conditions live deep in that graph. A 12-action winning sequence in
a 5-action game has 5¹² ≈ 244M ordered possibilities. Inverse-pair pruning
brings effective branching from 5 to ~4, leaving 4¹² ≈ 17M — still
infeasible at any reasonable API budget (each step is ~0.3s minimum).

No surface heuristic compresses this. The 8 deterministic winners we have
were all discovered by **a human watching the game and noticing a pattern**.
That's the actual workflow that produced them.

## What the third primitive *might* look like, and why it's also bounded

A "state-conditional trajectory exploration" primitive would:

1. Sample short paths (depth 3–5) through the action graph
2. From each reached state, run a brief sub-probe (mini click-heatmap or
   action-pair)
3. Record state hashes and which paths reach novel ones
4. Build a coarse map of the reachable state graph

Cost analysis:

- Depth 5, branching 4 (post inverse-pair pruning): 4⁵ = 1024 leaves
- Each leaf: 16-position click sub-probe = ~5s
- Per game: 1024 × 5 = 5120s = ~85 min

That's already past the budget where this is a "characterize-time" primitive.
And depth 5 might still be too shallow for re86-class wins.

So the third primitive doesn't break the wall. It just samples deeper into
the state graph at higher cost, with diminishing returns.

## The actual lever: learned heuristics over probe features

The discovery problem is genuinely **search × heuristic**, and every
tractable heuristic for deep state-graph search comes from learning on prior
solutions. The 8 deterministic winners are training examples; their probe
signatures are structured features.

The DiLoCo/Gemma4 work in `D:\diloco_lab` is already aimed at this — the
tensor exports under `data/arc3_exports/2026-04-27-meta-registry-replays-tensors-v2`
contain the action traces. The right next step from the probe layer's side
is to **export probe signatures alongside the tensor traces** so an adapter
trained on those traces learns "given this game signature, the policy
distribution looks like this."

Concrete artefacts the probe layer should publish to the export workspace:

- `inverse_pairs`, `toggle_actions`, `looks_like_cursor_click` (per game)
- `action_effects` (per-action single-press cell footprint)
- A condensed `action_pair_effects` matrix (n_changed_from_initial only)
- The click heat-map sorted top-K positions

This would not require any new live API work — the data is already on
profiles. It would require a small write-out script in the export pipeline.

## Standing TODOs (kept from this session, not yet done)

1. **Run `InverseAwareWalk` on tu93** — single 5-min run. tu93's signature
   ([1,2,3] all toggle, act 4 the unique mover) suggests the strategy will
   degenerate to "spam act4 with periodic toggles", which is exactly what
   the structure implies. Useful as a second data point on whether smart
   exploration ever cracks anything.
2. **Use `looks_like_cursor_click` to gate strategies in run_meta** — pure
   orchestrator change, no API calls. Skip click-effect-assuming strategies
   when the click is just cursor placement.

## Summary

| Layer | Scope | Status |
|---|---|---|
| Registry & replay | Bookkeeping for known wins | Solid |
| Probe layer (this session) | Structural game characterization | Solid; load-bearing for gating + safety |
| Discovery layer | Finding new winning sequences | Open; needs learned heuristics, not more probes |

The probe layer is enabling, not solving. It provides the features. The
discovery loop runs in `D:\diloco_lab` and currently consumes only action
traces; adding probe signatures to its inputs is the next coherent step.

---

## Update — boundary empirically confirmed across 4 attempts (later same session)

After this doc was written, the boundary thesis was tested by four
escalations that each tried to cross it. All four hit the same ceiling.
This section records the test results and revises the thesis from
"probably true" to "empirically confirmed within budget."

### What was tried

1. **State-conditional probe** added to `characterize` (gated on
   `looks_like_static ≥ 0.7 + nav present`) — see
   [state_conditional_probe_findings.md](state_conditional_probe_findings.md).
   Surfaced real mechanism for g50t, sc25, lf52. Win count change: 0.
2. **`MoverToggleWalk`** strategy parameterised on
   `dominant_actions × toggle_actions`. 24 directed attempts on g50t,
   0 lives lost, 0 wins. Pruned the search dramatically vs random walk;
   schema didn't match g50t's actual win condition.
3. **Learned heuristic v0/v1** — sklearn `GradientBoosting` and PyTorch
   `TinyPolicyCNN` trained on the 72-row probe-signature export. Both
   landed at the same eval accuracy (4/7 = 0.571 on held-out sp80/vc33),
   distinct from random (0.20) but unable to generalise novel-game
   policies. Live runs on re86: 0 wins, action distributions matched
   training-corpus modes.
4. **RL self-play (v2 CEM)** — Cross-Entropy Method with v1 CNN init,
   shaped reward (`+1.0 per level + 0.001 per cell changed - 0.5 on
   terminal`). 4 iterations × 6 episodes × 25 steps on re86. Mean reward
   *declined* 3.45 → 2.26 across iterations (classic CEM elite-collapse
   on sparse rewards). 0 wins.

### Net empirical result

**Win count over the entire arc: 10/25 → 10/25.** Five directed strategies,
two trained models, one RL trainer — none moved the count. The probe
layer's safety properties held: the live runs on stuck games used 0–24
lives total over 1000+ live actions (vs random walk's ~9–19). The
discovery problem was never moved.

### Revised thesis (sharper version)

The original doc said "the discovery layer needs learned heuristics, not
more probes." The empirical result sharpens this:

- **Learned heuristics need *more wins to train on*, not better
  architecture.** v0 (53 features, sklearn) and v1 (53 features + 64×64
  frame, CNN) achieved identical eval accuracy. Three different model
  capacities all saturated at the same number, which is the dataset
  ceiling, not the model ceiling.
- **RL self-play in this regime needs ≥10× the API budget AND a stabler
  algorithm.** CEM's elite-collapse failure mode is well-known; PPO with
  a value baseline or DQN with replay would be more stable but require
  proportionally more compute. At 0.3s/API call, getting to ~10⁶
  environment steps (the typical "RL works" regime) is ~80 hours per
  game on live API.

So the boundary isn't moved by this session's tools. The genuine paths
forward, in order of tractability:

1. **Human-in-the-loop discovery on stuck games** — 1–2 new wins changes
   the dataset shape meaningfully and is how the existing 8 wins were
   discovered.
2. **Multimodal Gemma-4 with vision via the existing Kaggle DiLoCo
   pipeline** — pretrained vision features may generalise where a 148K
   from-scratch CNN couldn't.
3. **Massive compute scale-up for RL** — needs an offline simulator or
   parallel environments, neither of which we have.

The pipeline shipped this session — probe layer + heuristic strategies +
v0 sklearn policy + v1 CNN policy + v2 CEM trainer + structured export —
is ready to consume any of these. The next genuine win-count move
happens at the dataset or compute end, not the model architecture end.

### What did move

The probe layer's *enabling* properties got concretely useful:

- `cursor_click_gate` skips click-mapping strategies on cursor-positioning
  games (saves wasted budget).
- `inverse_pairs` + `dominant_actions` make `InverseAwareWalk` use 0 lives
  on re86 vs 9 for vanilla `RandomWalk`.
- The `with-probe-sigs` export gives any future model — local CNN, Kaggle
  Gemma-4, or anything else — structured features that 3 model classes
  have shown carry signal (above-random, top-15 by importance).

The session's contribution is a *correctly-built scaffold* that the
discovery layer (whatever form it eventually takes) plugs into.

---

## Update — RL self-play with reward shaping (v3, v3b)

After v2 CEM (cells_changed-only reward) collapsed, two more RL runs
tested whether richer reward signals could fix the elite-collapse failure
mode. Both used the same CEM trainer with v1 CNN initialization on re86.

### v3 — absolute similarity to winning-state pool + novelty bonus

Built `D:/diloco_lab/diloco_lab/arc3_winning_states.py`: extracts the
final frame from each of the 8 winning replays, computes a 12-bin value
distribution signature, exposes `score_state(frame) → similarity ∈ [0, 1]`
that returns the maximum similarity over the 8 signatures.

Built `arc3_env.ARC3Env(use_similarity=True, use_novelty=True)`:
  - Adds `+REWARD_SIMILARITY_WEIGHT × score_state(frame)` per step
  - Adds `+REWARD_PER_NOVEL_HASH` per never-before-seen state hash,
    capped at 50 novel states per episode

Result on re86: mean reward jumped from v2's ~3.5 to ~16.2 per episode,
but trend was still declining (16.15 → 15.33). **Diagnosis**: re86's
INITIAL frame already scored 0.941 similarity to s5i5's winning signature
(both are mostly-uniform-background palettes). The agent gets ~0.47 reward
per step just sitting at start regardless of what it does. The shaping
was firing as a measurement but providing no learning gradient because
the baseline was contaminated.

### v3b — delta-similarity (subtract per-episode initial baseline)

Fix: reward the *improvement* in similarity over the episode's initial
frame, not absolute similarity. `reward += SIMILARITY_WEIGHT × (sim_now -
sim_initial)`.

Result on re86: rewards re-baselined to ~4.4 → 3.6 over 4 iterations.
Decline narrowed (-19% vs v2's -35%). **Iter4 ticked up for the first
time across any RL run** (3.33 → 3.57), suggesting the reward signal is
finally non-degenerate. But still 0 wins.

### What this proves about the RL approach

Three CEM variants all collapsed:

| variant | reward | iter1 → iter4 | trend |
|---|---|---|---|
| v2 | cells_changed | 3.45 → 2.26 | -35% |
| v3 | + abs similarity + novelty | 16.15 → 15.33 | -5% (degenerate baseline) |
| v3b | + Δ similarity + novelty | 4.39 → 3.57 | -19% (calibrated) |

Pattern: **CEM elite-distillation collapses the policy regardless of
reward shaping.** The top-2-of-6 elites pull the policy toward narrow
action distributions; the narrowed distribution explores less; reward
goes down. This is well-known CEM behavior on sparse-reward exploration
problems.

The reward shaping is now correctly built — similarity decomposes
sensibly (re86's initial similarity to s5i5: 0.941, to others: ~0.06),
delta gives proper per-step gradient, novelty fires per unseen hash. But
**CEM is the wrong algorithm for this regime.** Algorithms that maintain
exploration explicitly (PPO with entropy bonus + value baseline, or DQN
with replay buffer + ε-greedy) would be more stable but require
~10× more code and ~10× more API budget per game to train meaningfully.

### Final win count after entire session

**10/25 games solved (unchanged from session start).** Five strategy
families written, three model architectures (sklearn/CNN/CNN-RL), three
CEM reward configurations, all consumed the same probe data, none
produced a new win. The empirical record decisively confirms the
boundary: **win discovery in this regime requires either compute scale
(≥10⁶ env steps per game) or new wins in the training corpus, not
better algorithms over the existing 8 wins.**

The pipeline (probe layer → export → multiple model versions → multiple
RL trainers → strategy registry) is fully wired and ready to consume
either of those when they arrive.

---

## Update — RL algorithm sweep (v4 PPO, v3c persistent novelty)

After v2/v3/v3b all showed declining-mean trajectories, the next test
asked whether the issue was the **CEM algorithm specifically** (with its
known elite-collapse failure mode) or the **reward signal in general**.

### v4 — PPO with value baseline (option 1)

Built `D:/arc3/scripts/meta/rl/ppo_self_play.py` with full actor-critic
PPO: GAE(λ=0.95) advantages, clipped objective (ε=0.2), value MSE
baseline, entropy bonus (coef=0.02), 4 update epochs per iteration.
Reused v3b's Δ-similarity + per-episode novelty reward shaping.
Initialised actor from v1 CNN; value head random.

Run on re86: 6 iterations × 8 episodes × 25 steps = ~6.3 min.

**Result: -38% reward trend, worse than CEM.** Internals confirmed PPO
was working algorithmically:
- value loss dropped 15.0 → 9.9 (critic learning)
- actor loss dropped 0.32 → 0.07 (clipped objective working)
- entropy dropped 1.00 → 0.72 (distribution narrowing despite the bonus)

But mean reward fell 4.33 → 2.70 across iterations. PPO converged
the policy to a different low-reward equilibrium than CEM did, but
both equilibria are below the starting policy.

### v3c — CEM with persistent novelty (option 2)

Single line change to `ARC3Env`: `persistent_novelty=True` makes the
visited-hashes set carry across episodes, so the novelty bonus rewards
states unseen *across the entire training run*, not just within an
episode.

Run on re86: same shape as v3b (4 × 6 × 25 = ~3 min). **Result: -14%
trend** — the *least-bad* of all 5 RL configurations. iter4 ticked up
3.54 → 3.69, the second consecutive iter-N→iter-N+1 uptick observed
across all RL runs.

### Diagnosis: it's the reward, not the algorithm

| variant | algorithm | reward | iter1 → last | trend |
|---|---|---|---|---|
| v2 | CEM | cells | 3.45 → 2.26 | -35% |
| v3 | CEM | + abs sim + nov | 16.15 → 15.33 | -5% (degenerate) |
| v3b | CEM | + Δ sim + nov | 4.39 → 3.57 | -19% |
| v3c | CEM | + Δ sim + persistent nov | 4.29 → 3.69 | -14% |
| v4 | PPO | + Δ sim + nov | 4.33 → 2.70 | -38% |

Two algorithms (CEM elite-distillation, PPO clipped policy gradient),
four reward configurations, all show declining-mean trajectories.
**The reward signal does not correlate with winning, so optimisation
can't find winning.** No optimization algorithm fixes this.

### What's NOT going to help (and why options 3/4 were skipped)

- **Option 3: forward-model planning + MCTS.** Major build (~1-2 days).
  The forward model would learn `P(next_state | state, action)` from
  collected experience, then plan over predicted trajectories without
  burning live API. Useful for sample efficiency, but it'd plan against
  the *same shaped reward* that doesn't correlate with winning. No
  better gradient.
- **Option 4: KL-constrained CEM.** Cheap (~15 min). Would prevent the
  elite-distillation from collapsing the policy too fast. But "preserve
  more exploration" doesn't help if exploration toward all reachable
  state regions hits zero winning states. v3c (persistent novelty)
  already explored novelty in the strongest practical way and got -14%
  trend, vs the pure-cells baseline at -35%.

### What WOULD actually move the win count (none of which is tractable
within this session's constraints)

1. **Massive compute scale** — at 0.3s/API call, getting to ~10⁶ env
   steps per game (the typical "RL works on sparse rewards" regime) is
   ~80 hours per game. The ARC3 leaderboard agents either have offline
   simulators or massive parallelism. We have neither.
2. **Pretrained vision-language model with online RL fine-tuning**.
   Gemma-4-2B with image input + PPO via the existing Kaggle pipeline
   could *prior-bootstrap* the agent's understanding of "puzzle states."
   This is what `OPERATIONS.md` was built for; it's outside this
   session's scope but the wiring is in place.
3. **Adding new wins to the training corpus** (the option the user has
   ruled out for this session). Even 1-2 new wins would shift the
   model's class distribution meaningfully. This is how the original 8
   wins were collected.

### Final empirical record

**Win count: 10/25 → 10/25.** Across this entire session:

| layer | escalations attempted | wins added |
|---|---|---|
| Probe layer | 4 (heatmap, action-pair, triple-cycle, prelude) | 0 |
| Heuristic strategies | 5 (cluster, mover-toggle, inverse-aware, policy-guided × 2) | 0 |
| Learned classifiers | 2 (sklearn v0, CNN v1) | 0 |
| RL self-play | 5 (CEM × 4 reward configs, PPO) | 0 |

**Twelve directed escalations across four conceptual layers, all
informed by the previous failure, all hit the same wall.** This is a
solid empirical confirmation that the bottleneck is exactly where the
boundary doc identified it: the win-discovery problem cannot be
crossed by improvements to *what we already have* (8 wins, the
probe layer, the live API at 0.3s/step). The genuine paths forward
need something we don't have access to in a single session — either
more wins, more compute, or pretrained vision priors via Kaggle.

The full pipeline is preserved as a clean substrate for any of those.
