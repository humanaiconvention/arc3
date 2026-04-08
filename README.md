# ARC Prize 2026 — ARC-AGI-3

**Team:** Ben (Bazzer) + Guilherme Ferrari Brescia (guiferrarib)
**Competition:** https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
**Milestone #1:** 2026-06-30
**Brief:** `docs/research-brief.md` (mirrored from `C:\Users\benja\Downloads\arc-agi-3-research-brief.md`)

## Architecture (target)

MuZero (world model + MCTS) + Neuro-Symbolic rule extraction (clingo/ASP → PDDL → pyperplan, after Lorang et al. CoRL 2025 / Duggan et al. ICRA 2026) + TTT (test-time training fast weights, from Genesis-152M lineage). Wrapped in the HAIC Viability Convention loop: `M = C(t) − E(t) ≥ 0`, where E(t) is per-game environment drift and C(t) is the agent's corrective capacity (symbolic σ + MuZero refinement + TTT updates).

The pipeline is structured as eight phases (A → H), with the symbolic side (A → F) running on CPU and the neural side (G, H) only firing when the symbolic plan needs concretization. See `docs/research-brief.md` §3.3–§3.5 for the full Lorang-method mapping table, honest caveats, and dependency list.

## Layout

```
arc3/
├── agents/                # ARC-AGI-3 agent implementations (random baseline → full stack)
├── muzero_integration/    # muzero-general ↔ arc_agi env adapter
├── neurosym/              # ASP rule extraction, PDDL planner, action→effect mapping
├── ttt/                   # Test-time training fast-weights layer (Genesis-152M derived)
├── prism_integration/     # Prism geometry probes on agent's hidden states (drift = E(t))
├── viability/             # M = C − E evaluator over agent runs
├── scorecards/            # Captured arc.get_scorecard() outputs per run
├── scripts/               # Run launchers, baseline sweeps, profiling
├── external/              # Cloned repos: ARC-AGI-3-Agents/, muzero-general/
└── docs/
    └── research-brief.md
```

## Quick start

```bash
pip install arc-agi
cd D:\arc3\external
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
git clone https://github.com/werner-duvaud/muzero-general.git
```

Random baseline: `scripts/run_random_baseline.py` — scores all 0% (see `scorecards/BASELINE_SUMMARY.md`).

Action-labeled recording: `--agent=randomrecordactions` via `scripts/run_random_baseline.py` (registered in `external/ARC-AGI-3-Agents/agents/__init__.py`).

Phase B (transition graph): `python neurosym/phase_b_graph.py --collapse-complex`

Phase D (STRIPS domain induction):
```bash
python neurosym/phase_d_asp.py --selftest                                    # SAT in 0.01s
python neurosym/phase_d_asp.py --game ft09 --max-nodes 20 --time-limit 30   # SAT in 2.35s
python neurosym/phase_d_asp.py --game ls20 --max-nodes 15 --time-limit 30   # SAT in 15.95s
```

See `neurosym/README.md` for the full Phase B v2 + Phase D results table and gotchas log.

## Connection to gemma4good / Prism / Genesis-v3

- **Viability Condition** is shared infrastructure with `D:\gemma4good`. Reuse `viability/viability_condition.py` pattern; per-game E(t) replaces per-conversation E(t).
- **Prism** provides activation-level drift measurement on the MuZero representation function `h()` and on the TTT fast-weight deltas. Prior Genesis-v3 runs (`D:\prism\logs\model-runs\genesis-v3\`) were `partial` — structural audit passed, validation sweep needs work. Same Triton-disabled Windows fallback applies here.
- **Genesis-152M** (`guiferrarib/genesis-152m-instruct`) is the donor for the TTT layer.

See `docs/research-brief.md` for full architecture, references, and milestones.
