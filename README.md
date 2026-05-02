# ARC Prize 2026 - ARC-AGI-3

**Team:** Ben (Bazzer) + Guilherme Ferrari Brescia (guiferrarib)
**Competition:** https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
**Milestone #1:** 2026-06-30

## Current State

This checkout is now the self-contained ARC3 workspace. Follow-up agents should
be able to work from `D:\arc3` without depending on `D:\diloco_lab`.

Solved L1+ games: **10/25**

- solved: `cn04`, `lf52`, `lp85`, `m0r0`, `s5i5`, `sb26`, `sc25`, `sp80`, `tr87`, `vc33`
- unsolved: `ar25`, `bp35`, `cd82`, `dc22`, `ft09`, `g50t`, `ka59`, `ls20`, `r11l`, `re86`, `sk48`, `su15`, `tn36`, `tu93`, `wa30`

Active project lanes:

- `scripts/meta/` - live ARC meta-solver, probe layer, strategies, replay capture, and safe orchestration
- `scripts/submission/` - conservative submission manifest, offline report, package builder, and Kaggle/static audit
- `scripts/training/arc3/` - ARC3 evidence-dataset and local evaluation tooling
- `diloco_lab/` - mirrored ARC3-only helper modules used by local training/runtime code
- `kaggle/notebooks/arc3-evidence-ranker/` and `kaggle/scripts/` - Kaggle notebook builder and evidence-ranker notebook assets
- `data/arc3_exports/` - local evidence/action-prior exports
- `state/arc3_policy_*` - learned-policy and adapter artifacts
- `results/meta/` and `results/kaggle/` - game profiles, registry, and trial outputs

Current conservative package candidate:

- 9 deterministic replayable public L1 wins
- 1 known but non-replayable public L1 win: `lp85`
- 15 unsolved public L1 games
- latest dry package: `results\submission\packages\arc3_candidate_latest`
  with 186 files and strict audit status `pass_with_warnings`

The older MuZero, neuro-symbolic, and Colab replay docs are historical design
and experiment records. They are not the current live-solving path.

## Main Findings

The current empirical boundary is recorded in
[`docs/meta_solver_boundary.md`](D:/arc3/docs/meta_solver_boundary.md).

Short version:

- probe additions surfaced real structure but did not raise the win count
- heuristic strategies, sklearn/CNN policies, CEM, and PPO did not improve beyond the 10 solved games
- the v3 compact Gemma evidence ranker is usable as an opt-in action prior
- v5 click-ID training fixed schema issues but failed the post-hoc live-readiness gate, so live coordinate-click exploration should not be promoted from it
- the current submission candidate has 9 deterministic replayable public L1 wins,
  1 known but non-replayable win (`lp85`), and 15 unsolved games
- the reward-correlation audit did not clear similarity or novelty rewards for
  live steering; all saved RL reward rows had zero verified wins
- a 128-call hashed live A/B found zero wins for both random and
  probe-stratified policies, and probe-stratified did not beat random on
  unique frame hashes per call
- the progress scorer failed promotion: held-out Spearman `0.255413`, below
  the `0.35` gate; the new-frame scorer is coverage-only, not win-progress
- the no-leak frontier table is useful for candidate ranking and measurement,
  but remains `not_live_ready`: it passes several mixed-label holdouts and
  still fails the `g50t` triangular inverse/toggle diagnostic
- useful next work is submission packaging, `g50t`/triangular frontier
  isolation, broader v6 evidence, and low-waste live testing only after a
  corrected gate passes falsifiable checks

## Layout

```
arc3/
|-- agents/                # ARC-AGI-3 agent implementations (random baseline -> full stack)
|-- data/arc3_exports/     # local ARC3 evidence/action-prior exports
|-- diloco_lab/            # ARC3 helper modules mirrored into this repo
|-- muzero_integration/    # muzero-general <-> arc_agi env adapter
|-- neurosym/              # ASP rule extraction, PDDL planner, action->effect mapping
|-- kaggle/                # Kaggle notebooks and notebook-build scripts
|-- results/               # meta-solver profiles, registry, and trial outputs
|-- scripts/meta/          # current live meta-solver surface
|-- scripts/submission/    # manifest/report/audit/package harness
|-- scripts/training/arc3/ # evidence dataset/eval tooling
|-- state/                 # learned ARC3 policy and adapter artifacts
|-- ttt/                   # historical test-time training fast-weights layer
|-- prism_integration/     # Prism geometry probes on agent's hidden states (drift = E(t))
|-- viability/             # M = C - E evaluator over agent runs
|-- scorecards/            # Captured arc.get_scorecard() outputs per run
|-- external/              # Cloned repos: ARC-AGI-3-Agents/, muzero-general/
`-- docs/
    |-- meta_solver_boundary.md
    |-- scientific_todo.md
    |-- research_followup_2026.md
    `-- research-brief.md
```

## GitHub / Colab

Repo is public: `https://github.com/humanaiconvention/arc3` (default branch: `master`).

Clone includes `scripts/meta/`, `diloco_lab/`, and `state/arc3_policy_v5_evidence_ranker/`. No separate setup needed — `requirements.txt` covers runtime deps.

```bash
git clone https://github.com/humanaiconvention/arc3.git
pip install -r arc3/requirements.txt arc-agi arcengine
ARC3_EVIDENCE_RANKER_ENABLE=1 python -u arc3/scripts/meta/run_meta.py --game cd82 --strategy evidence_ranker_guided_walk --budget 7200
```

## Quick start

Use the ARC SDK virtual environment for ARC3 scripts:

```powershell
cd D:\arc3
$env:META_STEP_DELAY = "0.3"
D:\arc3\external\ARC-AGI-3-Agents\.venv\Scripts\python.exe -u scripts\meta\run_meta.py --game sp80 --replay --strategy replay_known
```

Generate the current conservative submission report without live API calls:

```powershell
cd D:\arc3
D:\arc3\external\ARC-AGI-3-Agents\.venv\Scripts\python.exe scripts\submission\evaluate_submission_candidate.py --out-dir results\submission\offline_latest
```

Live-smoke selected deterministic plans without re-characterizing:

```powershell
cd D:\arc3
$env:META_STEP_DELAY = "0.3"
D:\arc3\external\ARC-AGI-3-Agents\.venv\Scripts\python.exe scripts\submission\evaluate_submission_candidate.py --out-dir results\submission\live_replay_smoke --live-replay --games sp80 vc33 --budget 30
```

Run the score-facing offline package tests:

```powershell
cd D:\arc3
D:\arc3\external\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m unittest tests.test_submission_candidate
```

Run the strict static package audit:

```powershell
cd D:\arc3
D:\arc3\external\ARC-AGI-3-Agents\.venv\Scripts\python.exe scripts\submission\audit_submission_package.py --strict-kaggle --out results\submission\package_audit_strict.json
```

Re-run the latest reward/progress/frontier measurement layer:

```powershell
cd D:\arc3
D:\arc3\external\ARC-AGI-3-Agents\.venv\Scripts\python.exe scripts\submission\audit_reward_correlations.py --out results\research\reward_correlation_audit.json
D:\arc3\external\ARC-AGI-3-Agents\.venv\Scripts\python.exe scripts\submission\build_frontier_table.py --trace-root results\research\live_policy_ab --out-dir results\research\frontier_table_v1 --eval-games sk48
```

Stage a dry submission package:

```powershell
cd D:\arc3
D:\arc3\external\ARC-AGI-3-Agents\.venv\Scripts\python.exe scripts\submission\build_submission_package.py --out-dir results\submission\packages\arc3_candidate_latest
```

Validate the current v5 evidence export locally:

```powershell
cd D:\arc3
D:\arc3\external\ARC-AGI-3-Agents\.venv\Scripts\python.exe scripts\training\arc3\validate_arc3_action_evidence_v5.py data\arc3_exports\2026-04-29-arc3-action-evidence-v5-clickids
```

Recompute text baselines:

```powershell
cd D:\arc3
D:\arc3\external\ARC-AGI-3-Agents\.venv\Scripts\python.exe scripts\training\arc3\evaluate_arc3_action_evidence_baseline.py data\arc3_exports\2026-04-29-arc3-action-evidence-v5-clickids
```

Build the current Kaggle evidence-ranker notebook:

```powershell
cd D:\arc3
D:\arc3\external\ARC-AGI-3-Agents\.venv\Scripts\python.exe kaggle\scripts\build_arc3_evidence_ranker_nb.py --trial-name v5clickids_cap32_s90_lora16_len1792 --dataset-slug benhaslam/arc3-action-evidence-v5-clickids --max-steps 90 --learning-rate 0.00004 --balance-cap 32 --max-length 1792 --focus-probe 2 --focus-pairwise 1 --lora-r 16 --lora-alpha 32
```

## Docs

- [`docs/scientific_todo.md`](D:/arc3/docs/scientific_todo.md) - active running TODO
- [`docs/meta_solver_boundary.md`](D:/arc3/docs/meta_solver_boundary.md) - empirical boundary and trial record
- [`docs/research_followup_2026.md`](D:/arc3/docs/research_followup_2026.md) - 2026 research triage and experiment queue
- [`scripts/meta/README.md`](D:/arc3/scripts/meta/README.md) - meta-solver architecture
- [`scripts/submission/README.md`](D:/arc3/scripts/submission/README.md) - submission manifest/report/audit/package tools
- [`scripts/training/arc3/README.md`](D:/arc3/scripts/training/arc3/README.md) - local training/eval tools
- [`kaggle/notebooks/arc3-evidence-ranker/README.md`](D:/arc3/kaggle/notebooks/arc3-evidence-ranker/README.md) - Kaggle evidence-ranker notebook workflow
- [`docs/research-brief.md`](D:/arc3/docs/research-brief.md) - historical long-horizon architecture memo
