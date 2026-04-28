# Public Stage Set

This document defines the current "public-ready staged set" for [D:\arc3](D:/arc3).

## Included in the staged set

- core training/runtime files:
  - [train_pipeline.py](D:/arc3/train_pipeline.py)
  - [models.py](D:/arc3/muzero_patch/models.py)
  - [colab_stage2_meta_replays.py](D:/arc3/scripts/training/colab_stage2_meta_replays.py)
  - [start_colab_local_runtime.ps1](D:/arc3/scripts/training/start_colab_local_runtime.ps1)
- meta-solver package:
  - [scripts/meta](D:/arc3/scripts/meta)
- public-facing docs:
  - [README.md](D:/arc3/README.md)
  - [research-brief.md](D:/arc3/docs/research-brief.md)
  - [COLAB_META_REPLAY_TRAINING.md](D:/arc3/docs/COLAB_META_REPLAY_TRAINING.md)
  - [PUBLIC_REPO_SCRUB.md](D:/arc3/docs/PUBLIC_REPO_SCRUB.md)
- intentionally clean notebooks:
  - [arc3_muzero_finetune.ipynb](D:/arc3/kaggle/arc3_muzero_finetune.ipynb)
  - [arc3_muzero_ttt.ipynb](D:/arc3/kaggle/arc3_muzero_ttt.ipynb)
  - [arc3_meta_replay_colab.ipynb](D:/arc3/kaggle/arc3_meta_replay_colab.ipynb)

## Explicitly excluded from the staged set

- local artifacts and generated data:
  - [checkpoints](D:/arc3/checkpoints)
  - [recordings](D:/arc3/recordings)
  - [results](D:/arc3/results)
  - [content](D:/arc3/content)
  - [environment_files](D:/arc3/environment_files)
  - [external](D:/arc3/external)
- handoffs, notes, and operator-specific files:
  - [STATUS.md](D:/arc3/STATUS.md)
  - [ARC3_AGENT_PROMPT.md](D:/arc3/ARC3_AGENT_PROMPT.md)
  - [ARC3_HANDOFF.md](D:/arc3/ARC3_HANDOFF.md)
- scorecard logs and Kaggle kernel staging files:
  - [scorecards](D:/arc3/scorecards)
  - [kaggle/kernels](D:/arc3/kaggle/kernels)
- one-off experiment scripts outside the cohesive meta/training surface

## Purpose

This staged set is meant to provide a smaller, safer public snapshot before any repository visibility change.

It is not a claim that every other file is wrong to publish. It is a conservative default set intended to minimize accidental exposure of local artifacts, internal notes, and generated outputs.
