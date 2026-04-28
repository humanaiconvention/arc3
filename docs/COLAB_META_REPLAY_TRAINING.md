# Colab Meta-Replay Training

This is the fastest current path to train on fresh ARC3 solve data in Colab GPU.

## Current replay corpus

As of April 21, 2026 the clean exported replay set is:

- `sp80`
- `tr87`
- `vc33`

Files live in [recordings/meta_replays](D:/arc3/recordings/meta_replays).

Two known L1 wins are **not** yet replayable as training data:

- `lp85`
- `m0r0`

Those older registry entries only stored bare `ACTION6` steps without coordinates, so replay/export fails until we rediscover them with coordinate-aware capture.

## Recommended setup

Use a GPU runtime and keep the run scoped to Stage 2 fine-tuning if you want hosted compute.

Recommended starting settings for the current 3-episode replay corpus:

- `stage2-steps=15000`
- `batch=256`
- `--compile` on if the Colab GPU/runtime supports it cleanly

This is intentionally conservative because the replay dataset is still tiny.

## Managed Colab bootstrap

```bash
git clone --depth=1 https://github.com/humanaiconvention/arc3.git
git clone --depth=1 https://github.com/werner-duvaud/muzero-general.git
pip install -q torch torchvision ray[default] tensorboard Pillow numpy
cp arc3/muzero_patch/models.py muzero-general/models.py
```

If your replay recordings and Stage 1 checkpoint are in Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Run Stage 2 on meta replays only

From the repo root:

```bash
python scripts/training/colab_stage2_meta_replays.py \
  --arc3-dir /content/drive/MyDrive/arc3/meta_replays \
  --stage1-ckpt /content/drive/MyDrive/arc3/stage1_pretrain.checkpoint \
  --muzero-dir /content/muzero-general \
  --out-dir /content/drive/MyDrive/arc3/colab_meta_replay_results \
  --stage2-steps 15000 \
  --batch 256 \
  --compile
```

This uses [train_pipeline.py](D:/arc3/train_pipeline.py) in the existing repo flow, but skips fresh Stage 1 training by loading the checkpoint you provide.

## Local runtime option

Local runtime is the cleanest workaround if you want to use the Colab UI against the existing local checkout at [D:\arc3](D:/arc3).

Important tradeoff:

- local runtime uses your local CPU/GPU, not Colab's hosted GPU
- `drive.mount()` is not available the same way on a local runtime, so prefer direct local filesystem paths

### Start the local backend on Windows

From PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File D:\arc3\scripts\training\start_colab_local_runtime.ps1
```

Then in Colab:

- click `Connect`
- choose `Connect to local runtime...`
- paste the `http://127.0.0.1:8888/?token=...` URL printed by Jupyter

### Run Stage 2 against the local checkout

Use a fresh code cell after connecting to the local runtime:

```python
import pathlib, subprocess, sys

ROOT = pathlib.Path(r'D:\arc3')
MUZERO_DIR = ROOT / 'external' / 'muzero-general'
ARC3_REC_DIR = ROOT / 'recordings' / 'meta_replays'
STAGE1_CKPT = ROOT / 'checkpoints' / 'stage1_pretrain_2026-04-08--22-03-52.checkpoint'
OUT_DIR = ROOT / 'results' / 'colab_local_meta_replays'

MUZERO_DIR.parent.mkdir(parents=True, exist_ok=True)
if not MUZERO_DIR.exists():
    subprocess.run([
        'git', 'clone', '--depth=1',
        'https://github.com/werner-duvaud/muzero-general.git',
        str(MUZERO_DIR),
    ], check=True)

subprocess.run([
    sys.executable,
    str(ROOT / 'scripts' / 'training' / 'colab_stage2_meta_replays.py'),
    '--arc3-dir', str(ARC3_REC_DIR),
    '--stage1-ckpt', str(STAGE1_CKPT),
    '--muzero-dir', str(MUZERO_DIR),
    '--out-dir', str(OUT_DIR),
    '--stage2-steps', '15000',
    '--batch', '256',
    '--compile',
], check=True, cwd=str(ROOT))
```

This avoids GitHub access for the private ARC3 repo completely because the runtime is executing against the local checkout you already have.

## What to expect

- The loader should report `3` ARC3 histories if you use the current exported replay set only.
- Output checkpoint will be named like `stage2_finetune_<timestamp>.checkpoint`.
- That checkpoint can feed the existing TTT/live evaluation path afterward.

## Best next data improvement

The biggest lever is still more clean action-aware recordings, not bigger GPU.

Best next sources:

- push a slower meta pass and export any new replayable wins
- rediscover `lp85` and `m0r0` with coordinate-aware capture
- keep future click-game wins coordinate-aware via updated random-walk/meta capture
