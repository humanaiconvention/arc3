"""Colab-friendly launcher for Stage 2 fine-tuning on exported meta replays."""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]


def _run(cmd: list[str], cwd: pathlib.Path | None = None) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc3-dir", required=True,
                    help="Directory of action-aware ARC3 recordings, e.g. recordings/meta_replays.")
    ap.add_argument("--stage1-ckpt", required=True,
                    help="Path to a Stage 1 checkpoint to warm-start from.")
    ap.add_argument("--muzero-dir", default=str(ROOT / "muzero-general"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "colab_meta_replays"))
    ap.add_argument("--stage2-steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-skip-action-blind", action="store_true",
                    help="Pass through to train_pipeline.py if you intentionally want old blind recordings.")
    args = ap.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "train_pipeline.py"),
        "--arc3-dir", args.arc3_dir,
        "--stage1-ckpt", args.stage1_ckpt,
        "--muzero-dir", args.muzero_dir,
        "--out-dir", args.out_dir,
        "--stage2-steps", str(args.stage2_steps),
        "--batch", str(args.batch),
    ]
    if args.compile:
        cmd.append("--compile")
    if args.no_skip_action_blind:
        cmd.append("--no-skip-action-blind")

    _run(cmd, cwd=ROOT)


if __name__ == "__main__":
    main()
