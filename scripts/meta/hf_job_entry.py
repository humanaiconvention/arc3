"""Remote Hugging Face Job entrypoint for ARC3 meta runs."""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess


def _run(cmd: list[str], cwd: pathlib.Path | None = None) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _ensure_git() -> None:
    if shutil.which("git"):
        return
    if shutil.which("apt-get"):
        _run(["apt-get", "update"])
        _run(["apt-get", "install", "-y", "git"])
        return
    raise RuntimeError("git is required in the job container but was not found.")


def _checkout_repo(repo_url: str, git_ref: str, workspace: pathlib.Path) -> pathlib.Path:
    repo_dir = workspace / "arc3"
    _ensure_git()
    _run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)])
    _run(["git", "fetch", "--depth", "1", "origin", git_ref], cwd=repo_dir)
    _run(["git", "checkout", "FETCH_HEAD"], cwd=repo_dir)
    return repo_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", required=True)
    ap.add_argument("--git-ref", required=True,
                    help="Remote branch, tag, or commit reachable from origin.")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--budget", type=float, default=240.0)
    ap.add_argument("--step-delay", type=float, default=0.35)
    ap.add_argument("--game", action="append", default=[])
    ap.add_argument("--strategy", action="append", default=[])
    ap.add_argument("--replay", action="store_true")
    args = ap.parse_args()

    workspace = pathlib.Path("/tmp/hf-job")
    workspace.mkdir(parents=True, exist_ok=True)
    repo_dir = _checkout_repo(args.repo_url, args.git_ref, workspace)

    agents_dir = repo_dir / "external" / "ARC-AGI-3-Agents"
    _run(["uv", "sync"], cwd=agents_dir)

    runner = [
        "uv", "run",
        "--project", str(agents_dir),
        "python", "-u",
        str(repo_dir / "scripts" / "meta" / "run_meta_parallel.py"),
        "--workers", str(args.workers),
        "--budget", str(args.budget),
    ]
    for game in args.game:
        runner.extend(["--game", game])
    for strategy in args.strategy:
        runner.extend(["--strategy", strategy])
    if args.replay:
        runner.append("--replay")

    env = dict(os.environ)
    env["META_STEP_DELAY"] = str(args.step_delay)
    print(f"Running meta solver with META_STEP_DELAY={env['META_STEP_DELAY']}", flush=True)
    print(f"Workspace: {repo_dir}", flush=True)
    subprocess.run(runner, cwd=str(repo_dir), env=env, check=True)


if __name__ == "__main__":
    main()
