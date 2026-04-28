"""Submit a slower-paced ARC3 meta run to Hugging Face Jobs."""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

from huggingface_hub import HfApi, SpaceHardware


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_REPO_URL = "https://github.com/humanaiconvention/arc3.git"
DEFAULT_SCRIPT = ROOT / "scripts" / "meta" / "hf_job_entry.py"
PASS_THROUGH_SECRETS = ["ARC_API_KEY", "ARC_BASE_URL"]


def _git_output(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _default_git_ref() -> str:
    branch = _git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return _git_output(["git", "rev-parse", "HEAD"]) if branch == "HEAD" else branch


def _is_dirty() -> bool:
    return bool(_git_output(["git", "status", "--porcelain"]))


def _load_env_file() -> None:
    env_file = ROOT / "external" / "ARC-AGI-3-Agents" / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _build_secrets() -> dict[str, str]:
    _load_env_file()
    secrets = {}
    for key in PASS_THROUGH_SECRETS:
        value = os.environ.get(key)
        if value:
            secrets[key] = value
    return secrets


def _parse_flavor(value: str) -> SpaceHardware:
    for item in SpaceHardware:
        if item.value == value:
            return item
    raise argparse.ArgumentTypeError(f"Unknown HF hardware flavor: {value}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--git-ref", default=None,
                    help="Remote branch, tag, or commit to run. Default: current branch or HEAD commit.")
    ap.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--budget", type=float, default=240.0)
    ap.add_argument("--step-delay", type=float, default=0.35)
    ap.add_argument("--flavor", type=_parse_flavor, default=SpaceHardware.CPU_UPGRADE)
    ap.add_argument("--timeout", default="2h")
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--game", action="append", default=[])
    ap.add_argument("--strategy", action="append", default=[])
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--follow", action="store_true",
                    help="Stream job logs after submission.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the submission payload without creating a job.")
    args = ap.parse_args()

    git_ref = args.git_ref or _default_git_ref()
    dirty = _is_dirty()

    script_args = [
        "--repo-url", args.repo_url,
        "--git-ref", git_ref,
        "--workers", str(args.workers),
        "--budget", str(args.budget),
        "--step-delay", str(args.step_delay),
    ]
    for game in args.game:
        script_args.extend(["--game", game])
    for strategy in args.strategy:
        script_args.extend(["--strategy", strategy])
    if args.replay:
        script_args.append("--replay")

    payload = {
        "script": str(DEFAULT_SCRIPT),
        "script_args": script_args,
        "flavor": args.flavor.value,
        "timeout": args.timeout,
        "namespace": args.namespace,
        "secrets": sorted(_build_secrets()),
        "dirty_worktree": dirty,
    }
    print("HF job payload:")
    for key, value in payload.items():
        print(f"  {key}: {value}")
    if dirty:
        print("Warning: local uncommitted changes are not included in remote jobs.", file=sys.stderr)

    if args.dry_run:
        return

    api = HfApi()
    job = api.run_uv_job(
        script=str(DEFAULT_SCRIPT),
        script_args=script_args,
        flavor=args.flavor,
        timeout=args.timeout,
        namespace=args.namespace,
        secrets=_build_secrets() or None,
        env={"PYTHONUNBUFFERED": "1"},
        labels={"project": "arc3", "runner": "meta"},
    )
    print(f"Submitted job {job.id}")
    print(job.url)

    if args.follow:
        for line in api.fetch_job_logs(job_id=job.id, namespace=args.namespace, follow=True):
            print(line, end="")


if __name__ == "__main__":
    main()
