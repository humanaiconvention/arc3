# Public Repo Scrub

This repo currently mixes source code with many local experiment artifacts.

Before making the repository public, keep the public snapshot focused on:

- source code under [neurosym](D:/arc3/neurosym), [scripts](D:/arc3/scripts), [muzero_patch](D:/arc3/muzero_patch), and selected docs
- intentional notebooks under [kaggle](D:/arc3/kaggle)
- stable summaries that are meant for outside readers

The updated [.gitignore](D:/arc3/.gitignore) now excludes common non-public material:

- local caches and runtime dirs
- checkpoints, recordings, and result folders
- logs, numpy dumps, and progress files
- overnight wrappers and one-off local launchers
- handoff notes and session summaries
- `.env` and common secret-bearing key files

Manual review still recommended before flipping visibility:

- inspect tracked notebooks for saved outputs or local filesystem paths
- inspect tracked logs in [scorecards](D:/arc3/scorecards) and decide whether they belong in a public repo
- confirm [STATUS.md](D:/arc3/STATUS.md) contains only material you want public
- confirm no private repo URLs, tokens, or collaborator-only notes remain in markdown files

Suggested publish strategy:

1. Stage only the source, docs, and notebooks you intentionally want public.
2. Leave experiments, checkpoints, recordings, and local notes untracked.
3. Make the repository public only after that curated commit is ready.
