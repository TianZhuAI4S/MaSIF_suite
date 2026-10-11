# Stage 22 — Initial GitHub publication

## Purpose

Publish the standalone MaSIF Torch source repository to the user-selected GitHub
remote while retaining the existing clean repository rules and numerical scope.

## Intended work

Initialize the standalone Git repository, configure
`https://github.com/TianZhuAI4S/MaSIF_torch.git` as `origin`, use branch `main`,
review the actual staged files, create the initial source commit and push `main`
with its upstream tracking configuration. The initial remote has no advertised
branches; no existing history needs replacement.

## Implemented changes

The local repository has been initialized on `main` and the requested remote has
been configured. Source, Hydra resources/examples, portable installation recipes,
tools, tests, license, references, usage documentation, numerical evidence and the
weight catalog/manifest are selected for the initial source commit. Binary weights,
local datasets, upstream checkouts, run results, caches, builds and machine-specific
environment snapshots remain ignored.

No scientific code, model parameters, arithmetic or task defaults are modified by
publication. The prior weight-layout gate records exact checkpoint copies and all
28 original/copy CPU/CUDA forward comparisons. Publication does not certify missing
historical metrics or remove the existing training/checkpoint limitations.

## Changed files

- Git metadata: standalone repository initialization, `origin`, branch and upstream.
- `docs/torch_migration/22_github_publication.md`: this publication step record.
- `docs/torch_migration/REPORT.md`: source publication destination and scope.

## Verification

The final staged inventory, numerical regression outcome and independent review
are recorded below before the initial commit. Remote delivery is checked by matching
the pushed remote `main` object ID against the local commit and confirming upstream
tracking and a clean tracked worktree. No force push is required or authorized.


### Initial commit checks

- Actual staged inventory: **217 files**, approximately **1.84 MB**.
- Implementation/resources: **47 Python files**, **30 packaged Hydra YAML files**,
  **18 example YAML files**.
- Installation, license, citations, organizer, test suite, weight README and YAML
  manifest are present. No binary weights, local artifact trees, build/editor
  caches or machine-specific environment snapshots are staged.
- `git diff --cached --check` passes.
- Existing source-oracle numerical suite: **11 tests pass**, zero errors,
  failures or skips, 1.430 seconds. No source code changes accompany publication.
- Independent staged-file review passes: required application/install files are
  included, generated/binary artifacts are excluded, and branch/remote are correct.


### Confirmed source delivery

On 2026-10-11, the initial source commit `b016e8de20dc8dad1fd728e8104f1353375de7c1` was successfully pushed to
`origin/main`. Git's advertised remote `refs/heads/main` exactly matched the local
commit, upstream tracking was `origin/main`, and the tracked worktree was clean
before this delivery record was added.

The HTTPS push initially lacked terminal credentials. The existing SSH login
successfully pushed the same user-selected repository. `origin` retains the
requested HTTPS fetch URL and now uses
`git@github.com:TianZhuAI4S/MaSIF_torch.git` as its repository-specific push URL,
so subsequent ordinary `git push` commands use the working authentication.
No credentials were copied into the project. This documentation record is committed
and pushed as a normal follow-up; remote/local equality is checked again afterward.
