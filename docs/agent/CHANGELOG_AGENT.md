# CHANGELOG_AGENT

Agent-authored change log (high level).

## 2026-02-16
- Record M3.1 completion in TASKSTATE + roll NEXT_TASKS forward to M3.2.

## 2026-02-15
- Close M2 milestone (CI gate passed locally); roll NEXT_TASKS forward to M3.
- CI.FIX.M2: fix docs deployment workflow by ensuring deploy job checks out the repo (actions-gh-pages expects a git workspace).
