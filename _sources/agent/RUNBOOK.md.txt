# RUNBOOK

## Local dev / acceptance

Acceptance commands are configured in `portfolio/projects.yaml` and should be run from the repo root.

### Toolchain baseline (last verified: 2026-02-13)

- `uv --version` → `uv 0.10.0`
- Note: `python` may not be on `$PATH` in this environment; prefer `uv run python ...`.
- `uv run ruff --version` → `ruff 0.15.0`
- `uv run pytest -q` → `94 passed` (≈ 13s on kana-XPS-13-9370)

### Commands

```bash
uv run ruff check .
uv run pytest -q
```

If acceptance fails: stop, record the blocker in `docs/agent/PROJECT_STATE.md` (include the failing command and log path), and do not continue.

## Branch policy

Automation commits directly to the configured default branch.
