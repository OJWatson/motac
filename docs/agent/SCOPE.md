# SCOPE

This repository is managed under OpenClaw Portfolio OS.

## Operational sources of truth

- Source of truth for next work: `docs/agent/NEXT_TASKS.md`
- Source of truth for state: `docs/agent/PROJECT_STATE.md`
- Source of truth for completed tasks: `docs/agent/TRACE.md`

## Milestone scope notes (M0/M1)

These are the **guard rails** for early milestones.
If a task starts drifting beyond these, move it to a later milestone instead of
forcing it into M0/M1.

- **M0 (reset + schema):**
  - canonical event schema (single source of truth)
  - schema validation tests
  - stable repo layout (pyproject + src/; CI runs)
  - docs that point to the canonical schema

- **M1 (substrate cache artefacts v1):**
  - versioned cache bundle format
  - deterministic writes
  - stable provenance hash (tested)
  - short docs: what’s inside + how to load

Explicitly out-of-scope for M0/M1: loaders/ETL, full workflows (forecast/backtest),
and any significant CLI UX work.
