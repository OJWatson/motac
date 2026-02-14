# M0/M1 milestone DoD (canonical)

This file is the **single source of truth** for the definition-of-done (DoD)
for the early reset milestones.

It exists to stop M0/M1 work drifting into later-milestone feature work.

If this file and any other doc disagree, treat this file as authoritative.

## M0 — Reset + schema (DoD)

M0 is done when all of the following are true:

- **Canonical event schema exists** (one source of truth in code).
- **Schema validation is tested** (unit tests that fail on invalid events).
- **Repo layout is stable**:
  - `pyproject.toml` configured
  - `src/` layout used
  - `uv run pytest -q` passes in CI
- **Docs point to the canonical schema** (so later chapters do not invent new formats).

Practical verification:

- Canonical schema location is documented and discoverable.
- Tests cover at least: required fields, types, and a representative invalid payload.

## M1 — Substrate cache artefacts v1 (DoD)

M1 is done when all of the following are true:

- **Cache bundle format is defined and versioned** (reader can reject unknown versions).
- **Writes are deterministic** (same inputs → identical bundle bytes / file tree).
- **Provenance hash is stable** (guarded by a regression test).
- **Docs describe the artefact contract**:
  - what the bundle contains (high level)
  - how to load it (one minimal code snippet or API pointer)

## Explicit out-of-scope for M0/M1

Track these under later milestones instead:

- End-to-end forecasting / backtesting workflows.
- Full observed-data ingestion pipelines (ACLED/Chicago loaders, ETL, cleaning).
- Significant CLI UX work beyond minimal entrypoints.
