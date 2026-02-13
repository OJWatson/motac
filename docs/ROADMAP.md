# Roadmap (reset)

This roadmap has been **reset** to focus on the near-term, executable milestones.
Legacy/previous roadmap text is preserved under `docs/legacy/` for reference.

## M0 — Project reset + spec alignment

### M0.1 — Documentation reset
- Move legacy, generated Sphinx build artefacts under `docs/legacy/`.
- Reset `docs/ROADMAP.md` to a short, near-term plan.

### M0.2 — Confirm build/test baseline
- Ensure `uv run ruff check .` and `uv run pytest -q` pass on a clean checkout.
- Record any constraints (Python/uv versions) in `docs/agent/RUNBOOK.md`.

### M0.3 — Event schema + validation scaffold
- Define the canonical event schema in code.
- Add validation helpers and unit tests.

## M1+ — Deferred

Higher milestones will be reintroduced once M0 is complete and the repo is in a
stable, buildable state.
