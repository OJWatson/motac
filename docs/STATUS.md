```pm-status
milestone: M5
state: running
headSha: edfb3f190e8b063eb93b48c7905b3787936b46a7
ciRunUrl: https://github.com/OJWatson/motac/actions/runs/21936678278
updatedAtUtc: 2026-02-12T06:56:10Z
```

## Status
- M4 is **complete**:
  - Spec checklist + mapping: `docs/spec_alignment.md`
  - Architecture summary: `docs/architecture.md`
  - Package boundaries extracted (structure-first):
    - `motac.loaders`, `motac.substrate`, `motac.model`, `motac.sim`, `motac.eval`, `motac.inference`, `motac.cli`, `motac.configs`
  - CLI internal boundary extracted: `motac.cli.commands` (while keeping the entry point stable)
  - Import-path smoke tests added for new packages.

- M5 is **in progress**.

## Next step (M5)
- Implement minimal predict/eval plumbing for the parametric model and add toy-data tests (shapes/consistency), per `docs/ROADMAP.md`.

## Notes
- Documentation builds are tracked in CI; the status header points at the last meaningful code gate.
- Keep commits small and CI-gated; update this file on each push/gate.
