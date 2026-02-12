```pm-status
milestone: M4
state: waiting-for-ci
headSha: edfb3f190e8b063eb93b48c7905b3787936b46a7
ciRunUrl: https://github.com/OJWatson/motac/actions/runs/21936278420
updatedAtUtc: 2026-02-12T06:40:30Z
```

## Status
- M4 is **in progress**. Completed so far:
  - Spec checklist + mapping: `docs/spec_alignment.md`
  - Architecture summary: `docs/architecture.md`
  - Package boundaries extracted: `motac.loaders`, `motac.eval`, `motac.inference`, `motac.cli`, `motac.configs`
  - CLI internal boundary extracted: `motac.cli.commands` (while keeping the entry point stable)
  - Import-path smoke tests added for new packages.

## Next step (M4)
- Re-check M4 DoD against `docs/ROADMAP.md` now that an additional boundary has landed.
- If satisfied, mark M4 complete and advance to M5.

## Next step (after M4 → M5)
- Resume predict/eval plumbing work (some utilities already landed early; treat them as M5 scope).

## Notes
- Documentation builds are tracked in CI; the status header points at the last meaningful code gate.
- Keep commits small and CI-gated; update this file on each push/gate.
