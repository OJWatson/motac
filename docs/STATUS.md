```pm-status
milestone: M4
state: running
headSha: 0963643fd9601afccda0a7b51d7473f244ab7c65
ciRunUrl: https://github.com/OJWatson/motac/actions/runs/21934839731
updatedAtUtc: 2026-02-12T05:37:59Z
```

## Status
- M4 is **in progress**. Completed so far:
  - Spec checklist + mapping: `docs/spec_alignment.md`
  - Architecture summary: `docs/architecture.md`
  - Package boundaries extracted: `motac.loaders`, `motac.eval`, `motac.inference`

## Next step (M4)
- Continue structure-first package layout separation (keep imports stable):
  - extract one more boundary (suggested small/safe: create `motac.cli` subpackage boundary while keeping `motac.cli:app` stable, or reorganise `motac.sim` internals),
  - add import-path smoke test(s) for any new package.
- Re-check M4 DoD against `docs/ROADMAP.md` after the next boundary is extracted; if satisfied, mark M4 complete and advance to M5.

## Next step (after M4 → M5)
- Resume predict/eval plumbing work (some utilities already landed early; treat them as M5 scope).

## Notes
- Documentation builds are tracked in CI; the status header points at the last meaningful code gate.
- Keep commits small and CI-gated; update this file on each push/gate.
