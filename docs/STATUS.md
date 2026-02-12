```pm-status
milestone: M4
state: running
headSha: bff1c950cc6f6b35a7550297f94d258599c0f02c
ciRunUrl: https://github.com/OJWatson/motac/actions/runs/21934183342
updatedAtUtc: 2026-02-12T05:29:49Z
```

## Status
- M4 is **in progress**. Completed so far:
  - Spec checklist + mapping: `docs/spec_alignment.md`
  - Architecture summary: `docs/architecture.md`
  - Package boundaries extracted: `motac.loaders`, `motac.eval`

## Next step (M4)
- Continue structure-first package layout separation (keep imports stable):
  - introduce an `motac.inference` package stub (even if minimal) to establish the boundary,
  - move/alias any inference-like utilities currently living elsewhere,
  - add a small import-path test.
- Ensure `docs/spec_alignment.md` and `docs/architecture.md` reflect the updated layout.

## Next step (after M4 → M5)
- Resume predict/eval plumbing work (some utilities already landed early; treat them as M5 scope).

## Notes
- Documentation builds are tracked in CI; the status header points at the last meaningful code gate.
- Keep commits small and CI-gated; update this file on each push/gate.
