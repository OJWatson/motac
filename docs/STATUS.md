```pm-status
milestone: M4
state: running
headSha: 7ad07d7cddf90a2383d719586c85c51de57376f4
ciRunUrl: https://github.com/OJWatson/motac/actions/runs/21924221901, https://github.com/OJWatson/motac/actions/runs/21924221903
updatedAtUtc: 2026-02-11T21:54:20Z
```

## Status
- M3 is **complete** (CI green).
- Note: `forecast_intensity_horizon` + `mean_negative_log_likelihood` landed early; this work belongs to **M5** per `docs/ROADMAP.md` and will be treated as an early M5 slice.

## Next step (M4)
- Continue package layout separation (structure-first):
  - decide next boundary to extract (e.g. `eval/` or `inference/`),
  - keep stable imports (re-export stubs as needed),
  - add/update a small test if import paths change.
- Keep `docs/spec_alignment.md` and `docs/architecture.md` in sync.

## Next step (after M4, M5 continuation)
- Add an end-to-end toy example (fit -> forecast -> score) as a unit test.

## Notes
- Documentation builds are tracked in CI; the status header points at the last meaningful code gate.
- Keep commits small and CI-gated; update this file on each push/gate.
