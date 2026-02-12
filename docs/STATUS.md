```pm-status
milestone: M4
state: waiting-for-ci
headSha: bff1c950cc6f6b35a7550297f94d258599c0f02c
ciRunUrl: https://github.com/OJWatson/motac/actions/runs/21934183342
updatedAtUtc: 2026-02-12T04:59:28Z
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
