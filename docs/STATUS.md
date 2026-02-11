```pm-status
milestone: M4
state: running
headSha: 267c6cce6d07888ffe2580ae3762a6444e8fc3aa
ciRunUrl: https://github.com/OJWatson/motac/actions/runs/21923108678
updatedAtUtc: 2026-02-11T21:11:30Z
```

## Status
- M3 is **complete** (CI green).
- Note: `forecast_intensity_horizon` + `mean_negative_log_likelihood` landed early; this work belongs to **M5** per `docs/ROADMAP.md` and will be treated as an early M5 slice.

## Next step (M4)
- Create `docs/spec_alignment.md`:
  - checklist of PDF-required components,
  - what exists vs missing,
  - mapping from milestones to modules/code locations.
- Package layout (structure-first): introduce/confirm clear top-level module separation for:
  - loaders/schema, substrate, models, inference, sim, eval, cli, configs.

## Next step (after M4, M5 continuation)
- Add an end-to-end toy example (fit -> forecast -> score) as a unit test.

## Notes
- Documentation builds are tracked in CI; the status header points at the last meaningful code gate.
- Keep commits small and CI-gated; update this file on each push/gate.
