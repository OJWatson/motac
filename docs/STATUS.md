```pm-status
milestone: M4
state: waiting-for-ci
headSha: 8add516adf501d5b838a5f5a88b2c3ac81b240e4
ciRunUrl: https://github.com/OJWatson/motac/actions/runs/21923665484
updatedAtUtc: 2026-02-11T21:26:15Z
```

## Status
- M3 is **complete** (CI green).
- Note: `forecast_intensity_horizon` + `mean_negative_log_likelihood` landed early; this work belongs to **M5** per `docs/ROADMAP.md` and will be treated as an early M5 slice.

## Next step (M4)
- Package layout (structure-first): introduce/confirm clear top-level module separation for:
  - loaders/schema, substrate, models, inference, sim, eval, cli, configs.
- Update `docs/ROADMAP.md` only if any deliberate deviations are introduced.

## Next step (after M4, M5 continuation)
- Add an end-to-end toy example (fit -> forecast -> score) as a unit test.

## Notes
- Documentation builds are tracked in CI; the status header points at the last meaningful code gate.
- Keep commits small and CI-gated; update this file on each push/gate.
