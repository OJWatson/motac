```pm-status
milestone: M4
state: running
headSha: 68dc1ae6d8d5bb506451dd85f138d3f29d383b73
ciRunUrl: https://github.com/OJWatson/motac/actions/runs/21922218173
updatedAtUtc: 2026-02-11T20:55:14Z
```

## Status
- M3 is **complete** at headSha `68dc1ae6d8d5bb506451dd85f138d3f29d383b73` (CI green).

## Next step (M4)
- Implement minimal predict/eval plumbing v1:
  - small utility to produce next-step intensity forecasts over a horizon,
  - a tiny evaluation metric on toy data (e.g. mean negative log-likelihood),
  - CI-safe unit tests for shapes and determinism.

Notes:
- Recovery tests use median-over-seeds with coarse tolerances to reduce CI flakiness.

## Notes
- Documentation builds are tracked in CI; the status header points at the last meaningful code gate.
- Keep commits small and CI-gated; update this file on each push/gate.
