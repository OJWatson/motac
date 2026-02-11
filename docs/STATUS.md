```pm-status
milestone: M4
state: running
headSha: 267c6cce6d07888ffe2580ae3762a6444e8fc3aa
ciRunUrl: https://github.com/OJWatson/motac/actions/runs/21923108678
updatedAtUtc: 2026-02-11T21:09:18Z
```

## Status
- M3 is **complete** at headSha `68dc1ae6d8d5bb506451dd85f138d3f29d383b73` (CI green).

## Next step (M4)
- Extend predict/eval plumbing v1:
  - add an end-to-end toy example (fit -> forecast -> score) as a unit test,
  - optionally add Negative Binomial mean NLL metric and a smoke test.

Notes:
- Recovery tests use median-over-seeds with coarse tolerances to reduce CI flakiness.

## Notes
- Documentation builds are tracked in CI; the status header points at the last meaningful code gate.
- Keep commits small and CI-gated; update this file on each push/gate.
