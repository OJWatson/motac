```pm-status
milestone: M3
state: running
headSha: 68dc1ae6d8d5bb506451dd85f138d3f29d383b73
ciRunUrl: https://github.com/OJWatson/motac/actions/runs/21922218173
updatedAtUtc: 2026-02-11T20:40:14Z
```

## Next step
- Land M3 completion decision:
  - Review M3 DoD in `docs/ROADMAP.md` (sparse road intensity; Poisson+NegBin; recovery tests).
  - If satisfied, mark M3 complete and advance to M4.
  - If not, add any missing small API/docs glue (keep CI-safe).

Notes:
- Recovery tests use median-over-seeds with coarse tolerances to reduce CI flakiness.
- Minimal prediction wrappers are now available for in-sample and next-step intensity.

## Notes
- Documentation builds are tracked in CI; the status header points at the last meaningful code gate.
- Keep commits small and CI-gated; update this file on each push/gate.
