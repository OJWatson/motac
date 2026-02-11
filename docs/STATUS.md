```pm-status
milestone: M3
state: running
headSha: c2958c64968250dba66a7a60f97bef37a9e0f3ef
ciRunUrl: https://github.com/OJWatson/motac/actions/runs/21920166104
updatedAtUtc: 2026-02-11T19:37:43Z
```

## Next step
- Parameter-recovery test is landed (multi-seed, tiny offline road matrix).
- Next: decide whether M3 DoD is met; if not, extend coverage to:
  - Negative Binomial recovery (fit with `family=negbin` on simulated data), or
  - add a minimal predict wrapper for next-step intensity/in-sample intensity.

Notes on the test:
- Uses median-over-seeds with coarse tolerances to reduce CI flakiness.

## Notes
- Documentation builds are tracked in CI; the status header points at the last meaningful code gate.
- Keep commits small and CI-gated; update this file on each push/gate.
