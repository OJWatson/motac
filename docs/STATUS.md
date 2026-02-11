```pm-status
milestone: M3
state: running
headSha: c2958c64968250dba66a7a60f97bef37a9e0f3ef
ciRunUrl: https://github.com/OJWatson/motac/actions/runs/21920166104
updatedAtUtc: 2026-02-11T19:37:43Z
```

## Next step
- Land an offline multi-seed parameter-recovery test on a tiny substrate/graph:
  - simulate counts from known parameters using the sparse road kernel,
  - fit with the MLE entrypoint,
  - assert recovered parameters are within coarse tolerances.

## Notes
- Documentation builds are tracked in CI; the status header points at the last meaningful code gate.
- Keep commits small and CI-gated; update this file on each push/gate.
