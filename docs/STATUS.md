```pm-status
milestone: M3
state: running
headSha: d139fd66d929b6b71b2439715828afe6b4542fbb
ciRunUrl: https://github.com/OJWatson/motac/actions/runs/21916209192
updatedAtUtc: 2026-02-11T17:44:26Z
```

## Next step
- Land an offline multi-seed parameter-recovery test on a tiny substrate/graph:
  - simulate counts from known parameters using the sparse road kernel,
  - fit with the MLE entrypoint,
  - assert recovered parameters are within coarse tolerances.

## Notes
- Keep commits small and CI-gated; update this file on each push/gate.
