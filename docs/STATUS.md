```pm-status
milestone: M3
state: running
headSha: a9e2039ec83489275353ee44f1e38be14098824e
ciRunUrl: https://github.com/OJWatson/motac/actions/runs/21915687990
updatedAtUtc: 2026-02-11T18:17:57Z
```

## Next step
- Land an offline multi-seed parameter-recovery test on a tiny substrate/graph:
  - simulate counts from known parameters using the sparse road kernel,
  - fit with the MLE entrypoint,
  - assert recovered parameters are within coarse tolerances.

## Notes
- Documentation builds are tracked in CI; the status header points at the last meaningful code gate.
- Keep commits small and CI-gated; update this file on each push/gate.
