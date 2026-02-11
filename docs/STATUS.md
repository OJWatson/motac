```pm-status
milestone: M3
state: running
headSha: 357d779163b35662333c7926618e4dc7281d24f4
ciRunUrl: https://github.com/OJWatson/motac/actions/runs/21916380859
updatedAtUtc: 2026-02-11T17:56:47Z
```

## Next step
- Land an offline multi-seed parameter-recovery test on a tiny substrate/graph:
  - simulate counts from known parameters using the sparse road kernel,
  - fit with the MLE entrypoint,
  - assert recovered parameters are within coarse tolerances.

## Notes
- Documentation build for this push: https://github.com/OJWatson/motac/actions/runs/21916380831
- Keep commits small and CI-gated; update this file on each push/gate.
