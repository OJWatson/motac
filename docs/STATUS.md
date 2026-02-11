```pm-status
milestone: M3
state: running
headSha: 683e1a55638a195a7de74b3364bbd762f70a6a08
ciRunUrl: https://github.com/OJWatson/motac/actions/runs/21916972221, https://github.com/OJWatson/motac/actions/runs/21916972228
updatedAtUtc: 2026-02-11T18:09:59Z
```

## Next step
- Land an offline multi-seed parameter-recovery test on a tiny substrate/graph:
  - simulate counts from known parameters using the sparse road kernel,
  - fit with the MLE entrypoint,
  - assert recovered parameters are within coarse tolerances.

## Notes
- Documentation build for this push: https://github.com/OJWatson/motac/actions/runs/21916380831
- Keep commits small and CI-gated; update this file on each push/gate.
