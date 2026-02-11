```pm-status
milestone: M3
state: running
headSha: f6cf59b9029744cb5c284ebff89d9778384c654b
ciRunUrl: https://github.com/OJWatson/motac/actions/runs/21916684805, https://github.com/OJWatson/motac/actions/runs/21916684848
updatedAtUtc: 2026-02-11T18:04:51Z
```

## Next step
- Land an offline multi-seed parameter-recovery test on a tiny substrate/graph:
  - simulate counts from known parameters using the sparse road kernel,
  - fit with the MLE entrypoint,
  - assert recovered parameters are within coarse tolerances.

## Notes
- Documentation build for this push: https://github.com/OJWatson/motac/actions/runs/21916380831
- Keep commits small and CI-gated; update this file on each push/gate.
