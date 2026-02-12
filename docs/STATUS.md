```pm-status
milestone: M6
state: running
headSha: c17c199319b6ed1bf971de234147323354400070
ciRunUrl: https://github.com/OJWatson/motac/actions?query=branch%3Amain
updatedAtUtc: 2026-02-12T08:32:37Z
```

## Status
- M4 is **complete**:
  - Spec checklist + mapping: `docs/spec_alignment.md`
  - Architecture summary: `docs/architecture.md`
  - Package boundaries extracted (structure-first):
    - `motac.loaders`, `motac.substrate`, `motac.model`, `motac.sim`, `motac.eval`, `motac.inference`, `motac.cli`, `motac.configs`
  - CLI internal boundary extracted: `motac.cli.commands` (while keeping the entry point stable)
  - Import-path smoke tests added for new packages.

- M5 is **complete**:
  - Minimal fit→forecast→score backtest helper exists (`motac.eval.backtest.backtest_fit_forecast_nll`).
  - Toy unit test covers basic shape/consistency expectations.

## Next step (M6)
- Add a small committed mobility fixture (e.g. `.npy`) and a sanity test that `load_y_obs_matrix(..., mobility_path=...)` records `meta["mobility_source"]` and rejects mismatched shapes.

## Notes
- Documentation builds are tracked in CI; the status header points at the last meaningful code gate.
- Keep commits small and CI-gated; update this file on each push/gate.
