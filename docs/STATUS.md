```pm-status
milestone: M5
state: running
headSha: c647b5d41c50df3ddf8d194d1eb391d1959d3e52
ciRunUrl: https://github.com/OJWatson/motac/actions/runs/21937018719
updatedAtUtc: 2026-02-12T07:29:00Z
```

## Status
- M4 is **complete**:
  - Spec checklist + mapping: `docs/spec_alignment.md`
  - Architecture summary: `docs/architecture.md`
  - Package boundaries extracted (structure-first):
    - `motac.loaders`, `motac.substrate`, `motac.model`, `motac.sim`, `motac.eval`, `motac.inference`, `motac.cli`, `motac.configs`
  - CLI internal boundary extracted: `motac.cli.commands` (while keeping the entry point stable)
  - Import-path smoke tests added for new packages.

- M5 is **in progress**.

## Next step (M5)
- Confirm M5 DoD is satisfied (minimal predict/eval utilities + toy tests).
  - Added a tiny backtest helper wiring fit→forecast→NLL (`motac.eval.backtest_fit_forecast_nll`) + unit test.

## Notes
- Documentation builds are tracked in CI; the status header points at the last meaningful code gate.
- Keep commits small and CI-gated; update this file on each push/gate.
