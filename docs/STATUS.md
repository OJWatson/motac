```pm-status
milestone: M8
state: running
headSha: 837291ed72e1991e340c27fd7e55f71f1ed62f67
ciRunUrl: https://github.com/OJWatson/motac/actions?query=branch%3Amain
updatedAtUtc: 2026-02-12T12:28:40Z
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

- M6 is **complete**:
  - Chicago raw loader exists (CSV + optional mobility `.npy`) and is deterministic (order preserved).
  - Minimal config + CLI entry point exists (`motac data chicago-load`).
  - Fixture-driven sanity tests cover happy path + basic validation (shape mismatch rejection).

## Next step (M8)
- Add a minimal, CI-safe test that exercises an observed-data workflow end-to-end (fit → forecast/predictive sample → score) on toy data.

## Notes
- Documentation builds are tracked in CI; the status header points at the last meaningful code gate.
- Keep commits small and CI-gated; update this file on each push/gate.
