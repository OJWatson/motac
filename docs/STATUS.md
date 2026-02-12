```pm-status
milestone: M8
state: running
headSha: 5456d0b4dec55cbf5117dd9a7feff5d56ac739b8
ciRunUrl: https://github.com/OJWatson/motac/actions?query=branch%3Amain
updatedAtUtc: 2026-02-12T12:48:55Z
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
- Review M8 DoD coverage; if the observed-data workflow tests are sufficient, advance STATUS to M9 (exact observed log-likelihood v1).

## Notes
- Documentation builds are tracked in CI; the status header points at the last meaningful code gate.
- Keep commits small and CI-gated; update this file on each push/gate.
