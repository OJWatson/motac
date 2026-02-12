```pm-status
milestone: M11
state: running
headSha: e57e4e143da6f5f6f19c747bded6411f067aa02a
ciRunUrl: https://github.com/OJWatson/motac/actions?query=branch%3Amain
updatedAtUtc: 2026-02-12T15:37:47Z
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

- M7 is **complete**:
  - ACLED loader exists (CSV) and is deterministic (order preserved).
  - Minimal config + CLI entry point exists (`motac data acled-load`).
  - Fixture-driven sanity tests cover happy path + basic validation.

- M8 is **complete**:
  - CI-safe observed-data end-to-end unit test covers fit → predictive sample → score on toy data.

- M9 is **complete**:
  - Exact observed log-likelihood for thinning+clutter implemented (`motac.sim.hawkes_loglik_observed_exact`).
  - Tests cover hand-computable toy case and exact-vs-approx parity assertions.

- M10 is **complete**:
  - Comparison harness + tests exist for exact-vs-approx observed log-likelihood.
  - Docs note the interpretation gap: exact is conditional on `y_true`; Poisson-approx depends on Hawkes intensity/history proxying.

## Next step (M11)
- Wait for CI green on `e57e4e1` (docs-only). If green, mark M11 as complete and decide the next milestone to add to `docs/ROADMAP.md`.

## Notes
- Documentation builds are tracked in CI; the status header points at the last meaningful code gate.
- Keep commits small and CI-gated; update this file on each push/gate.
