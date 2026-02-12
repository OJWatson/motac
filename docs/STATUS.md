```pm-status
milestone: M12
state: running
headSha: 4db512497db12e9b7e29f875d7f806096362a380
ciRunUrl: https://github.com/OJWatson/motac/actions?query=branch%3Amain
updatedAtUtc: 2026-02-12T16:20:58Z
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

- M11 is **complete**:
  - CI-safe paper artefact generation stub exists (`motac paper generate-artifacts`).
  - Synthetic eval JSON artefact now includes a minimal manifest (SHA/seed/config summary/timestamp).
  - Unit tests cover artefact write + manifest contract.

## Next step (M12)
- Implement deterministic substrate cache bundle write/load (see `docs/ROADMAP.md` → M12).
  - Add an offline unit test that validates expected cache files + provenance/config hash.

## Notes
- Documentation builds are tracked in CI; the status header points at the last meaningful code gate.
- Keep commits small and CI-gated; update this file on each push/gate.
