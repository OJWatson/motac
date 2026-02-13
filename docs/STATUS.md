```pm-status
milestone: M16
state: running
headSha: a4d62763622e8fd5757ca6d9555f0956cc19b80f
ciRunUrl: PENDING
updatedAtUtc: 2026-02-13T05:57:34Z
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

## M12 — Substrate cache artefacts v1 (implementation)
- **Complete**:
  - Deterministic substrate cache bundle writes/loads via `motac.substrate.builder.SubstrateBuilder`.
  - Offline unit test validates expected cache files + provenance/config hash.

## M2 — POIs + baseline features v1
- **Complete**:
  - POI count features (incl. optional tag/value breakouts) implemented and tested.
  - Travel-time-to-nearest-POI (min travel time) features implemented and tested.
  - User tutorial added describing POI feature names + configuration knobs (`docs/tutorials/02_poi_features.md`).

## M3 — Parametric road-constrained Hawkes v1 (Poisson + NegBin)
- **Complete**:
  - Sparse road-constrained intensity based on travel-time neighbourhoods and W(d_travel).
  - Poisson + NegBin likelihood support under a stable API.
  - Offline parameter-recovery tests on a tiny substrate for Poisson and NegBin.

## M14 — Neural kernel scaffolding v1
- **Complete**:
  - Stable import path established: `motac.model.neural_kernels`.
  - Tiny deterministic toy kernel (`ExpDecayKernel`) + CI-safe unit tests.
  - Sphinx API docs page exists and is wired into the docs index.

## M15 — Neural kernel contract hardening v1
- **Complete**:
  - `validate_kernel_fn` asserts the v1 kernel contract (shape/value checks) and fails fast.
  - CI-safe unit tests cover happy path and key rejection cases (negative outputs, wrong shapes, non-array returns, non-finite weights).

## Next step
- Continue M16: extend a minimal fitting/estimation path to accept an optional validated `kernel_fn` (now supported by `road_intensity_matrix`/`road_loglik` and the one-step forecast helper).

## Notes
- Documentation builds are tracked in CI; the status header points at the last meaningful code gate.
- Keep commits small and CI-gated; update this file on each push/gate.
