# Spec alignment (PDF)

This document is a structure-first alignment checklist against the PDF spec.
It is intentionally concise and actionable.

## Scope notes (M0/M1)

These notes pin down what “done” means for the early reset milestones.
They exist to prevent M0/M1 from expanding into later-milestone feature work.

### M0 (reset + schema): definition of done

M0 is **in-scope** when all of the following are true:

- **Canonical event schema exists** (single source of truth in code).
- **Schema validation is tested** (unit tests that fail on invalid events).
- **Repo layout is stable** (pyproject + src/ layout; `uv run pytest -q` passes in CI).
- **Docs point to the canonical schema** (so later chapters don’t invent new formats).

Practical verification:

- Canonical schema location is documented and discoverable.
- Tests cover at least: required fields, types, and one representative invalid payload.

### M1 (substrate cache artefacts v1): definition of done

M1 is **in-scope** when all of the following are true:

- **Cache bundle format is defined and versioned** (a reader can reject unknown versions).
- **Writes are deterministic** (same inputs → identical bundle bytes / file tree).
- **Provenance hash is stable** (and guarded by a regression test).
- **Docs describe the artefact contract**:
  - what the bundle contains (high level)
  - how to load it (one minimal code snippet or API pointer)

### Explicit out-of-scope for M0/M1

The following are **not** M0/M1 work (track them under later milestones instead):

- End-to-end forecasting / backtesting workflows.
- Full observed-data ingestion pipelines (ACLED/Chicago loaders, ETL, cleaning).
- “Production” CLI polish (flags, UX, error handling) beyond minimal entrypoints.

## Checklist of required components

### Data + schema
- [x] Canonical event schema + validation
- [ ] Dataset loaders (Chicago, ACLED)
- [ ] Time binning and observation matrix construction

### Substrate / spatial
- [x] Road graph build/load (offline GraphML supported)
- [x] Grid definition + cell centroids
- [x] Travel-time neighbourhoods (sparse)
- [x] POIs + baseline features (count + travel-time-to-nearest)
- [x] Cache bundle + provenance

### Models
- [x] Parametric road-constrained Hawkes (Poisson)
- [x] Parametric road-constrained Hawkes (NegBin)
- [ ] Marked Hawkes variants (later)
- [ ] Neural / learned kernels (later)

### Inference
- [ ] Likelihoods and optimisers (JAX/jit kernels as required by PDF)
- [ ] Constraints/parameterisations for stability

### Simulation
- [ ] Simulator + observation models

### Evaluation
- [ ] Forecast/backtest workflows
- [ ] Metrics + calibration
- [ ] Plotting/reporting hooks

### CLI + configs
- [ ] CLI entrypoints (fetch/build-substrate/fit/forecast/backtest/simulate)
- [ ] Experiment configuration files

## What exists vs missing (repo reality)

### Exists (implemented + tested)
- Canonical schema + validation tests: `src/motac/schema.py`, `tests/test_schema_m0.py`
- Substrate cache bundle + provenance hash test: `tests/test_substrate_cache_bundle_m1.py`
- POI features:
  - count + tag/value breakouts: `src/motac/substrate/builder.py`
  - min travel-time features behind flag: `src/motac/substrate/features.py`
  - documented: `docs/api/substrate.md`
- Road-constrained parametric model pieces (M3):
  - sparse exp travel-time kernel + intensity: `src/motac/model/road_hawkes.py`
  - Poisson + NegBin likelihood: `src/motac/model/likelihood.py`
  - MLE fitter: `src/motac/model/fit.py`
  - minimal predict wrappers: `src/motac/model/predict.py`
  - recovery tests: `tests/test_parameter_recovery_m3.py`, `tests/test_negbin_parameter_recovery_m3.py`

### Missing / out of place (to be built)
- Clear package/module separation per PDF across:
  - loaders/schema vs substrate vs models vs inference vs sim vs eval vs cli vs configs
- Inference components specifically requiring JAX/jit (if mandated by PDF)
- Evaluation/backtesting workflows beyond toy metric utilities
- CLI wiring for end-to-end workflows
- Dataset loaders and full observed-data workflows as per later milestones

## Milestones → modules / code map

- M0 (reset + schema): `src/motac/schema.py`, docs + tests
- M1 (cache artefacts): `src/motac/substrate/*`, cache tests
- M2 (POIs + baseline features): `src/motac/substrate/{builder,features}.py`, `docs/api/substrate.md`
- M3 (parametric road Hawkes v1): `src/motac/model/{dataset,road_hawkes,likelihood,fit,predict}.py`, M3 tests
- M4 (this milestone):
  - spec alignment doc: `docs/spec_alignment.md`
  - package layout adjustments (structure-first): create/confirm module boundaries
- M5 (predict/eval plumbing v1): `src/motac/model/{forecast,metrics}.py` + eval utilities/tests

## Deviations / out-of-scope notes

- A small slice of M5 (forecast + mean NLL) landed early for CI-safe plumbing; it is tracked as M5 work even if merged earlier.
- This doc does not restate the full PDF; it is a checklist + mapping for implementation planning.
