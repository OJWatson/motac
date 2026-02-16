# NEXT_TASKS

Now:
- M4.0 — Sim: discrete-time Hawkes generator wired to substrate neighbours + deterministic fixture

Next:
- M4.1 — Sim: observation noise model (thinning/jitter/missingness) + golden regression tests
- M4.2 — Sim: scenario knobs incl. road-closure travel-time perturbations + docs
- M4.END — Milestone end: M4 complete (gate on CI)
- CI.FIX.M4 — CI fix: make GitHub Actions green for M4 boundary

Done:
- CI.FIX.M3 — CI fix: make GitHub Actions green for M3 boundary
- M3.END — Milestone end: M3 complete (gate on CI)
- M3.3 — Validation: parameter recovery harness (multi-seed) + tolerances + docs
- M3.2 — Model: parametric road-kernel Hawkes fit/predict API + end-to-end sim smoke
- M3.1 — Inference: Poisson/NegBin likelihoods + gradient checks on tiny fixtures
- M3.0 — Inference: JAX sparse neighbour convolution ops (jit-ready) + tests
- CI.FIX.M2 — CI fix: make GitHub Actions green for M2 boundary
- M2.END — Milestone end: M2 complete (gate on CI)
- M2.4 — Substrate: include POI baseline features in cache bundle v1 + deterministic tests
- M2.3 — Spatial: implement OSM POI extraction + per-cell baseline feature matrix
- M2.2 — Docs: add end-to-end example (ingest -> spatial join -> query)
- M2.1 — Ingestion tests: fixture roundtrip + schema validation
- M2.0 — Ingestion: add minimal pipeline to load raw events into canonical table
- M1.0 — Substrate cache bundle v1: document format + add one end-to-end smoke test
- M1.1 — Make cache bundle v1 fully deterministic + add provenance-hash regression test
- M1.2 — Docs: add minimal “load bundle + validate version” example snippet (and link from README)
- M1.3 — Spatial: add lon/lat → cell_id lookup helper
- M1.4 — Spatial: validate cell_id lookup boundaries + add property-based tests
- M1.5 — CLI: add `motac spatial cell-id` command + smoke test
