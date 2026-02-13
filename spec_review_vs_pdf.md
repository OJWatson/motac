# motac — Spec Review vs PDF plan (Road-Constrained Spatio‑Temporal Hawkes, v0.1, 2026‑02‑07)

This is a **gap analysis** of the current `OJWatson/motac` repository against the original PDF plan (which calls the package `road_hawkes`; we treat that as naming only).

## Executive summary

The repo currently provides a **discrete-time, network-coupled Hawkes-like simulator + basic MLE fitting + predictive sampling** and a minimal synthetic evaluation helper.
However, relative to the PDF’s stated MVP (“road‑constrained substrate + parametric Hawkes + forecasting + rolling backtests + benchmarks on synthetic, Chicago, ACLED”), the implementation is **materially incomplete**:

- **Road constraint** is only partially represented (there is an OSMnx-based `substrate` builder, but it is not integrated end-to-end into simulation/inference/evaluation in a config-driven benchmark pipeline).
- **Datasets** (Chicago/ACLED) are **placeholder** ingestion paths, not faithful benchmark replications.
- **Evaluation/backtesting** is not implemented to spec (no rolling-origin protocol, scoring rules beyond a minimal NLL/RMSE/MAE, no calibration diagnostics, no baselines/ablations).
- **Paper artefact** is explicitly a stub; no NeurIPS-grade experiments/figures exist.
- **Documentation** is far below the `trace` standard (no docs site, no rendered notebooks).

## What the PDF requires (MVP)

From the PDF (sections 2, 3, 4, 5, 6, 7):

MVP deliverable must:
1) build a **road-constrained substrate** (OSM roads + travel-time neighbours + POIs/covariates, cached)
2) fit a **parametric discrete-time Hawkes** model with road-travel-time kernel
3) forecast **1/3/7 days** ahead
4) run **rolling backtests** and output proper scoring rules + calibration diagnostics
5) reproduce benchmark results on:
   - synthetic simulation
   - Chicago crime (GraphST-style)
   - ACLED Gaza

## Current repository: status vs spec

### (A) Canonical event schema (PDF §3.1)
**Spec:** dataset-agnostic `EventRecord` schema with mapping for all datasets.

**Current:** `motac.schema` exists (needs verification), but Chicago/ACLED loaders do not map real data into a unified schema used by the model; they output dense matrices under a placeholder schema.

**Status:** **PARTIAL**.

### (B) Roads + travel-time constraints (PDF §3.2)
**Spec:** OSM road graph, travel-time shortest paths, sparse neighbour sets, persisted artefacts, JAX-friendly arrays.

**Current:** `motac.substrate.builder.SubstrateBuilder` uses OSMnx and NetworkX, builds a grid and computes neighbour sets via Dijkstra cutoff.

Gaps:
- No documented **artefact format/versioning/provenance** for cached substrate.
- No explicit “top‑k vs threshold” neighbour policy option (only cutoff in seconds).
- No pipeline connecting substrate outputs into core model fitting/prediction.

**Status:** **PARTIAL** (good start, not integrated).

### (C) POIs/covariates (PDF §3.3)
**Spec:** extract POIs from OSM tags, compute travel-time-to-nearest POI, baseline features.

**Current:** substrate builder has a POI hook; unclear completeness and no baseline model consumes features.

**Status:** **MOSTLY MISSING**.

### (D) Parametric discrete-time road-constrained Hawkes (PDF §4.1)
**Spec:** model on counts `y[j,t]`, road kernel `W[j,k]` based on travel-time distance, Poisson/NB likelihood, constrained parameters, Optax optimisation, stable APIs `fit/predict/loglik/simulate`.

**Current:** `motac.sim.*` provides a discrete-time Hawkes-like model with a generic mobility matrix; fitting via MLE exists.

Gaps:
- Road kernel `W(d_travel)` not yet wired into model core (it appears the model uses a precomputed mobility matrix, not the substrate travel-time distances/neighbor sets).
- No NB likelihood (Poisson only).
- No explicit stable model API module boundary (current functions are serviceable but not at spec-level clarity).

**Status:** **PARTIAL**.

### (E) Simulator requirements (PDF §5)
**Spec:** simulator shares schema and substrate; observation model with thinning/jitter/rounding/missingness; scenario knobs; parameter recovery + SBC.

**Current:** simulator exists with detection+clutter option; does not share substrate schema in a way that matches real pipeline; no SBC.

**Status:** **PARTIAL**.

### (F) Chicago crime benchmark (PDF §6.1, §9 M5)
**Spec:** Socrata fetcher + cleaning + mapping to schema; replication notebook; baseline comparisons.

**Current:** `motac.chicago.load_y_obs_matrix()` is a placeholder loader for a dense matrix and optional mobility.

**Status:** **MISSING**.

### (G) ACLED Gaza benchmark (PDF §6.2, §9 M6)
**Spec:** ACLED API + CSV ingestion; Gaza bounds; mark taxonomy; substrate builder.

**Current:** `motac.acled.load_acled_events_csv()` parses a simplified CSV and aggregates by unique (lat,lon) pairs; no Gaza bounds config; no mark taxonomy; no ACLED API support.

**Status:** **MISSING / PLACEHOLDER ONLY**.

### (H) Evaluation + backtesting + reporting (PDF §7)
**Spec:** rolling-origin, horizons 1/3/7, metrics (log score/NLL, CRPS), calibration (coverage, PIT), spatial utility metrics, baselines + ablations.

**Current:** `motac.eval.evaluate_synthetic()` runs a tiny synthetic holdout with NLL/RMSE/MAE/coverage. No rolling backtest, no baselines, no calibration plots.

**Status:** **MISSING (beyond minimal toy)**.

### (I) Documentation & paper artefact (PDF §8–9, §9 M8)
**Spec:** docs + notebooks + reproducible paper figures, “one command reproduce”.

**Current:** no docs site; `paper/` is a stub; no rendered tutorial notebooks.

**Status:** **MISSING**.

## Risks / drift drivers
- Early work focused on a minimal parametric discrete-time kernel without the full substrate/dataset/eval stack.
- Placeholder dataset loaders have created the *appearance* of coverage but do not satisfy the benchmark requirements.
