# Roadmap

This document tracks the planned milestone sequence and Definition-of-Done (DoD)
for `motac`.

## M0 — Project reset + spec alignment
- DoD:
  - Repository documentation is user-facing and buildable.
  - Canonical event schema implemented with validation + unit tests.

## M1 — Substrate cache artefacts v1
- DoD:
  - Substrate cache bundle writes/loads deterministically.
  - Offline unit test validates expected cache files + provenance/config hash.

## M2 — POIs + baseline features v1
- DoD:
  - POI count features (incl. optional tag/value breakouts).
  - Travel-time-to-nearest style POI features available (min travel time), with tests.
  - User docs describe feature names and configuration.

## M3 — Parametric road-constrained Hawkes v1 (Poisson + NegBin)
- DoD:
  - Sparse road-constrained intensity based on travel-time neighbourhoods and W(d_travel).
  - Poisson and Negative Binomial likelihood support under a stable API.
  - Offline parameter-recovery test on a tiny substrate to sanity-check end-to-end fitting.

## M4 — Predict/eval plumbing v1
- DoD:
  - Minimal prediction/evaluation utilities wired to the parametric model.
  - Tests for shapes/consistency on toy data.

## M5 — Chicago loader v1
- DoD:
  - Deterministic loader and basic sanity tests.

## M6 — ACLED loader v1
- DoD:
  - Deterministic loader and basic sanity tests.

## M7 — Observed-data likelihood/predictive sampling v1
- DoD:
  - Core observed-data workflows supported with tests.

## M8 — Exact observed log-likelihood v1
- DoD:
  - Exact observed log-likelihood implemented and tested.

## M9 — Exact vs approx observed log-likelihood comparison
- DoD:
  - Comparison harness + tests on toy data.

## M10 — Paper artefacts pipeline v1
- DoD:
  - Reproducible artefact generation with CI-safe tests.
