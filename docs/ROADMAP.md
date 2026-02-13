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

## M4 — PDF spec re-alignment + package layout (structure-first)
- Goal: bring the repository into close alignment with the PDF’s *explicit* architecture/goals (including later milestones), so subsequent work is not drifting.
- DoD:
  - A short, concrete spec-alignment doc exists (e.g. `docs/spec_alignment.md`) containing:
    - a checklist of the PDF-required components,
    - what already exists in the repo,
    - what is missing / out of place,
    - and how the remaining milestones map to code + docs.
  - Package layout is aligned to the PDF’s conceptual modules (names may differ, but the separation must be clear and discoverable):
    - dataset loaders + schema validation
    - substrate/spatial (grid/graph, travel-time neighbours, POIs)
    - models (parametric Hawkes; marked Hawkes later; neural kernel modules later)
    - inference (JAX likelihoods, jit kernels, optimisers/constraints)
    - sim (simulator + observation models)
    - eval (backtesting, metrics, calibration, plotting)
    - cli (fetch, build-substrate, fit, forecast, backtest, simulate)
    - configs (experiment configs)
  - Roadmap updated (this file) to reflect the PDF sequencing accurately, including an explicit note for any deliberate deviations.

## M5 — Predict/eval plumbing v1
- DoD:
  - Minimal prediction/evaluation utilities wired to the parametric model.
  - Tests for shapes/consistency on toy data.

## M6 — Chicago loader v1
- DoD:
  - Deterministic loader and basic sanity tests.

## M7 — ACLED loader v1
- DoD:
  - Deterministic loader and basic sanity tests.

## M8 — Observed-data likelihood/predictive sampling v1
- DoD:
  - Core observed-data workflows supported with tests.

## M9 — Exact observed log-likelihood v1
- DoD:
  - Exact observed log-likelihood implemented and tested.

## M10 — Exact vs approx observed log-likelihood comparison
- DoD:
  - Comparison harness + tests on toy data.

## M11 — Paper artefacts pipeline v1
- DoD:
  - Reproducible artefact generation with CI-safe tests.

## M12 — Substrate cache artefacts v1 (implementation)
- Goal: implement the M1 DoD in code (the roadmap already defines the desired outcome; this milestone is the concrete build-out).
- DoD:
  - Substrate cache bundle writes/loads deterministically.
  - Offline unit test validates expected cache files + provenance/config hash.

## M13 — Marked Hawkes scaffolding v1
- Goal: introduce a minimal, stable *API surface* for marked Hawkes variants (event types/severities), without committing to a final modelling choice.
- DoD:
  - A small `motac.model.marked_hawkes` module exists with:
    - a mark representation (categorical/int) suitable for unit tests,
    - and a thin dataset wrapper or validation helpers.
  - CI-safe unit test asserts import-path stability for the new module/API.

## M14 — Neural kernel scaffolding v1
- Goal: establish a stable import path + minimal interface for learned/neural kernels (used by later nonparametric variants).
- DoD:
  - A small `motac.model.neural_kernels` (or similarly named) module exists with:
    - a tiny, documented kernel interface (call signature + expected shapes),
    - a toy implementation sufficient for unit tests.
  - CI-safe import-path stability unit test for the new module/API.

## M15 — Neural kernel contract hardening v1
- Goal: harden the neural-kernel scaffold into a small utility surface that other
  modules can safely depend on (without introducing a full neural model).
- DoD:
  - A tiny validation helper exists (e.g. `validate_kernel_fn`) that asserts the
    v1 shape/value contract and fails fast on contract violations.
  - CI-safe unit tests cover at least:
    - happy-path validation on a reference kernel,
    - rejection of negative outputs,
    - rejection of incorrect output shapes.
