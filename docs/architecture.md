# Architecture

This document describes the intended package/module separation.

## Target top-level modules

- `motac.loaders`: dataset loaders + schema validation glue
- `motac.substrate`: spatial substrate (grid/graph, travel-time neighbours, POIs, caching)
- `motac.model`: parametric models (road-constrained Hawkes) and related utilities
- `motac.inference`: likelihoods/optimisers/constraints (incl. accelerated kernels where required)
- `motac.sim`: simulation and observation models
- `motac.eval`: backtesting, metrics, calibration utilities
- `motac.cli`: CLI entrypoints
- `motac.configs`: experiment configuration structures

## Current package layout progress

- Loaders moved under `motac.loaders` with backwards-compatible re-exports from
  `motac.acled` and `motac.chicago` to avoid breaking imports.
- Evaluation module moved to a package: `motac.eval` is now a subpackage.
- Inference boundary created: `motac.inference` (structure-first stub).
