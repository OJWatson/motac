# Changelog

## 0.1.0

Initial full implementation of `motac` in Python 3.11.

### Added

- Core package scaffold with `pyproject.toml` and editable install support.
- Data model layer:
  - `GridSpec`, `EventTable`, `CountsTensor`, `TrainTestSplit`.
  - rolling-origin split utilities.
- Spatial connectivity:
  - `ConvStencil`, `EdgeList`.
  - `make_grid_stencil`, `make_grid_edgelist`.
  - spatial apply operators using conv and `segment_sum`.
- Simulator:
  - `HawkesSimParams`, `HawkesSimNoise`.
  - scan-based simulation for Poisson and NB2 observations.
- Probabilistic model:
  - `HawkesModelSpec`, `MobilityHawkesModel`.
  - constrained parameter transforms and scan-based log-likelihood.
- Inference:
  - MAP ensemble with optax.
  - SVI and small NUTS pathways.
- Forecasting:
  - posterior predictive rollout consistent with latent recurrence.
- Evaluation:
  - rolling backtest, CRPS-like metric, coverage, hotspot recall.
- Dataset adapters:
  - Chicago and ACLED fetch helpers with cache-first behavior.
  - event-table to counts conversion helpers.
- Plotting helpers for intensity and calibration views.
- Benchmark helpers for conv vs edge backend comparisons.
- Test suite (unit, integration smoke, performance smoke skip-by-default).
- Docs stack (Sphinx + MyST + nbsphinx), quickstart and modeling notes.

### Decisions reflected in v1

- BCOO backend deferred behind feature-flag future path.
- ZINB2 deferred; v1 supports Poisson and NB2.
- `optax` is a hard dependency.
- Packaging uses `pyproject.toml` only.
- Docs are local-build artifacts, not notebook CI gates.
