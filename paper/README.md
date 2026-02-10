# motac — Paper / Reproducibility Stub (M8)

This directory is a lightweight placeholder for the paper and reproducibility
materials.

## Repro: synthetic evaluation (M4)

The repo contains a config-driven synthetic evaluation pipeline:

- `motac.eval.evaluate_synthetic(EvalConfig(...))` simulates data, fits the
  parametric model, produces 1–7 day forecasts, and computes basic metrics.

To generate a small example artifact:

```bash
python -m motac.paper.generate_artifacts --out-dir paper/artifacts
```

This writes a JSON file containing:
- `config`
- `fit`
- `forecasts`
- `metrics`

## Repro: CLI workflows

Observed-only forecasting workflow:

```bash
motac sim forecast-observed --y-obs y_obs.csv --horizon 7 --n-paths 200 --q 0.05,0.5,0.95
```

Notes:
- CSV is rows=locations, cols=time.
- Output JSON includes `meta`, `fit`, and `predict`.

## Figures (stub)

Planned figures for the paper (placeholders):
- Parameter recovery (alpha/beta) on synthetic simulation
- Forecast accuracy (RMSE/MAE) vs horizon
- Calibration / coverage of predictive intervals

Plotting is intentionally not part of the CI path.
