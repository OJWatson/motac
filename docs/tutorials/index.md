# Tutorials

These notebook vignettes are designed to be executed locally (typically on your CUDA-enabled setup), saved with outputs, and then rendered by Sphinx.

Each published tutorial has a single canonical notebook.

```{toctree}
:maxdepth: 1

01_simulator_to_forecast_deep_dive
02_backend_benchmark_and_device_dive
03_calibration_and_spatial_error_analysis
04_chicago_dataset
05_acled_dataset
07_acled_best_fit
```

## Experimental notebooks kept in the repository

Some ACLED notebooks are intentionally kept out of the public docs navigation because they are narrower, more overlapping, or more diagnostics-heavy than the main tutorial path.

- `06_acled_hawkes_deep_dive_and_improvements.ipynb`: deeper modelling diagnosis and iteration notes
- `08_acled_nuts_truncated.ipynb`: truncated NUTS-focused diagnostics notebook

These can still be useful for development and research, but `05` and `07` are the main public ACLED notebooks.

## Running tutorial notebooks locally

1. Install docs + runtime dependencies in the repo environment.
2. Execute each tutorial notebook end-to-end locally (JIT enabled by default; GPU used automatically when available).
3. Save outputs in-place so rendered docs include both code and results.
4. Build docs with Sphinx.

The commands below assume your local environment is already activated.

```bash
uv pip install -e .[docs]
python -m jupyter nbconvert --to notebook --execute --inplace docs/tutorials/01_simulator_to_forecast_deep_dive.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace docs/tutorials/02_backend_benchmark_and_device_dive.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace docs/tutorials/03_calibration_and_spatial_error_analysis.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace docs/tutorials/04_chicago_dataset.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace docs/tutorials/05_acled_dataset.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace docs/tutorials/07_acled_best_fit.ipynb
python -m sphinx -b html docs docs/_build/html
```
