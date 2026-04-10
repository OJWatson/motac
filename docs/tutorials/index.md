# Tutorials

These notebook vignettes are designed to be executed locally (typically on your CUDA-enabled setup), saved with outputs, and then rendered by Sphinx.

Each tutorial has a single canonical notebook (no `_full` variant).

```{toctree}
:maxdepth: 1

01_simulator_to_forecast_deep_dive
02_backend_benchmark_and_device_dive
03_calibration_and_spatial_error_analysis
04_chicago_dataset
05_acled_dataset
```

## Running tutorial notebooks locally

1. Install docs + runtime dependencies in the repo environment.
2. Execute each tutorial notebook end-to-end locally (JIT enabled by default; GPU used automatically when available).
3. Save outputs in-place so rendered docs include both code and results.
4. Build docs with Sphinx.

```bash
python -m pip install -e .[docs]
python -m jupyter nbconvert --to notebook --execute --inplace docs/tutorials/01_simulator_to_forecast_deep_dive.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace docs/tutorials/02_backend_benchmark_and_device_dive.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace docs/tutorials/03_calibration_and_spatial_error_analysis.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace docs/tutorials/04_chicago_dataset.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace docs/tutorials/05_acled_dataset.ipynb
python -m sphinx -b html docs docs/_build/html
```
