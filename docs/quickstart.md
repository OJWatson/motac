# Quickstart

This quickstart demonstrates the intended end-to-end workflow.

1. Build a grid and connectivity (`make_grid_stencil` or `make_grid_edgelist`).
2. Simulate `CountsTensor` data with `simulate_counts`.
3. Fit `MobilityHawkesModel` with `method="map_ensemble"` or `"svi"`.
4. Produce 7-day posterior predictive forecasts with `forecast`.
5. Run rolling-origin evaluation with `rolling_backtest`.

## Minimal example

```python
import jax.numpy as jnp
from motac import (
    FitConfig,
    GridSpec,
    HawkesModelSpec,
    HawkesSimParams,
    MobilityHawkesModel,
    make_grid_stencil,
    simulate_counts,
)

grid = GridSpec(shape=(30, 30))
stencil = make_grid_stencil(kind="gaussian", size=7, sigma=1.5)
marks = ("battle", "explosion", "civilians")

params = HawkesSimParams(
    mu=0.02,
    alpha_from=jnp.array([0.3, 0.25, 0.2], dtype=jnp.float32),
    P_from_to=jnp.array(
        [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.3, 0.2, 0.5]],
        dtype=jnp.float32,
    ),
    rho=jnp.array([[0.6, 0.86], [0.55, 0.82], [0.5, 0.78]], dtype=jnp.float32),
    w_time=jnp.array([[0.7, 0.3], [0.65, 0.35], [0.6, 0.4]], dtype=jnp.float32),
    obs="nb2",
    concentration=20.0,
)

data = simulate_counts(
    T=120,
    grid=grid,
    num_nodes=900,
    marks=marks,
    connectivity=stencil,
    params=params,
    seed=42,
)

spec = HawkesModelSpec(
    marks=marks,
    temporal_bases=2,
    observation="nb2",
    normalise_time_kernel=False,
    normalise_spatial_kernel=True,
)
model = MobilityHawkesModel(spec)
fit = model.fit(
    data,
    method="map_ensemble",
    config=FitConfig(map_steps=40, map_ensemble_size=2, map_patience=8),
    seed=0,
)
fc = model.forecast(data, fit, horizon=7, num_samples=200, seed=1)

print("mean excitation spectral radius:", fit.diagnostics["excitation_spectral_radius_mean"])
```
