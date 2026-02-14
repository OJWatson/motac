# motac

Road-constrained spatio-temporal Hawkes processes for event forecasting on road networks.

- Documentation: https://ojwatson.github.io/motac
- Paper artefact: [`paper/`](paper/)
- Spec gap review (PDF vs repo): [`spec_review_vs_pdf.md`](spec_review_vs_pdf.md)

## Install

Requirements: Python 3.11+

Using [`uv`](https://github.com/astral-sh/uv) (recommended):

```bash
git clone https://github.com/OJWatson/motac
cd motac
uv sync
```

Run the CLI:

```bash
uv run motac --help
```

## Quickstart (simulation → fit → forecast)

Canonical data schema (used by dataset loaders) lives in `motac.schema`:
- `EventRecord`
- `EventTable`

Fit an exponential-kernel Hawkes model to a saved simulator parquet:

```bash
uv run motac sim fit-kernel --parquet sim.parquet --n-lags 6
```

Forecast from an observed count matrix (CSV with rows=locations, cols=time):

```bash
uv run motac sim forecast-observed \
  --y-obs y_obs.csv \
  --horizon 20 \
  --n-paths 200 \
  --p-detect 0.7 \
  --false-rate 0.2
```

Python API (deterministic multi-step intensity forecast):

```python
import numpy as np
from motac.sim import (
    HawkesDiscreteParams,
    discrete_exponential_kernel,
    generate_random_world,
    predict_hawkes_intensity_multi_step,
)

world = generate_random_world(n_locations=5, seed=0, lengthscale=0.5)
params = HawkesDiscreteParams(
    mu=np.full((world.n_locations,), 0.2),
    alpha=0.7,
    kernel=discrete_exponential_kernel(n_lags=4, beta=0.8),
)

y_history = np.zeros((world.n_locations, 10), dtype=int)
lam = predict_hawkes_intensity_multi_step(world=world, params=params, y_history=y_history, horizon=7)
assert lam.shape == (world.n_locations, 7)
```

## Docs

- Full documentation: https://ojwatson.github.io/motac
- Substrate cache bundles (build/load/versioning):
  https://ojwatson.github.io/motac/api/substrate.html#loading-a-cached-bundle-with-version-validation

Build the docs locally:

```bash
uv sync --group docs
cd docs
make html
# open docs/_build/html/index.html
```

## Reproducibility

See [`paper/README.md`](paper/README.md) for the current artefact / reproduction entrypoints.

## Development

Run tests and lint:

```bash
uv run pytest
uv run ruff check .
```

## License

TBD.

## Citation

TBD (will be added once a paper/preprint is available).
