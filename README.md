# motac

Road-constrained spatio-temporal Hawkes processes for event forecasting on road networks.

This repository is being realigned to the original project PDF spec ("Road-Constrained Spatio-Temporal Hawkes").
The near-term goal is a clean, reproducible MVP: **build a road substrate → fit a parametric Hawkes count model → forecast → evaluate via rolling backtests**.

- Documentation (GitHub Pages): https://ojwatson.github.io/motac
- Spec gap review: [`spec_review_vs_pdf.md`](spec_review_vs_pdf.md)
- Milestones / acceptance criteria: [`milestones_v2.md`](milestones_v2.md)

## Install

Requirements: Python 3.11+

Using `uv` (recommended):

```bash
git clone https://github.com/OJWatson/motac
cd motac
uv sync
```

Run the CLI:

```bash
uv run motac --help
```

## Minimal quickstart (simulation → fit → forecast)

The current codebase includes a lightweight discrete-time Hawkes-like simulator and Poisson MLE fitting utilities.

### CLI

Fit an exponential kernel Hawkes model to a saved simulation parquet:

```bash
uv run motac sim fit-kernel --parquet sim.parquet --n-lags 6
```

Forecast from an observed count matrix (CSV rows=locations, cols=time):

```bash
uv run motac sim forecast-observed \
  --y-obs y_obs.csv \
  --horizon 20 \
  --n-paths 200 \
  --p-detect 0.7 \
  --false-rate 0.2
```

### Python

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

## Documentation

Build the docs locally:

```bash
uv sync --group docs
cd docs
make html
# open docs/_build/html/index.html
```

## Status (milestones v2)

This section tracks progress against [`milestones_v2.md`](milestones_v2.md).

- **M0 — Project reset + spec alignment:** IN PROGRESS
  - Spec review checked in: ✅
  - README rewritten (user-facing): ✅
  - Docs scaffold (Sphinx + API pages + tutorial placeholders): ✅
  - Canonical schema module (`EventRecord`/`EventTable` + validation): ⏳ (next)

- **M1 — Substrate artefacts v1:** PLANNED
- **M2 — POIs + baseline features v1:** PLANNED
- **M3 — Parametric road-constrained Hawkes (counts) v1:** PLANNED
- **M4 — Simulator v2:** PLANNED
- **M5 — Evaluation harness:** PLANNED
- **M6 — Chicago benchmark:** PLANNED
- **M7 — ACLED Gaza benchmark:** PLANNED
- **M8 — Neural kernel:** PLANNED
- **M9 — Paper-grade artefact:** PLANNED
- **M10 — Docs site + executed tutorials:** PARTIAL (site scaffold in place; executed notebooks planned)

## Citation

**TODO:** add a proper citation entry once the paper artefact is in place.
