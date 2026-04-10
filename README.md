# motac

`motac` is an experimental JAX/NumPyro package for discrete-time mobility-constrained Hawkes count models.

## Status

This repository is a **public work in progress**.

- the core package API is usable for experimentation
- documentation and tutorials are being curated for a first public release
- research notebooks and modelling workflows may change substantially
- results shown in notebooks should be treated as exploratory unless stated otherwise

## Python version

This project currently targets **Python 3.11**.

## Installation

### Recommended local development install (`uv`)

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .[dev]
```

### Docs build install

```bash
uv pip install -e .[docs]
```

The commands below assume your local environment is already activated.

### Standard library `venv` alternative

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

### Optional GPU/JAX setup

If you want CUDA-backed JAX locally, install the project first and then install the JAX CUDA wheels appropriate for your system.

```bash
source .venv/bin/activate
uv pip install -e .[dev]
uv pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python -c "import jax; print(jax.devices())"
```

Adjust CUDA-specific details to match your machine.

## Quick start

```python
from motac import HawkesModelSpec, MobilityHawkesModel

spec = HawkesModelSpec(marks=("event",), temporal_bases=2, observation="poisson")
model = MobilityHawkesModel(spec)
```

See the docs and tutorials for end-to-end examples using the simulator, backtesting helpers, and ACLED/Chicago datasets.

## Documentation

- package docs live under `docs/`
- the published tutorial set is intentionally curated
- more experimental notebooks may remain in the repo without being part of the rendered docs site

Build docs locally with:

```bash
python -m sphinx -b html docs docs/_build/html
```

## Tests

Run the main test suite with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

## Benchmarks

```bash
python scripts/run_bench.py
# or, after install:
motac-bench
```

## Repository layout

- `src/motac/`: package source
- `tests/`: unit and integration tests
- `docs/`: Sphinx, MyST, and notebook-backed documentation
- `docs/tutorials/`: curated tutorial notebooks

## Limitations

Current limitations and modelling caveats are documented in `docs/limitations.md` and in the ACLED notebook series.

## Contributing

If you want to use or extend the project, start with `CONTRIBUTING.md`.
