# Contributing

This repository is an active research/WIP codebase, so please prefer small, reviewable changes and document any behavioural changes to notebooks or public APIs.

## Development setup

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .[dev]
```

If you prefer standard library `venv`, use:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## Recommended checks

```bash
python -m py_compile src/motac/*.py tests/*.py
python -m ruff check --select F src/motac tests docs/conf.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m build
```

## Pre-commit hooks

```bash
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Docs (local artifact build)

```bash
uv pip install -e .[docs]
python -m sphinx -b html docs docs/_build/html
```

## Optional long smoke/performance runs

```bash
RUN_SLOW_SMOKE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -m slow
```

Use `motac.bench.compare_backends` for backend runtime comparisons.

## Tutorials

- notebooks listed in `docs/tutorials/index.md` are the curated public tutorial set
- additional notebooks may remain in the repo as experimental/development material
- avoid adding large generated outputs or local data caches to git
