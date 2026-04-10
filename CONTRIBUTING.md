# Contributing

## Development setup

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -e .[dev]
```

## Recommended checks

```bash
python3.11 -m py_compile motac/*.py tests/*.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3.11 -m pytest -q
```

## Pre-commit hooks

```bash
python3.11 -m pip install --break-system-packages --user pre-commit
pre-commit install
pre-commit run --all-files
```

## Docs (local artifact build)

```bash
python3.11 -m pip install --break-system-packages --user -e .[docs]
python3.11 -m sphinx -b html docs docs/_build/html
```

## Optional long smoke/performance runs

```bash
RUN_SLOW_SMOKE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3.11 -m pytest -q -m slow
```

Use `motac.bench.compare_backends` for backend runtime comparisons.
