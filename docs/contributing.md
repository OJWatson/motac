# Contributing

## Development install

```bash
uv sync --group dev
```

## Lint + tests

```bash
uv run ruff check .
uv run python -m pytest
```

## Build docs

```bash
uv sync --group docs
cd docs
make html
```

## Project status

See `milestones_v2.md` in the repository root.
