# Documentation scaffold

This project uses **Sphinx** with:
- **MyST Markdown** (`myst_parser`) for narrative docs (`.md`)
- **nbsphinx** for notebooks (`docs/tutorials/*.ipynb`)

Tooling choices are intentionally lightweight for M0; notebooks are rendered but
not executed in CI (`nbsphinx_execute = "never"`).

## Tree

- `docs/index.md` — landing page
- `docs/installation.md` — installation
- `docs/model/` — modelling narrative (math + assumptions)
- `docs/api/` — API reference pages (hand-written stubs that Sphinx renders)
- `docs/tutorials/` — tutorial notebooks (rendered)

## Build

From repo root:

```bash
uv sync --group docs
cd docs
make html
open _build/html/index.html
```

## CI

Docs are built in GitHub Actions via `.github/workflows/docs.yml` and deployed
to GitHub Pages on pushes to `main`.

## Legacy docs build

A previous generated Sphinx HTML build is preserved under .
