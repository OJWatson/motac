# Paper artefacts pipeline (M11)

This repository aims to support **reproducible paper artefacts** (tables/figures) in a way that is:

- deterministic (seeded)
- CI-safe (no large downloads)
- easy to run locally

## Current CI-safe stub

The smallest artefact generator is implemented as a **synthetic evaluation** run that writes a single JSON payload.

CLI:

```bash
motac paper generate-artifacts --out-dir paper/artifacts --seed 7
```

This writes:

- `paper/artifacts/synthetic_eval_seed7.json`

with keys:

- `config` — the evaluation configuration used
- `fit` — fitted parameters / fit diagnostics
- `forecasts` — forecast summaries for a small horizon
- `metrics` — basic scalar metrics

Python module entry point (equivalent):

```bash
python -m motac.paper.generate_artifacts --out-dir paper/artifacts --seed 7
```

Notes:

- The artefact is deliberately small and uses **toy/synthetic data**.
- Plotting is intentionally not part of the CI path.

## Intended artefacts (planned)

As the paper solidifies, this pipeline is expected to produce:

- a small, versioned **artefact manifest** (what was generated, from which config, git SHA)
- core paper tables (e.g. accuracy/calibration summaries)
- figure-ready data exports (CSV/JSON) for plotting outside CI

The goal is that a single command can regenerate the minimal set of artefacts required to validate the paper’s claims.
