# motac — Milestones v2 (re-aligned to PDF spec + NeurIPS-quality artefact)

This replaces the current implicit milestones. It is aligned to the PDF plan’s MVP and its acceptance criteria, and adds explicit **documentation + reproducibility** requirements (matching the `trace` standard).

## Guiding definition of done
A milestone is “done” only when:
- there is a **single command** (or short script) that reproduces the claimed artefacts,
- outputs are deterministic (seeded) and saved with provenance metadata,
- docs contain both **API reference** and **math / modelling** narrative,
- tutorial notebooks are **executed in CI** (at least a lightweight subset), and rendered on the docs site.

---

## M0 — Project reset + spec alignment
**Goal:** remove drift, clarify external-facing story.

Deliverables:
- `spec_review_vs_pdf.md` checked in.
- README rewritten: no internal milestone chatter; concise Quickstart; links to docs + paper + reproduce.
- Define canonical schema module: `EventRecord` + `EventTable` + validation.

DoD:
- README contains only user-facing content.
- `python -m motac --help` + key subcommands documented.

## M1 — Substrate artefacts (OSM → travel-time neighbours) v1
**Goal:** build and cache road-constrained substrate.

Deliverables:
- `motac.substrate` produces:
  - road graph artefact (GraphML)
  - grid definition
  - sparse neighbour sets with travel-time distances
  - provenance metadata (OSM query, bbox, timestamp, config hash)
- cache format versioned + documented.

DoD:
- Unit tests: reproducibility + neighbour-set sanity.
- CLI: `motac substrate build --config ...` writes artefacts.

## M2 — POIs + baseline features v1
**Goal:** implement POI extraction + feature construction.

Deliverables:
- OSM POI extraction from tags; per-cell features (counts and/or travel-time to nearest POI).
- Feature set documented + unit tests.

DoD:
- `motac substrate build` optionally includes POIs.

## M3 — Parametric road-constrained Hawkes (counts) v1
**Goal:** implement the discrete-time model in the PDF, with road kernel based on travel-time.

Deliverables:
- Likelihood: Poisson + NegBin options.
- Excitation uses sparse neighbour sets + travel-time kernel `W(d_travel)`.
- Stable API:
  - `fit(dataset, substrate, config)`
  - `predict(fitted, start_t, horizon, n_samples)`
  - `loglik(fitted, dataset)`
  - `simulate(params, substrate, T, seed)`

DoD:
- Parameter recovery on synthetic data (multiple seeds) within tolerances.

## M4 — Simulator v2 (shared schema + observation model)
**Goal:** simulator shares identical schema/substrate representation and supports observation noise.

Deliverables:
- Simulator uses substrate neighbour sets.
- Observation model: thinning, clutter, time rounding, missingness.
- Scenario knobs (incl. road closures via perturbed travel times).

DoD:
- Regression tests + golden outputs on tiny fixtures.

## M5 — Evaluation harness (rolling backtests + diagnostics)
**Goal:** implement the evaluation protocol required by the PDF.

Deliverables:
- Rolling-origin backtesting (expanding/fixed windows).
- Horizons: 1/3/7 (optional 14).
- Metrics: NLL/log score, CRPS (MC approx), calibration (coverage + randomised PIT), hotspot metrics.
- Baselines:
  - inhomogeneous Poisson regression (no excitation)
  - Euclidean Hawkes baseline
  - road-constrained Hawkes

DoD:
- `motac eval backtest --config ...` produces a report bundle (JSON + figures).

## M6 — Chicago crime benchmark (GraphST-style)
**Goal:** real dataset ingestion + reproducible benchmark.

Deliverables:
- Socrata fetcher (or documented cached parquet) + cleaning.
- Mapping to canonical schema + grid partition.
- Repro notebook: end-to-end benchmark.

DoD:
- `motac repro chicago --config ...` reproduces figures/tables.

## M7 — ACLED Gaza benchmark
**Goal:** real dataset ingestion + reproducible benchmark with terms-respecting caching.

Deliverables:
- ACLED ingestion: API (credentialed) + offline CSV.
- Gaza bounds, mark taxonomy.
- Repro notebook + benchmark outputs.

DoD:
- `motac repro acled --config ...` reproduces figures/tables.

## M8 — Neural kernel (GNN W) behind stable interface
**Goal:** implement neural mobility kernel with strict ablations.

Deliverables:
- GNN embeddings + edge influence `W[j,k]` (nonnegative) with identical neighbour sets.
- Ablation scripts + compute profiling.

DoD:
- Improves at least one benchmark without degrading calibration; produces ablation table.

## M9 — Paper-grade artefact + release checklist
**Goal:** a submission-quality artefact.

Deliverables:
- `paper/` contains:
  - figure scripts
  - tables
  - `reproduce.sh` / `make reproduce`
  - cached artefacts with provenance
- Model card / limitations.

DoD:
- Fresh machine can run `./paper/reproduce.sh` to regenerate key results.

## M10 — Documentation site (Sphinx like trace) + tutorial notebooks
**Goal:** bring docs up to trace standard.

Deliverables:
- Sphinx docs site deployed via GitHub Pages.
- Rendered notebooks (nbsphinx/myst) for:
  - synthetic
  - chicago
  - acled
  - ablations
- API reference pages generated.

DoD:
- Docs build in CI; Pages updated on main.

