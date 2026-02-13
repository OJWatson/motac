# MOTAC Rebuild: Road-Constrained Hawkes (JAX) — Task Graph & Commit-Level Plan
**Date:** 2026-02-13  
**Audience:** agentic coding systems (OpenClaw), maintainers  
**Tone:** directive. This is a reset plan.

This document is a **foolproof, commit-by-commit task graph** to realign `motac` to the road-constrained Hawkes package described in the Hawkes-only plan. fileciteturn3file0  
It is intended to be dropped into the repo as `ROADMAP_REBUILD.md` or `MOTAC_REBUILD_TASKGRAPH.md`.

---

## 0) What is going wrong (diagnosis) and how to fix it

### Symptoms (as reported and commonly observed)
1. **TODO sprawl**: TODOs scattered through core modules, meaning unfinished design decisions are embedded in production paths.
2. **Milestone-doc drift**: planning documents sitting alongside code → unclear “source of truth”.
3. **Broken module boundaries**: spatial preprocessing, modelling, evaluation, and notebooks intermixed → changes in one place break another.
4. **Unstable or undocumented APIs**: no canonical `EventRecord`/`Dataset`/`Substrate` contracts → dataset-specific hacks creep into models.
5. **No reproducible benchmark**: “works on my machine” code; no single command that reproduces metrics/figures.
6. **Documentation without end-to-end examples**: no fully rendered “Simulation → Fit → Forecast → Score” tutorial.
7. **Architectural drift** from the agreed target: mobility-constrained Hawkes with road travel-time kernels, simulator, Chicago crime benchmark, ACLED Gaza benchmark, and optional neural GNN kernel/surrogate.

### Root causes
- **No enforced contracts** (schemas + model interface).
- **No gating** (code merged without passing unit tests, backtest harness, or doc builds).
- **No experiment discipline** (configs/artefacts not versioned, seeds not fixed).
- **Neural exploration before parametric baseline** (no anchor; no ablation structure).

### Fix strategy
- **Hard reset to a minimal, strict architecture** (modules + contracts first).
- **Parametric Hawkes + simulator + evaluation harness are non-negotiable**.
- **Neural/GNN kernel only after M0–M6 pass** (see milestones).
- **Every task is one atomic commit** with objective acceptance checks.

---

## 1) Target scope (must not change)
`motac` becomes a **research-grade road-constrained spatio-temporal Hawkes package** (JAX), with:

- **Spatial substrate** built from OpenStreetMap roads + POIs, producing travel-time neighbour sets.
- **Discrete-time Hawkes count model** (Poisson and NegBin options) with baseline + excitation.
- **Simulator** with observation noise + regime shifts for parameter recovery and stress tests.
- **Benchmarks** on:
  - **Simulator** (synthetic)
  - **Chicago crime** (Socrata open data; replication of a common ML benchmark) fileciteturn3file0
  - **ACLED Gaza** (user-provided via API keys or CSV; cached locally) fileciteturn3file0
- **Optional neural kernel / surrogate** (Jraph + Equinox), trained on the *same likelihood*.

Everything else is deleted or quarantined.

---

## 2) Non-negotiable contracts (write these first)

### 2.1 Canonical schema
All datasets map into this schema:

```python
@dataclass(frozen=True)
class EventRecord:
    event_id: str
    t: int               # day index (0..T-1) for discrete-time core
    lat: float
    lon: float
    cell_id: int         # grid cell id
    mark: int | None = None
```

### 2.2 Dataset bundle
```python
@dataclass(frozen=True)
class Dataset:
    events: pd.DataFrame           # canonical events table
    counts: jnp.ndarray            # [J, T] or [J, M, T] for marked
    meta: dict                     # bbox, start_date, grid_spec, marks map, etc
```

### 2.3 Substrate bundle
```python
@dataclass(frozen=True)
class Substrate:
    grid: GridSpec
    road_graph: RoadGraph
    neighbours: NeighbourSet       # sparse: indices + travel-time distances
    poi_features: jnp.ndarray      # [J, P]
```

### 2.4 Model API
```python
class HawkesModel(Protocol):
    def fit(self, dataset: Dataset, substrate: Substrate, cfg: dict) -> "FittedModel": ...
    def predict(self, fitted: "FittedModel", start_t: int, horizon: int, n_samples: int, seed: int) -> "Forecast": ...
    def loglik(self, fitted: "FittedModel", dataset: Dataset) -> float: ...
```

---

## 3) Repo restructure (must happen early)
Target structure (minimal):

```
motac/
  data/        # chicago, acled, sim
  spatial/     # grid, roads, travel time, POIs, neighbours
  models/      # parametric, marked, neural
  inference/   # likelihood + optimisation
  sim/         # simulator + observation noise
  eval/        # backtest + metrics + calibration + plots
  cli/         # run entrypoints
configs/
notebooks/
paper/
tests/
docs/
```

---

## 4) Task Graph (commit-level, agent-parsable)

### How to read this
Each task is an atomic commit with:
- `id`: stable identifier
- `depends_on`: prerequisites
- `commit_title`: suggested git commit message
- `goal`: what it accomplishes
- `touch`: key files/directories
- `commands`: minimal commands to run
- `acceptance`: objective checks (must pass)

Agents should execute tasks in topological order.

---

# M0 — Reset & Scaffolding (must be done before any modelling)

### TASK M0.1
- id: M0.1
- depends_on: []
- commit_title: "chore: hard reset roadmap + move legacy docs"
- goal: Move milestone drafts/legacy planning docs out of the main tree; add this roadmap.
- touch: `docs/legacy/`, `ROADMAP_REBUILD.md`
- commands: `python -m compileall motac || true`
- acceptance:
  - No milestone docs in repo root
  - `ROADMAP_REBUILD.md` present

### TASK M0.2
- id: M0.2
- depends_on: [M0.1]
- commit_title: "chore: standardise tooling (ruff, mypy, pytest) + pre-commit"
- goal: Enforce code quality gates.
- touch: `pyproject.toml`, `.pre-commit-config.yaml`, `ruff.toml` (or ruff in pyproject), `mypy.ini`, `pytest.ini`
- commands: `ruff check .`, `pytest -q`
- acceptance:
  - CI-ready: lint and tests runnable locally
  - Pre-commit hooks installed and pass

### TASK M0.3
- id: M0.3
- depends_on: [M0.2]
- commit_title: "ci: add github actions for lint+tests"
- goal: Add minimal CI.
- touch: `.github/workflows/ci.yml`
- acceptance:
  - Workflow runs on PR
  - Fails on lint/test failures

### TASK M0.4
- id: M0.4
- depends_on: [M0.2]
- commit_title: "feat: add canonical dataclasses (EventRecord, Dataset, Substrate)"
- goal: Create stable contracts used everywhere.
- touch: `motac/core/types.py`, `tests/test_types.py`
- commands: `pytest -q`
- acceptance:
  - `tests/test_types.py` passes
  - No other module defines duplicate schema objects

### TASK M0.5
- id: M0.5
- depends_on: [M0.4]
- commit_title: "feat: add config + run directory conventions"
- goal: Introduce config parsing (YAML) and run artefact layout.
- touch: `motac/core/config.py`, `motac/core/run_dir.py`, `tests/test_run_dir.py`, `configs/example.yaml`
- acceptance:
  - `run_dir` creates deterministic output folders with metadata json including seed + git hash

---

# M1 — Spatial substrate (roads, travel-time, POIs, neighbours)

### TASK M1.1
- id: M1.1
- depends_on: [M0.5]
- commit_title: "feat(spatial): grid builder + coordinate transforms"
- goal: Implement grid creation and lat/lon → cell_id assignment.
- touch: `motac/spatial/grid.py`, `tests/test_grid.py`
- acceptance:
  - Given a bbox and resolution, grid cell count J is deterministic
  - Round-trip tests for cell indexing pass

### TASK M1.2
- id: M1.2
- depends_on: [M1.1]
- commit_title: "feat(spatial): OSM road graph fetch + travel time weights"
- goal: Download/build road graph with travel-time edge weights.
- touch: `motac/spatial/road_graph.py`, `tests/test_road_graph.py`
- acceptance:
  - Graph builds for a tiny bbox fixture
  - Edge weights include travel time (seconds/minutes)

### TASK M1.3
- id: M1.3
- depends_on: [M1.2]
- commit_title: "feat(spatial): travel-time distances from grid centroids"
- goal: Map grid centroids to graph nodes and compute shortest path travel time.
- touch: `motac/spatial/travel_time.py`, `tests/test_travel_time.py`
- acceptance:
  - For fixture graph, travel time matrix computed for a small set of nodes
  - Caching works (second run loads from disk)

### TASK M1.4
- id: M1.4
- depends_on: [M1.3]
- commit_title: "feat(spatial): neighbour sets (sparse) in travel-time space"
- goal: Build sparse neighbour indices/distances N(j).
- touch: `motac/spatial/neighbours.py`, `tests/test_neighbours.py`
- acceptance:
  - For each j, neighbour count <= k or within threshold
  - Stored as CSR/COO and loaded into JAX-friendly arrays

### TASK M1.5
- id: M1.5
- depends_on: [M1.2]
- commit_title: "feat(spatial): POI extraction + travel-time-to-POI features"
- goal: Extract POIs from OSM tags and compute features per cell.
- touch: `motac/spatial/poi.py`, `tests/test_poi.py`
- acceptance:
  - POIs extracted deterministically for fixture bbox
  - Feature matrix shape [J, P] is stable

### TASK M1.6
- id: M1.6
- depends_on: [M1.4, M1.5]
- commit_title: "feat(spatial): build Substrate object + cache artefacts"
- goal: One function builds all substrate artefacts and returns `Substrate`.
- touch: `motac/spatial/build.py`, `tests/test_substrate_build.py`
- acceptance:
  - `build_substrate(cfg)` writes cached artefacts and returns `Substrate`
  - Reloading uses cache (hash-based)

---

# M2 — Simulator (synthetic truth + observation noise)

### TASK M2.1
- id: M2.1
- depends_on: [M0.4, M1.6]
- commit_title: "feat(sim): world generator (covariates + POIs) for synthetic runs"
- goal: Generate synthetic baseline covariates on the grid.
- touch: `motac/sim/world.py`, `tests/test_sim_world.py`
- acceptance:
  - Covariate fields deterministic with seed
  - Covariates match grid shape [J, ...]

### TASK M2.2
- id: M2.2
- depends_on: [M2.1]
- commit_title: "feat(sim): hawkes event generator (discrete-time counts)"
- goal: Generate counts y[j,t] given baseline and excitation params using travel-time neighbours.
- touch: `motac/sim/event_generator.py`, `tests/test_sim_hawkes.py`
- acceptance:
  - With excitation=0, events follow baseline only
  - With excitation>0, clustering increases (measurable statistic)

### TASK M2.3
- id: M2.3
- depends_on: [M2.2]
- commit_title: "feat(sim): observation model (thinning, jitter, missingness)"
- goal: Apply under-reporting and jitter to events.
- touch: `motac/sim/observation.py`, `tests/test_sim_observation.py`
- acceptance:
  - Under-reporting reduces counts by expected rate
  - Jitter preserves total count but changes spatial allocation

### TASK M2.4
- id: M2.4
- depends_on: [M2.3]
- commit_title: "feat(sim): sim dataset IO (parquet) + ground truth params"
- goal: Save/load simulated datasets with ground truth.
- touch: `motac/data/sim_loader.py`, `tests/test_sim_io.py`
- acceptance:
  - Saved dataset reloads identically
  - Ground truth params stored in meta

---

# M3 — Parametric Hawkes (JAX likelihood + fitting)

### TASK M3.1
- id: M3.1
- depends_on: [M0.4, M1.4]
- commit_title: "feat(inference): JAX core ops for sparse neighbour convolution"
- goal: Implement fast excitation computation with jit.
- touch: `motac/inference/sparse_ops.py`, `tests/test_sparse_ops.py`
- acceptance:
  - Matches numpy reference on small fixtures
  - JIT compilation succeeds

### TASK M3.2
- id: M3.2
- depends_on: [M3.1]
- commit_title: "feat(inference): likelihoods (Poisson, NegBin) + transforms"
- goal: Implement log-likelihood for Lambda = mu + e.
- touch: `motac/inference/likelihood.py`, `tests/test_likelihood.py`
- acceptance:
  - Gradient check vs finite differences (small fixture)
  - Likelihood increases after a few optimisation steps on toy data

### TASK M3.3
- id: M3.3
- depends_on: [M3.2]
- commit_title: "feat(models): parametric road-kernel Hawkes model"
- goal: Implement model class with W=exp(-d/sigma) and optional power-law.
- touch: `motac/models/hawkes_parametric.py`, `tests/test_model_parametric.py`
- acceptance:
  - `fit` runs end-to-end on sim dataset
  - Parameters remain non-negative via softplus

### TASK M3.4
- id: M3.4
- depends_on: [M3.3]
- commit_title: "feat(inference): optimiser + regularisation + early stopping"
- goal: Add Optax optimiser, penalties, and stable training loop.
- touch: `motac/inference/fit_jax.py`, `tests/test_fit_loop.py`
- acceptance:
  - Training reduces NLL on synthetic
  - Deterministic given seed

### TASK M3.5
- id: M3.5
- depends_on: [M3.4]
- commit_title: "feat(models): forecasting (1-7 day) with Monte Carlo intervals"
- goal: Generate predictive distributions via forward simulation in discrete time.
- touch: `motac/models/forecast.py`, `tests/test_forecast.py`
- acceptance:
  - Forecast outputs shape [J, horizon, n_samples]
  - Coverage sanity check on simulated data

---

# M4 — Evaluation harness (backtest + metrics + calibration)

### TASK M4.1
- id: M4.1
- depends_on: [M3.5]
- commit_title: "feat(eval): rolling-origin backtest splits"
- goal: Standard backtest generator (expanding or sliding window).
- touch: `motac/eval/backtest.py`, `tests/test_backtest.py`
- acceptance:
  - Splits reproducible; no leakage

### TASK M4.2
- id: M4.2
- depends_on: [M4.1]
- commit_title: "feat(eval): proper scoring rules + baseline models"
- goal: Log score, CRPS approx; Poisson baseline; Euclidean Hawkes baseline (ablation).
- touch: `motac/eval/metrics.py`, `motac/models/baselines.py`, `tests/test_metrics.py`
- acceptance:
  - Metrics match known values on fixtures
  - Baselines run in the same pipeline

### TASK M4.3
- id: M4.3
- depends_on: [M4.2]
- commit_title: "feat(eval): calibration diagnostics (coverage, PIT)"
- goal: Interval coverage + randomised PIT for counts.
- touch: `motac/eval/calibration.py`, `tests/test_calibration.py`
- acceptance:
  - PIT histogram produced for fixture
  - Coverage computed correctly

### TASK M4.4
- id: M4.4
- depends_on: [M4.3]
- commit_title: "feat(eval): standard report (HTML) + figure utilities"
- goal: Generate report with key plots and tables.
- touch: `motac/eval/report.py`, `motac/eval/plots.py`
- acceptance:
  - `run_backtest` produces `report.html` in run directory

---

# M5 — Chicago crime benchmark (public data)

### TASK M5.1
- id: M5.1
- depends_on: [M0.5]
- commit_title: "feat(data): chicago crime fetcher with caching (Socrata)"
- goal: Implement Socrata pull + local parquet cache.
- touch: `motac/data/chicago.py`, `tests/test_chicago_loader.py`
- acceptance:
  - Works with API token optional
  - Cached parquet used on repeat

### TASK M5.2
- id: M5.2
- depends_on: [M5.1, M1.1]
- commit_title: "feat(data): chicago cleaning + mapping to canonical events"
- goal: Dedup, parse timestamps, map types; grid assignment.
- touch: `motac/data/chicago.py`, `tests/test_chicago_clean.py`
- acceptance:
  - Output conforms to EventRecord schema
  - Stable mark mapping (if used)

### TASK M5.3
- id: M5.3
- depends_on: [M5.2, M4.4]
- commit_title: "notebook: chicago benchmark end-to-end"
- goal: Notebook that fetches (or loads cached), builds substrate, fits model, backtests, produces figures.
- touch: `notebooks/02_chicago.ipynb`
- acceptance:
  - Notebook runs start→finish on cached data
  - Exports figures into `paper/figures/chicago/`

---

# M6 — ACLED Gaza benchmark (credentialed or CSV)

### TASK M6.1
- id: M6.1
- depends_on: [M0.5]
- commit_title: "feat(data): ACLED loader (CSV) + canonical mapping"
- goal: Load local CSV, clean, filter to Gaza bbox, map event types to marks.
- touch: `motac/data/acled.py`, `tests/test_acled_csv.py`
- acceptance:
  - No credentials required for CSV pathway
  - Output conforms to EventRecord schema

### TASK M6.2
- id: M6.2
- depends_on: [M6.1]
- commit_title: "feat(data): ACLED API loader (optional) with safe caching"
- goal: API fetch behind env vars; cache locally; do not commit raw outputs.
- touch: `motac/data/acled.py`, `tests/test_acled_api_mock.py`
- acceptance:
  - API calls mockable; respects rate limiting hooks
  - Writes provenance metadata (query, date, hash)

### TASK M6.3
- id: M6.3
- depends_on: [M6.1, M1.6]
- commit_title: "feat(config): gaza benchmark config + substrate recipe"
- goal: Provide config template for Gaza bbox, grid resolution, POI tags.
- touch: `configs/gaza.yaml`
- acceptance:
  - `build_substrate` runs from config and caches artefacts

### TASK M6.4
- id: M6.4
- depends_on: [M6.3, M4.4]
- commit_title: "notebook: gaza benchmark end-to-end (ACLED CSV)"
- goal: Notebook that loads ACLED CSV, builds substrate, fits model, backtests, produces figures.
- touch: `notebooks/03_acled_gaza.ipynb`
- acceptance:
  - Runs end-to-end with local CSV
  - Exports figures into `paper/figures/gaza/`

---

# M7 — Neural kernel / surrogate (only after parametric baseline is stable)

### TASK M7.1
- id: M7.1
- depends_on: [M6.4]
- commit_title: "feat(models): neural mobility kernel (Jraph + Equinox)"
- goal: Replace W[j,k] with non-negative learned function; keep same likelihood.
- touch: `motac/models/hawkes_neural.py`, `motac/models/gnn.py`, `tests/test_neural_kernel_shapes.py`
- acceptance:
  - `fit` runs without changing eval harness
  - Shapes and non-negativity enforced

### TASK M7.2
- id: M7.2
- depends_on: [M7.1]
- commit_title: "eval: neural vs parametric ablations + compute profiling"
- goal: Produce ablation table and profiling logs.
- touch: `paper/ablation_neural_vs_parametric.py`, `motac/eval/profiling.py`
- acceptance:
  - Table generated for simulation + at least one real dataset
  - Calibration not worse than parametric by default threshold

### TASK M7.3
- id: M7.3
- depends_on: [M7.2]
- commit_title: "notebook: neural ablation and results"
- goal: Fully rendered notebook with identical splits showing comparison.
- touch: `notebooks/04_neural_ablation.ipynb`
- acceptance:
  - Generates the ablation table used in `paper/`

---

# M8 — Paper-grade artefact + documentation

### TASK M8.1
- id: M8.1
- depends_on: [M6.4]
- commit_title: "docs: full worked examples + API reference"
- goal: README + docs pages with executable examples and dataset setup.
- touch: `README.md`, `docs/`, `mkdocs.yml` (or sphinx)
- acceptance:
  - Docs build passes in CI
  - Contains three worked examples: simulation, Chicago, Gaza

### TASK M8.2
- id: M8.2
- depends_on: [M8.1]
- commit_title: "paper: reproduce script + release checklist"
- goal: One script to recreate all paper figures from cached data.
- touch: `paper/reproduce.sh`, `paper/README.md`
- acceptance:
  - Fresh venv reproduces key figures without manual steps (given cached datasets)

---

## 5) Agent execution rules (strict)
1. Do not start a task until all dependencies are merged.
2. One task = one commit. Do not bundle.
3. Every task adds or updates at least one test.
4. No notebook is added unless it runs end-to-end.
5. No neural work begins until M6 is complete.
6. Any new feature must declare where it plugs into the canonical schema and model API.

---

## 6) Definition of Done
`motac` is “back on track” when:

- `motac` can run:
  - `simulate → fit → forecast → score` (synthetic)
  - `chicago → fit → backtest` (cached)
  - `gaza(acled csv) → fit → backtest` (local)
- Results are reproduced by config-driven CLI commands.
- Docs include fully worked examples and build in CI.
- Neural kernel comparison is a clean ablation (optional, gated).
- There are no TODOs in core code paths.

---

## 7) Appendix: quick start commands (post-M4)
```bash
# build substrate
python -m motac.cli.run build-substrate --config configs/chicago.yaml
# backtest parametric
python -m motac.cli.run backtest --config configs/chicago.yaml --model parametric
# simulate + recovery
python -m motac.cli.run simulate --config configs/sim_default.yaml
python -m motac.cli.run fit --config configs/sim_default.yaml --model parametric
```

