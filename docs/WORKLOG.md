# Work log

Short, append-only notes intended to preserve continuity across short-lived builder runs.

Guidelines:
- One dated bullet per meaningful action.
- Record failures (CI link, error snippet) and the next concrete step.
- Keep this file small.

- 2026-02-12: Added tiny backtest helper `backtest_fit_forecast_nll` (fit→forecast→score) and a toy unit test; ran full pytest in venv (64 passed).
- 2026-02-12: Documented `motac.eval.backtest` in Sphinx API docs and added explicit eval-module re-exports; `uv run ruff check` + `uv run pytest` (64 passed); pushed `0855815`.
- 2026-02-12: CI green on `0855815`; advanced STATUS from M5→M6; ran `uv run ruff check` + `uv run pytest` (64 passed).
- 2026-02-12: Added committed Chicago loader CSV fixture + fixture-driven sanity test; ran `uv run ruff check .` + `uv run pytest` (65 passed); pushed `c17c199`.
- 2026-02-12: Added committed mobility `.npy` fixture + tests for `meta["mobility_source"]` and mismatched mobility shape rejection; ran `uv run ruff check .` + `uv run pytest` (66 passed); pushed `66bb0cd`.
- 2026-02-12: Asserted identity default `meta["mobility_source"] == "identity"` in Chicago loader test; ran `uv run ruff check .` + `uv run pytest` (66 passed); pushed `fdde12f`.
- 2026-02-12: Documented Chicago raw on-disk contract (v1 placeholder) and added test that loader preserves row ordering; ran `uv run ruff check .` + `uv run pytest` (67 passed); pushed `c1fdfe9`.
- 2026-02-12: Extended Chicago loader `load_y_obs_matrix` to accept a raw directory (expects `y_obs.csv`, autodetects optional `mobility.npy`); added tests; ran `uv run ruff check .` + `uv run pytest` (69 passed); pushed `3fd47e0`.
- 2026-02-12: Added minimal Chicago raw loader JSON config (`ChicagoRawConfig`) + CLI entry point `motac data chicago-load`; added CLI test; ran `uv run ruff check .` + `uv run pytest` (70 passed); pushed `1558da0`.
- 2026-02-12: Documented `motac data chicago-load` usage + example JSON config in `docs/loaders/chicago.md`; ran `uv run ruff check .` + `uv run pytest` (70 passed); pushed `2364677`.
- 2026-02-12: CI green; reviewed M6 DoD (Chicago loader deterministic + sanity tests) and advanced STATUS to M7 (ACLED loader next); pushed `1af27be`.
- 2026-02-12: Added committed ACLED CSV fixture + fixture-driven sanity test (identity mobility); ran `uv run ruff check .` + `uv run pytest` (71 passed); pushed `6e49a5e`.
- 2026-02-12: Added minimal ACLED loader docs page + wired `docs/loaders/` into Sphinx toctree; ran `uv run ruff check .` + `uv run pytest` (71 passed); pushed `e071651`.
- 2026-02-12: Added ACLED loader JSON config (`AcledEventsCsvConfig`) + CLI entry point `motac data acled-load`; added CLI test; ran `uv run ruff check .` + `uv run pytest` (72 passed); pushed `1ab9112`.
- 2026-02-12: Documented `motac data acled-load` usage + example JSON config in `docs/loaders/acled.md`; ran `uv run ruff check .` + `uv run pytest` (72 passed); pushed `837291e`.
- 2026-02-12: Added CI-safe observed-data end-to-end unit test (fit → predictive sample → score) on toy data; ran `uv run ruff check .` + `uv run pytest` (73 passed); pushed `e8bfbe3`.
- 2026-02-12: Updated `docs/STATUS.md` (M8 next step + header) and appended to worklog; pushed `5456d0b`.
- 2026-02-12: Adjusted STATUS header `headSha` to point at last code-gated commit (`e8bfbe3`); pushed `961bcb4`.
- 2026-02-12: Reviewed M8 DoD coverage (observed-data end-to-end test exists) and advanced STATUS to M9.
- 2026-02-12: Reviewed M9 DoD coverage (exact observed log-likelihood + tests exist) and advanced STATUS to M10.
- 2026-02-12: Added public comparison harness `compare_observed_loglik_exact_vs_poisson_approx` + unit test; ran `uv run ruff check .` + `uv run pytest` (74 passed); pushed `6cf9f8d`.
- 2026-02-12: Documented interpretation/expected gaps for exact-vs-Poisson-approx observed likelihoods in `docs/api/sim.md`; ran `uv run ruff check .` + `uv run pytest` (74 passed); pushed `aa905e7`.
- 2026-02-12: Added Sphinx page documenting the CI-safe paper artefacts stub (`motac paper generate-artifacts`) and wired it into docs; refactored artefact generation to return the written path; ran `uv run ruff check .` + `uv run pytest` (75 passed); pushed `2ac1744`.
- 2026-02-12: Added minimal manifest alongside synthetic eval JSON artifact (git SHA + seed + config summary + timestamp) with CI-safe unit test; ran `uv run ruff check .` + `uv run pytest` (75 passed); pushed `0fc1134`.
- 2026-02-12: Documented paper artefact manifest filename + fields in `docs/paper_artefacts.md`; ran `uv run ruff check .` + `uv run pytest` (75 passed); pushed `e57e4e1`.
