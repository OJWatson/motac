# Work log

Short, append-only notes intended to preserve continuity across short-lived builder runs.

Guidelines:
- One dated bullet per meaningful action.
- Record failures (CI link, error snippet) and the next concrete step.
- Keep this file small.

- 2026-02-12: Added tiny backtest helper `backtest_fit_forecast_nll` (fit→forecast→score) and a toy unit test; ran full pytest in venv (64 passed).
- 2026-02-12: Documented `motac.eval.backtest` in Sphinx API docs and added explicit eval-module re-exports; `uv run ruff check` + `uv run pytest` (64 passed); pushed `0855815`.
- 2026-02-12: CI green on `0855815`; advanced STATUS from M5→M6; ran `uv run ruff check` + `uv run pytest` (64 passed).
- 2026-02-12: Added committed Chicago loader CSV fixture + fixture-driven sanity test; ran `uv run ruff check` + `uv run pytest` (65 passed); pushed `c17c199`.
- 2026-02-12: Added committed mobility `.npy` fixture + tests for `meta["mobility_source"]` and mismatched mobility shape rejection; ran `uv run ruff check .` + `uv run pytest` (66 passed); pushed `66bb0cd`.
