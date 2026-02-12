# Work log

Short, append-only notes intended to preserve continuity across short-lived builder runs.

Guidelines:
- One dated bullet per meaningful action.
- Record failures (CI link, error snippet) and the next concrete step.
- Keep this file small.

- 2026-02-12: Added tiny backtest helper `backtest_fit_forecast_nll` (fit→forecast→score) and a toy unit test; ran full pytest in venv (64 passed).
- 2026-02-12: Documented `motac.eval.backtest` in Sphinx API docs and added explicit eval-module re-exports; `uv run ruff check` + `uv run pytest` (64 passed); pushed `0855815`.
