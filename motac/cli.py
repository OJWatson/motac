"""Command-line entrypoints for motac."""

from __future__ import annotations

import argparse

from .bench import compare_backends


def run_benchmarks(argv: list[str] | None = None) -> None:
    """Run conv vs edge backend benchmarks and print a compact table."""
    parser = argparse.ArgumentParser(
        description="Run motac backend benchmarks")
    parser.add_argument("--grid-h", type=int, default=40, help="Grid height")
    parser.add_argument("--grid-w", type=int, default=40, help="Grid width")
    parser.add_argument("--time-steps", type=int,
                        default=30, help="Number of time bins")
    parser.add_argument("--fit-steps", type=int, default=40,
                        help="MAP optimization steps")
    parser.add_argument("--seed", type=int, default=0,
                        help="Benchmark random seed")
    args = parser.parse_args(argv if argv is not None else [])

    results = compare_backends(
        grid_shape=(args.grid_h, args.grid_w),
        time_steps=args.time_steps,
        fit_steps=args.fit_steps,
        seed=args.seed,
    )
    print("backend\tsim_s\tfit_s\tforecast_s")
    for r in results:
        print(
            f"{r.backend}\t{r.simulation_seconds:.3f}\t{r.fit_seconds:.3f}\t{r.forecast_seconds:.3f}")
