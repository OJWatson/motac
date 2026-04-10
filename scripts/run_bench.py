#!/usr/bin/env python3.11
"""Run conv vs edges backend benchmark and print a compact summary table."""

from __future__ import annotations

from motac.cli import run_benchmarks


def main() -> None:
    run_benchmarks()


if __name__ == "__main__":
    main()
