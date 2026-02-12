"""Command-line interface.

The CLI is implemented as a small package so we can keep a stable entry point
while extracting internal boundaries structure-first.
"""

from __future__ import annotations

from ._app import app

__all__ = ["app", "main"]


def main() -> None:
    """Entry point for `motac`."""

    app()
