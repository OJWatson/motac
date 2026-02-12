from __future__ import annotations

import typer

from ..._version import __version__
from .._app import app


@app.callback()
def _root() -> None:
    """Road-constrained spatio-temporal Hawkes models (JAX)."""


@app.command()
def version() -> None:
    """Print the package version."""

    typer.echo(__version__)
