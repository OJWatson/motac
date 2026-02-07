from __future__ import annotations

import typer

from ._version import __version__

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def _root() -> None:
    """Road-constrained spatio-temporal Hawkes models (JAX)."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


def main() -> None:
    """Entry point for `motac`."""
    app()
