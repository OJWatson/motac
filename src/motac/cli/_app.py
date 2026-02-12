from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)
substrate_app = typer.Typer(add_completion=False, no_args_is_help=True)
sim_app = typer.Typer(add_completion=False, no_args_is_help=True)
data_app = typer.Typer(add_completion=False, no_args_is_help=True)
paper_app = typer.Typer(add_completion=False, no_args_is_help=True)

app.add_typer(substrate_app, name="substrate")
app.add_typer(sim_app, name="sim")
app.add_typer(data_app, name="data")
app.add_typer(paper_app, name="paper")

# Register commands.
#
# Commands are defined in submodules so we can keep the package boundary small and
# avoid growing :mod:`motac.cli.__init__` into a grab-bag.
from .commands import core as _core  # noqa: E402,F401
from .commands import data as _data  # noqa: E402,F401
from .commands import paper as _paper  # noqa: E402,F401
from .commands import sim as _sim  # noqa: E402,F401
from .commands import substrate as _substrate  # noqa: E402,F401
