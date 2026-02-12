from __future__ import annotations

from pathlib import Path

import typer

from motac.cli._app import paper_app
from motac.paper.generate_artifacts import generate_synthetic_eval_artifact

_OUT_DIR_OPT = typer.Option(
    ..., "--out-dir", help="Output directory for JSON artifacts (created if missing)."
)
_SEED_OPT = typer.Option(0, "--seed", help="Random seed for the synthetic evaluation.")


@paper_app.command("generate-artifacts")
def generate_artifacts(
    out_dir: Path = _OUT_DIR_OPT,
    seed: int = _SEED_OPT,
) -> None:
    """Generate CI-safe, small paper artifacts.

    Currently this produces a synthetic-evaluation JSON payload.
    """

    path = generate_synthetic_eval_artifact(out_dir=out_dir, seed=seed)
    typer.echo(str(path))
