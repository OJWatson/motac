from __future__ import annotations

from click.testing import CliRunner
from typer.main import get_command

from road_hawkes.cli import app


def test_version_command() -> None:
    runner = CliRunner()
    res = runner.invoke(get_command(app), ["version"])
    assert res.exit_code == 0
    assert res.stdout.strip() != ""
