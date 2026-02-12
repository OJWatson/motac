from __future__ import annotations

import importlib


def test_cli_import_paths_smoke() -> None:
    # The CLI entry point should remain stable while we refactor internals.
    cli_pkg = importlib.import_module("motac.cli")
    assert hasattr(cli_pkg, "app")
    assert hasattr(cli_pkg, "main")

    # Internal structure should also be importable (no missing modules).
    importlib.import_module("motac.cli._app")
    importlib.import_module("motac.cli.commands.core")
    importlib.import_module("motac.cli.commands.substrate")
    importlib.import_module("motac.cli.commands.sim")
