from __future__ import annotations


def test_cli_import_path_stable() -> None:
    from motac.cli import app

    assert app is not None
