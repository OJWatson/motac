from __future__ import annotations


def test_configs_import_path_exists() -> None:
    import motac.configs as cfg

    assert hasattr(cfg, "__all__")
