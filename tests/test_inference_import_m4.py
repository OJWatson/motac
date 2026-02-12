from __future__ import annotations


def test_inference_import_path_exists() -> None:
    import motac.inference as inf

    assert hasattr(inf, "__all__")
