from __future__ import annotations


def test_eval_import_path_stable() -> None:
    # Ensure motac.eval is a package and exposes expected symbols.
    import motac.eval as me

    assert hasattr(me, "EvalConfig")
    assert hasattr(me, "evaluate_synthetic")
