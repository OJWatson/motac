from __future__ import annotations


def test_marked_hawkes_scaffold_import_path_stable() -> None:
    # Import both the module path and the package re-export.
    from motac.model import MarkedRoadHawkesDataset  # noqa: F401
    from motac.model.marked_hawkes import MarkedRoadHawkesDataset as _  # noqa: F401
