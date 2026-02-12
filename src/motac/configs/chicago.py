from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ChicagoRawConfig:
    """Configuration for the Chicago loader raw on-disk contract (v1).

    This is intentionally minimal: it exists to provide a stable CLI entry point
    for loading the raw observed-count matrix.
    """

    path: str
    mobility_path: str | None = None

    @staticmethod
    def from_json(path: str | Path) -> ChicagoRawConfig:
        data = json.loads(Path(path).read_text())
        return ChicagoRawConfig(**data)
