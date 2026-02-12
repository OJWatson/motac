from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AcledEventsCsvConfig:
    """Configuration for the ACLED events CSV loader (v1 placeholder schema).

    This is intentionally minimal: it exists to provide a stable CLI entry point
    for loading and aggregating ACLED-style event rows into a daily count matrix.
    """

    path: str
    mobility_path: str | None = None
    value: str = "events"

    @staticmethod
    def from_json(path: str | Path) -> AcledEventsCsvConfig:
        data = json.loads(Path(path).read_text())
        return AcledEventsCsvConfig(**data)
