from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding='utf-8'))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError('config must be a mapping')
    return data
