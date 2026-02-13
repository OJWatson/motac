from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

def _git_head_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(['git','rev-parse','HEAD'], cwd=str(repo_root))
        return out.decode().strip()
    except Exception:
        return 'UNKNOWN'

def make_run_dir(base_dir: str | Path, *, seed: int, cfg: dict[str, Any], repo_root: str | Path) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    head = _git_head_sha(Path(repo_root))
    payload = json.dumps({'seed': seed, 'cfg': cfg, 'head': head}, sort_keys=True).encode()
    h = hashlib.sha256(payload).hexdigest()[:16]
    run = base / f'run_{h}'
    run.mkdir(parents=True, exist_ok=True)
    meta = {'seed': seed, 'git_head': head, 'cfg': cfg, 'run_id': h}
    (run / 'metadata.json').write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return run
