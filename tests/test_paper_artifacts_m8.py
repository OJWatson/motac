from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_generate_artifacts_script_runs_and_writes_json(tmp_path: Path) -> None:
    out_dir = tmp_path / "artifacts"

    subprocess.check_call(
        [
            "python",
            "-m",
            "motac.paper.generate_artifacts",
            "--out-dir",
            str(out_dir),
            "--seed",
            "0",
        ]
    )

    path = out_dir / "synthetic_eval_seed0.json"
    assert path.exists()

    payload = json.loads(path.read_text())
    assert set(payload.keys()) == {"config", "fit", "forecasts", "metrics"}

    for k in ["nll_test", "rmse", "mae"]:
        assert k in payload["metrics"]
