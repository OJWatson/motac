from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from motac.cli import app


def test_cli_data_chicago_load_emits_meta(tmp_path: Path) -> None:
    fixtures = Path(__file__).resolve().parent / "fixtures" / "chicago"
    y_path = fixtures / "y_obs_small.csv"

    cfg_path = tmp_path / "chicago_raw.json"
    cfg_path.write_text(json.dumps({"path": str(y_path)}))

    runner = CliRunner()
    result = runner.invoke(app, ["data", "chicago-load", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.stdout

    payload = json.loads(result.stdout)
    assert payload["meta"]["n_locations"] == 2
    assert payload["meta"]["n_steps"] == 4
    assert payload["meta"]["mobility_source"] == "identity"
    assert payload["y_obs_shape"] == [2, 4]
