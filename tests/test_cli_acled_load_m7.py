from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from motac.cli import app


def test_cli_data_acled_load_emits_meta(tmp_path: Path) -> None:
    fixtures = Path(__file__).resolve().parent / "fixtures" / "acled"
    csv_path = fixtures / "acled_small.csv"

    cfg_path = tmp_path / "acled_events.json"
    cfg_path.write_text(json.dumps({"path": str(csv_path)}))

    runner = CliRunner()
    result = runner.invoke(app, ["data", "acled-load", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.stdout

    payload = json.loads(result.stdout)
    assert payload["meta"]["n_locations"] == 2
    assert payload["meta"]["n_days"] == 2
    assert payload["meta"]["mobility_source"] == "identity"
    assert payload["y_obs_shape"] == [2, 2]
