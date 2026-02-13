from __future__ import annotations

import json

from click.testing import CliRunner
from typer.main import get_command

from motac.cli import app


def test_paper_generate_artifacts_writes_json_and_manifest(tmp_path) -> None:
    out_dir = tmp_path / "artifacts"

    runner = CliRunner()
    res = runner.invoke(
        get_command(app),
        ["paper", "generate-artifacts", "--out-dir", str(out_dir), "--seed", "7"],
    )

    assert res.exit_code == 0, res.stdout
    # Command prints the written path.
    path = res.stdout.strip()
    assert path != ""

    payload = json.loads((out_dir / "synthetic_eval_seed7.json").read_text())
    assert set(payload.keys()) == {"config", "fit", "forecasts", "metrics"}

    manifest = json.loads((out_dir / "synthetic_eval_seed7.manifest.json").read_text())
    assert set(manifest.keys()) == {
        "artifact",
        "gitSha",
        "seed",
        "configSummary",
        "generatedAtUtc",
    }
    assert manifest["artifact"] == "synthetic_eval_seed7.json"
    assert manifest["seed"] == 7
    assert isinstance(manifest["gitSha"], str)
    assert manifest["gitSha"] != ""
    assert manifest["configSummary"] == payload["config"]
