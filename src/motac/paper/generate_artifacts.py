from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from motac.eval import EvalConfig, evaluate_synthetic


def generate_synthetic_eval_artifact(*, out_dir: Path, seed: int = 0) -> Path:
    """Generate a small JSON artifact for the synthetic evaluation.

    This is intended to be CI-safe (no downloads) and fast.

    Returns
    -------
    Path
        The written JSON path.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = EvalConfig(seed=int(seed))
    out = evaluate_synthetic(cfg)

    # Ensure JSON serializable arrays.
    fit = out["fit"]
    fit_ser = {
        **fit,
        "mu": list(map(float, fit["mu"])),
    }

    forecasts = out["forecasts"]
    forecasts_ser = {
        "q": list(map(float, forecasts["q"])),
        "y_true_mean": [[float(x) for x in row] for row in forecasts["y_true_mean"]],
        "y_true_quantiles": [
            [[float(x) for x in row] for row in qmat]
            for qmat in forecasts["y_true_quantiles"]
        ],
    }

    payload = {
        "config": asdict(cfg) | {"q": list(cfg.q)},
        "fit": fit_ser,
        "forecasts": forecasts_ser,
        "metrics": out["metrics"],
    }

    path = out_dir / f"synthetic_eval_seed{seed}.json"
    path.write_text(json.dumps(payload))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate motac paper artifacts")
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for JSON artifacts (created if missing).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    generate_synthetic_eval_artifact(out_dir=Path(args.out_dir), seed=int(args.seed))


if __name__ == "__main__":
    main()
