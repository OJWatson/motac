from motac.bench import BenchmarkResult
from motac.cli import run_benchmarks


def test_run_benchmarks_output(monkeypatch, capsys):
    fake_results = [
        BenchmarkResult("conv", 1.0, 2.0, 3.0),
        BenchmarkResult("edges", 0.5, 1.5, 2.5),
    ]

    monkeypatch.setattr(
        "motac.cli.compare_backends", lambda **_: fake_results)

    run_benchmarks()
    out = capsys.readouterr().out

    assert "backend\tsim_s\tfit_s\tforecast_s" in out
    assert "conv\t1.000\t2.000\t3.000" in out
    assert "edges\t0.500\t1.500\t2.500" in out


def test_run_benchmarks_argparse_passthrough(monkeypatch):
    seen: dict[str, object] = {}

    def fake_compare_backends(**kwargs):
        seen.update(kwargs)
        return [BenchmarkResult("conv", 1.0, 2.0, 3.0), BenchmarkResult("edges", 0.5, 1.5, 2.5)]

    monkeypatch.setattr(
        "motac.cli.compare_backends", fake_compare_backends)

    run_benchmarks(["--grid-h", "12", "--grid-w", "13",
                   "--time-steps", "9", "--fit-steps", "7", "--seed", "42"])

    assert seen["grid_shape"] == (12, 13)
    assert seen["time_steps"] == 9
    assert seen["fit_steps"] == 7
    assert seen["seed"] == 42
