from motac.bench import (
    BenchmarkResult,
    benchmark_backend,
    benchmarks_to_frame,
    compare_backends,
    save_benchmarks_csv,
)


def test_benchmark_backend_smoke():
    result = benchmark_backend(grid_shape=(
        8, 8), time_steps=8, fit_steps=6, backend="conv", seed=0)
    assert result.backend == "conv"
    assert result.simulation_seconds >= 0.0
    assert result.fit_seconds >= 0.0
    assert result.forecast_seconds >= 0.0


def test_compare_backends_returns_two_results():
    results = compare_backends(grid_shape=(
        6, 6), time_steps=6, fit_steps=4, seed=2)
    names = {r.backend for r in results}
    assert names == {"conv", "edges"}


def test_benchmark_serialization_helpers(tmp_path):
    results = [
        BenchmarkResult("conv", 1.0, 2.0, 3.0),
        BenchmarkResult("edges", 0.5, 1.5, 2.5),
    ]

    df = benchmarks_to_frame(results)
    assert list(df.columns) == [
        "backend", "simulation_seconds", "fit_seconds", "forecast_seconds"]
    assert df.shape == (2, 4)

    out = save_benchmarks_csv(results, tmp_path / "bench.csv")
    assert out.exists()
