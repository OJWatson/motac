# Benchmarks

Use `motac.bench` to compare conv vs edge-list backends.

```python
from motac.bench import compare_backends

results = compare_backends(grid_shape=(40, 40), time_steps=30, fit_steps=40, seed=0)
for r in results:
    print(r)

from motac.bench import benchmarks_to_frame, save_benchmarks_csv
df = benchmarks_to_frame(results)
save_benchmarks_csv(results, "artifacts/benchmarks.csv")
```

For CI safety and local reproducibility, keep benchmark dimensions modest by default and scale up manually.

You can also run the provided helper script:

```bash
python3.11 scripts/run_bench.py
```

Or, after installation, use the packaged CLI entrypoint:

```bash
motac-bench

# configurable run
motac-bench --grid-h 60 --grid-w 60 --time-steps 45 --fit-steps 60 --seed 7
```
