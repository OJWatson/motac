# Parameter recovery (road Hawkes, M3)

`motac` includes a small **parameter recovery harness** for the parametric
road-constrained Hawkes Poisson fitter.

The goal is not perfect recovery on a tiny synthetic problem (which is noisy),
but a **regression test** that the end-to-end loop

> simulate → fit → compare parameters

stays in a reasonable ballpark across multiple random seeds.

## What is tested

On a tiny 3-cell road substrate, we simulate counts from:

- baseline rates `mu` (per cell)
- excitation scale `alpha`
- travel-time decay `beta`
- fixed discrete lag kernel `g(ℓ)`

Then we fit `(mu, alpha, beta)` by MLE (Poisson family) and check:

- the optimiser improves the log-likelihood vs initialisation
- median absolute errors across seeds stay below tolerant thresholds
- most seeds succeed (to reduce CI flakiness)

## Running locally

The harness is exercised by the unit test:

```bash
uv run pytest -q tests/test_parameter_recovery_m3.py
```

The underlying helper is exposed as:

- `motac.model.run_parameter_recovery_road_hawkes_poisson`

(see `motac.model.validation`).

## Notes on tolerances

Recovery on small simulated datasets is stochastic.
The test intentionally uses:

- **multi-seed** evaluation
- **median** error checks
- a minimum success count (e.g. 4/5 seeds)

so that minor numerical drift does not cause flaky CI failures while still
catching real regressions.
