# motac

## Quickstart (simulation + fit + predict)

This repo currently contains a lightweight discrete-time simulator and parametric
Poisson Hawkes utilities.

### Predict API

Given a fitted (or known) set of Hawkes parameters and a history of counts, you
can forecast intensities (conditional Poisson means):

- `predict_hawkes_intensity_one_step(...)` returns \(\lambda(t)\) for the next time step.
- `predict_hawkes_intensity_multi_step(...)` rolls forward deterministically using
  expected counts: it sets future \(y(t) := \lambda(t)\).

### CLI: kernel fitting / observed fit (quick QA)

If you have a simulation saved via `save_simulation_parquet`, you can fit
`(mu, alpha, beta)` with an exponential kernel:

```bash
motac sim fit-kernel --parquet sim.parquet --n-lags 6
```

For simulator output with detection + clutter enabled, you can also fit
`(mu, alpha)` from `y_obs` using a Poisson approximation:

```bash
motac sim fit-observed --parquet sim.parquet
```

Both commands print JSON with fitted parameters and log-likelihood diagnostics.

Milestones
----------

## M4: Predict API (parametric Hawkes)

This project includes a simple discrete-time, network-coupled Hawkes-like count model.

### One-step-ahead intensity forecast

Given a history of counts `y_history` with shape `(n_locations, n_steps_history)`,
compute the next-step intensity (Poisson mean):

```python
import numpy as np
from motac.sim import (
    HawkesDiscreteParams,
    discrete_exponential_kernel,
    generate_random_world,
    predict_hawkes_intensity_one_step,
)

world = generate_random_world(n_locations=5, seed=0, lengthscale=0.5)
params = HawkesDiscreteParams(
    mu=np.full((world.n_locations,), 0.2),
    alpha=0.7,
    kernel=discrete_exponential_kernel(n_lags=4, beta=0.8),
)

y_history = np.zeros((world.n_locations, 10), dtype=int)
lam_next = predict_hawkes_intensity_one_step(world=world, params=params, y_history=y_history)
```

### Multi-step intensity forecast

A deterministic multi-step forecast is produced by rolling the recursion forward
and substituting expected counts for future unknown draws (i.e., setting `y(t) = lambda(t)`
inside the forecast loop):

```python
from motac.sim import predict_hawkes_intensity_multi_step

lam = predict_hawkes_intensity_multi_step(
    world=world,
    params=params,
    y_history=y_history,
    horizon=7,
)
assert lam.shape == (world.n_locations, 7)
```

## Uncertainty workflow (sampling + summaries)

The deterministic `predict_*` helpers are useful for a quick forecast, but to get
uncertainty you can sample predictive paths and then summarize them.

### A) Latent model: fit on `y_true` + sample `(y_true, y_obs)` forward

```python
import numpy as np
from motac.sim import (
    HawkesDiscreteParams,
    discrete_exponential_kernel,
    fit_hawkes_mle_alpha_mu_beta,
    generate_random_world,
    sample_hawkes_predictive_paths,
    summarize_predictive_paths,
    simulate_hawkes_counts,
)

world = generate_random_world(n_locations=5, seed=0, lengthscale=0.5)

# Simulate some data (or replace with your own latent counts).
params_true = HawkesDiscreteParams(
    mu=np.full((world.n_locations,), 0.1),
    alpha=0.6,
    kernel=discrete_exponential_kernel(n_lags=6, beta=1.0),
    p_detect=0.7,
    false_rate=0.2,
)
out = simulate_hawkes_counts(world=world, params=params_true, n_steps=200, seed=1)

# Fit mu/alpha/beta from latent counts.
fit = fit_hawkes_mle_alpha_mu_beta(world=world, n_lags=6, y=out["y_true"])
params_hat = HawkesDiscreteParams(
    mu=np.asarray(fit["mu"], dtype=float),
    alpha=float(fit["alpha"]),
    kernel=np.asarray(fit["kernel"], dtype=float),
    p_detect=params_true.p_detect,
    false_rate=params_true.false_rate,
)

# Sample predictive paths forward from a history window.
y_hist = out["y_true"][:, :50]
paths = sample_hawkes_predictive_paths(
    world=world, params=params_hat, y_history=y_hist, horizon=30, n_paths=200, seed=123
)
summary = summarize_predictive_paths(paths=paths["y_obs"], q=(0.05, 0.5, 0.95))
# summary["mean"] has shape (n_locations, horizon)
```

### B) Observed model (approx): fit on `y_obs` + sample `y_obs` forward

The observation model in the simulator is
`y_obs = Binomial(y_true, p_detect) + Poisson(false_rate)`.
A cheap approximation treats `y_obs(t) ~ Poisson(p_detect*lambda(t) + false_rate)`.

```python
from motac.sim import (
    fit_hawkes_mle_alpha_mu_observed_poisson_approx,
    sample_hawkes_observed_predictive_paths_poisson_approx,
)

fit_obs = fit_hawkes_mle_alpha_mu_observed_poisson_approx(
    world=world,
    kernel=params_hat.kernel,
    y_true_for_history=out["y_true"],
    y_obs=out["y_obs"],
    p_detect=params_true.p_detect,
    false_rate=params_true.false_rate,
)

obs_paths = sample_hawkes_observed_predictive_paths_poisson_approx(
    world=world,
    mu=np.asarray(fit_obs["mu"], dtype=float),
    alpha=float(fit_obs["alpha"]),
    kernel=params_hat.kernel,
    y_history_for_intensity=out["y_true"][:, :50],
    horizon=30,
    n_paths=200,
    seed=123,
    p_detect=params_true.p_detect,
    false_rate=params_true.false_rate,
)
obs_summary = summarize_predictive_paths(paths=obs_paths["y_obs"], q=(0.05, 0.5, 0.95))
```
