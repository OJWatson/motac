# Simulation and Hawkes utilities

## Observed-data likelihoods: exact vs Poisson approximation

`motac` includes two likelihoods for *observed* counts under the simulator’s
thinning+clutter observation model:

- **Exact conditional likelihood**: `motac.sim.hawkes_loglik_observed_exact`
  computes **p(y_obs | y_true)** exactly (by summing out the unobserved detected
  count `y_det`). This is *conditional on knowing the latent true counts*
  `y_true` (and therefore does **not** directly depend on the Hawkes intensity).

- **Poisson-approx likelihood**: `motac.sim.hawkes_loglik_poisson_observed`
  uses the approximation **y_obs(t) ~ Poisson(p_detect * λ(t) + false_rate)**,
  where `λ(t)` is the latent Hawkes intensity. This is cheap and convenient, but
  it is *not* the exact marginal likelihood for real observed-only data because
  `λ(t)` depends on the unobserved latent history.

Interpretation / expected gaps

- The “exact” function is exact for the **observation model given `y_true`**.
  In real data you typically do not have `y_true`, so this exact score is mainly
  useful for simulator QA / controlled experiments.

- The Poisson approximation can be motivated by the standard thinning identity:
  if `y_true(t) | λ(t) ~ Poisson(λ(t))` and `y_det(t) | y_true(t) ~ Binomial(y_true(t), p_detect)`,
  then marginally `y_det(t) | λ(t) ~ Poisson(p_detect * λ(t))`. However, in a
  Hawkes model `λ(t)` depends on the (latent) past, so using `y_obs` (or expected
  observed counts) as a proxy history is misspecified.

- See `motac.sim.compare_observed_loglik_exact_vs_poisson_approx` for a small
  comparison harness used in tests.

```{eval-rst}
.. automodule:: motac.sim
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: motac.sim.hawkes
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: motac.sim.fit
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: motac.sim.predictive
   :members:
   :undoc-members:
   :show-inheritance:
```
