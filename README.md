# motac

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
