# Neural kernels API

The neural-kernel scaffold lives in `motac.model.neural_kernels`.

This milestone is intentionally small: it establishes a stable import path and a
minimal kernel interface that later nonparametric / learned variants can build
on.

## Contract validation helper

The `validate_kernel_fn` helper exists to fail fast if a kernel implementation
violates the minimal (v1) contract:

- input: a nonnegative `numpy.ndarray` of distances / travel-times `d`
- output: a `numpy.ndarray` of the **same shape** with **finite, nonnegative**
  weights

Typical use is in downstream modules (or your own experiments) at import-time or
construction-time:

```python
from motac.model.neural_kernels import ExpDecayKernel, validate_kernel_fn

kernel = ExpDecayKernel(lengthscale=1.5)
validate_kernel_fn(kernel)
```

```{eval-rst}
.. automodule:: motac.model.neural_kernels
   :members:
   :undoc-members:
   :show-inheritance:
```
