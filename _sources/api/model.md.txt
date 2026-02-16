# Model API

## Marked Hawkes scaffolding (M13)

The marked-Hawkes scaffold lives in `motac.model.marked_hawkes`.

### Categorical marks contract (v1)

For CI-safe unit tests and a stable, minimal API, marks are represented as an
**integer-coded categorical label matrix** aligned to the binned observation
matrix `y_obs`:

- `marks.shape == y_obs.shape == (n_cells, n_steps)`
- `marks` has an integer dtype
- `marks` is non-negative
- optionally, a category bound can be enforced via `n_marks` such that
  `0 <= marks < n_marks`

Use `validate_categorical_marks_matrix(marks, y_obs=..., n_marks=...)` to enforce
this contract.

### One-hot encoding helper

`encode_categorical_marks_onehot(marks, y_obs=..., n_marks=...)` produces a
one-hot tensor of shape `(n_cells, n_steps, n_marks)`.

```{eval-rst}
.. automodule:: motac.model.marked_hawkes
   :members:
   :undoc-members:
   :show-inheritance:
```
