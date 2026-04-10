import jax.numpy as jnp
import numpy as np

from motac.connectivity import (
    apply_edgelist,
    make_grid_edgelist,
    make_grid_stencil,
)
from motac.data import GridSpec


def test_stencil_normalises_to_one():
    stencil = make_grid_stencil(
        kind="gaussian", size=5, sigma=1.2, normalise=True)
    assert abs(float(jnp.sum(stencil.kernel)) - 1.0) < 1e-5


def test_edgelist_row_sums_close_to_one():
    grid = GridSpec(shape=(4, 4))
    edges = make_grid_edgelist(
        grid, neighbourhood="moore", normalise_rows=True)
    row_sums = jnp.zeros((edges.num_nodes,),
                         dtype=jnp.float32).at[edges.src].add(edges.weight)
    assert float(jnp.max(jnp.abs(row_sums - 1.0))) < 1e-4


def test_apply_edgelist_shape():
    grid = GridSpec(shape=(3, 3))
    edges = make_grid_edgelist(grid)
    x = jnp.ones((9, 2), dtype=jnp.float32)
    out = apply_edgelist(x, edges)
    assert out.shape == x.shape


def test_undirected_edgelist_has_no_duplicate_rows():
    grid = GridSpec(shape=(4, 4))
    edges = make_grid_edgelist(
        grid,
        neighbourhood="von_neumann",
        directed=False,
        normalise_rows=False,
    )
    pairs = list(zip(np.asarray(edges.src), np.asarray(edges.dst)))
    assert len(pairs) == len(set(pairs))
