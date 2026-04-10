import jax.numpy as jnp

from motac.connectivity import (
    ConvStencil,
    EdgeList,
    apply_conv_stencil,
    apply_edgelist,
)


def test_recurrence_update_matches_manual():
    rho = jnp.array([[0.5, 0.9]], dtype=jnp.float32)  # [M=1,B=2]
    h_prev = jnp.array([[[2.0, 4.0]]], dtype=jnp.float32)  # [J=1,M=1,B=2]
    y_prev = jnp.array([[3.0]], dtype=jnp.float32)  # [J=1,M=1]

    h_next = rho[None, :, :] * h_prev + y_prev[:, :, None]
    expected = jnp.array([[[4.0, 6.6]]], dtype=jnp.float32)
    assert jnp.allclose(h_next, expected)


def test_conv_and_edgelist_equivalence_on_identity_kernel():
    x = jnp.arange(9, dtype=jnp.float32).reshape(3, 3, 1)

    identity_kernel = jnp.zeros((3, 3), dtype=jnp.float32).at[1, 1].set(1.0)
    conv = ConvStencil(kernel=identity_kernel, normalise=False)
    out_conv = apply_conv_stencil(x, conv).reshape(9, 1)

    src = jnp.arange(9, dtype=jnp.int32)
    dst = jnp.arange(9, dtype=jnp.int32)
    w = jnp.ones((9,), dtype=jnp.float32)
    identity_edges = EdgeList(
        src=src, dst=dst, weight=w, num_nodes=9, distance=None, meta={})
    out_edge = apply_edgelist(x.reshape(9, 1), identity_edges)

    assert jnp.allclose(out_conv, out_edge)
