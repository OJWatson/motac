import jax.numpy as jnp
import numpy as np

from motac.data import CountsTensor, GridSpec


class DummyConnectivity:
    pass


def test_counts_tensor_shape_contract_and_splits():
    y = jnp.zeros((20, 16, 2), dtype=jnp.float32)
    data = CountsTensor(
        y=y,
        t0=np.datetime64("2020-01-01"),
        dt_days=1,
        num_time=20,
        num_nodes=16,
        marks=("a", "b"),
        node_coords=None,
        grid=GridSpec(shape=(4, 4)),
        connectivity=DummyConnectivity(),
        covariates={},
    )

    splits = data.rolling_origin_splits(horizon=5, step=5, min_train=10)
    assert len(splits) == 2
    assert splits[0].train.y.shape == (10, 16, 2)
    assert splits[0].test.y.shape == (5, 16, 2)
