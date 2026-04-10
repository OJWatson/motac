import jax.numpy as jnp

from motac.eval import (
    aggregate_daily_totals,
    aggregate_mark_totals,
    coverage,
    hotspot_recall,
    score_crps_counts,
    score_log_likelihood_nb2,
)


def test_metrics_shapes_and_ranges():
    y_true = jnp.array([[[1.0], [0.0]], [[2.0], [1.0]]],
                       dtype=jnp.float32)  # [H=2,J=2,M=1]
    y_samples = jnp.stack(
        [y_true, y_true + 1.0, y_true * 0.5], axis=0)  # [S,H,J,M]

    ll = score_log_likelihood_nb2(y_true, y_samples)
    crps = score_crps_counts(y_true, y_samples)

    q_lo = jnp.quantile(y_samples, 0.05, axis=0)
    q_hi = jnp.quantile(y_samples, 0.95, axis=0)
    cov = coverage(y_true, q_lo, q_hi)
    hr = hotspot_recall(y_true, jnp.mean(y_samples, axis=0), k=2)

    assert isinstance(ll, float)
    assert isinstance(crps, float)
    assert 0.0 <= cov <= 1.0
    assert 0.0 <= hr <= 1.0


def test_aggregation_helpers():
    y = jnp.array(
        [
            [[1.0, 0.0], [0.0, 2.0]],
            [[3.0, 1.0], [1.0, 0.0]],
        ],
        dtype=jnp.float32,
    )  # [T=2,J=2,M=2]

    daily = aggregate_daily_totals(y)
    marks = aggregate_mark_totals(y)

    assert list(daily.columns) == ["t", "total", "mark_0", "mark_1"]
    assert daily.shape[0] == 2
    assert marks.shape[0] == 2
    assert float(daily["total"].sum()) == float(y.sum())


def test_hotspot_recall_is_spatial_not_time_flattened():
    # True hotspot is node 1 at horizon 0; prediction puts mass on node 1 at horizon 1.
    # Time-flattened hotspot logic would incorrectly count this as a miss.
    y_true = jnp.array(
        [
            [[0.0], [10.0]],
            [[0.0], [0.0]],
        ],
        dtype=jnp.float32,
    )
    y_mean = jnp.array(
        [
            [[9.0], [0.0]],
            [[0.0], [10.0]],
        ],
        dtype=jnp.float32,
    )
    # Spatially aggregated, both rank node 1 highest.
    assert hotspot_recall(y_true, y_mean, k=1) == 1.0


def test_crps_blockwise_matches_reference_on_small_tensor():
    y_true = jnp.array([[[1.0], [0.0]]], dtype=jnp.float32)
    y_samples = jnp.array(
        [
            [[[1.0], [0.0]]],
            [[[2.0], [1.0]]],
            [[[0.0], [1.0]]],
        ],
        dtype=jnp.float32,
    )

    ref_s1 = jnp.mean(jnp.abs(y_samples - y_true[None, ...]))
    ref_s2 = jnp.mean(
        jnp.abs(y_samples[:, None, ...] - y_samples[None, :, ...]))
    ref = float(ref_s1 - 0.5 * ref_s2)

    got = score_crps_counts(y_true, y_samples, pairwise_block_size=2)
    assert abs(got - ref) < 1e-8
