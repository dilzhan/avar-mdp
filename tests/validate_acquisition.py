"""Mahalanobis kNN fails clearly on one observation and works on valid input."""

import jax.numpy as jnp

from src.acquisition import knn_distances_and_indices

candidates = jnp.array([[0.0, -1.0], [1.0, -0.5]])

try:
    knn_distances_and_indices(candidates, jnp.array([[0.2, -0.8]]), k=1)
except ValueError:
    print("One-observation rejection: PASSED")
else:
    raise AssertionError("Expected ValueError for only one rollout observation")

rollouts = jnp.array([
    [0.0, -1.0],
    [0.5, -0.7],
    [1.0, -1.2],
])
distances, indices = knn_distances_and_indices(candidates, rollouts, k=2)
assert jnp.asarray(distances).shape == (2, 2)
assert jnp.asarray(indices).shape == (2, 2)
assert bool(jnp.all(jnp.isfinite(jnp.asarray(distances))))
print("Regularized covariance and kNN: PASSED")
