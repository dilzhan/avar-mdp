import jax.numpy as jnp
from jax.scipy.stats.norm import cdf
from jaxtyping import Array, Float, Int
from sklearn.neighbors import BallTree


def knn_distances_and_indices(
    candidates: Float[Array, "n_candidates 2"],
    rollout_states: Float[Array, "n_rollouts 2"],
    k: int = 20,
    leaf_size: int = 40,
) -> tuple[Float[Array, "n_candidates k"], Int[Array, "n_candidates k"]]:
    def mahalanobis_transform(
        candidates: Float[Array, "n_candidates 2"], states: Float[Array, "n_rollouts 2"]
    ) -> tuple[Float[Array, "n_candidates 2"], Float[Array, "n_rollouts 2"]]:
        import jax

        mean = jnp.mean(states, axis=0)
        cov = jnp.cov(states.T) + 1e-6 * jnp.eye(2)
        L = jax.scipy.linalg.cholesky(cov, lower=True)

        states_scaled = jax.scipy.linalg.solve_triangular(
            L, (states - mean).T, lower=True
        ).T
        candidates_scaled = jax.scipy.linalg.solve_triangular(
            L, (candidates - mean).T, lower=True
        ).T

        return candidates_scaled, states_scaled

    if rollout_states.shape[0] < 2:
        raise ValueError("Mahalanobis kNN requires at least two rollout states.")

    candidates_scaled, states_scaled = mahalanobis_transform(candidates, rollout_states)

    tree = BallTree(states_scaled, leaf_size=leaf_size, metric="euclidean")
    dist, idx = tree.query(candidates_scaled, k=k)

    dist = jnp.asarray(dist)
    idx = jnp.asarray(idx)

    return dist, idx


def decision_score(
    q_mean: Float[Array, "n_candidates n_actions"],
    q_std: Float[Array, "n_candidates n_actions"],
) -> Float[Array, " n_candidates"]:
    def two_best(
        means: Float[Array, "n_candidates n_actions"],
        stds: Float[Array, "n_candidates n_actions"],
    ) -> tuple[
        Float[Array, "n_candidates 2"],
        Float[Array, "n_candidates 2"],
        Int[Array, "n_candidates 2"],
    ]:
        indices = jnp.argsort(means, axis=1)[:, :2]

        best_means = jnp.take_along_axis(means, indices, axis=1)
        best_stds = jnp.take_along_axis(stds, indices, axis=1)

        return best_means, best_stds, indices

    best_means, best_stds, _ = two_best(q_mean, q_std)

    delta_mean = best_means[:, 1] - best_means[:, 0]
    delta_std = jnp.sqrt(best_stds[:, 0] ** 2 + best_stds[:, 1] ** 2)

    return 2.0 * cdf(-delta_mean / (delta_std + 1e-6))


def reachability_score(
    knn_distances: Float[Array, "n_candidates k"],
) -> Float[Array, " n_candidates"]:
    radius = knn_distances[:, -1]
    score = 1.0 / (radius**2 + 1e-6)

    score /= score.max() + 1e-6

    return score


def tail_score(
    knn_indices: Int[Array, "n_canndidates k"], tail_flags: Int[Array, " n_rollouts"]
) -> Float[Array, " n_candidates"]:
    knn_flags = tail_flags[knn_indices]
    score = jnp.mean(knn_flags, axis=1)

    return score
