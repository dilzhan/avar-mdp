import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Int

# Performance metrics


def mean_cost(costs: Float[Array, "*batch"]) -> Float[Array, ""]:
    return jnp.mean(costs)


def value_mae(
    values: Float[Array, "*batch"], values_ref: Float[Array, "*batch"]
) -> Float[Array, ""]:
    return jnp.mean(jnp.abs(values - values_ref))


def value_rmse(
    values: Float[Array, "*batch"], values_ref: Float[Array, "*batch"]
) -> Float[Array, ""]:
    return jnp.sqrt(jnp.mean((values - values_ref) ** 2))


def value_linf(
    values: Float[Array, "*batch"], values_ref: Float[Array, "*batch"]
) -> Float[Array, ""]:
    return jnp.max(jnp.abs(values - values_ref))


def conditional_mean(
    rows: list[dict],
    value_key: str,
    flag_key: str,
) -> float:
    values = [row[value_key] for row in rows if row[flag_key]]
    return float("nan") if not values else float(jnp.mean(jnp.asarray(values)))


def mean_and_se(values: ArrayLike) -> tuple[float, float]:
    values = jnp.asarray(values, dtype=float)
    mean = float(jnp.mean(values))

    if values.size < 2:
        return mean, jnp.nan

    se = float(jnp.std(values, ddof=1) / jnp.sqrt(values.size))
    return mean, se


# Decision metrics


def policy_error(
    actions: Float[Array, "*batch"], actions_ref: Float[Array, "*batch"]
) -> Float[Array, ""]:
    return jnp.mean(jnp.asarray(actions != actions_ref))


def action_gap(q_values: Float[Array, "*batch n_actions"]) -> Float[Array, "*batch"]:
    q_values_sorted = jnp.sort(q_values, axis=-1)
    return q_values_sorted[..., 1] - q_values_sorted[..., 0]


def action_regret(
    q_values: Float[Array, "*batch n_actions"],
    action_idx: Int[Array, "*batch"],
) -> Float[Array, "*batch"]:
    """Regret of the selected action relative to the optimal action."""
    q_selected = jnp.take_along_axis(
        q_values,
        action_idx[..., None],
        axis=-1,
    )[..., 0]

    q_optimal = jnp.min(q_values, axis=-1)

    return q_selected - q_optimal


# Q-function metrics


def q_mae(
    q_values: Float[Array, "*batch n_actions"],
    q_values_ref: Float[Array, "*batch n_actions"],
) -> Float[Array, "*batch"]:
    return jnp.mean(jnp.abs(q_values - q_values_ref), axis=-1)


def q_rmse(
    q_values: Float[Array, "*batch n_actions"],
    q_values_ref: Float[Array, "*batch n_actions"],
) -> Float[Array, "*batch"]:
    return jnp.sqrt(jnp.mean((q_values - q_values_ref) ** 2, axis=-1))


def q_linf(
    q_values: Float[Array, "*batch n_actions"],
    q_values_ref: Float[Array, "*batch n_actions"],
) -> Float[Array, "*batch"]:
    return jnp.max(jnp.abs(q_values - q_values_ref), axis=-1)
