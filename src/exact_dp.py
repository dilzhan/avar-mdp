from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .model import ACTIONS, NOISE, PROBABILITIES, cost, terminal_cost, transition


def _batched_policy(scalar_policy: Callable) -> Callable:
    def policy(t, x, y):
        x, y = jnp.broadcast_arrays(
            jnp.asarray(x),
            jnp.asarray(y),
        )

        if x.ndim == 0:
            return scalar_policy(t, x, y)

        states = jnp.stack(
            (x.reshape(-1), y.reshape(-1)),
            axis=-1,
        )

        actions = jax.lax.map(
            lambda state: scalar_policy(t, state[0], state[1]),
            states,
        )

        return actions.reshape(x.shape)

    return policy


def _backup_counts(
    horizon: int,
    n_actions: int,
    n_noise: int,
) -> Int[Array, " horizon+1"]:

    if horizon < 0 or n_actions < 1 or n_noise < 1:
        raise ValueError("Invalid horizon, actions, or noise size.")

    max_int64 = (1 << 63) - 1
    branching = n_actions * n_noise

    counts = [0] * (horizon + 1)

    for t in range(horizon - 1, -1, -1):
        counts[t] = n_actions + branching * counts[t + 1]

        if counts[t] > max_int64:
            raise OverflowError(
                f"Backup count exceeds int64 at t={t}, " f"horizon={horizon}."
            )

    return jnp.asarray(counts, dtype=jnp.int64)


def risk_neutral_dp(
    horizon: int,
    actions: Float[Array, " n_actions"] = ACTIONS,
    noise: Float[Array, " n_samples"] = NOISE,
    probabilities: Float[Array, " n_samples"] = PROBABILITIES,
) -> tuple[Callable, Callable, Callable]:

    actions = jnp.asarray(actions)
    noise = jnp.asarray(noise)
    probabilities = jnp.asarray(probabilities)

    def value(t: int, x: Float[Array, ""]) -> Float[Array, ""]:
        if t == horizon:
            return terminal_cost(x)
        return jnp.min(q_values(t, x))

    def q_values(t: int, x: Float[Array, ""]) -> Float[Array, " n_actions"]:
        return jax.vmap(
            lambda a: cost(x, a)
            + jnp.sum(
                probabilities
                * jax.vmap(lambda eps: value(t + 1, transition(x, a, eps)))(noise)
            )
        )(actions)

    def scalar_policy(t, x, y):
        del y
        return actions[jnp.argmin(q_values(t, x))]

    batched_policy = _batched_policy(scalar_policy)

    def policy(t, x, y=None):
        if y is None:
            y = jnp.zeros_like(jnp.asarray(x))
        return batched_policy(t, x, y)

    backup_counts = _backup_counts(
        horizon,
        int(actions.shape[0]),
        int(noise.shape[0]),
    )

    def backup_count(t: int, x: None | Float[Array, ""] = None) -> Int[Array, ""]:
        return backup_counts[t]

    return value, policy, backup_count


def avar_dp(
    horizon: int,
    actions: Float[Array, " n_actions"] = ACTIONS,
    noise: Float[Array, " n_samples"] = NOISE,
    probabilities: Float[Array, " n_samples"] = PROBABILITIES,
) -> tuple[Callable, Callable, Callable]:

    actions = jnp.asarray(actions)
    noise = jnp.asarray(noise)
    probabilities = jnp.asarray(probabilities)

    def value(t: int, x: Float[Array, ""], y: Float[Array, ""]) -> Float[Array, ""]:
        if t == horizon:
            return jnp.maximum(y + terminal_cost(x), 0.0)
        return jnp.min(q_values(t, x, y))

    def q_values(
        t: int, x: Float[Array, ""], y: Float[Array, ""]
    ) -> Float[Array, " n_actions"]:
        return jax.vmap(
            lambda a: jnp.sum(
                probabilities
                * jax.vmap(
                    lambda eps: value(
                        t + 1,
                        transition(x, a, eps),
                        y + cost(x, a),
                    )
                )(noise)
            )
        )(actions)

    def scalar_policy(t, x, y):
        return actions[jnp.argmin(q_values(t, x, y))]

    policy = _batched_policy(scalar_policy)

    backup_counts = _backup_counts(
        horizon,
        int(actions.shape[0]),
        int(noise.shape[0]),
    )

    def backup_count(
        t: int, x: None | Float[Array, ""] = None, y: None | Float[Array, ""] = None
    ) -> Int[Array, ""]:
        return backup_counts[t]

    return value, policy, backup_count


def avar_dp_decomposed(
    horizon: int,
    actions: Float[Array, " n_actions"] = ACTIONS,
    noise: Float[Array, " n_samples"] = NOISE,
    probabilities: Float[Array, " n_samples"] = PROBABILITIES,
) -> tuple[Callable, Callable, Callable]:

    actions = jnp.asarray(actions)
    noise = jnp.asarray(noise)
    probabilities = jnp.asarray(probabilities)

    value_rn, policy_rn, backup_count_rn = risk_neutral_dp(
        horizon,
        actions=actions,
        noise=noise,
        probabilities=probabilities,
    )

    def value(t: int, x: Float[Array, ""], y: Float[Array, ""]) -> Float[Array, ""]:
        if t == horizon:
            return jnp.maximum(y + terminal_cost(x), 0.0)

        return jax.lax.cond(
            y >= 0.0,
            lambda: y + value_rn(t, x),
            lambda: jnp.min(q_values(t, x, y)),
        )

    def q_values(
        t: int, x: Float[Array, ""], y: Float[Array, ""]
    ) -> Float[Array, " n_actions"]:
        return jax.lax.map(
            lambda a: jnp.sum(
                probabilities
                * jax.lax.map(
                    lambda eps: value(
                        t + 1,
                        transition(x, a, eps),
                        y + cost(x, a),
                    ),
                    noise,
                )
            ),
            actions,
        )

    def scalar_policy(t, x, y):
        return jax.lax.cond(
            y >= 0.0,
            lambda: policy_rn(t, x),
            lambda: actions[jnp.argmin(q_values(t, x, y))],
        )

    policy = _batched_policy(scalar_policy)

    def backup_count(
        t: int, x: Float[Array, ""], y: Float[Array, ""]
    ) -> Int[Array, ""]:
        """Logical Bellman backups needed for one value/policy query."""
        if t == horizon:
            return jnp.asarray(0, dtype=jnp.int64)

        def decomposed_count() -> Int[Array, ""]:
            child_counts = jax.lax.map(
                lambda a: jax.lax.map(
                    lambda eps: backup_count(
                        t + 1,
                        transition(x, a, eps),
                        y + cost(x, a),
                    ),
                    noise,
                ),
                actions,
            )

            # One Bellman backup for each action at the current state.
            return jnp.asarray(actions.shape[0], dtype=jnp.int64) + jnp.sum(
                child_counts
            )

        return jax.lax.cond(
            y >= 0.0,
            lambda: jnp.asarray(backup_count_rn(t, x), dtype=jnp.int64),
            decomposed_count,
        )

    return value, policy, backup_count
