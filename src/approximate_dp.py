from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .model import ACTIONS, NOISE, PROBABILITIES, cost, terminal_cost, transition
from .risk_neutral_approx import predict_policy, predict_value


def avar_dp_rn_approx(
    horizon: int,
    models: Float[Array, "horizon n_actions n_features"],
    phi: Callable,
    actions: Float[Array, " n_actions"] = ACTIONS,
    noise: Float[Array, " n_samples"] = NOISE,
    probabilities: Float[Array, " n_samples"] = PROBABILITIES,
) -> tuple[Callable, Callable]:

    actions = jnp.asarray(actions)
    noise = jnp.asarray(noise)
    probabilities = jnp.asarray(probabilities)

    def value_rn(t: int, x: Float[Array, "*batch"]) -> Float[Array, "*batch"]:
        return predict_value(t, x, models, phi)

    def policy_rn(t: int, x: Float[Array, "*batch"]) -> Float[Array, "*batch"]:
        return predict_policy(t, x, models, phi, actions)

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

    def policy(t: int, x: Float[Array, ""], y: Float[Array, ""]) -> Float[Array, ""]:
        return jax.lax.cond(
            y >= 0.0,
            lambda: policy_rn(t, x),
            lambda: actions[jnp.argmin(q_values(t, x, y))],
        )

    return value, policy
