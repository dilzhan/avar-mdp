from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float

from .evaluation import make_noise_paths
from .model import (
    ACTIONS,
    NOISE,
    PROBABILITIES,
    cost,
    terminal_cost,
    transition,
)
from .risk_neutral_approx import (
    predict_policy as predict_rn_policy,
)
from .risk_neutral_approx import (
    predict_value as predict_rn_value,
)


def state_key(
    t: int,
    x: float | Float[Array, ""],
    y: float | Float[Array, ""],
) -> tuple[int, float, float]:
    return (
        int(t),
        round(float(x), 6),
        round(float(y), 6),
    )


def adp_q(
    adp_value: Callable,
    t: int,
    x: float | Float[Array, ""],
    y: float | Float[Array, ""],
    actions: Float[Array, " n_actions"] = ACTIONS,
    noise: Float[Array, " n_samples"] = NOISE,
    probabilities: Float[Array, " n_samples"] = PROBABILITIES,
) -> Float[Array, " n_actions"]:
    """Q-values induced by the current read-only ADP value approximation."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    q_values = []

    # Ordinary loops are intentional: adp_value is dictionary/Python based.
    for action in actions:
        y_next = y + cost(x, action)
        next_values = jnp.asarray(
            [adp_value(t + 1, transition(x, action, eps), y_next) for eps in noise]
        )
        q_values.append(jnp.sum(probabilities * next_values))

    return jnp.asarray(q_values)


def make_exact_q(
    exact_value: Callable,
    t: int,
    actions: Float[Array, " n_actions"] = ACTIONS,
    noise: Float[Array, " n_samples"] = NOISE,
    probabilities: Float[Array, " n_samples"] = PROBABILITIES,
) -> Callable:
    """Return a JIT-compiled exact Q(t, x, y, ·) evaluator."""

    @jax.jit
    def q_values(x, y):
        return jax.vmap(
            lambda a: jnp.sum(
                probabilities
                * jax.vmap(
                    lambda eps: exact_value(
                        t + 1,
                        transition(x, a, eps),
                        y + cost(x, a),
                    )
                )(noise)
            )
        )(actions)

    return q_values


def make_policy_initializer(
    horizon: int,
    rn_policy: Callable,
    rn_models: Float[Array, "horizon n_actions n_features"],
    rn_phi: Callable,
    key: Array,
    n_rollouts: int = 20,
    actions: Float[Array, " n_actions"] = ACTIONS,
    noise: Float[Array, " n_samples"] = NOISE,
    probabilities: Float[Array, " n_samples"] = PROBABILITIES,
) -> Callable:

    actions = jnp.asarray(actions)
    noise = jnp.asarray(noise)
    probabilities = jnp.asarray(probabilities)

    cache = {}

    def initializer(
        t: int, x: Float[Array, ""] | float, y: Float[Array, ""] | float
    ) -> float:
        nonlocal key

        current_key = state_key(
            t,
            x,
            y,
        )

        if current_key in cache:
            return cache[current_key]

        if t == horizon:
            value = jnp.maximum(
                jnp.asarray(y) + terminal_cost(x),
                0.0,
            )
            cache[current_key] = float(value)
            return cache[current_key]

        remaining = horizon - t

        key, rollout_key = random.split(key, 2)

        noise_paths = make_noise_paths(
            rollout_key, n_rollouts, remaining, noise=noise, probabilities=probabilities
        )

        def rollout(
            noise_path: Float[Array, " remaining_horizon"],
        ) -> Float[Array, ""]:
            x_current = jnp.asarray(x)
            z = jnp.asarray(y)

            for k in range(remaining):
                s = t + k

                action = rn_policy(s, x_current, rn_models, rn_phi, actions)

                z = z + cost(
                    x_current,
                    action,
                )

                x_current = transition(
                    x_current,
                    action,
                    noise_path[k],
                )

            return jnp.maximum(
                z + terminal_cost(x_current),
                0.0,
            )

        samples = jax.vmap(rollout)(noise_paths)

        estimate = jnp.mean(samples)

        # RN initializer
        lower_bound = jnp.maximum(
            y
            + predict_rn_value(
                t,
                jnp.asarray(x),
                rn_models,
                rn_phi,
            ),
            0.0,
        )

        estimate = jnp.maximum(
            estimate,
            lower_bound,
        )

        cache[current_key] = float(estimate)

        return cache[current_key]

    return initializer


def run_adp(
    horizon: int,
    eta: float,
    rn_models: Float[Array, "horizon n_actions n_features"],
    rn_phi: Callable,
    key: Array,
    actions: Float[Array, " n_actions"] = ACTIONS,
    noise: Float[Array, " n_samples"] = NOISE,
    probabilities: Float[Array, " n_samples"] = PROBABILITIES,
    n_iterations: int = 1000,
    initializer: Callable | None = None,
    tol: float | None = None,
    x0: float = 1.0,
) -> tuple[Callable, Callable, dict, Float[Array, " n_steps"], int]:

    actions = jnp.asarray(actions)
    noise = jnp.asarray(noise)
    probabilities = jnp.asarray(probabilities)

    values = {}
    n_backups = 0

    if initializer is None:

        def rn_initial_value(
            t: int, x: float | Float[Array, ""], y: float | Float[Array, ""]
        ) -> Float[Array, ""]:
            """Value used for states not yet visited by ADP."""

            x = jnp.asarray(x)
            y = jnp.asarray(y)

            if t == horizon:
                return jnp.maximum(
                    y + terminal_cost(x),
                    0.0,
                )

            return jnp.maximum(
                y
                + predict_rn_value(
                    t,
                    x,
                    rn_models,
                    rn_phi,
                ),
                0.0,
            )

        initializer = rn_initial_value

    def initial_value(
        t: int, x: Float[Array, ""] | float, y: Float[Array, ""] | float
    ) -> Float[Array, ""]:
        x = jnp.asarray(x)
        y = jnp.asarray(y)

        if t == horizon:
            return jnp.maximum(
                y + terminal_cost(x),
                0.0,
            )

        if y >= 0.0:
            return y + predict_rn_value(
                t,
                x,
                rn_models,
                rn_phi,
            )

        return initializer(t, x, y)

    def approximate_value(t: int, x: Float[Array, ""], y: Float[Array, ""]) -> float:
        """Read-only current ADP value approximation."""
        current_key = state_key(t, x, y)

        if current_key in values:
            return values[current_key]

        return float(initial_value(t, x, y))

    def value(
        t: int, x: Float[Array, ""], y: Float[Array, ""], key: Array
    ) -> tuple[float, Array]:
        nonlocal n_backups
        current_key = state_key(t, x, y)

        # Exact terminal value or decomposed state.
        if t == horizon or y >= 0:
            values[current_key] = float(initial_value(t, x, y))
            return values[current_key], key

        q_values = []

        for action in actions:
            y_next = y + cost(x, action)
            next_values = []

            for eps in noise:
                x_next = transition(x, action, eps)
                next_key = state_key(t + 1, x_next, y_next)

                if next_key not in values:
                    values[next_key] = float(
                        initial_value(
                            t + 1,
                            x_next,
                            y_next,
                        )
                    )

                next_values.append(values[next_key])

            q_values.append(jnp.sum(probabilities * jnp.asarray(next_values)))
            n_backups += 1

        q_values = jnp.asarray(q_values)

        # Follow one disturbance under the currently best action.
        best = int(jnp.argmin(q_values))
        action = actions[best]

        key, noise_key = random.split(key, 2)

        noise_index = random.choice(
            noise_key,
            noise.shape[0],
            p=probabilities,
        )
        eps = noise[noise_index]

        y_next = y + cost(x, action)
        x_next = transition(x, action, eps)

        _, key = value(
            t + 1,
            x_next,
            y_next,
            key,
        )

        # Recompute all actions after recursive update.
        updated_q_values = []

        for action in actions:
            y_next = y + cost(x, action)

            updated_values = jnp.asarray(
                [
                    approximate_value(
                        t + 1,
                        transition(x, action, eps),
                        y_next,
                    )
                    for eps in noise
                ]
            )

            updated_q_values.append(jnp.sum(probabilities * updated_values))
            n_backups += 1

        q_values = jnp.asarray(updated_q_values)

        values[current_key] = float(jnp.min(q_values))

        return values[current_key], key

    def scalar_policy(
        t: int, x: Float[Array, ""] | float, y: Float[Array, ""] | float
    ) -> Float[Array, ""]:
        """Greedy policy induced by the current ADP value approximation."""
        x = jnp.asarray(x)
        y = jnp.asarray(y)

        if y >= 0.0:
            return predict_rn_policy(t, x, rn_models, rn_phi, actions)

        q_values = []

        for action in actions:
            y_next = y + cost(x, action)

            next_values = jnp.asarray(
                [
                    approximate_value(
                        t + 1,
                        transition(x, action, eps),
                        y_next,
                    )
                    for eps in noise
                ]
            )

            q_values.append(jnp.sum(probabilities * next_values))

        return actions[jnp.argmin(jnp.asarray(q_values))]

    def policy(t, x, y):
        x, y = jnp.broadcast_arrays(
            jnp.asarray(x),
            jnp.asarray(y),
        )

        if x.ndim == 0:
            return jnp.asarray(scalar_policy(t, x, y))

        actions = [
            scalar_policy(t, xi, yi)
            for xi, yi in zip(
                x.reshape(-1),
                y.reshape(-1),
            )
        ]

        return jnp.asarray(actions).reshape(x.shape)

    history = []
    iteration = 0

    while iteration < n_iterations:
        _, key = value(
            0,
            jnp.asarray(x0),
            jnp.asarray(-eta),
            key,
        )

        current_value = values[state_key(0, x0, -eta)]
        history.append(current_value)

        if tol is not None and len(history) >= 10:
            recent = jnp.asarray(history[-10:])
            if jnp.max(jnp.abs(jnp.diff(recent))) < tol:
                break

        iteration += 1

    return approximate_value, policy, values, jnp.asarray(history), n_backups
