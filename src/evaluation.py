from collections.abc import Callable

import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float

from .model import NOISE, PROBABILITIES, cost, terminal_cost, transition
from .risk import empirical_avar, rollout_ru_objective


def make_noise_paths(
    key: Array,
    n_rollouts: int,
    horizon: int,
    noise: Float[Array, " n_samples"] = NOISE,
    probabilities: Float[Array, " n_samples"] = PROBABILITIES,
) -> Float[Array, "n_rollouts horizon"]:
    idx = random.choice(
        key,
        noise.shape[0],
        shape=(n_rollouts, horizon),
        p=probabilities,
    )
    return jnp.asarray(noise[idx])


def make_rollouts(
    key: Array,
    x0: Float[Array, ""],
    y0: Float[Array, ""],
    n_rollouts: int,
    horizon: int,
    policy: Callable,
    noise: Float[Array, " n_samples"] = NOISE,
    probabilities: Float[Array, " n_samples"] = PROBABILITIES,
) -> tuple[Float[Array, "n_rollouts horizon+1 2"], Float[Array, " n_rollouts"]]:
    noise_paths = make_noise_paths(
        key=key,
        n_rollouts=n_rollouts,
        horizon=horizon,
        noise=noise,
        probabilities=probabilities,
    )

    x_cur = jnp.full(n_rollouts, x0)
    y_cur = jnp.full(n_rollouts, y0)

    paths = [jnp.stack([x_cur, y_cur], axis=1)]

    for t in range(horizon):
        action = jnp.asarray(policy(t, x_cur, y_cur))

        if action.shape != x_cur.shape:
            raise ValueError(
                f"Invalid action shape at t={t}: "
                f"expected {x_cur.shape}, got {action.shape}"
            )

        y_cur = y_cur + cost(
            x_cur,
            action,
        )

        x_cur = transition(
            x_cur,
            action,
            noise_paths[:, t],
        )

        paths.append(
            jnp.stack(
                [x_cur, y_cur],
                axis=1,
            )
        )

    total_costs = y_cur + terminal_cost(x_cur) - y0

    return (
        jnp.stack(paths, axis=1),
        total_costs,
    )


def evaluate_q_values(
    q_values: Callable,
    states: Float[Array, "n_states 3"],
) -> Float[Array, "n_states n_actions"]:
    """Evaluate Q(t, x, y, ·) on a collection of states."""
    return jnp.stack([q_values(int(t), x, y) for t, x, y in states])


def evaluate_values(
    value: Callable,
    states: Float[Array, "n_states 3"],
) -> Float[Array, " n_states"]:
    """Evaluate V(t, x, y) on a collection of states."""
    return jnp.stack([value(int(t), x, y) for t, x, y in states])


def evaluate_actions(
    policy: Callable,
    states: Float[Array, "n_states 3"],
) -> Float[Array, " n_states"]:
    """Evaluate a policy on a collection of states."""
    return jnp.stack([policy(int(t), x, y) for t, x, y in states])


def policy_metrics(
    total_costs: Float[Array, " n_rollouts"],
    alpha: float,
) -> dict[str, float]:
    avar_value, var_value = empirical_avar(total_costs, alpha)

    return {
        "mean": float(jnp.mean(total_costs)),
        "var": var_value,
        "avar": avar_value,
    }


def evaluate_policy(
    policy: Callable,
    horizon: int,
    alpha: float,
    eta: float,
    key: Array,
    x0: float = 1.0,
    noise: Float[Array, " n_samples"] = NOISE,
    probabilities: Float[Array, " n_samples"] = PROBABILITIES,
    n_rollouts: int = 10_000,
) -> dict[str, float | Float[Array, " n_rollouts"]]:

    _, total_costs = make_rollouts(
        key=key,
        x0=x0,
        y0=-eta,
        n_rollouts=n_rollouts,
        horizon=horizon,
        policy=policy,
        noise=noise,
        probabilities=probabilities,
    )

    metrics = policy_metrics(
        total_costs,
        alpha,
    )

    ru_obj = rollout_ru_objective(total_costs, eta, alpha)

    return {
        **metrics,
        "total_costs": total_costs,
        "ru_objective": ru_obj,
    }
