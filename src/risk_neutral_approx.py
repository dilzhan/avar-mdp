from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float

from .model import ACTIONS, NOISE, PROBABILITIES, cost, terminal_cost, transition


def make_phi(
    x_min: float, x_max: float, delta_center: float = 0.5, padding_centers: int = 2
) -> Callable:
    padding = padding_centers * delta_center

    centers = jnp.arange(
        x_min - padding,
        x_max + padding + delta_center,
        delta_center,
    )

    ell = 1.5 * delta_center

    def phi(x: Float[Array, "*batch"]) -> Float[Array, "*batch n_features"]:
        x = jnp.asarray(x)

        rbf = jnp.exp(-((x[..., None] - centers) ** 2) / (2 * ell**2))

        z = 2.0 * (x - x_min) / (x_max - x_min) - 1.0

        return jnp.concatenate(
            [
                jnp.ones_like(z)[..., None],
                z[..., None],
                (z**2)[..., None],
                rbf,
            ],
            axis=-1,
        )

    return phi


def make_training_grid(
    x_min: float,
    x_max: float,
    horizon: int,
    mesh: float,
    actions: Float[Array, " n_actions"] = ACTIONS,
    noise: Float[Array, " n_samples"] = NOISE,
) -> Float[Array, " grid_size"]:
    step = jnp.max(jnp.abs(actions)) + jnp.max(jnp.abs(noise))

    train_min = x_min - horizon * step
    train_max = x_max + horizon * step

    return jnp.arange(train_min, train_max + mesh, mesh)


def fit_risk_neutral_q(
    horizon: int,
    phi: Callable,
    x_grid: Float[Array, " grid_size"],
    actions: Float[Array, " n_actions"] = ACTIONS,
    noise: Float[Array, " n_samples"] = NOISE,
    probabilities: Float[Array, " n_samples"] = PROBABILITIES,
) -> Float[Array, "horizon n_actions n_features"]:

    Phi = phi(x_grid)
    models = jnp.zeros((horizon, len(actions), Phi.shape[-1]))

    for t in range(horizon - 1, -1, -1):
        for i in range(len(actions)):
            next_x = transition(
                x_grid[:, None],
                actions[i],
                noise[None, :],
            )

            if t == horizon - 1:
                next_v = terminal_cost(next_x)

            else:
                next_v = jnp.min(phi(next_x) @ models[t + 1].T, axis=2)

            expectation = jnp.sum(probabilities[None, :] * next_v, axis=1)

            Q_value = cost(x_grid, actions[i]) + expectation

            weights = jnp.linalg.lstsq(Phi, Q_value, rcond=None)[0]
            models = models.at[t, i].set(weights)

    return models


def predict_q(
    t: int,
    x: Float[Array, "*batch"],
    models: Float[Array, "horizon n_actions n_features"],
    phi: Callable,
) -> Float[Array, "*batch n_actions"]:

    x = jnp.asarray(x)
    scalar = x.ndim == 0
    x = jnp.atleast_1d(x)

    q = phi(x) @ models[t].T

    return q[0] if scalar else q


def predict_value(
    t: int,
    x: Float[Array, "*batch"],
    models: Float[Array, "horizon n_actions n_features"],
    phi: Callable,
) -> Float[Array, "*batch"]:

    return jnp.min(
        predict_q(t, x, models, phi),
        axis=-1,
    )


def predict_policy(
    t: int,
    x: Float[Array, "*batch"],
    models: Float[Array, "horizon n_actions n_features"],
    phi: Callable,
    actions: Float[Array, " n_actions"] = ACTIONS,
) -> Float[Array, "*batch"]:

    return actions[
        jnp.argmin(
            predict_q(t, x, models, phi),
            axis=-1,
        )
    ]


def make_policy(
    models: Float[Array, "horizon n_actions n_features"],
    phi: Callable,
    actions: Float[Array, " n_actions"] = ACTIONS,
) -> Callable:

    actions = jnp.asarray(actions)

    def policy(t, x, y):
        x, _ = jnp.broadcast_arrays(
            jnp.asarray(x),
            jnp.asarray(y),
        )

        return predict_policy(t, x, models, phi, actions)

    return policy
