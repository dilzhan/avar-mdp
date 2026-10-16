"""One fixed-domain RBF Gaussian process per discrete action."""

import jax.numpy as jnp
from jaxtyping import Array, Float
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def fit_q_gps(
    states: Float[Array, "n_samples 2"],
    q_values: Float[Array, "n_samples n_actions"],
    eta: float | None = None,
    seed: int = 47,
    x_bounds: tuple[float, float] = (-8.0, 8.0),
    length_scales: tuple[float, float] = (0.16, 0.16),
    gp_alpha: float = 1e-6,
    gp_y_bounds: tuple[float, float] | None = None,
) -> list[GaussianProcessRegressor]:
    """Fit action-wise GPs with fixed, *training-independent* input scaling.

    The legacy `eta` argument corresponds to gp_y_bounds=(-eta, 0.0).
    Passing gp_y_bounds lets a single GP collection cover a threshold family.
    Kernel length_scales are in the normalized coordinates, not physical units.
    No clipping is applied to out-of-domain prediction states.
    """
    left, right = x_bounds
    if not left < right:
        raise ValueError("Require x_bounds[0] < x_bounds[1]")
    if gp_y_bounds is None:
        if eta is None or eta <= 0:
            raise ValueError("Require eta > 0 or explicit gp_y_bounds")
        gp_y_bounds = (-float(eta), 0.0)
    y_min, y_max = gp_y_bounds
    if not y_min < y_max <= 0:
        raise ValueError("Require gp_y_bounds[0] < gp_y_bounds[1] <= 0")
    if any(ell <= 0 for ell in length_scales) or gp_alpha <= 0:
        raise ValueError("GP length scales and gp_alpha must be positive")

    states = jnp.asarray(states, dtype=float)
    q_values = jnp.asarray(q_values, dtype=float)
    mean = jnp.asarray([left, y_min], dtype=float)
    scale = jnp.asarray([right - left, y_max - y_min], dtype=float)
    features = (states - mean) / scale

    gps = []
    for a in range(q_values.shape[1]):
        gp = GaussianProcessRegressor(
            kernel=RBF(
                length_scale=jnp.asarray(length_scales), length_scale_bounds="fixed"
            ),
            alpha=gp_alpha,
            optimizer=None,
            normalize_y=True,
            random_state=seed,
        )
        gp.fit(features, q_values[:, a])
        gp._state_mean = mean  # type: ignore[attr-defined]
        gp._state_scale = scale  # type: ignore[attr-defined]
        gps.append(gp)
    return gps


def predict_q(
    gps: list[GaussianProcessRegressor], states: Float[Array, "n_states 2"]
) -> tuple[Float[Array, "n_states n_actions"], Float[Array, "n_states n_actions"]]:
    states = jnp.atleast_2d(jnp.asarray(states, dtype=float))
    means, stds = [], []
    for gp in gps:
        features = (states - gp._state_mean) / gp._state_scale  # type: ignore[attr-defined]
        mean, std = gp.predict(  # pyright: ignore[reportAssignmentType]
            features, return_std=True
        )
        means.append(mean)
        stds.append(std)
    return jnp.asarray(jnp.column_stack(means)), jnp.asarray(jnp.column_stack(stds))


def predict_value(
    gps: list[GaussianProcessRegressor], states: Float[Array, "n_states 2"]
) -> Float[Array, " n_states"]:
    """Only predictive means are needed for Bellman continuations."""
    states = jnp.atleast_2d(jnp.asarray(states, dtype=float))
    means = [gp.predict((states - gp._state_mean) / gp._state_scale) for gp in gps]  # type: ignore[attr-defined]
    return jnp.asarray(jnp.min(jnp.column_stack(means), axis=1))


def predict_policy(
    gps: list[GaussianProcessRegressor], states: Float[Array, "n_states 2"]
) -> Float[Array, " n_states"]:
    states = jnp.atleast_2d(jnp.asarray(states, dtype=float))
    means = [gp.predict((states - gp._state_mean) / gp._state_scale) for gp in gps]  # type: ignore[attr-defined]
    return jnp.asarray(jnp.argmin(jnp.column_stack(means), axis=1))
