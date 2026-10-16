import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("Require 0 < alpha < 1.")


def _validate_costs(costs: Float[Array, " n_samples"]) -> Float[Array, " n_samples"]:
    costs = jnp.asarray(costs)

    if costs.ndim != 1 or costs.size == 0 or not bool(jnp.all(jnp.isfinite(costs))):
        raise ValueError("Costs must be a nonempty, finite 1D array.")

    return costs


def _validate_probabilities(
    costs: Float[Array, " n_samples"],
    probabilities: Float[Array, " n_samples"],
) -> Float[Array, " n_samples"]:
    probabilities = jnp.asarray(probabilities)

    if probabilities.shape != costs.shape:
        raise ValueError("Costs and probabilities must have matching shapes.")

    if (
        not bool(jnp.all(jnp.isfinite(probabilities)))
        or not bool(jnp.all(probabilities >= 0.0))
        or not bool(
            jnp.isclose(
                jnp.sum(probabilities),
                1.0,
                rtol=0.0,
                atol=1e-8,
            )
        )
    ):
        raise ValueError("Probabilities must be finite, nonnegative, and sum to one.")

    return probabilities


def ru_objective(
    costs: Float[Array, " n_samples"],
    probabilities: Float[Array, " n_samples"],
    eta: float,
    alpha: float = 0.95,
) -> float:
    _validate_alpha(alpha)
    costs = _validate_costs(costs)
    probabilities = _validate_probabilities(costs, probabilities)

    costs = jnp.asarray(costs)
    probabilities = jnp.asarray(probabilities)

    if not bool(jnp.isfinite(jnp.asarray(eta))) or jnp.isnan(eta):
        raise ValueError("eta must be finite.")

    return float(
        eta + jnp.sum(probabilities * jnp.maximum(costs - eta, 0.0)) / (1.0 - alpha)
    )


def rollout_ru_objective(costs: ArrayLike, eta: float, alpha: float = 0.95) -> float:
    _validate_alpha(alpha)
    costs = _validate_costs(costs)

    if not bool(jnp.isfinite(jnp.asarray(eta))) or jnp.isnan(eta):
        raise ValueError("eta must be finite.")

    inner = jnp.mean(jnp.maximum(jnp.asarray(costs) - eta, 0.0))
    return float(eta + inner / (1.0 - alpha))


def avar(
    costs: Float[Array, " n_samples"],
    probabilities: Float[Array, " n_samples"],
    alpha: float = 0.95,
) -> tuple[float, float]:
    _validate_alpha(alpha)
    costs = _validate_costs(costs)
    probabilities = _validate_probabilities(costs, probabilities)

    order = jnp.argsort(costs)
    sorted_costs = costs[order]
    cumulative_probability = jnp.cumsum(probabilities[order])
    eta = float(sorted_costs[jnp.searchsorted(cumulative_probability, alpha)])

    return ru_objective(costs, probabilities, eta, alpha), eta


def empirical_avar(
    costs: Float[Array, " n_samples"], alpha: float = 0.95
) -> tuple[float, float]:
    _validate_alpha(alpha)
    costs = _validate_costs(costs)

    probabilities = jnp.ones_like(costs) / costs.size
    return avar(costs, probabilities, alpha)
