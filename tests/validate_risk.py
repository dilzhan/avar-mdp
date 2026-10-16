"""Weighted/empirical AVaR calculations and invalid-input handling."""

import jax.numpy as jnp

from src.risk import avar, empirical_avar, rollout_ru_objective, ru_objective

costs = jnp.array([0.0, 10.0, 20.0])
probabilities = jnp.array([0.7, 0.2, 0.1])
alpha = 0.75

weighted_avar, eta = avar(costs, probabilities, alpha)
assert abs(weighted_avar - 14.0) < 1e-10
assert abs(eta - 10.0) < 1e-10
# Keyword arguments survive either positional eta/alpha ordering.
assert abs(
    ru_objective(costs, probabilities, eta=eta, alpha=alpha) - 14.0
) < 1e-10
print("Weighted AVaR, quantile atom, and RU objective: PASSED")

empirical, _ = empirical_avar(jnp.array([0.0, 10.0, 20.0, 30.0]), 0.5)
assert abs(empirical - 25.0) < 1e-10
assert abs(
    rollout_ru_objective(jnp.array([0.0, 10.0, 20.0, 30.0]), eta=10.0, alpha=0.5)
    - 25.0
) < 1e-10
print("Empirical AVaR and RU objective: PASSED")

invalid_weighted = [
    (jnp.array([]), jnp.array([]), alpha),
    (jnp.array([0.0, jnp.nan, 20.0]), probabilities, alpha),
    (costs, jnp.array([0.7, -0.1, 0.4]), alpha),
    (costs, jnp.array([0.7, 0.2, 0.2]), alpha),
    (costs, probabilities[:2], alpha),
    (costs, probabilities, 1.0),
    (costs, probabilities, -0.1),
]
for c, p, a in invalid_weighted:
    try:
        avar(c, p, a)
    except ValueError:
        pass
    else:
        raise AssertionError("avar accepted invalid weighted input")

for invalid_costs in (jnp.array([]), jnp.array([jnp.inf])):
    for fn in (
        lambda: empirical_avar(invalid_costs, 0.95),
        lambda: rollout_ru_objective(invalid_costs, eta=0.0, alpha=0.95),
    ):
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("An empirical risk function accepted invalid costs")

for invalid_eta in (jnp.nan, jnp.inf):
    try:
        ru_objective(costs, probabilities, eta=invalid_eta, alpha=alpha)
    except ValueError:
        pass
    else:
        raise AssertionError("RU objective accepted nonfinite eta")

print("Input validation: PASSED")
print("ALL RISK TESTS PASSED")
