"""Verify pre-import float64 and safe integer backup counters."""

import jax
import jax.numpy as jnp

from src.exact_dp import avar_dp, avar_dp_decomposed, risk_neutral_dp
from src.model import ACTIONS, NOISE, PROBABILITIES

assert jax.config.read("jax_enable_x64")
assert ACTIONS.dtype == jnp.float64
assert NOISE.dtype == jnp.float64
assert PROBABILITIES.dtype == jnp.float64
print("Model precision: PASSED")

actions = jnp.array([-1.0, 0.0, 1.0])
noise = jnp.linspace(-1.0, 1.0, 201)
probabilities = jnp.ones(201) / 201
model = dict(actions=actions, noise=noise, probabilities=probabilities)

# Construction only: do not attempt to evaluate the enormous recursion.
T = 7
expected = sum(len(actions) * (len(actions) * len(noise)) ** k for k in range(T))
for solver in (risk_neutral_dp, avar_dp):
    _, _, backup_count = solver(T, **model)
    assert int(backup_count(0)) == expected, solver.__name__
    assert jnp.asarray(backup_count(0)).dtype == jnp.int64, solver.__name__
print("64-bit backup counters: PASSED")

for solver in (risk_neutral_dp, avar_dp, avar_dp_decomposed):
    try:
        solver(T + 1, **model)
    except OverflowError:
        pass
    else:
        raise AssertionError(f"{solver.__name__}: missing overflow check")
print("Counter overflow protection: PASSED")

small_model = dict(
    actions=actions,
    noise=jnp.array([-0.5, 0.5]),
    probabilities=jnp.array([0.5, 0.5]),
)
_, _, rn_count = risk_neutral_dp(2, **small_model)
_, _, decomposed_count = avar_dp_decomposed(2, **small_model)
assert int(decomposed_count(0, 1.0, 0.5)) == int(rn_count(0))
print("Decomposed count on y >= 0: PASSED")
print("ALL PRECISION TESTS PASSED")
