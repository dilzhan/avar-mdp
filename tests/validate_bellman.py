"""Independent tiny-tree reference for exact augmented-state Bellman DP."""

from functools import lru_cache

import jax.numpy as jnp

from src.exact_dp import avar_dp, avar_dp_decomposed, risk_neutral_dp

T = 2
ACTIONS = (-1.0, 0.0, 1.0)
NOISE = (-0.5, 1.0)
PROBABILITIES = (0.3, 0.7)


@lru_cache(None)
def reference_rn(t, x):
    if t == T:
        return x * x
    return min(
        x * x + a * a
        + sum(p * reference_rn(t + 1, x + a + w)
              for w, p in zip(NOISE, PROBABILITIES))
        for a in ACTIONS
    )


@lru_cache(None)
def reference_avar(t, x, y):
    if t == T:
        return max(y + x * x, 0.0)
    return min(
        sum(p * reference_avar(t + 1, x + a + w, y + x * x + a * a)
            for w, p in zip(NOISE, PROBABILITIES))
        for a in ACTIONS
    )


model = dict(
    actions=jnp.array(ACTIONS),
    noise=jnp.array(NOISE),
    probabilities=jnp.array(PROBABILITIES),
)
rn_value, _, _ = risk_neutral_dp(T, **model)
full_value, _, _ = avar_dp(T, **model)
decomp_value, _, _ = avar_dp_decomposed(T, **model)

for t, x, y in ((0, 1.0, -3.0), (0, -0.5, 0.75), (1, 0.3, -1.0)):
    expected = reference_avar(t, x, y)
    assert abs(float(full_value(t, x, y)) - expected) < 1e-10
    assert abs(float(decomp_value(t, x, y)) - expected) < 1e-10
    if y >= 0:
        expected_rn = reference_rn(t, x)
        assert abs(float(decomp_value(t, x, y)) - (y + expected_rn)) < 1e-10
    print(f"Bellman check t={t}, x={x}, y={y}: PASSED")

assert abs(float(rn_value(0, 1.0)) - reference_rn(0, 1.0)) < 1e-10
print("Risk-neutral independent recursion: PASSED")
print("ALL BELLMAN TESTS PASSED")
