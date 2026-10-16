"""Selected candidate indices refer to the ORIGINAL unfiltered input array."""

import jax.numpy as jnp
from jax import random

from src.probabilistic_adp import fit_stage

candidates = jnp.array([
    [-1.0, 0.5],
    [-0.5, -1.0],
    [0.0, 0.2],
    [0.5, -0.8],
    [1.0, -1.5],
])
actions = jnp.array([-1.0, 0.0, 1.0])
phi = lambda x: jnp.ones(jnp.shape(x) + (1,))
rn_models = jnp.zeros((2, len(actions), 1))

_, diagnostics = fit_stage(
    t=1,
    candidates=candidates,
    rollout_states=None,
    tail_flags=None,
    next_gps=[],
    rn_models=rn_models,
    rn_phi=phi,
    key=random.PRNGKey(42),
    n_backups=3 * len(actions),
    n_initial=3,
    sampling_method="random",
    eta=2.0,
    gp_x_bounds=(-2.0, 2.0),
    gp_y_bounds=(-2.0, 0.0),
    actions=actions,
    noise=jnp.array([-0.5, 0.5]),
    probabilities=jnp.array([0.5, 0.5]),
)

indices = jnp.asarray(diagnostics["selected_indices"])
states = jnp.asarray(diagnostics["training_states"])
assert bool(jnp.array_equal(jnp.sort(indices), jnp.array([1, 3, 4])))
assert bool(jnp.array_equal(candidates[indices], states))
assert bool(jnp.all(states[:, 1] < 0.0))
print("Original candidate index mapping: PASSED")
