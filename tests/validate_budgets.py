"""Requested/actual training and root-backup budget accounting."""

import jax.numpy as jnp
from jax import random

from src.probabilistic_adp import probabilistic_adp

T = 2
actions = jnp.array([-1.0, 0.0, 1.0])
noise = jnp.array([-0.5, 0.5])
probabilities = jnp.array([0.5, 0.5])
candidates = jnp.array([[-1.0, -1.5], [-0.5, -1.0]])
phi = lambda x: jnp.ones(jnp.shape(x) + (1,))
rn_models = jnp.zeros((T, len(actions), 1))


def run(requested):
    _, _, _, diagnostics = probabilistic_adp(
        horizon=T,
        eta=2.0,
        alpha=0.95,
        candidates_by_stage={1: candidates},
        rn_models=rn_models,
        rn_phi=phi,
        key=random.PRNGKey(42),
        n_backups_per_stage=requested,
        n_initial=2,
        sampling_method="random",
        gp_x_bounds=(-2.0, 2.0),
        actions=actions,
        noise=noise,
        probabilities=probabilities,
    )
    return diagnostics


# 8 requested: only 2 full action vectors (6 backups) fit.
d = run(8)
assert d["requested_training_backups"] == 8
assert d["actual_training_backups"] == 6
assert d["root_backups"] == 3
assert d["requested_total_backups"] == 11
assert d["total_backups"] == 9
print("Budget rounding: PASSED")

# 20 requested: only two candidate states are available.
d = run(20)
assert d["requested_training_backups"] == 20
assert d["actual_training_backups"] == 6
assert d["requested_total_backups"] == 23
assert d["total_backups"] == 9
assert d["stages"][1]["requested_backups"] == 20
assert d["stages"][1]["actual_backups"] == 6
assert d["n_backups"] == d["actual_training_backups"]
print("Candidate availability cap: PASSED")
print("ALL BUDGET TESTS PASSED")
