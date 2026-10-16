"""All returned policies accept scalar and batched JAX (x, y) inputs."""

import jax.numpy as jnp
from jax import random

from src.avar_adp import run_adp
from src.evaluation import evaluate_policy
from src.exact_dp import avar_dp, avar_dp_decomposed, risk_neutral_dp
from src.probabilistic_adp import probabilistic_adp
from src.risk_neutral_approx import fit_risk_neutral_q, make_phi, make_policy

T = 2
ETA = 2.0
ALPHA = 0.95
ACTIONS = jnp.array([-1.0, 0.0, 1.0])
NOISE = jnp.array([-0.5, 0.5])
PROBABILITIES = jnp.array([0.5, 0.5])
MODEL = dict(actions=ACTIONS, noise=NOISE, probabilities=PROBABILITIES)

exact_value, exact_policy, _ = avar_dp(T, **MODEL)
decomp_value, decomp_policy, _ = avar_dp_decomposed(T, **MODEL)
_, exact_rn_policy, _ = risk_neutral_dp(T, **MODEL)

phi = make_phi(-4.0, 4.0, delta_center=1.0)
rn_models = fit_risk_neutral_q(T, phi, jnp.linspace(-5.0, 5.0, 61), **MODEL)
rn_policy = make_policy(rn_models, phi, ACTIONS)

_, baseline_policy, _, _, _ = run_adp(
    horizon=T,
    eta=ETA,
    rn_models=rn_models,
    rn_phi=phi,
    key=random.PRNGKey(1),
    n_iterations=2,
    **MODEL,
)

candidates = jnp.array([
    [-1.5, -1.5],
    [-0.8, -0.7],
    [-0.2, -1.0],
    [0.4, -1.7],
    [1.0, -0.4],
    [1.5, -1.3],
])
_, gp_policy, _, _ = probabilistic_adp(
    horizon=T,
    eta=ETA,
    alpha=ALPHA,
    candidates_by_stage={1: candidates},
    rn_models=rn_models,
    rn_phi=phi,
    key=random.PRNGKey(2),
    n_backups_per_stage=15,
    n_initial=5,
    sampling_method="random",
    gp_x_bounds=(-4.0, 4.0),
    **MODEL,
)

policies = {
    "Exact": exact_policy,
    "Decomposed": decomp_policy,
    "Exact RN": exact_rn_policy,
    "Fitted RN": rn_policy,
    "Baseline": baseline_policy,
    "Probabilistic": gp_policy,
}

x = jnp.array([-0.8, 0.2, 1.0])
y = jnp.array([-1.5, 0.2, -0.5])

for name, policy in policies.items():
    scalar = jnp.asarray(policy(1, x[0], y[0]))
    batched = jnp.asarray(policy(1, x, y))
    broadcasted = jnp.asarray(policy(1, x[0], y))
    individual = jnp.stack([policy(1, xi, yi) for xi, yi in zip(x, y)])

    assert scalar.shape == (), name
    assert batched.shape == x.shape, name
    assert broadcasted.shape == y.shape, name
    assert bool(jnp.array_equal(batched, individual)), name
    assert bool(jnp.all(jnp.isin(batched, ACTIONS))), name

    result = evaluate_policy(
        policy=policy,
        horizon=T,
        alpha=ALPHA,
        eta=ETA,
        key=random.PRNGKey(47),
        x0=1.0,
        n_rollouts=12,
        noise=NOISE,
        probabilities=PROBABILITIES,
    )
    assert result["total_costs"].shape == (12,), name
    assert bool(jnp.all(jnp.isfinite(result["total_costs"]))), name
    print(f"{name}: PASSED")

for xi, yi in zip(x, y):
    assert bool(jnp.allclose(exact_value(1, xi, yi), decomp_value(1, xi, yi), atol=1e-10))

print("Exact/decomposed consistency: PASSED")
print("ALL POLICY-INTERFACE TESTS PASSED")
