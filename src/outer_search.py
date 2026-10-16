from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def solve_avar_exact(
    dp_solver: Callable,
    alpha: float,
    horizon: int,
    eta_grid: Float[Array, " grid_size"],
    x0: float | Float[Array, ""] = 1.0,
    **dp_arguments
) -> tuple[Float[Array, ""], Float[Array, ""], Callable, Float[Array, "grid_size 2"]]:
    eta_grid = jnp.asarray(eta_grid)

    value, policy, _ = dp_solver(horizon, **dp_arguments)

    def objective(eta):
        return eta + value(0, x0, -eta) / (1.0 - alpha)

    objectives = jax.lax.map(objective, eta_grid)

    best = jnp.argmin(objectives)
    eta = eta_grid[best]
    trace = jnp.column_stack((eta_grid, objectives))

    return objectives[best], eta, policy, trace


def solve_avar_probabilistic(
    horizon: int,
    alpha: float,
    eta_grid: Float[Array, " n_thresholds"],
    candidates_by_stage: dict[int, Float[Array, "n_candidates 2"]],
    rn_models: Float[Array, "horizon n_actions n_features"],
    rn_phi: Callable,
    key: Array,
    n_backups_per_stage: int,
    sampling_method: str = "decision",
    x0: float = 1.0,
    n_initial: int = 10,
    batch_size: int = 10,
    gp_x_bounds: tuple[float, float] = (-8.0, 8.0),
    gp_length_scales: tuple[float, float] = (0.16, 0.16),
    gp_alpha: float = 1e-6,
    actions: Float[Array, " n_actions"] | None = None,
    noise: Float[Array, " n_samples"] | None = None,
    probabilities: Float[Array, " n_samples"] | None = None,
) -> tuple[float, float, Callable, dict, dict]:
    """Optimize static AVaR over eta_grid with a single backward GP fit.

    Only random, uncertainty and decision acquisition are supported here.
    `candidates_by_stage` must cover the relevant difficult (x,y) region for
    *all* thresholds. Stage zero is evaluated directly at (x0,-eta), not fitted.

    Returns (predicted_RU_minimum, eta_star, policy, stage_gps, diagnostics).
    The predicted RU minimum is NOT an empirical AVaR of the returned policy.
    Evaluate that policy separately with `evaluation.evaluate_policy`.

    GP length_scales are in normalized coordinates. When changing the y-domain,
    pass adjusted scaled length scales if preserving physical smoothness matters.
    """
    import numpy as np

    from .model import ACTIONS, NOISE, PROBABILITIES
    from .probabilistic_adp import (
        make_probabilistic_policy,
        normalize_sampling_method,
        root_q_values,
        train_stage_gps,
    )

    method = normalize_sampling_method(sampling_method)
    if method not in {("random",), ("uncertainty",), ("decision",)}:
        raise ValueError(
            "Outer eta solver supports only random, uncertainty or decision"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError("Require 0 < alpha < 1")

    eta_grid = jnp.asarray(eta_grid, dtype=float)
    if eta_grid.ndim != 1 or eta_grid.size == 0:
        raise ValueError("eta_grid must be a nonempty 1D threshold family")
    if not bool(jnp.all(jnp.isfinite(eta_grid))) or bool(jnp.any(eta_grid < 0)):
        raise ValueError("eta_grid must contain finite, nonnegative thresholds")
    if len(np.unique(np.asarray(eta_grid))) != len(eta_grid):
        raise ValueError("eta_grid cannot contain duplicate thresholds")
    eta_max = float(jnp.max(eta_grid))
    if eta_max <= 0:
        raise ValueError("Require at least one positive threshold for GP training")

    if actions is None:
        actions = ACTIONS
    if noise is None:
        noise = NOISE
    if probabilities is None:
        probabilities = PROBABILITIES
    actions = jnp.asarray(actions)

    stage_gps, diagnostics = train_stage_gps(
        horizon=horizon,
        candidates_by_stage=candidates_by_stage,
        rn_models=rn_models,
        rn_phi=rn_phi,
        key=key,
        n_backups_per_stage=n_backups_per_stage,
        gp_y_bounds=(-eta_max, 0.0),
        reference_eta=None,
        sampling_method=method,
        n_initial=n_initial,
        batch_size=batch_size,
        gp_x_bounds=gp_x_bounds,
        gp_length_scales=gp_length_scales,
        gp_alpha=gp_alpha,
        x0=x0,
        actions=actions,
        noise=noise,
        probabilities=probabilities,
    )

    # Each row evaluates all actions at a candidate root (x0,-eta).
    root_q_matrix = jnp.stack(
        [
            root_q_values(
                float(eta),
                x0,
                stage_gps,
                rn_models,
                rn_phi,
                actions=actions,
                noise=noise,
                probabilities=probabilities,
            )
            for eta in eta_grid
        ]
    )
    root_values = jnp.min(root_q_matrix, axis=1)
    objectives = eta_grid + root_values / (1.0 - alpha)
    index = int(jnp.argmin(objectives))
    eta_star = float(eta_grid[index])
    avar_estimate = float(objectives[index])
    root_action = float(actions[int(jnp.argmin(root_q_matrix[index]))])
    policy = make_probabilistic_policy(
        root_action, stage_gps, rn_models, rn_phi, actions=actions
    )

    diagnostics.update(
        {
            "eta_grid": eta_grid,
            "root_states": jnp.column_stack((jnp.full_like(eta_grid, x0), -eta_grid)),
            "root_q_values": root_q_matrix,
            "objective_trace": jnp.column_stack((eta_grid, objectives)),
            "eta_star": eta_star,
            "root_action": root_action,
            "predicted_ru": avar_estimate,
            "root_backups": int(len(eta_grid) * len(actions)),
            "total_backups": diagnostics["n_backups"]
            + int(len(eta_grid) * len(actions)),
        }
    )
    return avar_estimate, eta_star, policy, stage_gps, diagnostics
