"""Finite-horizon probabilistic ADP for static AVaR.

The fixed-threshold and threshold-family solvers share one backward GP routine.
RN-policy acquisition rollouts are generated only for reachability/tail rules.
"""

from collections.abc import Callable, Sequence
from time import perf_counter

import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float
from sklearn.gaussian_process import GaussianProcessRegressor

from .acquisition import (
    decision_score,
    knn_distances_and_indices,
    reachability_score,
    tail_score,
)
from .evaluation import make_rollouts
from .gp import fit_q_gps, predict_q, predict_value
from .gp import predict_policy as predict_gp_policy
from .model import ACTIONS, NOISE, PROBABILITIES, cost, terminal_cost, transition
from .risk_neutral_approx import predict_policy as predict_rn_policy
from .risk_neutral_approx import predict_value as predict_rn_value

VALID_METHODS = frozenset({"random", "uncertainty", "decision", "reachability", "tail"})


def normalize_sampling_method(method: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(method, str):
        method = (method,)
    method = tuple(method)
    unknown = set(method) - VALID_METHODS
    if not method or unknown:
        raise ValueError(
            f"Unknown/empty sampling_method: {method!r}; valid: {sorted(VALID_METHODS)}"
        )
    if len(set(method)) != len(method):
        raise ValueError(f"Duplicate acquisition components: {method!r}")
    if ("random" in method or "uncertainty" in method) and len(method) != 1:
        raise ValueError("'random' and 'uncertainty' must be used alone")
    return method


def bounded_gp_value(
    t: int,
    x: Float[Array, " n_samples"],
    y: Float[Array, ""],
    gps: list[GaussianProcessRegressor],
    rn_models: Float[Array, "horizon n_actions n_features"],
    rn_phi: Callable,
) -> Float[Array, " n_samples"]:
    """Continuing value in y<0; RN-based bound is approximate, zero is exact."""
    states = jnp.column_stack((x, jnp.full_like(x, y)))
    return jnp.maximum(
        jnp.maximum(
            predict_value(gps, states),
            y + predict_rn_value(t, x, rn_models, rn_phi),
        ),
        0.0,
    )


def bellman_backup(
    t: int,
    x: float | Float[Array, ""],
    y: float | Float[Array, ""],
    next_gps: list[GaussianProcessRegressor],
    rn_models: Float[Array, "horizon n_actions n_features"],
    rn_phi: Callable,
    actions: Float[Array, " n_actions"] = ACTIONS,
    noise: Float[Array, " n_samples"] = NOISE,
    probabilities: Float[Array, " n_samples"] = PROBABILITIES,
) -> Float[Array, " n_actions"]:
    x, y = jnp.asarray(x), jnp.asarray(y)
    values = []
    for a in actions:
        next_x = transition(x, a, noise)
        next_y = y + cost(x, a)
        if t + 1 == rn_models.shape[0]:
            continuation = jnp.maximum(next_y + terminal_cost(next_x), 0.0)
        elif next_y >= 0:
            continuation = next_y + predict_rn_value(t + 1, next_x, rn_models, rn_phi)
        else:
            continuation = bounded_gp_value(
                t + 1, next_x, next_y, next_gps, rn_models, rn_phi
            )
        values.append(jnp.sum(continuation * probabilities))
    return jnp.asarray(values)


def fit_stage(
    t: int,
    candidates: Float[Array, "n_candidates 2"],
    rollout_states: Float[Array, "n_rollouts 2"] | None,
    tail_flags: Array | None,
    next_gps: list[GaussianProcessRegressor],
    rn_models: Float[Array, "horizon n_actions n_features"],
    rn_phi: Callable,
    key: Array,
    n_backups: int,
    eta: float | None = None,
    n_initial: int = 10,
    batch_size: int = 10,
    k: int = 20,
    sampling_method: str | Sequence[str] = "decision",
    gp_x_bounds: tuple[float, float] = (-8.0, 8.0),
    gp_length_scales: tuple[float, float] = (0.16, 0.16),
    gp_alpha: float = 1e-6,
    tail_lambda: float = 0.0,
    actions: Float[Array, " n_actions"] = ACTIONS,
    noise: Float[Array, " n_samples"] = NOISE,
    probabilities: Float[Array, " n_samples"] = PROBABILITIES,
    gp_y_bounds: tuple[float, float] | None = None,
) -> tuple[list[GaussianProcessRegressor], dict]:

    method = normalize_sampling_method(sampling_method)
    candidates = jnp.asarray(candidates)

    original_indices = jnp.where(candidates[:, 1] < 0)[0]
    candidates = candidates[original_indices]

    n_candidates = len(candidates)
    n_actions = len(actions)
    n_points = min(n_candidates, n_backups // n_actions)

    if n_candidates == 0 or n_points < 1 or n_initial <= 0:
        raise ValueError("Require y<0 candidate points and a positive feasible budget")

    n_initial = min(n_initial, n_points)
    if batch_size < 1 and n_points > n_initial:
        raise ValueError("batch_size must be positive")
    if "tail" in method and not 0.0 <= tail_lambda <= 1.0:
        raise ValueError("Require 0 <= tail_lambda <= 1")
    if ("tail" in method or "reachability" in method) and k < 1:
        raise ValueError("k must be positive for rollout-based acquisition")

    reachability = tail_relevance = None
    if "reachability" in method or "tail" in method:
        if rollout_states is None:
            raise ValueError("Rollout states required for reachability/tail")
        if len(rollout_states) < 2:
            raise ValueError("At least two rollout states required")
        distances, neighbors = knn_distances_and_indices(
            candidates, jnp.asarray(rollout_states), k=min(k, len(rollout_states))
        )
        if "reachability" in method:
            reachability = reachability_score(distances)
        if "tail" in method:
            if tail_flags is None:
                raise ValueError("Tail flags required for tail acquisition")
            tail_relevance = tail_score(neighbors, jnp.asarray(tail_flags))

    def backup_points(points):
        return jnp.stack(
            [
                bellman_backup(
                    t,
                    x,
                    y,
                    next_gps,
                    rn_models,
                    rn_phi,
                    actions=actions,
                    noise=noise,
                    probabilities=probabilities,
                )
                for x, y in points
            ]
        )

    def fit_gps(points, targets):
        return fit_q_gps(
            points,
            targets,
            x_bounds=gp_x_bounds,
            eta=eta,
            length_scales=gp_length_scales,
            gp_alpha=gp_alpha,
            gp_y_bounds=gp_y_bounds,
        )

    key, init_key = random.split(key, 2)
    initial = random.choice(init_key, n_candidates, shape=(n_initial,), replace=False)
    indices = [int(i) for i in initial]
    scores = [float("nan")] * n_initial
    used = jnp.zeros(n_candidates, dtype=bool).at[initial].set(True)

    # Random sampling is nonadaptive: select all remaining states and fit once.
    if method == ("random",) and n_points > n_initial:
        key, random_key = random.split(key, 2)
        remaining = jnp.where(~used)[0]
        additions = random.choice(
            random_key, remaining, shape=(n_points - n_initial,), replace=False
        )
        indices.extend(int(i) for i in additions)
        scores.extend([1.0] * len(additions))
        points = candidates[jnp.asarray(indices)]
        targets = backup_points(points)
        gps = fit_gps(points, targets)
    else:
        points = candidates[initial]
        targets = backup_points(points)
        gps = fit_gps(points, targets)
        while len(indices) < n_points:
            remaining = jnp.where(~used)[0]
            n_select = min(batch_size, n_points - len(indices))
            means, stds = predict_q(gps, candidates[remaining])
            if method == ("uncertainty",):
                acquisition = jnp.max(stds, axis=1)
            else:
                acquisition = jnp.ones(len(remaining))
                if "decision" in method:
                    acquisition = acquisition * decision_score(means, stds)
                if "reachability" in method:
                    assert reachability is not None
                    acquisition = acquisition * reachability[remaining]
                if "tail" in method:
                    assert tail_relevance is not None
                    acquisition = acquisition * (
                        tail_lambda + (1 - tail_lambda) * tail_relevance[remaining]
                    )
            choices = jnp.argsort(acquisition)[-n_select:]
            chosen = remaining[choices]
            indices.extend(int(i) for i in chosen)
            scores.extend(float(acquisition[i]) for i in choices)
            used = used.at[chosen].set(True)
            points = jnp.concatenate((points, candidates[chosen]), axis=0)
            targets = jnp.concatenate(
                (targets, backup_points(candidates[chosen])), axis=0
            )
            gps = fit_gps(points, targets)

    actual_backups = int(len(points) * n_actions)

    diagnostics = {
        "training_states": points,
        "q_values": targets,
        "selected_indices": original_indices[jnp.asarray(indices)],
        "selected_scores": jnp.asarray(scores),
        "reachability": reachability,
        "tail_lambda": tail_lambda,
        "tail_relevance": tail_relevance,
        "requested_backups": int(n_backups),
        "actual_backups": actual_backups,
        "n_backups": actual_backups,
    }
    return gps, diagnostics


def train_stage_gps(
    horizon: int,
    candidates_by_stage: dict[int, Float[Array, "n_candidates 2"]],
    rn_models: Float[Array, "horizon n_actions n_features"],
    rn_phi: Callable,
    key: Array,
    n_backups_per_stage: int,
    gp_y_bounds: tuple[float, float],
    reference_eta: float | None = None,
    sampling_method: str | Sequence[str] = "decision",
    n_rollouts: int = 1000,
    n_initial: int = 10,
    batch_size: int = 10,
    k: int = 20,
    gp_x_bounds: tuple[float, float] = (-8.0, 8.0),
    gp_length_scales: tuple[float, float] = (0.16, 0.16),
    gp_alpha: float = 1e-6,
    tail_lambda: float = 0.0,
    x0: float = 1.0,
    actions: Float[Array, " n_actions"] = ACTIONS,
    noise: Float[Array, " n_samples"] = NOISE,
    probabilities: Float[Array, " n_samples"] = PROBABILITIES,
) -> tuple[dict[int, list[GaussianProcessRegressor]], dict]:
    """Train one GP collection on stages T-1,...,1, independent of root queries.

    `reference_eta` is required only for reachability/tail acquisition. It sets
    the initial augmented state for RN rollouts and the RU exceedance threshold.
    `gp_y_bounds` determines GP normalization and is independent of this value.
    """
    method = normalize_sampling_method(sampling_method)
    if horizon < 2 or rn_models.shape[0] != horizon:
        raise ValueError("Require horizon >= 2 and RN models for each stage")
    if not (gp_y_bounds[0] < gp_y_bounds[1] <= 0.0):
        raise ValueError("Require gp_y_bounds[0] < gp_y_bounds[1] <= 0")
    if "tail" in method and not 0.0 <= tail_lambda <= 1.0:
        raise ValueError("Require 0 <= tail_lambda <= 1")
    if ("tail" in method or "reachability" in method) and (
        reference_eta is None or reference_eta <= 0
    ):
        raise ValueError("Rollout-based acquisition requires reference_eta > 0")

    actions = jnp.asarray(actions)
    noise = jnp.asarray(noise)
    probabilities = jnp.asarray(probabilities)
    rollout_paths = total_costs = tail_flags = None
    rollout_seconds = 0.0

    if "reachability" in method or "tail" in method:
        if reference_eta is None or reference_eta <= 0:
            raise ValueError("Rollout-based acquisition requires reference_eta > 0")
        if n_rollouts < 2:
            raise ValueError("At least two rollout states required")

        start = perf_counter()

        def rollout_policy(t, x, y):
            del y
            return predict_rn_policy(t, x, rn_models, rn_phi, actions)

        # Do not advance the stage-design PRNG stream when rollouts are needed.
        rollout_key = random.fold_in(key, 0)
        rollout_paths, total_costs = make_rollouts(
            key=rollout_key,
            x0=jnp.asarray(x0),
            y0=jnp.asarray(-reference_eta),
            n_rollouts=n_rollouts,
            horizon=horizon,
            policy=rollout_policy,
            noise=noise,
            probabilities=probabilities,
        )
        rollout_paths.block_until_ready()
        total_costs.block_until_ready()
        if "tail" in method:
            tail_flags = (total_costs > reference_eta).astype(jnp.int32)
        rollout_seconds = perf_counter() - start

    stage_gps = {}
    stage_diagnostics = {}
    for t in range(horizon - 1, 0, -1):
        if t not in candidates_by_stage:
            raise ValueError(f"Missing candidate states at t={t}")
        key, stage_key = random.split(key, 2)
        stage_gps[t], stage_diagnostics[t] = fit_stage(
            t=t,
            candidates=candidates_by_stage[t],
            rollout_states=None if rollout_paths is None else rollout_paths[:, t, :],
            tail_flags=tail_flags,
            next_gps=[] if t == horizon - 1 else stage_gps[t + 1],
            rn_models=rn_models,
            rn_phi=rn_phi,
            key=stage_key,
            n_backups=n_backups_per_stage,
            eta=reference_eta,
            n_initial=n_initial,
            batch_size=batch_size,
            k=k,
            sampling_method=method,
            gp_x_bounds=gp_x_bounds,
            gp_y_bounds=gp_y_bounds,
            gp_length_scales=gp_length_scales,
            gp_alpha=gp_alpha,
            actions=actions,
            noise=noise,
            probabilities=probabilities,
            tail_lambda=tail_lambda,
        )

    requested_training_backups = n_backups_per_stage * (horizon - 1)

    actual_training_backups = sum(
        d["actual_backups"] for d in stage_diagnostics.values()
    )

    diagnostics = {
        "stages": stage_diagnostics,
        "rollout_paths": rollout_paths,
        "rollout_costs": total_costs,
        "tail_flags": tail_flags,
        "tail_threshold": reference_eta if "tail" in method else None,
        "rollout_seconds": rollout_seconds,
        "requested_training_backups": requested_training_backups,
        "actual_training_backups": actual_training_backups,
        "n_backups": actual_training_backups,
        "gp_x_bounds": tuple(gp_x_bounds),
        "gp_y_bounds": tuple(gp_y_bounds),
        "gp_length_scales": tuple(gp_length_scales),
        "gp_alpha": gp_alpha,
        "sampling_method": method,
    }
    return stage_gps, diagnostics


def root_q_values(
    eta: float,
    x0: float,
    stage_gps: dict[int, list[GaussianProcessRegressor]],
    rn_models: Float[Array, "horizon n_actions n_features"],
    rn_phi: Callable,
    actions: Float[Array, " n_actions"] = ACTIONS,
    noise: Float[Array, " n_samples"] = NOISE,
    probabilities: Float[Array, " n_samples"] = PROBABILITIES,
) -> Float[Array, " n_actions"]:
    """Evaluate one root (x0,-eta) directly, without fitting a stage-0 GP."""
    return bellman_backup(
        0,
        x0,
        -eta,
        stage_gps[1],
        rn_models,
        rn_phi,
        actions=actions,
        noise=noise,
        probabilities=probabilities,
    )


def make_probabilistic_value(
    stage_gps: dict[int, list[GaussianProcessRegressor]],
    rn_models: Float[Array, "horizon n_actions n_features"],
    rn_phi: Callable,
    actions: Float[Array, " n_actions"] = ACTIONS,
    noise: Float[Array, " n_samples"] = NOISE,
    probabilities: Float[Array, " n_samples"] = PROBABILITIES,
) -> Callable:
    """Return the scalar-state continuation approximation of the trained GPs."""
    horizon = rn_models.shape[0]

    def value(t, x, y):
        x, y = jnp.asarray(x), jnp.asarray(y)
        if t == horizon:
            return jnp.maximum(y + terminal_cost(x), 0.0)
        if y >= 0:
            return y + predict_rn_value(t, x, rn_models, rn_phi)
        if t == 0:
            return jnp.min(
                bellman_backup(
                    0,
                    x,
                    y,
                    stage_gps[1],
                    rn_models,
                    rn_phi,
                    actions=actions,
                    noise=noise,
                    probabilities=probabilities,
                )
            )
        return bounded_gp_value(
            t, jnp.atleast_1d(x), y, stage_gps[t], rn_models, rn_phi
        )[0]

    return value


def make_probabilistic_policy(
    root_action: float,
    stage_gps: dict[int, list[GaussianProcessRegressor]],
    rn_models: Float[Array, "horizon n_actions n_features"],
    rn_phi: Callable,
    actions: Float[Array, " n_actions"] = ACTIONS,
) -> Callable:

    actions = jnp.asarray(actions)

    def policy(t, x, y):
        x, y = jnp.broadcast_arrays(
            jnp.asarray(x),
            jnp.asarray(y),
        )

        # Fixed action for the selected initial state.
        if t == 0:
            return jnp.full(
                x.shape,
                root_action,
                dtype=actions.dtype,
            )

        shape = x.shape
        xf = x.reshape(-1)
        yf = y.reshape(-1)

        rn_mask = yf >= 0.0
        selected = jnp.zeros(xf.shape, dtype=actions.dtype)

        # Risk-neutral region.
        if bool(jnp.any(rn_mask)):
            rn_actions = predict_rn_policy(
                t,
                xf[rn_mask],
                rn_models,
                rn_phi,
                actions,
            )

            selected = selected.at[rn_mask].set(rn_actions)

        # GP region.
        if bool(jnp.any(~rn_mask)):
            states = jnp.stack(
                (xf[~rn_mask], yf[~rn_mask]),
                axis=1,
            )

            indices = jnp.asarray(
                predict_gp_policy(stage_gps[t], states),
                dtype=jnp.int32,
            )

            selected = selected.at[~rn_mask].set(actions[indices])

        return selected.reshape(shape)

    return policy


def probabilistic_adp(
    horizon: int,
    eta: float,
    alpha: float,
    candidates_by_stage: dict[int, Float[Array, "n_candidates 2"]],
    rn_models: Float[Array, "horizon n_actions n_features"],
    rn_phi: Callable,
    key: Array,
    n_backups_per_stage: int,
    n_rollouts: int = 1000,
    n_initial: int = 10,
    batch_size: int = 10,
    k: int = 20,
    sampling_method: str | Sequence[str] = "decision",
    gp_x_bounds: tuple[float, float] = (-8.0, 8.0),
    gp_length_scales: tuple[float, float] = (0.16, 0.16),
    gp_alpha: float = 1e-6,
    tail_lambda: float = 0.0,
    x0: float = 1.0,
    actions: Float[Array, " n_actions"] = ACTIONS,
    noise: Float[Array, " n_samples"] = NOISE,
    probabilities: Float[Array, " n_samples"] = PROBABILITIES,
) -> tuple[Callable, Callable, dict[int, list[GaussianProcessRegressor]], dict]:
    """Fixed-threshold compatibility interface, including all acquisition rules.

    The returned diagnostic `n_backups` counts training action backups only.
    `total_backups` additionally counts the single direct root Q-vector.
    """
    if not (0.0 < alpha < 1.0) or eta <= 0:
        raise ValueError("Require 0 < alpha < 1 and eta > 0")
    stage_gps, diagnostics = train_stage_gps(
        horizon=horizon,
        candidates_by_stage=candidates_by_stage,
        rn_models=rn_models,
        rn_phi=rn_phi,
        key=key,
        n_backups_per_stage=n_backups_per_stage,
        gp_y_bounds=(-float(eta), 0.0),
        reference_eta=eta,
        sampling_method=sampling_method,
        n_rollouts=n_rollouts,
        n_initial=n_initial,
        batch_size=batch_size,
        k=k,
        gp_x_bounds=gp_x_bounds,
        gp_length_scales=gp_length_scales,
        gp_alpha=gp_alpha,
        tail_lambda=tail_lambda,
        x0=x0,
        actions=actions,
        noise=noise,
        probabilities=probabilities,
    )
    root_q = root_q_values(
        eta,
        x0,
        stage_gps,
        rn_models,
        rn_phi,
        actions=actions,
        noise=noise,
        probabilities=probabilities,
    )
    root_action = float(jnp.asarray(actions)[int(jnp.argmin(root_q))])
    value = make_probabilistic_value(
        stage_gps,
        rn_models,
        rn_phi,
        actions=actions,
        noise=noise,
        probabilities=probabilities,
    )
    policy = make_probabilistic_policy(
        root_action, stage_gps, rn_models, rn_phi, actions
    )

    diagnostics.update(
        {
            "root_q": root_q,
            "root_action": root_action,
            "root_backups": len(actions),
            "requested_total_backups": (
                diagnostics["requested_training_backups"] + len(actions)
            ),
            "total_backups": (diagnostics["actual_training_backups"] + len(actions)),
        }
    )
    return value, policy, stage_gps, diagnostics
