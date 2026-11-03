"""Compute the exact AVaR reference and its policy cost distribution.

Run: python experiments/00_exact_dp.py
The full disturbance grid is used. Reruns replace the matching reference.
Exact DP grows exponentially with the horizon.
"""

import os
import sys
from pathlib import Path
from time import perf_counter

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import (
    ETA_MAX,
    ETA_MIN,
    ETA_STEP,
    RESULTS_ROOT,
    environment_metadata,
    load_json,
    matches_problem,
    save_csv,
    save_json,
)

from src.exact_dp import avar_dp
from src.model import ACTIONS, NOISE, PROBABILITIES, cost, terminal_cost, transition
from src.risk import avar, ru_objective

HORIZON = int(os.environ.get("T", "3"))
ALPHA = float(os.environ.get("ALPHA", "0.95"))
X0 = float(os.environ.get("X0", "1.0"))


def save_policy_distribution(policy, eta, out):
    """Enumerate all weighted trajectories of the selected fixed-eta policy."""
    x = jnp.array([X0])
    accumulated = jnp.zeros(1)
    weights = jnp.ones(1)
    policy_rows = []

    for t in range(HORIZON):
        y = accumulated - eta
        actions = policy(t, x, y)
        policy_rows.append(
            {
                "t": jnp.full(len(x), t),
                "path_index": jnp.arange(len(x)),
                "x": x,
                "y": y,
                "action": actions,
                "probability": weights,
            }
        )
        accumulated = jnp.repeat(accumulated + cost(x, actions), len(NOISE))
        x = transition(x[:, None], actions[:, None], NOISE).ravel()
        weights = (weights[:, None] * PROBABILITIES).ravel()

    save_csv(
        out / "optimal_policy.csv",
        {
            name: jnp.concatenate([row[name] for row in policy_rows]).tolist()
            for name in policy_rows[0]
        },
    )

    total_costs = accumulated + terminal_cost(x)
    costs, inverse = jnp.unique(total_costs, return_inverse=True)
    weights = jnp.bincount(inverse, weights=weights, length=len(costs))
    save_csv(
        out / "cost_distribution.csv",
        {"cost": costs.tolist(), "probability": weights.tolist()},
    )
    return costs, weights


def run():
    out = RESULTS_ROOT / "00_exact_dp" / f"t{HORIZON}_a{ALPHA}_x{X0}"
    eta_grid = jnp.arange(ETA_MIN, ETA_MAX + ETA_STEP / 2, ETA_STEP)

    value, policy, backup_count = avar_dp(HORIZON, ACTIONS, NOISE, PROBABILITIES)
    value = jax.jit(value, static_argnums=0)
    policy = jax.jit(policy, static_argnums=0)

    rows = []
    for eta in eta_grid:
        start = perf_counter()
        inner_value = float(value(0, X0, -eta))
        ru = float(eta + inner_value / (1 - ALPHA))
        rows.append(
            {
                "eta": float(eta),
                "inner_value": inner_value,
                "ru_objective": ru,
                "bellman_backups": int(backup_count(0, X0, -eta)),
                "seconds": perf_counter() - start,
            }
        )
        # Save each completed threshold: exact evaluations can be expensive.
        save_csv(out / "eta_grid.csv", rows)
        print(f"eta={eta:g}, RU={ru:.9f}", flush=True)

    best = min(rows, key=lambda row: row["ru_objective"])
    eta = best["eta"]

    start = perf_counter()
    costs, weights = save_policy_distribution(policy, eta, out)
    policy_avar, policy_var = avar(costs, weights, ALPHA)
    policy_ru = ru_objective(costs, weights, eta, ALPHA)
    evaluation_seconds = perf_counter() - start

    assert bool(jnp.isclose(jnp.sum(weights), 1.0, rtol=0, atol=1e-10))
    assert bool(jnp.isclose(policy_ru, best["ru_objective"], rtol=0, atol=1e-8))

    problem = {
        "horizon": HORIZON,
        "alpha": ALPHA,
        "x0": X0,
        "actions": ACTIONS.tolist(),
        "noise": NOISE.tolist(),
        "probabilities": PROBABILITIES.tolist(),
    }
    reference = {
        "eta_star_grid": eta,
        "exact_ru_grid_minimum": best["ru_objective"],
        "policy_avar": policy_avar,
        "policy_var": policy_var,
        "policy_mean_cost": float(weights @ costs),
        "policy_ru_at_eta": policy_ru,
        "eta_grid": [row["eta"] for row in rows],
        "ru_curve": [row["ru_objective"] for row in rows],
        "cost_distribution_csv": str(
            out.relative_to(RESULTS_ROOT) / "cost_distribution.csv"
        ),
        "optimal_policy_csv": str(out.relative_to(RESULTS_ROOT) / "optimal_policy.csv"),
    }
    summary = {
        **problem,
        **reference,
        "eta_min": ETA_MIN,
        "eta_max": ETA_MAX,
        "eta_step": ETA_STEP,
        "eta_search_seconds": sum(row["seconds"] for row in rows),
        "eta_search_bellman_backups": sum(row["bellman_backups"] for row in rows),
        "evaluation_seconds": evaluation_seconds,
        "evaluation_policy_bellman_backups": sum(
            len(NOISE) ** t * int(backup_count(t, None, None)) for t in range(HORIZON)
        ),
        "evaluation": "Full weighted path enumeration, including terminal cost",
        "environment": environment_metadata(),
    }
    save_json(out / "summary.json", summary)

    # Maintain one reference per problem, without deleting other configurations.
    ref_path = RESULTS_ROOT / "references" / "exact.json"
    records = load_json(ref_path) if ref_path.exists() else []
    records = [r for r in records if not matches_problem(r["problem"], problem)]
    records.append({"problem": problem, "reference": reference})
    save_json(ref_path, records)

    print(
        f"Best eta={eta:g}, grid RU={best['ru_objective']:.9f}, "
        f"policy AVaR={policy_avar:.9f}"
    )
    if eta == float(eta_grid[0]) or eta == float(eta_grid[-1]):
        print("Eta minimum is on the grid boundary; check the search interval.")
    print(f"Saved to {out}")


if __name__ == "__main__":
    run()
