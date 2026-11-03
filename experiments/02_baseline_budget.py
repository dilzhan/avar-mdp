"""Baseline ADP across Bellman-backup budgets and training seeds.

Run: python experiments/02_baseline_budget.py
Requires an eta estimate from experiment 01. Uses exact T=3 policy states
for optional policy disagreement when experiment 00 is available.
"""

import os
import sys
from pathlib import Path
from time import perf_counter

import jax

jax.config.update("jax_enable_x64", True)
from typing import cast

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import (
    BACKUPS,
    EVAL_ROLLOUTS,
    EVAL_SEED,
    NUM_SEEDS,
    RESULTS_ROOT,
    SEED,
    environment_metadata,
    load_csv,
    load_eta,
    load_json,
    load_npz,
    load_reference,
    plot_budget_summary,
    prepare_rn,
    save_csv,
    save_json,
    save_npz,
    summarize_seeds,
)

from src.avar_adp import make_policy_initializer, run_adp
from src.evaluation import evaluate_actions, evaluate_policy
from src.metrics import policy_error
from src.model import ACTIONS, NOISE, PROBABILITIES
from src.risk_neutral_approx import predict_policy as predict_rn_policy

HORIZON = int(os.environ.get("T", "3"))
ALPHA = float(os.environ.get("ALPHA", "0.95"))
X0 = float(os.environ.get("X0", "1.0"))

INITIALIZER_ROLLOUTS = 20
EXACT_TEST_STATES_PER_STAGE = 50


def train(n_episodes, seed, eta, rn_models, rn_phi):
    """Train afresh with reproducible, separate ADP and initializer keys."""
    key = jax.random.key(seed)
    initializer = make_policy_initializer(
        HORIZON,
        predict_rn_policy,
        rn_models,
        rn_phi,
        jax.random.fold_in(key, 1),
        n_rollouts=INITIALIZER_ROLLOUTS,
        actions=ACTIONS,
        noise=NOISE,
        probabilities=PROBABILITIES,
    )
    return run_adp(
        horizon=HORIZON,
        eta=eta,
        rn_models=rn_models,
        rn_phi=rn_phi,
        key=key,
        actions=ACTIONS,
        noise=NOISE,
        probabilities=PROBABILITIES,
        n_episodes=n_episodes,
        initializer=initializer,
        tol=None,
        x0=X0,
    )


def calibration_history(seed, eta, rn_models, rn_phi, out):
    """Train long enough to determine the last episode within each budget."""
    path = out / f"calibration_seed{seed}.npz"
    if path.exists():
        return load_npz(path)["backup_history"]

    # Rough initial guess; double if the largest budget has not been reached.
    n_episodes = max(1, max(BACKUPS) // (2 * len(ACTIONS) * HORIZON))
    while True:
        start = perf_counter()
        _, _, _, roots, backups = train(n_episodes, seed, eta, rn_models, rn_phi)
        print(
            f"Calibration seed={seed}: {n_episodes} episodes, "
            f"{int(backups[-1])} backups, {perf_counter() - start:.1f}s",
            flush=True,
        )
        if int(backups[-1]) >= max(BACKUPS):
            save_npz(path, root_history=roots, backup_history=backups)
            return backups
        n_episodes *= 2


def exact_test_states(exact, eta, out):
    """Sample states weighted by exact-policy occupancy at each time stage."""
    if exact is None or not bool(jnp.isclose(eta, exact["eta_star_grid"], atol=1e-8)):
        return None

    path = out / "exact_test_states.npz"
    if path.exists():
        return load_npz(path)

    policy_path = RESULTS_ROOT / exact["optimal_policy_csv"]
    rows = load_csv(policy_path)
    states, actions = [], []
    key = jax.random.key(EVAL_SEED)
    for t in range(HORIZON):
        stage = rows[rows["t"] == t].reset_index(drop=True)
        probabilities = jnp.asarray(stage["probability"].to_numpy())
        probabilities = probabilities / probabilities.sum()
        key, subkey = jax.random.split(key)
        indices = jax.random.choice(
            subkey,
            len(stage),
            shape=(1 if t == 0 else EXACT_TEST_STATES_PER_STAGE,),
            replace=True,
            p=probabilities,
        )
        selected = stage.iloc[indices.tolist()]
        states.extend(
            (t, float(row.x), float(row.y)) for row in selected.itertuples(index=False)
        )
        actions.extend(float(a) for a in selected["action"])

    result = {
        "states": jnp.asarray(states),
        "exact_actions": jnp.asarray(actions),
    }
    save_npz(path, **result)
    return result


def run():
    problem = {
        "horizon": HORIZON,
        "alpha": ALPHA,
        "x0": X0,
        "actions": ACTIONS.tolist(),
        "noise": NOISE.tolist(),
        "probabilities": PROBABILITIES.tolist(),
    }
    eta = load_eta(problem)
    exact = load_reference(problem, "exact", mode="auto")
    out = RESULTS_ROOT / "02_baseline_budget" / f"t{HORIZON}_a{ALPHA}_x{X0}"

    config = {
        "problem": problem,
        "eta": eta,
        "budgets": list(BACKUPS),
        "seeds": list(range(SEED, SEED + NUM_SEEDS)),
        "initializer_rollouts": INITIALIZER_ROLLOUTS,
        "eval_rollouts": EVAL_ROLLOUTS,
        "eval_seed": EVAL_SEED,
        "exact_test_states_per_stage": EXACT_TEST_STATES_PER_STAGE,
        "exact_eta": None if exact is None else exact["eta_star_grid"],
    }
    config_path = out / "config.json"
    if config_path.exists() and load_json(config_path) != config:
        raise ValueError(
            f"Different settings already saved in {out}; use a new directory"
        )
    save_json(config_path, config)
    test = exact_test_states(exact, eta, out)
    rn_models, rn_phi = prepare_rn(HORIZON)

    runs_path = out / "runs.csv"
    rows = load_csv(runs_path).to_dict("records") if runs_path.exists() else []

    for seed in config["seeds"]:
        if all(
            any(r["seed"] == seed and r["target_budget"] == b for r in rows)
            for b in BACKUPS
        ):
            continue

        history = calibration_history(seed, eta, rn_models, rn_phi, out)
        for target in BACKUPS:
            if any(r["seed"] == seed and r["target_budget"] == target for r in rows):
                continue
            # Cumulative backup counts are nondecreasing. Do not exceed target.
            episodes = int(jnp.sum(history <= target))
            if episodes == 0:
                raise ValueError(f"Budget {target} is smaller than the first episode")

            start = perf_counter()
            _, policy, _, roots, backups = train(episodes, seed, eta, rn_models, rn_phi)
            train_seconds = perf_counter() - start
            actual = int(backups[-1])
            if actual != int(history[episodes - 1]) or actual > target:
                raise RuntimeError("Training does not reproduce the calibrated budget")

            start = perf_counter()
            metrics = evaluate_policy(
                policy=policy,
                horizon=HORIZON,
                alpha=ALPHA,
                eta=eta,
                key=jax.random.key(EVAL_SEED),
                x0=X0,
                noise=NOISE,
                probabilities=PROBABILITIES,
                n_rollouts=EVAL_ROLLOUTS,
            )
            eval_seconds = perf_counter() - start

            actions = None
            disagreement = float("nan")
            if test is not None:
                actions = evaluate_actions(policy, test["states"])
                disagreement = float(policy_error(actions, test["exact_actions"]))

            file = out / "runs" / f"seed{seed}_budget{target}.npz"
            arrays = {
                "root_history": roots,
                "backup_history": backups,
                "total_costs": metrics["total_costs"],
            }
            if actions is not None:
                arrays["policy_actions"] = actions
            save_npz(file, **arrays)

            rows.append(
                {
                    "method": "baseline",
                    "seed": seed,
                    "target_budget": target,
                    "n_episodes": episodes,
                    "actual_backups": actual,
                    "root_value": float(roots[-1]),
                    "root_ru": float(eta + roots[-1] / (1 - ALPHA)),
                    "avar": metrics["avar"],
                    "var": metrics["var"],
                    "mean": metrics["mean"],
                    "ru_objective": metrics["ru_objective"],
                    "policy_error": disagreement,
                    "train_seconds": train_seconds,
                    "eval_seconds": eval_seconds,
                }
            )
            save_csv(runs_path, rows)
            print(
                f"Seed {seed}, target {target}: {actual} backups, AVaR={metrics['avar']:.4f}",
                flush=True,
            )

    data = pd.DataFrame(rows)
    summary = summarize_seeds(
        data,
        ("method", "target_budget"),
        (
            "actual_backups",
            "avar",
            "mean",
            "ru_objective",
            "root_ru",
            "policy_error",
            "train_seconds",
            "eval_seconds",
        ),
    )
    summary["budget"] = summary["actual_backups_mean"]
    save_csv(out / "summary.csv", summary)

    # Monte Carlo AVaR is not directly comparable to enumerated exact AVaR.
    # Experiment 03 plots a paired comparison on identical disturbances.
    for metric in ("avar", "mean", "ru_objective"):
        ax = plot_budget_summary(summary, metric)
        cast(Figure, ax.figure).savefig(
            out / f"{metric}_vs_backups.png", bbox_inches="tight", dpi=160
        )
        plt.close(ax.figure)

    save_json(out / "environment.json", environment_metadata())
    print(f"Saved to {out}")


if __name__ == "__main__":
    run()
