"""Train and save a reusable high-budget baseline ADP policy at fixed eta.

Run: T=7 python experiments/04_baseline_reference.py
Requires experiment 01. Policies are used by experiment 05 for paired evaluation.
Evaluation here is optional: EVAL_BASELINE=1 T=7 python experiments/04_baseline_reference.py

Requires cloudpickle to serialize the stateful ADP policy closure.
"""

import os
import sys
from pathlib import Path
from time import perf_counter

import cloudpickle
import jax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import (
    EVAL_ROLLOUTS,
    EVAL_SEED,
    NUM_SEEDS,
    RESULTS_ROOT,
    SEED,
    environment_metadata,
    load_csv,
    load_eta,
    load_json,
    prepare_rn,
    save_csv,
    save_json,
)

from src.avar_adp import make_policy_initializer, run_adp
from src.evaluation import evaluate_policy
from src.model import ACTIONS, NOISE, PROBABILITIES
from src.risk_neutral_approx import predict_policy as predict_rn_policy

HORIZON = int(os.environ.get("T", "3"))
ALPHA = float(os.environ.get("ALPHA", "0.95"))
X0 = float(os.environ.get("X0", "1.0"))
NUM_SEEDS = int(os.environ.get("NUM_SEEDS", NUM_SEEDS))

N_EPISODES = int(os.environ.get("N_EPISODES", "1000"))
INITIALIZER_ROLLOUTS = 20
EVAL_BASELINE = os.environ.get("EVAL_BASELINE", "0") == "1"


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
    seeds = list(range(SEED, SEED + NUM_SEEDS))
    out = RESULTS_ROOT / "04_baseline_reference" / f"t{HORIZON}_a{ALPHA}_x{X0}"

    # Training configuration deliberately excludes evaluation options.
    config = {
        "problem": problem,
        "eta": eta,
        "seeds": seeds,
        "n_episodes": N_EPISODES,
        "initializer_rollouts": INITIALIZER_ROLLOUTS,
    }
    config_path = out / "config.json"
    if config_path.exists() and load_json(config_path) != config:
        raise ValueError(f"Different baseline training settings already saved in {out}")
    save_json(config_path, config)

    rn_models, rn_phi = prepare_rn(HORIZON)
    train_path = out / "training.csv"
    training = load_csv(train_path).to_dict("records") if train_path.exists() else []

    for seed in seeds:
        policy_path = out / "policies" / f"seed{seed}.pkl"
        if not policy_path.exists():
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
            start = perf_counter()
            _, policy, _, root_history, backup_history = run_adp(
                horizon=HORIZON,
                eta=eta,
                rn_models=rn_models,
                rn_phi=rn_phi,
                key=key,
                actions=ACTIONS,
                noise=NOISE,
                probabilities=PROBABILITIES,
                n_episodes=N_EPISODES,
                initializer=initializer,
                tol=None,
                x0=X0,
            )
            train_seconds = perf_counter() - start

            # Serialize *before* evaluation: a baseline initializer can cache
            # values on first use, so evaluation must not change the saved policy.
            policy_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = policy_path.with_suffix(".pkl.tmp")
            with tmp_path.open("wb") as f:
                cloudpickle.dump(policy, f)
            tmp_path.replace(policy_path)

            training.append(
                {
                    "seed": seed,
                    "n_episodes": N_EPISODES,
                    "actual_backups": int(backup_history[-1]),
                    "root_ru": float(eta + root_history[-1] / (1.0 - ALPHA)),
                    "train_seconds": train_seconds,
                }
            )
            save_csv(train_path, training)
            print(
                f"Saved seed {seed}: {int(backup_history[-1])} backups, "
                f"{train_seconds:.1f}s training",
                flush=True,
            )
        else:
            print(f"Reusing policy for seed {seed}: {policy_path}", flush=True)

        if EVAL_BASELINE:
            # Reload a fresh policy for independent evaluation; never modify the
            # serialized checkpoint through evaluation side effects.
            with policy_path.open("rb") as f:
                policy = cloudpickle.load(f)
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
            print(
                f"Evaluation seed {seed}: AVaR={float(metrics['avar']):.6f}, "
                f"{perf_counter() - start:.1f}s",
                flush=True,
            )

    save_json(out / "environment.json", environment_metadata())
    print(f"Policies saved to {out / 'policies'}")
    if not EVAL_BASELINE:
        print(
            "Skipped policy evaluation; experiment 05 evaluates these policies on shared noise paths."
        )


if __name__ == "__main__":
    run()
