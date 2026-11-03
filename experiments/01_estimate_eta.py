"""Estimate a near-optimal eta using exact DP or a pilot ADP policy.

Run: python experiments/01_estimate_eta.py
Exact references take priority. Otherwise, reuse a saved estimate or run
RN VaR -> pilot ADP -> policy VaR. The pilot estimate is heuristic.
"""

import os
import sys
from pathlib import Path
from time import perf_counter

import jax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import (
    EVAL_ROLLOUTS,
    EVAL_SEED,
    N_EPISODES,
    RESULTS_ROOT,
    SEED,
    environment_metadata,
    load_eta,
    load_json,
    load_reference,
    matches_problem,
    prepare_rn,
    save_csv,
    save_json,
)

from src.avar_adp import run_adp
from src.evaluation import evaluate_policy
from src.model import ACTIONS, NOISE, PROBABILITIES
from src.risk_neutral_approx import make_policy

HORIZON = int(os.environ.get("T", "3"))
ALPHA = float(os.environ.get("ALPHA", "0.95"))
X0 = float(os.environ.get("X0", "1.0"))


def run():
    problem = {
        "horizon": HORIZON,
        "alpha": ALPHA,
        "x0": X0,
        "actions": ACTIONS.tolist(),
        "noise": NOISE.tolist(),
        "probabilities": PROBABILITIES.tolist(),
    }
    estimates_path = RESULTS_ROOT / "01_estimate_eta" / "estimates.json"

    exact = load_reference(problem, "exact", mode="auto")
    if exact is not None:
        result = {
            "problem": problem,
            "eta": float(exact["eta_star_grid"]),
            "method": "exact_eta_grid",
            "exact_ru_grid_minimum": exact["exact_ru_grid_minimum"],
        }
    else:
        try:
            eta = load_eta(problem)
        except FileNotFoundError:
            pass
        else:
            print(f"Reusing eta={eta:.6f} from {estimates_path}")
            return

        out = RESULTS_ROOT / "01_estimate_eta" / f"t{HORIZON}_a{ALPHA}_x{X0}"

        start = perf_counter()
        rn_models, rn_phi = prepare_rn(HORIZON)
        rn_seconds = perf_counter() - start
        rn_policy = make_policy(rn_models, rn_phi, ACTIONS)

        start = perf_counter()
        rn_eval = evaluate_policy(
            policy=rn_policy,
            horizon=HORIZON,
            alpha=ALPHA,
            eta=0.0,
            key=jax.random.key(EVAL_SEED),
            x0=X0,
            noise=NOISE,
            probabilities=PROBABILITIES,
            n_rollouts=EVAL_ROLLOUTS,
        )
        rn_eval_seconds = perf_counter() - start
        eta_0 = rn_eval["var"]
        print(f"RN VaR={eta_0:.6f}; training pilot ADP", flush=True)

        start = perf_counter()
        _, policy, _, _, backup_history = run_adp(
            horizon=HORIZON,
            eta=eta_0,
            rn_models=rn_models,
            rn_phi=rn_phi,
            key=jax.random.key(SEED),
            actions=ACTIONS,
            noise=NOISE,
            probabilities=PROBABILITIES,
            n_episodes=N_EPISODES,
            initializer=None,
            tol=None,
            x0=X0,
        )
        train_seconds = perf_counter() - start

        # Both evaluations use the same disturbance paths.
        start = perf_counter()
        adp_eval = evaluate_policy(
            policy=policy,
            horizon=HORIZON,
            alpha=ALPHA,
            eta=eta_0,
            key=jax.random.key(EVAL_SEED),
            x0=X0,
            noise=NOISE,
            probabilities=PROBABILITIES,
            n_rollouts=EVAL_ROLLOUTS,
        )
        adp_eval_seconds = perf_counter() - start

        save_csv(
            out / "pilot_metrics.csv",
            [
                {
                    "policy": name,
                    "eta": eta,
                    "empirical_var": metrics["var"],
                    "empirical_avar": metrics["avar"],
                    "mean_cost": metrics["mean"],
                    "ru_at_eta": metrics["ru_objective"],
                    "evaluation_seconds": seconds,
                }
                for name, eta, metrics, seconds in (
                    ("risk_neutral", 0.0, rn_eval, rn_eval_seconds),
                    ("pilot_adp", eta_0, adp_eval, adp_eval_seconds),
                )
            ],
        )
        result = {
            "problem": problem,
            "eta": float(adp_eval["var"]),
            "method": "pilot_adp_policy_var",
            "training_eta": float(eta_0),
            "empirical_avar": adp_eval["avar"],
            "mean_cost": adp_eval["mean"],
            "ru_at_training_eta": adp_eval["ru_objective"],
            "seed": SEED,
            "eval_seed": EVAL_SEED,
            "n_episodes": N_EPISODES,
            "n_rollouts": EVAL_ROLLOUTS,
            "bellman_backups": int(backup_history[-1]),
            "rn_prepare_seconds": rn_seconds,
            "rn_eval_seconds": rn_eval_seconds,
            "train_seconds": train_seconds,
            "adp_eval_seconds": adp_eval_seconds,
            "environment": environment_metadata(),
        }
        print(f"Pilot AVaR={adp_eval['avar']:.6f}, mean={adp_eval['mean']:.6f}")

    records = load_json(estimates_path) if estimates_path.exists() else []
    records = [r for r in records if not matches_problem(r["problem"], problem)]
    records.append(result)
    save_json(estimates_path, records)

    print(f"Eta estimate={result['eta']:.6f} ({result['method']})")
    print(f"Saved to {estimates_path}")


if __name__ == "__main__":
    run()
