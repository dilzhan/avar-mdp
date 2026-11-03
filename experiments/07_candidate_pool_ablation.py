"""Compare nested candidate-pool sizes for random and decision sampling.

Run: python experiments/07_candidate_pool_ablation.py
Requires experiment 01. Exact and baseline references are optional.
Uses the same full 5,000-point candidate pools as experiment 05.
Completed runs are reused; no GP objects are saved.
"""

import os
import sys
from pathlib import Path
from time import perf_counter
from typing import cast

import jax
from matplotlib.figure import Figure

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import (
    ACQUISITION_ROLLOUTS,
    BACKUPS,
    BATCH_SIZE,
    EVAL_SEED,
    GP_ALPHA,
    GP_LENGTH_SCALES,
    GP_X_BOUNDS,
    K_NEIGHBORS,
    N_INITIAL,
    NUM_SEEDS,
    RESULTS_ROOT,
    SEED,
    TAIL_LAMBDA,
    environment_metadata,
    load_csv,
    load_eta,
    load_json,
    load_npz,
    load_reference,
    make_candidates,
    matches_problem,
    plot_budget_summary,
    prepare_rn,
    save_csv,
    save_json,
    save_npz,
    summarize_seeds,
)

from src.evaluation import evaluate_actions, evaluate_policy
from src.metrics import policy_error
from src.model import ACTIONS, NOISE, PROBABILITIES
from src.probabilistic_adp import probabilistic_adp

HORIZON = int(os.environ.get("T", "7"))
ALPHA = float(os.environ.get("ALPHA", "0.95"))
X0 = float(os.environ.get("X0", "1.0"))

POOL_SIZES = (1000, 2000, 3000, 4000, 5000)
METHODS = ("random", "decision")

EVAL_ROLLOUTS = 20_000

# Optional old baseline AVaR until experiment 04 has finished.
PROVISIONAL_BASELINE_AVAR = None


def points_per_stage(budget):
    """Number of candidate states that fit under the total backup budget."""
    n_actions = len(ACTIONS)
    return (budget - n_actions) // (n_actions * (HORIZON - 1))


def get_candidates(seed, eta, out):
    """Generate one full pool per seed; smaller pools use its prefixes."""
    path = out / "candidates" / f"seed{seed}.npz"
    if path.exists():
        data = load_npz(path)
        return {t: data[f"stage{t}"] for t in range(1, HORIZON)}

    full = make_candidates(
        jax.random.fold_in(jax.random.key(seed), 1),
        HORIZON,
        eta,
        n_candidates=max(POOL_SIZES),
    )
    save_npz(path, **{f"stage{t}": states for t, states in full.items()})
    return full


def get_exact_test(problem, eta, exact):
    """Load the exact-policy test states already saved by experiment 02."""
    if exact is None or not bool(
        jnp.isclose(eta, exact["eta_star_grid"], rtol=0, atol=1e-10)
    ):
        return None

    folder = RESULTS_ROOT / "02_baseline_budget" / f"t{HORIZON}_a{ALPHA}_x{X0}"
    path = folder / "exact_test_states.npz"
    if not path.exists():
        print("Exact test states missing; skipping policy disagreement.")
        return None

    config = load_json(folder / "config.json")
    if not matches_problem(config["problem"], problem) or not bool(
        jnp.isclose(config["eta"], eta, rtol=0, atol=1e-10)
    ):
        raise ValueError("Exact test states belong to a different problem or eta")
    return load_npz(path)


def save_diagnostics(path, metrics, diagnostics, actions=None):
    arrays = {
        "total_costs": metrics["total_costs"],
        "root_q": diagnostics["root_q"],
    }
    if actions is not None:
        arrays["policy_actions"] = actions
    for t, stage in diagnostics["stages"].items():
        for name in (
            "training_states",
            "q_values",
            "selected_indices",
            "selected_scores",
        ):
            arrays[f"t{t}_{name}"] = stage[name]
    save_npz(path, **arrays)


def matching_experiment05(problem, eta, full):
    """Return experiment 05's runs only if its setup and full pools match ours."""
    folder = RESULTS_ROOT / "05_fixed_eta_benchmark" / f"t{HORIZON}_a{ALPHA}_x{X0}"
    config_path = folder / "config.json"
    runs_path = folder / "runs.csv"
    if not (config_path.exists() and runs_path.exists()):
        return None, None

    config = load_json(config_path)
    expected = {
        "eta": eta,
        "candidate_pool_size": max(POOL_SIZES),
        "gp_x_bounds": list(GP_X_BOUNDS),
        "gp_length_scales": list(GP_LENGTH_SCALES),
        "gp_alpha": GP_ALPHA,
        "n_initial": N_INITIAL,
        "batch_size": BATCH_SIZE,
        "k_neighbors": K_NEIGHBORS,
        "tail_lambda": TAIL_LAMBDA,
        "acquisition_rollouts": ACQUISITION_ROLLOUTS,
        "eval_rollouts": EVAL_ROLLOUTS,
        "eval_seed": EVAL_SEED,
    }
    if not matches_problem(config.get("problem", {}), problem) or not matches_problem(
        config, expected
    ):
        return None, None

    # Experiment 05 used the same candidate RNG key. Confirm equality, not just seeds.
    for seed, full_pool in full.items():
        path = folder / "candidates" / f"seed{seed}.npz"
        if not path.exists():
            return None, None
        saved = load_npz(path)
        if any(
            f"stage{t}" not in saved
            or not bool(jnp.array_equal(saved[f"stage{t}"], full_pool[t]))
            for t in range(1, HORIZON)
        ):
            return None, None

    return folder, load_csv(runs_path)


def plot_results(summary, differences, out, exact, baseline):
    avar_refs = {}
    if exact is not None:
        avar_refs["Exact DP"] = exact["policy_avar"]
    if baseline is not None:
        avar_refs["Baseline ADP (1,000 episodes)"] = baseline["avar_mean"]
    elif PROVISIONAL_BASELINE_AVAR is not None:
        avar_refs["Baseline ADP (provisional)"] = PROVISIONAL_BASELINE_AVAR

    for method in METHODS:
        part = summary[summary["method"] == method].copy()
        part["method"] = part["pool_size"].map(lambda n: f"{int(n):,} candidates")
        ax = plot_budget_summary(part, "avar", references=avar_refs)
        ax.set_title(f"{method.capitalize()} sampling")
        cast(Figure, ax.figure).savefig(
            out / f"avar_vs_backups_{method}.png", bbox_inches="tight", dpi=160
        )
        plt.close(ax.figure)

    # Paired AVaR differences relative to the 5,000-candidate pool.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, method in zip(axes, METHODS):
        part = differences[differences["method"] == method].copy()
        part["method"] = part["pool_size"].map(lambda n: f"{int(n):,} candidates")
        plot_budget_summary(part, "avar_difference", ax=ax)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("AVaR minus 5,000-pool AVaR")
        ax.set_title(method.capitalize())
    fig.tight_layout()
    fig.savefig(out / "avar_difference.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    if not summary["policy_error_mean"].isna().all():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, method in zip(axes, METHODS):
            part = summary[summary["method"] == method].copy()
            part["method"] = part["pool_size"].map(lambda n: f"{int(n):,} candidates")
            plot_budget_summary(
                part, "policy_error", ax=ax, references={"Exact DP": 0.0}
            )
            ax.set_title(method.capitalize())
        fig.tight_layout()
        fig.savefig(out / "policy_error_vs_backups.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def run():
    if HORIZON < 2:
        raise ValueError("Require T >= 2")

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
    baseline = load_reference(
        {**problem, "comparison": "fixed_eta", "eta": eta}, "baseline", mode="auto"
    )

    out = RESULTS_ROOT / "07_candidate_pool_ablation" / f"t{HORIZON}_a{ALPHA}_x{X0}"
    config = {
        "problem": problem,
        "eta": eta,
        "pool_sizes": list(POOL_SIZES),
        "budgets": list(BACKUPS),
        "methods": list(METHODS),
        "seeds": list(range(SEED, SEED + NUM_SEEDS)),
        "gp_x_bounds": list(GP_X_BOUNDS),
        "gp_length_scales": list(GP_LENGTH_SCALES),
        "gp_alpha": GP_ALPHA,
        "n_initial": N_INITIAL,
        "batch_size": BATCH_SIZE,
        "k_neighbors": K_NEIGHBORS,
        "tail_lambda": TAIL_LAMBDA,
        "acquisition_rollouts": ACQUISITION_ROLLOUTS,
        "eval_rollouts": EVAL_ROLLOUTS,
        "eval_seed": EVAL_SEED,
    }
    config_path = out / "config.json"
    if config_path.exists() and load_json(config_path) != config:
        raise ValueError(f"Different settings saved in {out}; use a new directory")
    save_json(config_path, config)

    test = get_exact_test(problem, eta, exact)
    full = {seed: get_candidates(seed, eta, out) for seed in config["seeds"]}
    source_folder, source_runs = matching_experiment05(problem, eta, full)
    if source_folder is not None:
        print(
            "Compatible experiment 05 results found; reusing complete 5,000-point runs."
        )

    rn_models, rn_phi = prepare_rn(HORIZON)
    runs_path = out / "runs.csv"
    rows = load_csv(runs_path).to_dict("records") if runs_path.exists() else []

    for seed in config["seeds"]:
        for pool_size in POOL_SIZES:
            candidates = {t: states[:pool_size] for t, states in full[seed].items()}
            for target in BACKUPS:
                n_points = points_per_stage(target)
                if n_points < 1 or n_points > pool_size:
                    continue  # Do not pretend exhausted pools used the requested budget.

                actual_expected = len(ACTIONS) * (1 + n_points * (HORIZON - 1))
                for method in METHODS:
                    completed = next(
                        (
                            r
                            for r in rows
                            if r["seed"] == seed
                            and r["pool_size"] == pool_size
                            and r["target_budget"] == target
                            and r["method"] == method
                        ),
                        None,
                    )
                    if completed is not None:
                        if not (RESULTS_ROOT / completed["diagnostics_path"]).exists():
                            raise FileNotFoundError(completed["diagnostics_path"])
                        continue

                    # Avoid repeating experiment 05's identical full-pool runs.
                    source_row = None
                    if pool_size == max(POOL_SIZES) and source_runs is not None:
                        subset = source_runs[
                            (source_runs["seed"] == seed)
                            & (source_runs["target_budget"] == target)
                            & (source_runs["method"] == method)
                        ]
                        if len(subset) == 1:
                            candidate_row = subset.iloc[0]
                            if source_folder is None:
                                raise RuntimeError(
                                    "Experiment 05 runs found but not its folder"
                                )
                            source_path = (
                                source_folder
                                / "runs"
                                / f"{method}_seed{seed}_budget{target}.npz"
                            )
                            valid_error = test is None or pd.notna(
                                candidate_row["policy_error"]
                            )
                            if (
                                source_path.exists()
                                and int(candidate_row["actual_backups"])
                                == actual_expected
                                and valid_error
                            ):
                                source_row = candidate_row

                    if source_row is not None:
                        row = source_row.to_dict()
                        row["source"] = "05_fixed_eta_benchmark"
                        path = source_path
                    else:
                        start = perf_counter()
                        _, policy, _, diagnostics = probabilistic_adp(
                            horizon=HORIZON,
                            eta=eta,
                            alpha=ALPHA,
                            candidates_by_stage=candidates,
                            rn_models=rn_models,
                            rn_phi=rn_phi,
                            key=jax.random.key(seed),
                            n_backups_per_stage=n_points * len(ACTIONS),
                            n_rollouts=ACQUISITION_ROLLOUTS,
                            n_initial=N_INITIAL,
                            batch_size=BATCH_SIZE,
                            k=K_NEIGHBORS,
                            sampling_method=method,
                            gp_x_bounds=GP_X_BOUNDS,
                            gp_length_scales=GP_LENGTH_SCALES,
                            gp_alpha=GP_ALPHA,
                            tail_lambda=TAIL_LAMBDA,
                            x0=X0,
                            actions=ACTIONS,
                            noise=NOISE,
                            probabilities=PROBABILITIES,
                        )
                        train_seconds = perf_counter() - start
                        if diagnostics["total_backups"] != actual_expected:
                            raise RuntimeError("Unexpected number of Bellman backups")

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
                            disagreement = float(
                                policy_error(actions, test["exact_actions"])
                            )

                        path = (
                            out
                            / "runs"
                            / f"{method}_pool{pool_size}_seed{seed}_budget{target}.npz"
                        )
                        save_diagnostics(path, metrics, diagnostics, actions)
                        row = {
                            "method": method,
                            "seed": seed,
                            "target_budget": target,
                            "actual_backups": diagnostics["total_backups"],
                            "root_ru": float(
                                eta + jnp.min(diagnostics["root_q"]) / (1 - ALPHA)
                            ),
                            "avar": metrics["avar"],
                            "var": metrics["var"],
                            "mean": metrics["mean"],
                            "ru_objective": metrics["ru_objective"],
                            "policy_error": disagreement,
                            "train_seconds": train_seconds,
                            "eval_seconds": eval_seconds,
                            "source": "07_candidate_pool_ablation",
                        }

                    row["pool_size"] = pool_size
                    row["candidate_utilization"] = n_points / pool_size
                    row["diagnostics_path"] = str(path.relative_to(RESULTS_ROOT))
                    rows.append(row)
                    save_csv(runs_path, rows)
                    print(
                        f"{method}, pool {pool_size}, seed {seed}, target {target}: "
                        f"{actual_expected} backups, AVaR={row['avar']:.4f}",
                        flush=True,
                    )

    data = pd.DataFrame(rows)
    summary = summarize_seeds(
        data,
        ("method", "pool_size", "target_budget"),
        (
            "actual_backups",
            "avar",
            "var",
            "mean",
            "ru_objective",
            "root_ru",
            "policy_error",
            "candidate_utilization",
            "train_seconds",
            "eval_seconds",
        ),
    )
    summary["budget"] = summary["actual_backups_mean"]
    save_csv(out / "summary.csv", summary)

    reference = data[data["pool_size"] == max(POOL_SIZES)][
        ["method", "seed", "target_budget", "actual_backups", "avar"]
    ].rename(columns={"actual_backups": "reference_backups", "avar": "reference_avar"})
    paired = data.merge(reference, on=["method", "seed", "target_budget"])
    paired = paired[
        (paired["pool_size"] < max(POOL_SIZES))
        & (paired["actual_backups"] == paired["reference_backups"])
    ].copy()
    paired["avar_difference"] = paired["avar"] - paired["reference_avar"]
    save_csv(out / "paired_differences.csv", paired)
    differences = summarize_seeds(
        paired,
        ("method", "pool_size", "target_budget"),
        ("actual_backups", "avar_difference"),
    )
    differences["budget"] = differences["actual_backups_mean"]
    save_csv(out / "paired_summary.csv", differences)

    plot_results(summary, differences, out, exact, baseline)
    save_json(out / "environment.json", environment_metadata())
    print(f"Saved to {out}")


if __name__ == "__main__":
    run()
