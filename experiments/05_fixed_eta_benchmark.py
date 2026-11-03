"""Compare six probabilistic ADP sampling rules at a fixed estimated eta.

Run: python experiments/05_fixed_eta_benchmark.py
Requires experiment 01. Exact and high-budget baseline references are optional.
Completed runs are reused; GPs are never saved.
Baseline policies are evaluated once per seed on the GP evaluation trajectories.
"""

import hashlib
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
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import (
    ACQUISITION_ROLLOUTS,
    BACKUPS,
    BATCH_SIZE,
    CANDIDATE_POOL_SIZE,
    EVAL_ROLLOUTS,
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

from src.evaluation import evaluate_actions, evaluate_policy, policy_metrics
from src.metrics import policy_error
from src.model import ACTIONS, NOISE, PROBABILITIES, cost, terminal_cost, transition
from src.probabilistic_adp import probabilistic_adp
from src.risk import rollout_ru_objective

HORIZON = int(os.environ.get("T", "7"))
ALPHA = float(os.environ.get("ALPHA", "0.95"))
X0 = float(os.environ.get("X0", "1.0"))
NUM_SEEDS = int(os.environ.get("NUM_SEEDS", NUM_SEEDS))

METHODS = {
    "random": "random",
    "uncertainty": "uncertainty",
    "decision": "decision",
    "decision_reachability": ("decision", "reachability"),
    "decision_tail": ("decision", "tail"),
    "decision_reachability_tail": ("decision", "reachability", "tail"),
}


def stage_budget(total_budget):
    """Equal action backups per stage, reserving one root Q-vector."""
    n_actions = len(ACTIONS)
    n_points = (total_budget - n_actions) // (n_actions * (HORIZON - 1))
    if n_points < 1 or n_points > CANDIDATE_POOL_SIZE:
        raise ValueError(f"Budget {total_budget} does not fit the candidate pool")
    return n_points * n_actions


def get_candidates(seed, eta, out):
    path = out / "candidates" / f"seed{seed}.npz"
    if path.exists():
        arrays = load_npz(path)
        return {t: arrays[f"stage{t}"] for t in range(1, HORIZON)}

    candidates = make_candidates(
        jax.random.fold_in(jax.random.key(seed), 1), HORIZON, eta
    )
    save_npz(path, **{f"stage{t}": states for t, states in candidates.items()})
    return candidates


def get_exact_test(problem, eta, exact):
    """Use precisely the exact-policy test states saved by experiment 02."""
    if exact is None or not bool(
        jnp.isclose(eta, exact["eta_star_grid"], rtol=0, atol=1e-10)
    ):
        return None

    name = f"t{HORIZON}_a{ALPHA}_x{X0}"
    directory = RESULTS_ROOT / "02_baseline_budget" / name
    path = directory / "exact_test_states.npz"
    if not path.exists():
        print("No exact test-state sample from experiment 02; skipping policy error.")
        return None

    config = load_json(directory / "config.json")
    if not matches_problem(config["problem"], problem) or not bool(
        jnp.isclose(config["eta"], eta, rtol=0, atol=1e-10)
    ):
        raise ValueError("Exact test states were saved for a different configuration")
    return load_npz(path)


def save_diagnostics(path, metrics, diagnostics, actions=None):
    arrays = {
        "total_costs": metrics["total_costs"],
        "root_q": diagnostics["root_q"],
    }
    if actions is not None:
        arrays["policy_actions"] = actions
    for name in ("rollout_paths", "rollout_costs", "tail_flags"):
        if diagnostics[name] is not None:
            arrays[name] = diagnostics[name]
    for t, stage in diagnostics["stages"].items():
        for name in (
            "training_states",
            "q_values",
            "selected_indices",
            "selected_scores",
            "reachability",
            "tail_relevance",
        ):
            if stage[name] is not None:
                arrays[f"t{t}_{name}"] = stage[name]
    save_npz(path, **arrays)


def evaluate_baseline_policies(problem, eta, out):
    """Evaluate final saved baseline ADP policy once per seed, on GP noise paths.

    The baseline *training* rollout count is irrelevant here. We deliberately
    reuse the GP evaluation key and GP evaluation rollout count.

    Experiment 04 saves the serialized final baseline policies. Its old
    scalar AVaR reference and rollout count are deliberately ignored.
    """
    import cloudpickle

    base_dir = RESULTS_ROOT / "04_baseline_reference" / f"t{HORIZON}_a{ALPHA}_x{X0}"
    config_path = base_dir / "config.json"
    if not config_path.exists():
        print("No experiment 04 baseline configuration; omitting baseline reference.")
        return {}

    base_config = load_json(config_path)
    if not matches_problem(base_config["problem"], problem) or not bool(
        jnp.isclose(base_config["eta"], eta, rtol=0, atol=1e-10)
    ):
        print("Experiment 04 uses different model/eta; omitting baseline reference.")
        return {}

    n_episodes = int(base_config["n_episodes"])

    policy_files = sorted((base_dir / "policies").glob("seed*.pkl"))
    policies = {int(path.stem[4:]): path for path in policy_files}

    if not policies:
        print("No saved baseline policies found.")
        return {}

    cache_dir = out / "baseline_evaluation"
    # Cache identity includes the *actual* policy bytes, not the old baseline
    # evaluation sample size. Changing the model/policy or GP evaluation settings
    # requires a new cache rather than silently using stale reference costs.
    policy_hashes = {
        str(seed): hashlib.sha256(path.read_bytes()).hexdigest()
        for seed, path in policies.items()
    }
    eval_config = {
        "problem": problem,
        "eta": eta,
        "seeds": sorted(policies),
        "baseline_n_episodes": n_episodes,
        "eval_rollouts": EVAL_ROLLOUTS,
        "eval_seed": EVAL_SEED,
        "policy_hashes": policy_hashes,
    }
    saved_config_path = cache_dir / "config.json"
    if saved_config_path.exists() and load_json(saved_config_path) != eval_config:
        raise ValueError(
            f"Baseline evaluation cache uses different settings/policies: "
            f"{cache_dir}. Remove that cache to reevaluate baseline policies."
        )
    save_json(saved_config_path, eval_config)

    result = {}
    metric_rows = []
    for seed, path in policies.items():
        cost_path = cache_dir / f"seed{seed}.npz"
        if cost_path.exists():
            costs = load_npz(cost_path)["total_costs"]
            if costs.ndim != 1 or len(costs) != EVAL_ROLLOUTS:
                raise ValueError(f"Incorrect cached baseline cost shape: {cost_path}")
            seconds = float("nan")
        else:
            start = perf_counter()
            with path.open("rb") as f:
                policy = cloudpickle.load(f)
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
            seconds = perf_counter() - start
            costs = metrics["total_costs"]
            save_npz(cost_path, total_costs=costs)
            print(
                f"Baseline seed={seed}: evaluated once on "
                f"{EVAL_ROLLOUTS} shared noise paths.",
                flush=True,
            )

        base_metrics = policy_metrics(costs, ALPHA)
        base_metrics["ru_objective"] = float(rollout_ru_objective(costs, eta, ALPHA))
        result[seed] = {**base_metrics, "total_costs": costs}
        metric_rows.append(
            {
                "seed": seed,
                "baseline_n_episodes": n_episodes,
                "avar": base_metrics["avar"],
                "mean": base_metrics["mean"],
                "ru_objective": base_metrics["ru_objective"],
                "eval_seconds": seconds,
            }
        )

    save_csv(cache_dir / "metrics.csv", metric_rows)
    return result


def save_paired_results(rows, baseline, out, eta):
    """Compute matched-path comparisons for *all* existing GP checkpoints."""
    if not baseline:
        for name in (
            "paired_vs_baseline.csv",
            "paired_summary.csv",
            "paired_avar_difference_vs_backups.png",
            "paired_ru_difference_vs_backups.png",
        ):
            (out / name).unlink(missing_ok=True)
        print("No baseline reference; removed stale paired results.")
        return

    baseline_costs = np.stack(
        [np.asarray(b["total_costs"], dtype=float) for b in baseline.values()]
    )

    baseline_avar = float(np.mean([b["avar"] for b in baseline.values()]))

    baseline_mean_costs = baseline_costs.mean(axis=0)
    baseline_mean_hinges = np.maximum(baseline_costs - eta, 0.0).mean(axis=0)

    paired_rows = []
    for r in rows:
        seed = int(r["seed"])
        target = int(r["target_budget"])
        method = str(r["method"])

        gp_path = out / "runs" / f"{method}_seed{seed}_budget{target}.npz"
        gp_costs = np.asarray(load_npz(gp_path)["total_costs"], dtype=float)
        if gp_costs.shape != baseline_mean_costs.shape:
            raise ValueError(f"Paired rollout-count mismatch in {gp_path}")

        # Fixed-eta RU differences are means of paired pathwise differences.
        d_ru = (np.maximum(gp_costs - eta, 0.0) - baseline_mean_hinges) / (1.0 - ALPHA)
        se = (
            float(np.std(d_ru, ddof=1) / np.sqrt(len(d_ru)))
            if len(d_ru) > 1
            else float("nan")
        )
        paired_rows.append(
            {
                "method": method,
                "seed": seed,
                "target_budget": target,
                "actual_backups": r["actual_backups"],
                "avar_difference": float(r["avar"]) - baseline_avar,
                "mean_difference": float(np.mean(gp_costs - baseline_mean_costs)),
                "n_baseline_policies": len(baseline),
                "ru_difference": float(np.mean(d_ru)),
                "ru_difference_mc_se": se,
                "ru_difference_mc_ci_low": float(np.mean(d_ru)) - 1.96 * se,
                "ru_difference_mc_ci_high": float(np.mean(d_ru)) + 1.96 * se,
            }
        )
    save_csv(out / "paired_vs_baseline.csv", paired_rows)
    paired_summary = summarize_seeds(
        pd.DataFrame(paired_rows),
        ("method", "target_budget"),
        ("actual_backups", "avar_difference", "mean_difference", "ru_difference"),
    )
    paired_summary["budget"] = paired_summary["actual_backups_mean"]
    save_csv(out / "paired_summary.csv", paired_summary)
    for metric in ("avar_difference", "ru_difference"):
        ax = plot_budget_summary(
            paired_summary, metric, references={"Baseline ADP": 0.0}
        )
        cast(Figure, ax.figure).savefig(
            out / f"paired_{metric}_vs_backups.png", bbox_inches="tight", dpi=160
        )
        plt.close(ax.figure)


def save_paired_exact(rows, exact, eta, out):
    """Evaluate the saved exact policy on the GP noise sample and pair RU costs."""
    if exact is None or not bool(
        jnp.isclose(eta, exact["eta_star_grid"], rtol=0, atol=1e-10)
    ):
        return

    # Experiment 00 enumerates the exact policy in noise-tree order.
    # Replaying its saved actions avoids solving exact DP again.
    path = RESULTS_ROOT / exact["optimal_policy_csv"]
    policy_rows = load_csv(path)
    n_noise = len(NOISE)
    noise_indices = np.asarray(
        jax.random.choice(
            jax.random.key(EVAL_SEED),
            n_noise,
            shape=(EVAL_ROLLOUTS, HORIZON),
            p=PROBABILITIES,
        )
    )
    path_indices = np.zeros(EVAL_ROLLOUTS, dtype=np.int64)
    x = jnp.full(EVAL_ROLLOUTS, X0)
    total_costs = jnp.zeros(EVAL_ROLLOUTS)

    for t in range(HORIZON):
        stage = policy_rows[policy_rows["t"] == t]
        if len(stage) != n_noise ** t or not np.array_equal(
            stage["path_index"].to_numpy(), np.arange(len(stage))
        ):
            raise ValueError(f"Invalid exact policy tree at stage {t}: {path}")
        actions = jnp.asarray(stage["action"].to_numpy()[path_indices])
        total_costs = total_costs + cost(x, actions)
        x = transition(x, actions, NOISE[noise_indices[:, t]])
        path_indices = path_indices * n_noise + noise_indices[:, t]

    exact_costs = np.asarray(total_costs + terminal_cost(x), dtype=float)
    save_npz(out / "exact_paired_costs.npz", total_costs=exact_costs)
    exact_avar = policy_metrics(jnp.asarray(exact_costs), ALPHA)["avar"]
    exact_ru = float(rollout_ru_objective(jnp.asarray(exact_costs), eta, ALPHA))

    paired_rows = []
    path_differences = {}
    for row in rows:
        method, seed, budget = (
            str(row["method"]),
            int(row["seed"]),
            int(row["target_budget"]),
        )
        gp_path = out / "runs" / f"{method}_seed{seed}_budget{budget}.npz"
        gp_costs = np.asarray(load_npz(gp_path)["total_costs"], dtype=float)
        if gp_costs.shape != exact_costs.shape:
            raise ValueError(f"Paired rollout count differs: {gp_path}")
        differences = (
            np.maximum(gp_costs - eta, 0.0) - np.maximum(exact_costs - eta, 0.0)
        ) / (1.0 - ALPHA)
        path_differences[(method, budget, seed)] = differences
        mu = float(np.mean(differences))
        se = float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
        paired_rows.append(
            {
                "method": method,
                "seed": seed,
                "target_budget": budget,
                "actual_backups": row["actual_backups"],
                "exact_mc_avar": exact_avar,
                "exact_mc_ru": exact_ru,
                "avar_gap": float(row["avar"]) - exact_avar,
                "ru_gap": mu,
                "ru_gap_mc_se": se,
                "ru_gap_mc_ci_low": mu - 1.96 * se,
                "ru_gap_mc_ci_high": mu + 1.96 * se,
            }
        )
    if not paired_rows:
        return
    save_csv(out / "paired_vs_exact.csv", paired_rows)

    # The same noise paths are used for every training seed. Average the
    # pathwise differences over seeds BEFORE estimating MC uncertainty.
    summary_rows = []
    for (method, budget), group in pd.DataFrame(paired_rows).groupby(
        ["method", "target_budget"]
    ):
        seeds = [int(s) for s in group["seed"]]
        differences = np.mean(
            [path_differences[(method, budget, seed)] for seed in seeds], axis=0
        )
        mu = float(np.mean(differences))
        se = float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
        summary_rows.append(
            {
                "method": method,
                "target_budget": budget,
                "budget": float(group["actual_backups"].mean()),
                "n_seeds": len(seeds),
                "n_rollouts": len(differences),
                "ru_gap_mean": mu,
                "ru_gap_mc_ci_low": mu - 1.96 * se,
                "ru_gap_mc_ci_high": mu + 1.96 * se,
            }
        )
    summary = pd.DataFrame(summary_rows)
    save_csv(out / "paired_exact_summary.csv", summary)

    fig, ax = plt.subplots()
    for method, group in summary.groupby("method"):
        group = group.sort_values("budget")
        x_values = group["budget"].to_numpy()
        (line,) = ax.plot(x_values, group["ru_gap_mean"], "-o", label=method)
        ax.fill_between(
            x_values,
            group["ru_gap_mc_ci_low"],
            group["ru_gap_mc_ci_high"],
            color=line.get_color(),
            alpha=0.15,
        )
    ax.axhline(0, linestyle="--", color="black", linewidth=1)
    ax.set(xlabel="Bellman backups", ylabel="Paired RU gap (GP - exact DP)")
    ax.legend()
    fig.savefig(out / "paired_exact_ru_vs_backups.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Exact policy on {EVAL_ROLLOUTS} shared paths: "
        f"empirical AVaR={exact_avar:.6f}, RU={exact_ru:.6f}"
    )


def plot_results(summary, out, exact, baseline, eta):
    exact_same_eta = exact is not None and bool(
        jnp.isclose(eta, exact["eta_star_grid"], rtol=0, atol=1e-10)
    )
    for metric in ("avar", "mean", "ru_objective", "policy_error", "train_seconds"):
        if metric == "policy_error" and summary["policy_error_mean"].isna().all():
            continue

        refs = {}
        if exact is not None:
            if metric == "avar":
                refs["Exact DP"] = exact["policy_avar"]
            elif metric == "mean":
                refs["Exact DP"] = exact["policy_mean_cost"]
            elif metric == "ru_objective" and exact_same_eta:
                refs["Exact DP"] = exact["policy_ru_at_eta"]
            elif metric == "policy_error" and exact_same_eta:
                refs["Exact DP"] = 0.0

        if baseline and metric in ("avar", "mean", "ru_objective"):
            refs["Baseline ADP (paired evaluation)"] = float(
                np.mean([b[metric] for b in baseline.values()])
            )

        ax = plot_budget_summary(summary, metric, references=refs)
        cast(Figure, ax.figure).savefig(
            out / f"{metric}_vs_backups.png", bbox_inches="tight", dpi=160
        )
        plt.close(ax.figure)


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

    out = RESULTS_ROOT / "05_fixed_eta_benchmark" / f"t{HORIZON}_a{ALPHA}_x{X0}"
    config = {
        "problem": problem,
        "eta": eta,
        "seeds": list(range(SEED, SEED + NUM_SEEDS)),
        "budgets": list(BACKUPS),
        "methods": {
            k: list(v) if isinstance(v, tuple) else [v] for k, v in METHODS.items()
        },
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
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

    if "--postprocess-only" in sys.argv:
        saved_config = load_json(config_path)

        if not matches_problem(saved_config["problem"], problem) or not np.isclose(
            saved_config["eta"], eta, rtol=0, atol=1e-10
        ):
            raise ValueError("Saved experiment uses different problem settings.")

        if (
            saved_config["eval_rollouts"] != EVAL_ROLLOUTS
            or saved_config["eval_seed"] != EVAL_SEED
        ):
            raise ValueError("Evaluation settings differ from saved GP runs.")

        rows = load_csv(out / "runs.csv").to_dict("records")

        expected = {
            (str(method), int(seed), int(budget))
            for method in saved_config["methods"]
            for seed in saved_config["seeds"]
            for budget in saved_config["budgets"]
        }
        actual = {
            (str(r["method"]), int(r["seed"]), int(r["target_budget"])) for r in rows
        }

        if actual != expected or len(rows) != len(expected):
            raise ValueError("Saved GP runs are incomplete or duplicated.")

        baseline = evaluate_baseline_policies(problem, eta, out)
        save_paired_results(rows, baseline, out, eta)

        if not baseline:
            raise RuntimeError("No compatible baseline reference available.")

        summary = load_csv(out / "summary.csv")
        plot_results(summary, out, exact, baseline, eta)

        print(f"Baseline comparisons regenerated in {out}")
        return

    if config_path.exists() and load_json(config_path) != config:
        raise ValueError(
            f"Different settings already saved in {out}; use a new directory"
        )
    if "--paired-exact-only" in sys.argv:
        save_paired_exact(
            load_csv(out / "runs.csv").to_dict("records"), exact, eta, out
        )
        return
    save_json(config_path, config)

    # Evaluate fixed baseline policy once per training seed. GP runs use the
    # identical EVAL_SEED/EVAL_ROLLOUTS at every method and budget.
    baseline = evaluate_baseline_policies(problem, eta, out)
    test = get_exact_test(problem, eta, exact)
    rn_models, rn_phi = prepare_rn(HORIZON)
    runs_path = out / "runs.csv"
    rows = load_csv(runs_path).to_dict("records") if runs_path.exists() else []

    for seed in config["seeds"]:
        candidates = get_candidates(seed, eta, out)
        for target in BACKUPS:
            per_stage = stage_budget(target)
            for name, method in METHODS.items():
                path = out / "runs" / f"{name}_seed{seed}_budget{target}.npz"
                if any(
                    r["seed"] == seed
                    and r["target_budget"] == target
                    and r["method"] == name
                    for r in rows
                ):
                    if not path.exists():
                        raise FileNotFoundError(
                            f"Missing diagnostics for saved run: {path}"
                        )
                    continue

                start = perf_counter()
                _, policy, _, diagnostics = probabilistic_adp(
                    horizon=HORIZON,
                    eta=eta,
                    alpha=ALPHA,
                    candidates_by_stage=candidates,
                    rn_models=rn_models,
                    rn_phi=rn_phi,
                    key=jax.random.key(seed),
                    n_backups_per_stage=per_stage,
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
                actual = diagnostics["total_backups"]
                if actual > target:
                    raise RuntimeError("Bellman backups exceeded target budget")

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

                save_diagnostics(path, metrics, diagnostics, actions)
                root_ru = float(eta + jnp.min(diagnostics["root_q"]) / (1 - ALPHA))
                rows.append(
                    {
                        "method": name,
                        "seed": seed,
                        "target_budget": target,
                        "actual_backups": actual,
                        "root_ru": root_ru,
                        "avar": metrics["avar"],
                        "var": metrics["var"],
                        "mean": metrics["mean"],
                        "ru_objective": metrics["ru_objective"],
                        "policy_error": disagreement,
                        "root_action": diagnostics["root_action"],
                        "train_seconds": train_seconds,
                        "eval_seconds": eval_seconds,
                        "acquisition_rollout_seconds": diagnostics["rollout_seconds"],
                    }
                )
                save_csv(runs_path, rows)
                print(
                    f"{name}, seed {seed}, target {target}: "
                    f"{actual} backups, AVaR={metrics['avar']:.4f}",
                    flush=True,
                )

    data = pd.DataFrame(rows)
    summary = summarize_seeds(
        data,
        ("method", "target_budget"),
        (
            "actual_backups",
            "root_ru",
            "avar",
            "var",
            "mean",
            "ru_objective",
            "policy_error",
            "train_seconds",
            "eval_seconds",
            "acquisition_rollout_seconds",
        ),
    )
    summary["budget"] = summary["actual_backups_mean"]
    save_csv(out / "summary.csv", summary)
    plot_results(summary, out, exact, baseline, eta)
    save_paired_results(rows, baseline, out, eta)
    save_paired_exact(rows, exact, eta, out)
    save_json(out / "environment.json", environment_metadata())
    print(f"Saved to {out}")


if __name__ == "__main__":
    run()
