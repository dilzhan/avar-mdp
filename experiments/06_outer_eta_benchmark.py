"""Outer-eta GP benchmark with a paired high-budget baseline ADP reference.

Run: python experiments/06_outer_eta_benchmark.py
Postprocess only: python experiments/06_outer_eta_benchmark.py --paired-only
Requires experiment 01 to train. Loads the final baseline policy saved by
experiment 04 and evaluates it once on the GP evaluation disturbances.
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
    BACKUPS,
    BATCH_SIZE,
    CANDIDATE_POOL_SIZE,
    ETA_MAX,
    ETA_MIN,
    ETA_STEP,
    EVAL_ROLLOUTS,
    EVAL_SEED,
    GP_ALPHA,
    GP_LENGTH_SCALES,
    GP_X_BOUNDS,
    N_INITIAL,
    NUM_SEEDS,
    RESULTS_ROOT,
    SEED,
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

from src.evaluation import evaluate_policy
from src.model import ACTIONS, NOISE, PROBABILITIES
from src.outer_search import solve_avar_probabilistic

HORIZON = int(os.environ.get("T", "7"))
ALPHA = float(os.environ.get("ALPHA", "0.95"))
X0 = float(os.environ.get("X0", "1.0"))
NUM_SEEDS = int(os.environ.get("NUM_SEEDS", NUM_SEEDS))

METHODS = ("random", "uncertainty", "decision")
BASELINE_SEED = SEED  # One fixed, high-budget policy for every GP run.


def stage_budget(total_budget, n_eta):
    """Equal backups per stage after reserving all root Q evaluations."""
    n_actions = len(ACTIONS)
    root_backups = n_eta * n_actions
    n_points = (total_budget - root_backups) // (n_actions * (HORIZON - 1))
    if n_points < N_INITIAL or n_points > CANDIDATE_POOL_SIZE:
        raise ValueError(f"Budget {total_budget} does not fit the GP training design")
    return n_points * n_actions


def get_candidates(seed, out):
    path = out / "candidates" / f"seed{seed}.npz"
    if path.exists():
        arrays = load_npz(path)
        return {t: arrays[f"stage{t}"] for t in range(1, HORIZON)}

    candidates = make_candidates(
        jax.random.fold_in(jax.random.key(seed), 1), HORIZON, ETA_MAX
    )
    save_npz(path, **{f"stage{t}": states for t, states in candidates.items()})
    return candidates


def save_diagnostics(path, metrics, diagnostics):
    arrays = {
        "total_costs": metrics["total_costs"],
        "objective_trace": diagnostics["objective_trace"],
        "root_q_values": diagnostics["root_q_values"],
    }
    for t, stage in diagnostics["stages"].items():
        for name in (
            "training_states",
            "q_values",
            "selected_indices",
            "selected_scores",
        ):
            arrays[f"t{t}_{name}"] = stage[name]
    save_npz(path, **arrays)


def plot_results(summary, out, exact, eta_reference, baseline=None):
    # Baseline AVaR uses the same Monte Carlo disturbances as the GP policies.
    # Exact DP AVaR, when available, is enumerated and not MC-paired.
    refs = {}
    if baseline is not None:
        refs["Baseline ADP (shared disturbances)"] = baseline["avar"]
    ax = plot_budget_summary(summary, "avar", references=refs)
    ax.set_ylabel("Empirical AVaR")
    cast(Figure, ax.figure).savefig(
        out / "avar_vs_backups.png", bbox_inches="tight", dpi=160
    )
    plt.close(ax.figure)

    eta_refs = {"Baseline ADP eta": eta_reference}
    if exact is not None:
        eta_refs["Exact DP eta"] = exact["eta_star_grid"]
    ax = plot_budget_summary(summary, "eta_star", references=eta_refs)
    cast(Figure, ax.figure).savefig(
        out / "eta_vs_backups.png", bbox_inches="tight", dpi=160
    )
    plt.close(ax.figure)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    predicted_refs = (
        {"Exact DP": exact["exact_ru_grid_minimum"]} if exact is not None else None
    )
    evaluated_refs = (
        {"Baseline ADP (at baseline eta)": baseline["ru_objective"]}
        if baseline is not None
        else None
    )
    plot_budget_summary(summary, "predicted_ru", ax=axes[0], references=predicted_refs)
    plot_budget_summary(summary, "ru_objective", ax=axes[1], references=evaluated_refs)
    axes[0].set_title("Predicted minimum RU")
    axes[1].set_title("Evaluated RU at respective eta")
    fig.tight_layout()
    fig.savefig(out / "ru_vs_backups.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    ax = plot_budget_summary(summary, "train_seconds")
    cast(Figure, ax.figure).savefig(
        out / "runtime_vs_backups.png", bbox_inches="tight", dpi=160
    )
    plt.close(ax.figure)


def _costs(path):
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["total_costs"], dtype=float)


def _ru_contributions(costs, eta, alpha):
    """Per-trajectory contribution to RU at a specified threshold."""
    return float(eta) + np.maximum(costs - float(eta), 0.0) / (1 - alpha)


def evaluate_baseline_policy(out, gp_config):
    """Evaluate the one saved experiment-04 ADP policy on GP disturbances.

    All GP methods, checkpoints, and seeds use the same noise paths, so this
    reference is evaluated at most once per policy and evaluation configuration.
    In --paired-only mode, use the *saved* GP evaluation settings.
    """
    import cloudpickle

    problem = gp_config["problem"]
    eta = float(gp_config["baseline_eta"])
    base_dir = (
        RESULTS_ROOT
        / "04_baseline_reference"
        / f"t{problem['horizon']}_a{problem['alpha']}_x{problem['x0']}"
    )
    config_path = base_dir / "config.json"
    policy_path = base_dir / "policies" / f"seed{BASELINE_SEED}.pkl"
    if not config_path.exists() or not policy_path.exists():
        print(f"Baseline policy unavailable: {policy_path}. Skipping paired reference.")
        return None

    base_config = load_json(config_path)
    if not matches_problem(base_config["problem"], problem) or not bool(
        jnp.isclose(base_config["eta"], eta, rtol=0, atol=1e-10)
    ):
        raise ValueError("Experiment 04 policy uses a different problem or eta")
    if BASELINE_SEED not in [int(s) for s in base_config["seeds"]]:
        raise ValueError(f"Baseline seed {BASELINE_SEED} not in experiment 04 config")

    n_rollouts = int(gp_config["eval_rollouts"])
    eval_seed = int(gp_config["eval_seed"])
    if n_rollouts < 2:
        raise ValueError("At least two evaluation rollouts required")

    cache_dir = out / "baseline_evaluation"
    cache_config = {
        "problem": problem,
        "eta": eta,
        "baseline_seed": BASELINE_SEED,
        "baseline_n_episodes": int(base_config["n_episodes"]),
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "eval_seed": eval_seed,
        "eval_rollouts": n_rollouts,
    }
    cache_config_path = cache_dir / "config.json"
    cache_costs_path = cache_dir / "total_costs.npz"

    if cache_costs_path.exists() and not cache_config_path.exists():
        raise ValueError("Baseline costs found without cache config")

    if cache_config_path.exists():
        if load_json(cache_config_path) != cache_config:
            raise ValueError("Baseline evaluation cache configuration mismatch")
    else:
        save_json(cache_config_path, cache_config)

    if cache_costs_path.exists():
        costs = _costs(cache_costs_path)
        if costs.shape != (n_rollouts,):
            raise ValueError("Cached baseline cost count does not match GP evaluations")
        print(f"Reusing baseline evaluation, seed={BASELINE_SEED}, n={n_rollouts}.")
    else:
        start = perf_counter()
        with policy_path.open("rb") as f:
            policy = cloudpickle.load(f)
        metrics = evaluate_policy(
            policy=policy,
            horizon=int(problem["horizon"]),
            alpha=float(problem["alpha"]),
            eta=eta,
            key=jax.random.key(eval_seed),
            x0=float(problem["x0"]),
            noise=NOISE,
            probabilities=PROBABILITIES,
            n_rollouts=n_rollouts,
        )
        costs = np.asarray(metrics["total_costs"], dtype=float)
        if costs.shape != (n_rollouts,):
            raise ValueError("Unexpected baseline policy rollout shape")
        save_npz(cache_costs_path, total_costs=costs)
        print(
            f"Baseline policy evaluated once: seed={BASELINE_SEED}, "
            f"{n_rollouts} shared paths, {perf_counter()-start:.1f}s.",
            flush=True,
        )

    alpha = float(problem["alpha"])
    # Recompute from the persisted costs, never trust an old scalar reference.
    from src.evaluation import policy_metrics

    base_metrics = policy_metrics(jnp.asarray(costs), alpha)
    base_metrics["ru_objective"] = float(np.mean(_ru_contributions(costs, eta, alpha)))
    base_metrics["total_costs"] = costs
    base_metrics["eta"] = eta
    base_metrics["seed"] = BASELINE_SEED
    save_json(
        cache_dir / "metrics.json",
        {k: v for k, v in base_metrics.items() if k != "total_costs"},
    )
    return base_metrics


def pair_with_baseline(out, gp_config, baseline):
    """Pair every saved GP run to the same fixed baseline policy and noise paths.

    RU confidence intervals are Monte Carlo intervals conditional on the trained
    policies. Same noise paths across GP seeds are aggregated *by path* before
    computing the MC standard error; training-seed variation is separate.
    """
    if baseline is None:
        return
    gp_runs_path = out / "runs.csv"
    if not gp_runs_path.exists():
        print("No GP runs.csv available for pairing.")
        return

    rows = load_csv(gp_runs_path)
    if rows.empty:
        return
    base_costs = np.asarray(baseline["total_costs"], dtype=float)
    n_rollouts = int(gp_config["eval_rollouts"])
    if base_costs.shape != (n_rollouts,):
        raise ValueError("Baseline and GP rollout counts differ")
    base_eta = float(baseline["eta"])
    alpha = float(gp_config["problem"]["alpha"])
    base_ru = _ru_contributions(base_costs, base_eta, alpha)

    paired_rows = []
    contributions = {}
    for row in rows.itertuples(index=False):
        seed, budget = int(row.seed), int(row.target_budget)
        gp_path = out / "runs" / f"{row.method}_seed{seed}_budget{budget}.npz"
        if not gp_path.exists():
            raise FileNotFoundError(f"Missing saved GP rollout costs: {gp_path}")
        gp_costs = _costs(gp_path)
        if gp_costs.shape != (n_rollouts,):
            raise ValueError(f"Wrong GP cost count: {gp_path}: {gp_costs.shape}")

        eta_gp = float(row.eta_star)
        # Same-eta RU compares policies at the baseline threshold; respective-
        # eta RU compares each policy using its own chosen/training threshold.
        d_fixed = _ru_contributions(gp_costs, base_eta, alpha) - base_ru
        d_selected = _ru_contributions(gp_costs, eta_gp, alpha) - base_ru
        paired_rows.append(
            {
                "method": row.method,
                "seed": seed,
                "target_budget": budget,
                "gp_actual_backups": int(row.actual_backups),
                "baseline_seed": int(baseline["seed"]),
                "baseline_eta": base_eta,
                "gp_eta": eta_gp,
                "gp_avar_mc": float(row.avar),
                "baseline_avar_mc": float(baseline["avar"]),
                "avar_gap_mc": float(row.avar) - float(baseline["avar"]),
                "mean_gap": float(np.mean(gp_costs - base_costs)),
                "ru_gap_fixed_eta": float(np.mean(d_fixed)),
                "ru_gap_selected_eta": float(np.mean(d_selected)),
            }
        )
        contributions[(row.method, budget, seed)] = (d_fixed, d_selected)

    save_csv(out / "paired_vs_baseline.csv", paired_rows)
    summaries = []
    for (method, budget), group in pd.DataFrame(paired_rows).groupby(
        ["method", "target_budget"]
    ):
        seeds = sorted(int(s) for s in group["seed"])
        # Every seed uses the SAME disturbance sample and the SAME baseline.
        fixed = np.mean(
            [contributions[(method, budget, seed)][0] for seed in seeds], axis=0
        )
        selected = np.mean(
            [contributions[(method, budget, seed)][1] for seed in seeds], axis=0
        )
        result = {
            "method": method,
            "target_budget": budget,
            "budget": float(group["gp_actual_backups"].mean()),
            "baseline_seed": int(baseline["seed"]),
            "n_gp_seeds": len(seeds),
            "n_rollouts": n_rollouts,
            "avar_gap_mc_mean": float(group["avar_gap_mc"].mean()),
            "avar_gap_mc_std": float(group["avar_gap_mc"].std()),
            "mean_gap_mean": float(group["mean_gap"].mean()),
        }
        for name, d in (("ru_gap_fixed_eta", fixed), ("ru_gap_selected_eta", selected)):
            mu = float(np.mean(d))
            se = float(np.std(d, ddof=1) / np.sqrt(n_rollouts))
            result[f"{name}_mean"] = mu
            result[f"{name}_ci_low"] = mu - 1.96 * se
            result[f"{name}_ci_high"] = mu + 1.96 * se
        summaries.append(result)

    summary = pd.DataFrame(summaries).sort_values(["method", "target_budget"])
    save_csv(out / "paired_summary.csv", summary)
    for metric, ylabel in (
        ("ru_gap_fixed_eta", "Paired RU gap at baseline eta (GP - baseline)"),
        ("ru_gap_selected_eta", "Paired RU gap at respective eta (GP - baseline)"),
    ):
        fig, ax = plt.subplots()
        for method, group in summary.groupby("method"):
            group = group.sort_values("budget")
            x = group["budget"].to_numpy()
            y = group[f"{metric}_mean"].to_numpy()
            lower = y - group[f"{metric}_ci_low"].to_numpy()
            upper = group[f"{metric}_ci_high"].to_numpy() - y
            ax.errorbar(x, y, yerr=[lower, upper], marker="o", capsize=2, label=method)
        ax.axhline(0, linestyle="--", color="black", linewidth=1)
        ax.set(xlabel="GP Bellman backups", ylabel=ylabel)
        ax.legend()
        fig.savefig(out / f"{metric}_vs_backups.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # GP minus a single constant baseline AVaR, shown across GP backup budgets.
    avar_summary = summary.rename(
        columns={
            "avar_gap_mc_mean": "avar_gap_mean",
            "avar_gap_mc_std": "avar_gap_std",
        }
    )
    ax = plot_budget_summary(avar_summary, "avar_gap", references={"Baseline ADP": 0.0})
    ax.set_ylabel("Paired empirical AVaR gap (GP - baseline)")
    ax.set_title("Shading: GP training-seed SD, not an MC confidence interval")
    cast(Figure, ax.figure).savefig(
        out / "paired_avar_vs_backups.png", bbox_inches="tight", dpi=160
    )
    plt.close(ax.figure)

    print(
        f"Paired {len(paired_rows)} GP runs to baseline seed={BASELINE_SEED}, "
        f"using {n_rollouts} shared disturbance paths each."
    )


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
    eta_reference = load_eta(problem)
    exact = load_reference(problem, "exact", mode="auto")
    eta_grid = jnp.arange(ETA_MIN, ETA_MAX + ETA_STEP / 2, ETA_STEP)

    out = RESULTS_ROOT / "06_outer_eta_benchmark" / f"t{HORIZON}_a{ALPHA}_x{X0}"
    config = {
        "problem": problem,
        "baseline_eta": eta_reference,
        "eta_grid": eta_grid.tolist(),
        "seeds": list(range(SEED, SEED + NUM_SEEDS)),
        "budgets": list(BACKUPS),
        "methods": list(METHODS),
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "gp_x_bounds": list(GP_X_BOUNDS),
        "gp_length_scales": list(GP_LENGTH_SCALES),
        "gp_alpha": GP_ALPHA,
        "n_initial": N_INITIAL,
        "batch_size": BATCH_SIZE,
        "eval_rollouts": EVAL_ROLLOUTS,
        "eval_seed": EVAL_SEED,
    }
    config_path = out / "config.json"
    if "--paired-only" in sys.argv:
        saved_config = load_json(config_path)
        if not matches_problem(saved_config["problem"], problem):
            raise ValueError("Saved GP results use a different MDP configuration")
        baseline = evaluate_baseline_policy(out, saved_config)
        pair_with_baseline(out, saved_config, baseline)
        summary_path = out / "summary.csv"
        if summary_path.exists():
            plot_results(
                load_csv(summary_path),
                out,
                exact,
                float(saved_config["baseline_eta"]),
                baseline,
            )
        return
    if config_path.exists() and load_json(config_path) != config:
        raise ValueError(
            f"Different settings already saved in {out}; use a new directory"
        )
    save_json(config_path, config)
    baseline = evaluate_baseline_policy(out, config)
    rn_models, rn_phi = prepare_rn(HORIZON)
    runs_path = out / "runs.csv"
    rows = load_csv(runs_path).to_dict("records") if runs_path.exists() else []

    for seed in config["seeds"]:
        candidates = get_candidates(seed, out)
        for target in BACKUPS:
            per_stage = stage_budget(target, len(eta_grid))
            for method in METHODS:
                path = out / "runs" / f"{method}_seed{seed}_budget{target}.npz"
                if any(
                    r["seed"] == seed
                    and r["target_budget"] == target
                    and r["method"] == method
                    for r in rows
                ):
                    if not path.exists():
                        raise FileNotFoundError(
                            f"Missing saved run diagnostics: {path}"
                        )
                    continue

                start = perf_counter()
                predicted_ru, eta_star, policy, _, diagnostics = (
                    solve_avar_probabilistic(
                        horizon=HORIZON,
                        alpha=ALPHA,
                        eta_grid=eta_grid,
                        candidates_by_stage=candidates,
                        rn_models=rn_models,
                        rn_phi=rn_phi,
                        key=jax.random.key(seed),
                        n_backups_per_stage=per_stage,
                        sampling_method=method,
                        x0=X0,
                        n_initial=N_INITIAL,
                        batch_size=BATCH_SIZE,
                        gp_x_bounds=GP_X_BOUNDS,
                        gp_length_scales=GP_LENGTH_SCALES,
                        gp_alpha=GP_ALPHA,
                        actions=ACTIONS,
                        noise=NOISE,
                        probabilities=PROBABILITIES,
                    )
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
                    eta=eta_star,
                    key=jax.random.key(EVAL_SEED),
                    x0=X0,
                    noise=NOISE,
                    probabilities=PROBABILITIES,
                    n_rollouts=EVAL_ROLLOUTS,
                )
                eval_seconds = perf_counter() - start

                save_diagnostics(path, metrics, diagnostics)
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "target_budget": target,
                        "actual_backups": actual,
                        "eta_star": eta_star,
                        "predicted_ru": predicted_ru,
                        "avar": metrics["avar"],
                        "var": metrics["var"],
                        "mean": metrics["mean"],
                        "ru_objective": metrics["ru_objective"],
                        "root_action": diagnostics["root_action"],
                        "train_seconds": train_seconds,
                        "eval_seconds": eval_seconds,
                    }
                )
                save_csv(runs_path, rows)
                print(
                    f"{method}, seed {seed}, target {target}: "
                    f"{actual} backups, eta={eta_star:.3f}, AVaR={metrics['avar']:.4f}",
                    flush=True,
                )

    summary = summarize_seeds(
        pd.DataFrame(rows),
        ("method", "target_budget"),
        (
            "actual_backups",
            "eta_star",
            "predicted_ru",
            "avar",
            "var",
            "mean",
            "ru_objective",
            "train_seconds",
            "eval_seconds",
        ),
    )
    summary["budget"] = summary["actual_backups_mean"]
    save_csv(out / "summary.csv", summary)
    plot_results(summary, out, exact, eta_reference, baseline)
    pair_with_baseline(out, config, baseline)
    save_json(out / "environment.json", environment_metadata())
    print(f"Saved to {out}")


if __name__ == "__main__":
    run()
