"""Paired T=3 validation of saved baseline ADP runs against exact DP.

Run: python experiments/03_baseline_vs_exact.py
Requires completed experiments 00, 01, and 02, including the per-run .npz files.
Does not retrain ADP, run new Monte Carlo rollouts, or recompute exact DP.
"""

import os
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import (
    RESULTS_ROOT,
    load_csv,
    load_eta,
    load_json,
    load_npz,
    load_reference,
    matches_problem,
    save_csv,
)

from src.model import ACTIONS, NOISE, PROBABILITIES, cost, terminal_cost, transition

HORIZON = 3
ALPHA = float(os.environ.get("ALPHA", "0.95"))
X0 = float(os.environ.get("X0", "1.0"))
Z95 = 1.959963984540054


def empirical_avar(costs):
    """Sample AVaR (descriptive; not a paired unbiased gap estimator)."""
    costs = jnp.sort(jnp.asarray(costs))
    threshold = costs[int(jnp.ceil(ALPHA * len(costs))) - 1]
    return float(threshold + jnp.maximum(costs - threshold, 0).mean() / (1 - ALPHA))


def exact_costs_on_saved_noise(policy_csv, config):
    """Replay the enumerated exact policy on experiment 02's noise paths.

    Experiment 00's path_index is a base-len(NOISE) prefix index, so this
    requires exactly the same ordered disturbance grid in both experiments.
    """
    policy_rows = load_csv(policy_csv)
    n = int(config["eval_rollouts"])
    m = len(NOISE)
    idx = jnp.asarray(
        jax.random.choice(
            jax.random.key(int(config["eval_seed"])),
            m,
            shape=(n, HORIZON),
            p=PROBABILITIES,
        )
    )

    x = jnp.full(n, X0)
    total = jnp.zeros(n)
    prefix = jnp.zeros(n, dtype=jnp.int64)

    for t in range(HORIZON):
        stage = policy_rows[policy_rows["t"] == t].sort_values("path_index")
        if not jnp.array_equal(stage["path_index"].to_numpy(), jnp.arange(m**t)):
            raise ValueError(f"Exact-policy CSV is incomplete at stage {t}")
        action = jnp.asarray(stage["action"].to_numpy())[prefix]
        total = total + cost(x, action)
        x = transition(x, action, NOISE[idx[:, t]])
        prefix = prefix * m + idx[:, t]

    return jnp.asarray(total + terminal_cost(x))


def mean_ci(values):
    """Approximate 95% CI for mean of independent trajectory contributions."""
    values = jnp.asarray(values, dtype=float)
    mean = float(values.mean())
    radius = Z95 * float(values.std(ddof=1)) / jnp.sqrt(len(values))
    return mean, mean - radius, mean + radius


def plot_root_convergence(baseline_dir, config, eta, exact_ru, out):
    """Show the root *value-estimation* error separately from policy quality."""
    histories = []
    for seed in config["seeds"]:
        saved = load_npz(baseline_dir / f"calibration_seed{seed}.npz")
        histories.append((saved["backup_history"], saved["root_history"]))

    left = max(int(b[0]) for b, _ in histories)
    right = min(max(config["budgets"]), min(int(b[-1]) for b, _ in histories))
    if left > right:
        raise ValueError("Calibration histories have no common budget range")

    budgets = jnp.unique(jnp.concatenate([b for b, _ in histories]))
    budgets = budgets[(budgets >= left) & (budgets <= right)]
    root_values = jnp.stack(
        [
            eta
            + roots[jnp.searchsorted(backups, budgets, side="right") - 1] / (1 - ALPHA)
            for backups, roots in histories
        ]
    )
    means = jnp.asarray(jnp.mean(root_values, axis=0))
    stds = (
        jnp.asarray(jnp.std(root_values, axis=0, ddof=1))
        if len(histories) > 1
        else jnp.zeros_like(means)
    )

    save_csv(
        out / "root_convergence.csv",
        {
            "budget": jnp.asarray(budgets),
            "root_ru_mean": means,
            "root_ru_std": stds,
        },
    )
    fig, ax = plt.subplots()
    ax.plot(budgets, means, label="Baseline root estimate")
    ax.fill_between(budgets, means - stds, means + stds, alpha=0.2)
    ax.axhline(exact_ru, linestyle="--", label="Exact DP root")
    ax.set(xlabel="Bellman backups", ylabel="Fixed-eta root RU estimate")
    ax.legend()
    fig.savefig(out / "root_convergence.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


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
    exact = load_reference(problem, "exact", mode=True)
    if exact is None:
        raise ValueError("Exact DP results are required for this experiment")

    if not jnp.isclose(eta, exact["eta_star_grid"], atol=1e-10, rtol=0):
        raise ValueError("Baseline eta differs from the exact reference eta")

    name = f"t{HORIZON}_a{ALPHA}_x{X0}"
    baseline_dir = RESULTS_ROOT / "02_baseline_budget" / name
    out = RESULTS_ROOT / "03_baseline_vs_exact_t3" / name
    config = load_json(baseline_dir / "config.json")
    if not matches_problem(config["problem"], problem) or not jnp.isclose(
        config["eta"], eta, rtol=0, atol=1e-10
    ):
        raise ValueError("Baseline results use different model settings or eta")

    runs = load_csv(baseline_dir / "runs.csv")
    expected = {
        (seed, budget) for seed in config["seeds"] for budget in config["budgets"]
    }
    actual = set(zip(runs["seed"], runs["target_budget"]))
    if len(runs) != len(expected) or actual != expected:
        raise ValueError("Baseline runs are incomplete or duplicated")

    exact_ru = float(exact["exact_ru_grid_minimum"])
    exact_costs = exact_costs_on_saved_noise(
        RESULTS_ROOT / exact["optimal_policy_csv"], config
    )
    exact_avar_mc = empirical_avar(exact_costs)
    exact_ru_mc = float(eta + jnp.maximum(exact_costs - eta, 0).mean() / (1 - ALPHA))

    details = []
    pathwise_by_budget = {b: [] for b in config["budgets"]}

    for row in runs.itertuples(index=False):
        seed, budget = int(row.seed), int(row.target_budget)
        with np.load(baseline_dir / "runs" / f"seed{seed}_budget{budget}.npz") as saved:
            costs = jnp.asarray(saved["total_costs"])
        if costs.shape != exact_costs.shape:
            raise ValueError(f"Wrong saved cost shape for seed={seed}, budget={budget}")
        if not jnp.isclose(empirical_avar(costs), row.avar, atol=1e-6, rtol=0):
            raise ValueError(
                f"Saved costs disagree with runs.csv: seed={seed}, budget={budget}"
            )

        # Common random paths make this an unbiased fixed-eta RU *gap* estimator.
        contributions = (
            jnp.maximum(costs - eta, 0) - jnp.maximum(exact_costs - eta, 0)
        ) / (1 - ALPHA)
        gap, low, high = mean_ci(contributions)
        pathwise_by_budget[budget].append(contributions)
        details.append(
            {
                "method": "baseline",
                "seed": seed,
                "target_budget": budget,
                "actual_backups": row.actual_backups,
                "paired_ru_gap": gap,
                "paired_ru_ci_low": low,
                "paired_ru_ci_high": high,
                "ru_control_variate": exact_ru + gap,
                "paired_avar_gap_mc": empirical_avar(costs) - exact_avar_mc,
                "paired_mean_gap_mc": float(jnp.mean(costs - exact_costs)),
                "root_ru_abs_error": abs(float(row.root_ru) - exact_ru),
                "policy_error": row.policy_error,
            }
        )

    errors = pd.DataFrame(details)
    summary = []
    for budget in config["budgets"]:
        subset = errors[errors["target_budget"] == budget]
        # All training seeds used the SAME noise paths. Average per path first;
        # treating (seeds x paths) as independent would understate uncertainty.
        per_path_mean = jnp.mean(jnp.stack(pathwise_by_budget[budget]), axis=0)
        gap, low, high = mean_ci(per_path_mean)
        summary.append(
            {
                "target_budget": budget,
                "budget": float(subset["actual_backups"].mean()),
                "paired_ru_gap_mean": gap,
                "paired_ru_gap_ci_low": low,
                "paired_ru_gap_ci_high": high,
                "paired_ru_gap_seed_std": float(subset["paired_ru_gap"].std(ddof=1)),
                "ru_control_variate_mean": exact_ru + gap,
                "paired_avar_gap_mc_mean": float(subset["paired_avar_gap_mc"].mean()),
                "paired_avar_gap_mc_seed_std": float(
                    subset["paired_avar_gap_mc"].std(ddof=1)
                ),
                "paired_mean_gap_mc_mean": float(subset["paired_mean_gap_mc"].mean()),
                "root_ru_abs_error_mean": float(subset["root_ru_abs_error"].mean()),
                "policy_error_mean": float(subset["policy_error"].mean()),
            }
        )

    result = pd.DataFrame(summary)
    save_csv(out / "errors.csv", errors)
    save_csv(out / "summary.csv", result)
    plot_root_convergence(baseline_dir, config, eta, exact_ru, out)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = result["budget"].to_numpy()
    ru = result["paired_ru_gap_mean"].to_numpy()
    low = result["paired_ru_gap_ci_low"].to_numpy()
    high = result["paired_ru_gap_ci_high"].to_numpy()
    axes[0].errorbar(
        x,
        ru,
        yerr=jnp.array([ru - low, high - ru]),
        fmt="-o",
        capsize=3,
        label="Baseline (paired)",
    )
    axes[0].axhline(0, linestyle="--", color="gray", label="Exact policy")
    axes[0].set(xlabel="Bellman backups", ylabel="Fixed-eta RU gap (95% MC CI)")
    axes[0].legend()

    av = result["paired_avar_gap_mc_mean"].to_numpy()
    av_sd = result["paired_avar_gap_mc_seed_std"].fillna(0).to_numpy()
    axes[1].plot(x, av, "-o", label="Baseline (paired)")
    axes[1].fill_between(x, av - av_sd, av + av_sd, alpha=0.2)
    axes[1].axhline(0, linestyle="--", color="gray", label="Exact policy")
    axes[1].set(xlabel="Bellman backups", ylabel="Paired empirical AVaR gap (seed SD)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "policy_performance.png", bbox_inches="tight", dpi=160)
    plt.close(fig)

    if errors["policy_error"].notna().any():
        fig, ax = plt.subplots()
        ax.plot(x, result["policy_error_mean"], "-o", label="Baseline")
        ax.axhline(0, linestyle="--", color="gray", label="Exact policy")
        ax.set(xlabel="Bellman backups", ylabel="Sampled policy disagreement")
        ax.legend()
        fig.savefig(out / "policy_disagreement.png", bbox_inches="tight", dpi=160)
        plt.close(fig)

    print(f"Exact DP enumerated AVaR: {exact['policy_avar']:.6f}")
    print(f"Exact policy MC AVaR:      {exact_avar_mc:.6f}")
    print(f"Exact policy MC RU:        {exact_ru_mc:.6f}")
    print(
        result[
            [
                "target_budget",
                "paired_ru_gap_mean",
                "paired_ru_gap_ci_low",
                "paired_ru_gap_ci_high",
                "paired_avar_gap_mc_mean",
            ]
        ].to_string(index=False)
    )
    print(f"Saved to {out}")


if __name__ == "__main__":
    run()
