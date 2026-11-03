"""Shared defaults and small helpers for the experiment scripts."""

import json
import platform
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"

# Reproducibility and evaluation
SEED = 47
NUM_SEEDS = 3
EVAL_SEED = 314_159
EVAL_ROLLOUTS = 2_000
N_EPISODES = 1_000
BACKUPS = (900, 1800, 2700, 3600, 5400, 7200, 9000)

# Outer eta search
ETA_MIN = 0.0
ETA_MAX = 40.0
ETA_STEP = 0.25

# Probabilistic ADP
CANDIDATE_POOL_SIZE = 5_000
N_INITIAL = 20
BATCH_SIZE = 10

# GP defaults (length scales are in normalized coordinates)
GP_X_BOUNDS = (-8.0, 8.0)
GP_ALPHA = 1e-6
GP_LENGTH_SCALES = (0.16, 0.16)

# GP acquisition
K_NEIGHBORS = 50
TAIL_LAMBDA = 0.5
ACQUISITION_ROLLOUTS = 2_000

# Risk-neutral approximation
RN_X_MIN = -5.0
RN_X_MAX = 5.0
RN_MESH = 0.25
RN_DELTA_CENTER = 0.5
RN_PADDING = 2


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=lambda x: x.tolist())


def load_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def save_csv(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False)


def load_csv(path):
    return pd.read_csv(path)


def append_csv(path, row):
    """Append one result dictionary; columns must match existing CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    jnp.savez(path, **arrays)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: jnp.asarray(data[name]) for name in data.files}


def environment_metadata():
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "jax": jax.__version__,
        "jax_precision": "float64" if jax.config.read("jax_enable_x64") else "float32",
        "jax_backend": jax.default_backend(),
    }


def summarize_seeds(data, group_by, metrics):
    """Mean and standard deviation for each group."""
    summary = data.groupby(list(group_by))[list(metrics)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def plot_budget_summary(summary, metric, ax=None, references=None):
    """Plot metric_mean +/- metric_std against actual backup budgets."""
    if ax is None:
        _, ax = plt.subplots()

    for method, rows in summary.groupby("method"):
        rows = rows.sort_values("budget")
        x = rows["budget"].tolist()
        mean = rows[f"{metric}_mean"].tolist()
        (line,) = ax.plot(x, mean, "-o", label=str(method))
        std_name = f"{metric}_std"
        if std_name in rows:
            std = rows[std_name].fillna(0).tolist()
            ax.fill_between(
                x,
                [m - s for m, s in zip(mean, std)],
                [m + s for m, s in zip(mean, std)],
                color=line.get_color(),
                alpha=0.15,
            )

    for name, value in (references or {}).items():
        linestyle = ":" if "exact" in name.lower() else "--"
        ax.axhline(value, linestyle=linestyle, label=name, color="grey")

    ax.set_xlabel("Bellman backups")
    ax.set_ylabel(metric)
    ax.legend()
    return ax


def matches_problem(saved, problem):
    """Check that every requested problem setting matches the saved settings."""
    for name, expected in problem.items():
        if name not in saved:
            return False
        actual = saved[name]
        if isinstance(expected, (str, bool)):
            if actual != expected:
                return False
        else:
            a, b = jnp.asarray(actual), jnp.asarray(expected)
            if a.shape != b.shape or not bool(jnp.allclose(a, b, rtol=0, atol=1e-10)):
                return False
    return True


def _find_record(path, problem):
    """Find one matching entry in a JSON list of {problem, ...} records."""
    path = Path(path)
    if not path.exists():
        return None
    matches = [
        record
        for record in load_json(path)
        if matches_problem(record["problem"], problem)
    ]
    if len(matches) > 1:
        raise ValueError(f"Multiple matching records in {path}")
    return matches[0] if matches else None


def load_eta(problem):
    """Load the threshold saved by experiment 01; fail if unavailable."""
    path = RESULTS_ROOT / "01_estimate_eta" / "estimates.json"
    record = _find_record(path, problem)
    if record is None:
        raise FileNotFoundError(f"No estimated eta for {problem} in {path}")
    return float(record["eta"])


def load_reference(problem, kind, mode="auto"):
    """Load an exact/baseline reference; mode is True, False, or 'auto'."""
    if kind not in ("exact", "baseline"):
        raise ValueError("kind must be 'exact' or 'baseline'")
    if mode is False:
        return None
    if mode is not True and mode != "auto":
        raise ValueError("mode must be True, False, or 'auto'")

    path = RESULTS_ROOT / "references" / f"{kind}.json"
    record = _find_record(path, problem)
    if record is None:
        if mode is True:
            raise FileNotFoundError(f"No {kind} reference for {problem} in {path}")
        return None
    return record["reference"]


def prepare_rn(horizon):
    """Load or fit risk-neutral Q approximations for a fixed horizon."""
    from src.model import ACTIONS, NOISE, PROBABILITIES
    from src.risk_neutral_approx import fit_risk_neutral_q, make_phi, make_training_grid

    grid = make_training_grid(RN_X_MIN, RN_X_MAX, horizon, RN_MESH, ACTIONS, NOISE)
    phi = make_phi(float(grid[0]), float(grid[-1]), RN_DELTA_CENTER, RN_PADDING)

    settings = {
        "horizon": horizon,
        "actions": ACTIONS,
        "noise": NOISE,
        "probabilities": PROBABILITIES,
        "rn_x_min": RN_X_MIN,
        "rn_x_max": RN_X_MAX,
        "rn_mesh": RN_MESH,
        "rn_delta_center": RN_DELTA_CENTER,
        "rn_padding": RN_PADDING,
    }
    path = RESULTS_ROOT / "rn" / f"t{horizon}.npz"
    meta_path = RESULTS_ROOT / "rn" / f"t{horizon}.json"

    if (
        path.exists()
        and meta_path.exists()
        and matches_problem(load_json(meta_path), settings)
    ):
        return load_npz(path)["models"], phi

    models = fit_risk_neutral_q(horizon, phi, grid, ACTIONS, NOISE, PROBABILITIES)
    models.block_until_ready()
    save_npz(path, models=models)
    save_json(meta_path, settings)
    return models, phi


def make_candidates(
    key, horizon, eta, n_candidates=CANDIDATE_POOL_SIZE, x_bounds=GP_X_BOUNDS
):
    """Uniform candidate states (x, y) for stages 1 through horizon - 1."""
    if horizon < 2 or eta <= 0 or n_candidates < 1:
        raise ValueError("Require horizon >= 2, eta > 0, and n_candidates >= 1")

    lower = jnp.array([x_bounds[0], -eta])
    upper = jnp.array([x_bounds[1], 0.0])
    keys = jax.random.split(key, horizon - 1)
    return {
        t: jax.random.uniform(
            keys[t - 1], (n_candidates, 2), minval=lower, maxval=upper
        )
        for t in range(1, horizon)
    }
