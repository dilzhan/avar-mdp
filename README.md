# AVaR control in MDPs

Research code for finite-horizon, static (pre-commitment) AVaR/CVaR control with continuous states and discrete actions.

The implementation uses the Rockafellar–Uryasev threshold formulation and an augmented state `(x, y)`, with `y` tracking accumulated cost minus the threshold. For nonnegative costs, the region `y >= 0` reduces to risk-neutral continuation.

Paper: [Decision-Aware Bellman Sampling for Static AVaR Control Using Gaussian Processes](paper/main.pdf) ([LaTeX source](paper/main.tex)).

## Contents

- `src/`: reference dynamic programming, baseline ADP, risk-neutral approximation, GP-based probabilistic ADP, sampling rules, threshold search, and policy evaluation.
- `tests/`: validations for Bellman recursion, risk measures, policy interfaces, precision, sampling utilities, and backup accounting.
- `validate_everything.py`: runs all validation scripts.
- [`experiments/`](experiments/README.md): exact and baseline references, fixed-threshold and outer-threshold benchmarks, and candidate-pool ablations.
- `results/`: saved configurations, reference policies, per-run data, summary tables, and figures.
- `paper/`: paper draft, LaTeX sources, bibliography, and figures.

The example model uses quadratic costs, actions `{-1, 0, 1}`, and discretized Gaussian noise. Probabilistic ADP includes random, uncertainty, decision, reachability, and tail-based sampling.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Enable JAX float64 before importing the source modules:

```bash
export JAX_ENABLE_X64=1
```

## Validation

Run from the repository root:

```bash
python validate_everything.py
```

The validation runner enables float64 automatically and stops on the first failure.

## Experiments and results

Experiments cover horizons `T=3` and `T=7`, with an exact DP reference at `T=3`. They compare baseline ADP and GP sampling rules using AVaR, mean cost, RU objective, policy disagreement where an exact reference is available, runtime, and Bellman-backup budgets.

Run scripts from the repository root. For example, estimate the threshold before running the fixed-threshold benchmark:

```bash
T=7 python experiments/01_estimate_eta.py
NUM_SEEDS=3 T=7 python experiments/05_fixed_eta_benchmark.py
```

See the [experiment guide](experiments/README.md) for the full sequence and evaluation notes. Shared defaults are in `experiments/_common.py`; scripts enable float64 automatically. Completed benchmark runs are generally reused.

Saved outputs are organized under `results/00_exact_dp/` through `results/07_candidate_pool_ablation/`, with risk-neutral fits in `results/rn/` and reference metadata in `results/references/`. Browse the included CSV summaries and figures without rerunning the experiments.
