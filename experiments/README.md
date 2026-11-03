# Experiments

This folder contains the experiments for AVaR control using baseline and probabilistic approximate dynamic programming (ADP).

| Script | Description |
|---|---|
| `00_exact_dp.py` | Compute the exact DP reference for T=3. |
| `01_estimate_eta.py` | Estimate the risk threshold η. |
| `02_baseline_budget.py` | Evaluate baseline ADP at different backup budgets. |
| `03_baseline_vs_exact.py` | Compare baseline ADP with exact DP for T=3. |
| `04_baseline_reference.py` | Train and save baseline policies. |
| `05_fixed_eta_benchmark.py` | Compare probabilistic ADP sampling methods at fixed η. |
| `06_outer_eta_benchmark.py` | Compare sampling methods with outer η optimization. |
| `07_candidate_pool_ablation.py` | Study the effect of candidate-pool size. |

## Running

Run scripts from the project root, in numerical order. For example:

```bash
T=3 python experiments/00_exact_dp.py
T=7 python experiments/01_estimate_eta.py
NUM_SEEDS=3 T=7 python experiments/05_fixed_eta_benchmark.py
```

Experiments use T=3 and/or T=7. Experiment 03 is T=3 only.

Shared settings are in `experiments/_common.py`. Results are saved in `results/`, including configurations, CSV files, figures, and diagnostics. Completed runs are generally reused.

**Note:** Baseline evaluations use randomized cached initialization, which may depend on evaluation order. Baseline comparisons should be interpreted with this limitation.
