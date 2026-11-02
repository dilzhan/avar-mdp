# AVaR control in MDPs

Research code for finite-horizon, static (pre-commitment) AVaR/CVaR control with continuous states and discrete actions.

The implementation uses the Rockafellar–Uryasev threshold formulation and an augmented state `(x, y)`, with `y` tracking accumulated cost minus the threshold. For nonnegative costs, the region `y >= 0` reduces to risk-neutral continuation.

## Contents

- `src/`: reference dynamic programming, baseline ADP, risk-neutral approximation, GP-based probabilistic ADP, sampling rules, threshold search, and policy evaluation.
- `tests/`: validations for Bellman recursion, risk measures, policy interfaces, precision, sampling utilities, and backup accounting.
- `validate_everything.py`: runs all validation scripts.

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
