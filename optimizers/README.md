# BO / EPO vs. Standard Hyperparameter Tuning

This code compares the Bowerbird Optimizer (BO) and Adaptive Emperor
Penguin Optimization (EPO) against grid search and random search under
an **identical function-evaluation budget** (30 × 100 = 3,000
evaluations) on the two non-differentiable optimisation targets used
in the pipeline:

1. **BO's target** — a 100-dimensional proposal score weighting vector
   `w` that scales DSR-CNN's raw detection confidence scores before
   non-maximum suppression.
2. **EPO's target** — a 3-dimensional MobDEAP hyperparameter search
   over learning rate, dropout, and the DA-EVA softmax temperature λ.

---

## Repository layout

```
optimizers/
  bowerbird_optimizer.py   BO implementation (chaotic init, attraction/decoration)
  adaptive_epo.py           EPO implementation (fuzzy-adaptive social forces)
  baseline_search.py        Grid search and random search baselines

experiments/
  fitness_functions.py      The two optimisation targets (Eq. 3 + surrogate loss)
  run_table16_comparison.py Main script — reproduces Table 16

tests/
  test_optimizers.py        Sanity checks (reproducibility, monotonicity, etc.)

results/                    Output directory (created automatically)
  table16_results.json
  table16.csv

requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

Only `numpy` is required. No GPU, no PyTorch, no external datasets —
this experiment is self-contained and runs in under 10 seconds on a
laptop CPU.

---

## Running the experiment

```bash
python experiments/run_table16_comparison.py
```

This prints a full breakdown to stdout and writes:

- `results/table16_results.json` — full numeric detail (per-seed
  values, best parameters found, convergence curves).
- `results/table16.csv` — the flat table for direct inclusion in
  spreadsheets or LaTeX table generators.

## Running the tests

```bash
python -m pytest tests/ -v
```

or, if `pytest` is not installed:

```bash
python tests/test_optimizers.py
```

---

## What the experiment demonstrates

**Grid search is combinatorially infeasible on BO's 100-D target.**
Even the coarsest possible grid (2 points per dimension) requires
2¹⁰⁰ ≈ 1.27 × 10³⁰ evaluations — twenty-six orders of magnitude beyond
the 3,000-evaluation budget. Grid search is therefore reported only on
EPO's 3-D target, where a 14-point-per-dimension grid (2,744
evaluations) is feasible but converges to a coarse local solution
because the fixed discretisation cannot represent the narrow true
optimum precisely.

**Random search is feasible on both tasks but is consistently
outperformed.** On the proposal weighting task, BO outperforms random
search under the matched budget. On the hyperparameter search task,
EPO converges to a substantially lower loss with far less inter-seed
variance than random search, which is highly inconsistent across
seeds at this budget.

**Important honest caveat.** On the BO task, the trivial baseline
(`w` = all-ones, i.e. using raw detection scores unmodified) is itself
a strong reference point on the synthetic data used here — its MSE
(≈0.2325) sits close to the analytical optimum (≈0.2273) obtainable by
per-proposal ordinary least squares. As a result, BO's mean final MSE
across 5 seeds does not always fall below this trivial baseline by a
large margin, and the manuscript should **not** claim that BO
dramatically outperforms a no-optimisation baseline. The defensible
and tested claim — and the one this code validates — is that **BO
outperforms random search under an identical evaluation budget**,
which is the comparison Reviewer #4 actually asked for. See the
docstring of `test_bowerbird_outperforms_random_search` in
`tests/test_optimizers.py` for the exact assertion this code makes and
why.

---

## Hyperparameters used (matching manuscript Table 13a / 13b)

| Optimizer | Population | Iterations | Other parameters |
|---|---|---|---|
| BO | 30 | 100 | α=0.5, β=0.5, mutation=0.10, chaotic init r=3.9 |
| EPO | 30 | 100 (early stop δ=1e-5) | T₀=1.0, decay=0.95, f=0.5, l=1.5, fuzzy adapt enabled |
| Grid search (EPO task only) | — | 14³ = 2,744 points | Log-spaced grid for `lr`, linear for `dropout`/`λ` |
| Random search | — | 3,000 samples | Uniform sampling within bounds; log-uniform for `lr` |

---