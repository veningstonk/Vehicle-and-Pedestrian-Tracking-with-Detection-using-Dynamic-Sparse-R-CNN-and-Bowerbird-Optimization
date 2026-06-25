"""
tests/test_optimizers.py

Sanity tests for the BO / EPO / grid search / random search
implementations.

Run with:
    python -m pytest tests/ -v

or, if pytest is not installed, run directly:
    python tests/test_optimizers.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from optimizers.bowerbird_optimizer import BowerbirdOptimizer, BOConfig
from optimizers.adaptive_epo import AdaptiveEPO, EPOConfig
from optimizers.baseline_search import (
    grid_search_feasibility,
    grid_search_low_dim,
    random_search_vector,
    random_search_dict,
)
from experiments.fitness_functions import (
    NUM_PROPOSALS,
    make_proposal_data,
    make_proposal_fitness_fn,
    hyperparameter_val_loss,
    EPO_BOUNDS,
    TRUE_OPTIMUM,
)


def test_bowerbird_outperforms_random_search():
    raw_scores, gt_labels = make_proposal_data()
    fitness_fn = make_proposal_fitness_fn(raw_scores, gt_labels)

    from optimizers.baseline_search import random_search_multi_seed_vector
    rs = random_search_multi_seed_vector(fitness_fn, dim=NUM_PROPOSALS,
                                          budget=3000, num_seeds=5)

    bo = BowerbirdOptimizer(BOConfig(population_size=30, max_iterations=100,
                                      num_seeds=5))
    bo_result = bo.multi_seed_convergence(fitness_fn, dim=NUM_PROPOSALS)

    assert bo_result["final_mean"] < rs["final_mean"], (
        "BO must outperform random search under the identical budget "
        f"(BO={bo_result['final_mean']:.6f}, "
        f"random_search={rs['final_mean']:.6f})"
    )
    print("PASS: test_bowerbird_outperforms_random_search "
          f"(BO={bo_result['final_mean']:.6f} < "
          f"random_search={rs['final_mean']:.6f})")


def test_bowerbird_curve_is_non_increasing():
    raw_scores, gt_labels = make_proposal_data()
    fitness_fn = make_proposal_fitness_fn(raw_scores, gt_labels)

    bo = BowerbirdOptimizer(BOConfig(population_size=10, max_iterations=30))
    _, curve = bo.optimise(fitness_fn, dim=NUM_PROPOSALS, seed=1)

    assert all(curve[i] >= curve[i + 1] for i in range(len(curve) - 1)), (
        "BO's best-fitness-so-far curve must be monotonically non-increasing "
        "by construction (it always retains the best solution seen)."
    )
    print("PASS: test_bowerbird_curve_is_non_increasing")


def test_epo_recovers_known_optimum():
    epo = AdaptiveEPO(EPOConfig(population_size=20, max_iterations=80))
    best_params, curve = epo.optimise(hyperparameter_val_loss, EPO_BOUNDS, seed=0)

    assert abs(best_params["lr"] - TRUE_OPTIMUM["lr"]) < 1e-3
    assert abs(best_params["dropout"] - TRUE_OPTIMUM["dropout"]) < 0.05
    assert abs(best_params["da_eva_lambda"] - TRUE_OPTIMUM["da_eva_lambda"]) < 0.10
    assert curve[-1] < 1.0, "EPO should reduce the surrogate loss substantially."
    print("PASS: test_epo_recovers_known_optimum")


def test_grid_search_infeasible_for_high_dimension():
    n_required = grid_search_feasibility(num_dims=NUM_PROPOSALS, points_per_dim=2)
    budget = 30 * 100
    assert n_required > budget, (
        "Grid search over the 100-D BO target must require more "
        "evaluations than the matched budget, confirming infeasibility."
    )
    print(f"PASS: test_grid_search_infeasible_for_high_dimension "
          f"({n_required:.3e} > {budget})")


def test_grid_search_feasible_for_low_dimension():
    best_params, best_value, n_evals = grid_search_low_dim(
        hyperparameter_val_loss, EPO_BOUNDS,
        points_per_dim=14, log_scale_keys=["lr"])
    assert n_evals == 14 ** 3
    assert best_value < 10.0, "Grid search should find a reasonable local solution."
    print(f"PASS: test_grid_search_feasible_for_low_dimension "
          f"(n_evals={n_evals}, best_value={best_value:.4f})")


def test_random_search_runs_with_matched_budget():
    raw_scores, gt_labels = make_proposal_data()
    fitness_fn = make_proposal_fitness_fn(raw_scores, gt_labels)
    _, best_value = random_search_vector(fitness_fn, dim=NUM_PROPOSALS,
                                          budget=300, seed=0)
    assert np.isfinite(best_value)
    print(f"PASS: test_random_search_runs_with_matched_budget "
          f"(best_value={best_value:.6f})")


def test_reproducibility_same_seed_same_result():
    raw_scores, gt_labels = make_proposal_data()
    fitness_fn = make_proposal_fitness_fn(raw_scores, gt_labels)

    bo1 = BowerbirdOptimizer(BOConfig(population_size=10, max_iterations=15))
    bo2 = BowerbirdOptimizer(BOConfig(population_size=10, max_iterations=15))
    _, curve1 = bo1.optimise(fitness_fn, dim=NUM_PROPOSALS, seed=7)
    _, curve2 = bo2.optimise(fitness_fn, dim=NUM_PROPOSALS, seed=7)

    assert curve1 == curve2, "Same seed must produce identical results."
    print("PASS: test_reproducibility_same_seed_same_result")


if __name__ == "__main__":
    test_bowerbird_outperforms_random_search()
    test_bowerbird_curve_is_non_increasing()
    test_epo_recovers_known_optimum()
    test_grid_search_infeasible_for_high_dimension()
    test_grid_search_feasible_for_low_dimension()
    test_random_search_runs_with_matched_budget()
    test_reproducibility_same_seed_same_result()
    print("\nAll tests passed.")
