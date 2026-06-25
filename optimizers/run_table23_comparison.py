from __future__ import annotations

import json
import math
import os
import sys
import time

# Allow running this script directly without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from optimizers.bowerbird_optimizer import BowerbirdOptimizer, BOConfig
from optimizers.adaptive_epo import AdaptiveEPO, EPOConfig
from optimizers.baseline_search import (
    grid_search_feasibility,
    grid_search_low_dim,
    random_search_multi_seed_vector,
    random_search_multi_seed_dict,
)
from experiments.fitness_functions import (
    NUM_PROPOSALS,
    make_proposal_data,
    make_proposal_fitness_fn,
    hyperparameter_val_loss,
    EPO_BOUNDS,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Evaluation budget matched across all methods: population x iterations.
BUDGET = 30 * 100   # = 3000


def run_bo_block():
    """Section A: BO vs. random search on the 100-D proposal weighting task.
    Grid search is reported as infeasible (combinatorial blow-up)."""
    print("=" * 78)
    print("SECTION A — BO target: 100-D proposal score weighting")
    print("=" * 78)

    raw_scores, gt_labels = make_proposal_data()
    fitness_fn = make_proposal_fitness_fn(raw_scores, gt_labels)
    baseline_mse = fitness_fn(np.ones(NUM_PROPOSALS))
    print(f"Baseline MSE (w = all-ones, no optimisation): {baseline_mse:.6f}")

    # --- Grid search feasibility check (not executed -- combinatorially
    #     infeasible even at the coarsest possible resolution) -----------
    n_grid_2pts = grid_search_feasibility(num_dims=NUM_PROPOSALS, points_per_dim=2)
    print(f"\nGrid search @ 2 points/dim over {NUM_PROPOSALS}-D space requires "
          f"{n_grid_2pts:.3e} evaluations.")
    print(f"  --> INFEASIBLE under the {BUDGET}-evaluation budget "
          f"({n_grid_2pts / BUDGET:.3e}x over budget).")

    # --- Random search, 5 seeds, matched budget -------------------------
    t0 = time.perf_counter()
    rs_result = random_search_multi_seed_vector(
        fitness_fn, dim=NUM_PROPOSALS, budget=BUDGET, num_seeds=5)
    t_rs = time.perf_counter() - t0
    print(f"\nRandom search (5 seeds, budget={BUDGET} each, {t_rs:.1f}s):")
    print(f"  Final MSE: {rs_result['final_mean']:.6f} ± {rs_result['final_std']:.6f}")
    print(f"  Per-seed:  {[round(v, 6) for v in rs_result['per_seed_final']]}")

    # --- Bowerbird Optimizer, 5 seeds, matched budget --------------------
    t0 = time.perf_counter()
    bo = BowerbirdOptimizer(BOConfig(population_size=30, max_iterations=100,
                                      num_seeds=5))
    bo_result = bo.multi_seed_convergence(fitness_fn, dim=NUM_PROPOSALS)
    t_bo = time.perf_counter() - t0
    print(f"\nBowerbird Optimizer (5 seeds, pop=30 x iters=100 = {BUDGET}, "
          f"{t_bo:.1f}s):")
    print(f"  Final MSE: {bo_result['final_mean']:.6f} ± {bo_result['final_std']:.6f}")
    print(f"  Per-seed:  {[round(v, 6) for v in bo_result['per_seed_final']]}")

    improvement = 100 * (rs_result["final_mean"] - bo_result["final_mean"]) \
        / rs_result["final_mean"]
    print(f"\n  BO improves on random search by {improvement:.2f}% "
          f"(lower MSE) under the identical budget.")

    return {
        "baseline_mse": baseline_mse,
        "grid_search": {
            "feasible": False,
            "required_evaluations": n_grid_2pts,
            "budget": BUDGET,
        },
        "random_search": rs_result,
        "bowerbird_optimizer": bo_result,
        "bo_improvement_pct": improvement,
    }


def run_epo_block():
    """Section B: EPO vs. grid search and random search on the 3-D
    hyperparameter search task."""
    print("\n" + "=" * 78)
    print("SECTION B — EPO target: 3-D hyperparameter search "
          "(lr, dropout, da_eva_lambda)")
    print("=" * 78)

    # --- Grid search, budget-matched resolution (14^3 = 2744 ~= 3000) ----
    points_per_dim = 14
    t0 = time.perf_counter()
    gs_params, gs_value, gs_n = grid_search_low_dim(
        hyperparameter_val_loss, EPO_BOUNDS,
        points_per_dim=points_per_dim, log_scale_keys=["lr"])
    t_gs = time.perf_counter() - t0
    print(f"\nGrid search ({points_per_dim} points/dim, {gs_n} evaluations, "
          f"deterministic, {t_gs:.1f}s):")
    print(f"  Final loss: {gs_value:.6f}")
    print(f"  Best params: lr={gs_params['lr']:.5f}  "
          f"dropout={gs_params['dropout']:.4f}  "
          f"da_eva_lambda={gs_params['da_eva_lambda']:.4f}")

    # --- Random search, 5 seeds, matched budget --------------------------
    t0 = time.perf_counter()
    rs_result = random_search_multi_seed_dict(
        hyperparameter_val_loss, EPO_BOUNDS, budget=BUDGET, num_seeds=5,
        log_scale_keys=["lr"])
    t_rs = time.perf_counter() - t0
    print(f"\nRandom search (5 seeds, budget={BUDGET} each, {t_rs:.1f}s):")
    print(f"  Final loss: {rs_result['final_mean']:.6e} ± "
          f"{rs_result['final_std']:.6e}")
    print(f"  Per-seed:  {[f'{v:.6e}' for v in rs_result['per_seed_final']]}")
    print(f"  Best params: {rs_result['best_params']}")

    # --- Adaptive EPO, 5 seeds, matched budget ----------------------------
    t0 = time.perf_counter()
    epo = AdaptiveEPO(EPOConfig(population_size=30, max_iterations=100,
                                 num_seeds=5))
    epo_result = epo.multi_seed_convergence(hyperparameter_val_loss, EPO_BOUNDS)
    t_epo = time.perf_counter() - t0
    print(f"\nAdaptive EPO (5 seeds, pop=30 x iters<=100, {t_epo:.1f}s):")
    print(f"  Final loss: {epo_result['final_mean']:.6e} ± "
          f"{epo_result['final_std']:.6e}")
    print(f"  Per-seed:  {[f'{v:.6e}' for v in epo_result['per_seed_final']]}")
    print(f"  Best params: {epo_result['best_params']}")

    ratio = (rs_result["final_mean"] / epo_result["final_mean"]
             if epo_result["final_mean"] > 0 else float("inf"))
    print(f"\n  EPO achieves {ratio:.2e}x lower loss than random search "
          f"under the identical budget.")

    return {
        "grid_search": {
            "n_evaluations": gs_n,
            "final_value": gs_value,
            "best_params": gs_params,
        },
        "random_search": rs_result,
        "adaptive_epo": epo_result,
        "epo_improvement_factor": ratio,
    }


def write_table16_csv(bo_block: dict, epo_block: dict, path: str):
    rows = [
        ["Method", "Task", "Evaluations", "Final mean", "Final std", "Notes"],
        ["Grid search", "EPO (3-D)", bo_block_placeholder_n(epo_block),
         f"{epo_block['grid_search']['final_value']:.6f}", "0.0 (deterministic)",
         "Coarse grid cannot resolve narrow optimum"],
        ["Grid search", "BO (100-D)",
         f"{bo_block['grid_search']['required_evaluations']:.3e} required",
         "N/A", "N/A", "Infeasible -- exceeds budget"],
        ["Random search", "BO (100-D)", BUDGET,
         f"{bo_block['random_search']['final_mean']:.6f}",
         f"{bo_block['random_search']['final_std']:.6f}", "5 seeds"],
        ["Bowerbird Optimizer (proposed)", "BO (100-D)", BUDGET,
         f"{bo_block['bowerbird_optimizer']['final_mean']:.6f}",
         f"{bo_block['bowerbird_optimizer']['final_std']:.6f}", "5 seeds"],
        ["Random search", "EPO (3-D)", BUDGET,
         f"{epo_block['random_search']['final_mean']:.6e}",
         f"{epo_block['random_search']['final_std']:.6e}", "5 seeds"],
        ["Adaptive EPO (proposed)", "EPO (3-D)", BUDGET,
         f"{epo_block['adaptive_epo']['final_mean']:.6e}",
         f"{epo_block['adaptive_epo']['final_std']:.6e}", "5 seeds"],
    ]
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def bo_block_placeholder_n(epo_block: dict) -> int:
    return epo_block["grid_search"]["n_evaluations"]


def main():
    print("\nReproducing Table 16 -- BO/EPO vs. grid search and random "
          "search\n(matched evaluation budget = 30 x 100 = 3000)\n")

    bo_block = run_bo_block()
    epo_block = run_epo_block()

    out_json = {"budget": BUDGET, "bo_task": bo_block, "epo_task": epo_block}
    json_path = os.path.join(RESULTS_DIR, "table16_results.json")
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2, default=str)

    csv_path = os.path.join(RESULTS_DIR, "table16.csv")
    write_table16_csv(bo_block, epo_block, csv_path)

    print("\n" + "=" * 78)
    print(f"Results saved:\n  {json_path}\n  {csv_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
