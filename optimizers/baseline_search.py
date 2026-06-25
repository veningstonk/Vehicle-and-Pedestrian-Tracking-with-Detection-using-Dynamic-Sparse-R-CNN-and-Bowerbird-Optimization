"""
optimizers/baseline_search.py

Grid search and random search baselines used in Section 3.5
("Comparison with standard hyperparameter tuning") to demonstrate
why standard search strategies cannot replace BO and EPO under an
identical function-evaluation budget (Reviewer #4, Comment 2).
"""
from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Grid search
# ============================================================================
def grid_search_low_dim(
    objective_fn: Callable[[Dict[str, float]], float],
    bounds: Dict[str, Tuple[float, float]],
    points_per_dim: int = 14,
    log_scale_keys: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], float, int]:
    """Exhaustive grid search over a low-dimensional space.

    Args:
        objective_fn: callable(params: dict) -> float, lower is better.
        bounds: search space, e.g. {"lr": (1e-5, 1e-2), ...}.
        points_per_dim: grid resolution per dimension.
        log_scale_keys: parameter names that should use a log-spaced
            grid instead of a linear grid (e.g. learning rates).

    Returns:
        best_params: dict of the best grid point found.
        best_value: objective value at best_params.
        n_evaluations: total number of grid points evaluated
            (= points_per_dim ** num_dimensions).
    """
    log_scale_keys = log_scale_keys or []
    keys = list(bounds.keys())

    grids = []
    for k in keys:
        lo, hi = bounds[k]
        if k in log_scale_keys:
            grids.append(np.geomspace(lo, hi, points_per_dim))
        else:
            grids.append(np.linspace(lo, hi, points_per_dim))

    best_value = np.inf
    best_params: Optional[Dict[str, float]] = None
    n_evals = 0

    for combo in itertools.product(*grids):
        params = dict(zip(keys, combo))
        value = objective_fn(params)
        n_evals += 1
        if value < best_value:
            best_value, best_params = value, params

    return best_params, float(best_value), n_evals


def grid_search_feasibility(num_dims: int, points_per_dim: int = 2) -> float:
    """Return the number of grid points required for a num_dims-dimensional
    grid at the given resolution -- used to demonstrate combinatorial
    infeasibility for high-dimensional targets (e.g. BO's 100-D task).
    """
    return float(points_per_dim) ** num_dims


# ============================================================================
# Random search
# ============================================================================
def random_search_vector(
    fitness_fn: Callable[[np.ndarray], float],
    dim: int,
    budget: int,
    seed: int = 0,
    low: float = 0.0,
    high: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Random search over a real-valued vector of dimension `dim`
    (used for the BO-equivalent 100-D proposal weighting task).

    Args:
        fitness_fn: callable(w: np.ndarray) -> float, lower is better.
        dim: dimensionality of the search vector.
        budget: number of random samples to evaluate.
        seed: random seed.
        low, high: per-dimension sampling bounds.

    Returns:
        best_w: best vector found.
        best_value: objective value at best_w.
    """
    rng = np.random.RandomState(seed)
    best_value = np.inf
    best_w = None
    for _ in range(budget):
        w = rng.uniform(low, high, dim)
        value = fitness_fn(w)
        if value < best_value:
            best_value, best_w = value, w
    return best_w, float(best_value)


def random_search_dict(
    objective_fn: Callable[[Dict[str, float]], float],
    bounds: Dict[str, Tuple[float, float]],
    budget: int,
    seed: int = 0,
    log_scale_keys: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], float]:
    """Random search over a named hyperparameter dict (used for the
    EPO-equivalent 3-D hyperparameter search task).

    Args:
        objective_fn: callable(params: dict) -> float, lower is better.
        bounds: search space, e.g. {"lr": (1e-5, 1e-2), ...}.
        budget: number of random samples to evaluate.
        seed: random seed.
        log_scale_keys: parameter names sampled log-uniformly
            instead of uniformly (e.g. learning rates).

    Returns:
        best_params: best hyperparameter dict found.
        best_value: objective value at best_params.
    """
    log_scale_keys = log_scale_keys or []
    rng = np.random.RandomState(seed)
    keys = list(bounds.keys())

    best_value = np.inf
    best_params: Optional[Dict[str, float]] = None

    for _ in range(budget):
        params = {}
        for k in keys:
            lo, hi = bounds[k]
            if k in log_scale_keys:
                params[k] = 10 ** rng.uniform(np.log10(lo), np.log10(hi))
            else:
                params[k] = rng.uniform(lo, hi)
        value = objective_fn(params)
        if value < best_value:
            best_value, best_params = value, params

    return best_params, float(best_value)


def random_search_multi_seed_dict(
    objective_fn: Callable[[Dict[str, float]], float],
    bounds: Dict[str, Tuple[float, float]],
    budget: int,
    num_seeds: int = 5,
    log_scale_keys: Optional[List[str]] = None,
) -> dict:
    """Run random_search_dict across multiple seeds and report mean/std,
    matching the reporting convention used for BO/EPO."""
    results = [
        random_search_dict(objective_fn, bounds, budget, seed=s,
                            log_scale_keys=log_scale_keys)
        for s in range(num_seeds)
    ]
    finals = [r[1] for r in results]
    best_idx = int(np.argmin(finals))
    return {
        "final_mean": float(np.mean(finals)),
        "final_std": float(np.std(finals)),
        "per_seed_final": finals,
        "best_params": results[best_idx][0],
        "num_seeds": num_seeds,
    }


def random_search_multi_seed_vector(
    fitness_fn: Callable[[np.ndarray], float],
    dim: int,
    budget: int,
    num_seeds: int = 5,
) -> dict:
    """Run random_search_vector across multiple seeds and report mean/std,
    matching the reporting convention used for BO."""
    results = [
        random_search_vector(fitness_fn, dim, budget, seed=s)
        for s in range(num_seeds)
    ]
    finals = [r[1] for r in results]
    best_idx = int(np.argmin(finals))
    return {
        "final_mean": float(np.mean(finals)),
        "final_std": float(np.std(finals)),
        "per_seed_final": finals,
        "best_weights": results[best_idx][0],
        "num_seeds": num_seeds,
    }
