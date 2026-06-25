"""
optimizers/bowerbird_optimizer.py

Satin Bowerbird Optimizer (BO) for the DSR-CNN proposal score
re-weighting task.

BO optimises a weight vector w in R^100 (one scalar per detection
proposal) that scales raw confidence scores before non-maximum
suppression. The optimisation target sits after the Hungarian
matching step in the pipeline, which is combinatorial and therefore
non-differentiable -- this is why a population-based metaheuristic
is used instead of a gradient-based optimiser (SGD / Adam).

Fitness function:

    F(w) = (1/N) * sum_n ( w_n * s_n - g_n )^2

where:
    s_n  = raw proposal confidence score for proposal n
    g_n  = ground-truth presence indicator in {0, 1}
    N    = number of (image, proposal) pairs evaluated

"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class BOConfig:
    population_size: int = 30
    max_iterations: int = 100
    alpha: float = 0.5          # attraction weight (exploitation)
    beta: float = 0.5           # decoration weight (exploration)
    mutation_prob: float = 0.10
    mutation_sigma: float = 0.01
    chaotic_init: bool = True
    chaotic_r: float = 3.9      # logistic map r in chaotic regime (3.57, 4)
    num_seeds: int = 5          # seeds used for multi-seed convergence analysis


class BowerbirdOptimizer:
    """Population-based metaheuristic with chaotic-map initialisation.

    Each "bower" is a candidate weight vector w in [0, 1]^dim. Every
    iteration applies, per individual, either an attraction step
    (move toward the global best) or a decoration step (random
    displacement), followed by an optional Gaussian mutation.
    """

    def __init__(self, cfg: BOConfig | None = None):
        self.cfg = cfg or BOConfig()

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #
    def _init_population(self, dim: int, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        if not self.cfg.chaotic_init:
            return rng.uniform(0.0, 1.0, (self.cfg.population_size, dim))

        # Logistic chaotic map: x_{n+1} = r * x_n * (1 - x_n)
        r = self.cfg.chaotic_r
        pop = np.zeros((self.cfg.population_size, dim))
        x = rng.uniform(0.01, 0.99, dim)
        for i in range(self.cfg.population_size):
            pop[i] = x
            x = r * x * (1.0 - x)
        return np.clip(pop, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    # Single-seed optimisation
    # ------------------------------------------------------------------ #
    def optimise(
        self,
        fitness_fn,
        dim: int,
        seed: int = 0,
    ) -> Tuple[np.ndarray, List[float]]:
        """Run BO for one seed.

        Args:
            fitness_fn: callable(w: np.ndarray) -> float, lower is better.
            dim: dimensionality of the weight vector.
            seed: random seed for this run.

        Returns:
            best_w: optimised weight vector, shape (dim,)
            curve:  best-fitness-so-far per iteration, length = max_iterations
        """
        rng = np.random.RandomState(seed)
        pop = self._init_population(dim, seed)
        fit = np.array([fitness_fn(p) for p in pop])

        best_idx = int(np.argmin(fit))
        best_w, best_f = pop[best_idx].copy(), float(fit[best_idx])
        curve: List[float] = []

        for _ in range(self.cfg.max_iterations):
            for i in range(self.cfg.population_size):
                r_vec = rng.rand(dim)

                # Attraction phase (exploitation): move toward global best
                cand_a = np.clip(
                    pop[i] + self.cfg.alpha * (best_w - pop[i]) * r_vec,
                    0.0, 1.0,
                )
                # Decoration phase (exploration): random displacement
                cand_b = np.clip(
                    pop[i] + self.cfg.beta * rng.randn(dim),
                    0.0, 1.0,
                )
                cand = cand_a if rng.rand() > 0.5 else cand_b

                if rng.rand() < self.cfg.mutation_prob:
                    cand = np.clip(
                        cand + rng.randn(dim) * self.cfg.mutation_sigma,
                        0.0, 1.0,
                    )

                f = fitness_fn(cand)
                if f < best_f:
                    best_f, best_w = f, cand.copy()
                pop[i] = cand

            curve.append(best_f)

        return best_w, curve

    # ------------------------------------------------------------------ #
    # Multi-seed convergence analysis (Fig. 19a)
    # ------------------------------------------------------------------ #
    def multi_seed_convergence(self, fitness_fn, dim: int) -> dict:
        """Run BO across cfg.num_seeds seeds and report mean +/- std."""
        curves, weights = [], []
        for s in range(self.cfg.num_seeds):
            w, curve = self.optimise(fitness_fn, dim, seed=s)
            curves.append(curve)
            weights.append(w)

        arr = np.array(curves)               # (num_seeds, max_iterations)
        best_seed = int(np.argmin(arr[:, -1]))
        return {
            "mean_curve": arr.mean(axis=0).tolist(),
            "std_curve": arr.std(axis=0).tolist(),
            "final_mean": float(arr[:, -1].mean()),
            "final_std": float(arr[:, -1].std()),
            "per_seed_final": arr[:, -1].tolist(),
            "best_weights": weights[best_seed],
            "num_seeds": self.cfg.num_seeds,
        }
