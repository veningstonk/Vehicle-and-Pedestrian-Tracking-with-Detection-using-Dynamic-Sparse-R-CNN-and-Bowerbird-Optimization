"""
optimizers/adaptive_epo.py

Adaptive Emperor Penguin Optimization (EPO) for the MobDEAP
hyperparameter search task.

EPO tunes three hyperparameters that govern the MobDEAP training
procedure itself:

    lr           -- AdamW learning rate
    dropout      -- dropout rate before the final classification layer
    da_eva_lambda -- softmax temperature lambda in DA-EVA

These hyperparameters are inaccessible to training-time gradients
because they control the training procedure rather than participate
in the forward-pass computation graph -- this is why a population-
based metaheuristic is used instead of gradient-based hyperparameter
optimisation.

Fuzzy-adaptive social forces: the exploration force `f` decays
linearly to zero and the exploitation force `l` grows linearly,
both as a function of iteration progress, which improves convergence
stability relative to the original EPO's fixed social forces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EPOConfig:
    """All Adaptive EPO hyperparameters, explicit and documented"""

    population_size: int = 30
    max_iterations: int = 100
    temperature_init: float = 1.0
    temperature_decay: float = 0.95      # per iteration
    social_force_f: float = 0.5          # exploration (decreases via fuzzy adapt)
    social_force_l: float = 1.5          # exploitation (increases via fuzzy adapt)
    fuzzy_adapt: bool = True
    early_stop_delta: float = 1e-5
    num_seeds: int = 5
    search_space: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "lr": (1e-5, 1e-2),
            "dropout": (0.10, 0.50),
            "da_eva_lambda": (0.50, 2.00),
        }
    )


class AdaptiveEPO:
    """Population-based hyperparameter optimiser with fuzzy-adaptive
    social forces and geometric temperature decay."""

    def __init__(self, cfg: Optional[EPOConfig] = None):
        self.cfg = cfg or EPOConfig()

    # ------------------------------------------------------------------ #
    def _init_population(self, bounds: Dict[str, Tuple[float, float]],
                          seed: int) -> Tuple[np.ndarray, List[str]]:
        rng = np.random.RandomState(seed)
        keys = list(bounds.keys())
        pop = np.array([
            [rng.uniform(*bounds[k]) for k in keys]
            for _ in range(self.cfg.population_size)
        ])
        return pop, keys

    def _fuzzy_forces(self, iteration: int) -> Tuple[float, float]:
        progress = iteration / max(self.cfg.max_iterations, 1)
        f = self.cfg.social_force_f * (1 - progress)   # exploration shrinks
        l = self.cfg.social_force_l * (1 + progress)   # exploitation grows
        return f, l

    # ------------------------------------------------------------------ #
    # Single-seed optimisation
    # ------------------------------------------------------------------ #
    def optimise(
        self,
        objective_fn: Callable[[Dict[str, float]], float],
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        seed: int = 0,
    ) -> Tuple[Dict[str, float], List[float]]:
        """Run EPO for one seed.

        Args:
            objective_fn: callable(params: dict) -> float, lower is better.
            bounds: search space; defaults to cfg.search_space.
            seed: random seed for this run.

        Returns:
            best_params: dict of optimised hyperparameters.
            curve: best-fitness-so-far per iteration.
        """
        rng = np.random.RandomState(seed)
        bounds = bounds or self.cfg.search_space
        pop, keys = self._init_population(bounds, seed)

        fitness = np.array([objective_fn(dict(zip(keys, p))) for p in pop])
        best_idx = int(np.argmin(fitness))
        best_p, best_f = pop[best_idx].copy(), float(fitness[best_idx])
        temp = self.cfg.temperature_init
        curve: List[float] = []

        for it in range(self.cfg.max_iterations):
            f_force, l_force = (
                self._fuzzy_forces(it) if self.cfg.fuzzy_adapt
                else (self.cfg.social_force_f, self.cfg.social_force_l)
            )

            for i in range(self.cfg.population_size):
                delta = pop[i] - best_p
                new_p = (
                    pop[i]
                    - f_force * rng.rand() * delta * temp
                    + l_force * rng.rand() * (best_p - pop[i])
                )
                for j, k in enumerate(keys):
                    lo, hi = bounds[k]
                    new_p[j] = np.clip(new_p[j], lo, hi)

                fv = objective_fn(dict(zip(keys, new_p)))
                if fv < best_f:
                    best_f, best_p = fv, new_p.copy()
                pop[i] = new_p

            temp *= self.cfg.temperature_decay
            curve.append(best_f)

            if (len(curve) > 5
                    and abs(curve[-5] - curve[-1]) < self.cfg.early_stop_delta):
                break

        return dict(zip(keys, best_p)), curve

    # ------------------------------------------------------------------ #
    # Multi-seed convergence analysis (Fig. 19b)
    # ------------------------------------------------------------------ #
    def multi_seed_convergence(
        self,
        objective_fn: Callable[[Dict[str, float]], float],
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> dict:
        results = [
            self.optimise(objective_fn, bounds, seed=s)
            for s in range(self.cfg.num_seeds)
        ]
        curves = [r[1] for r in results]
        max_len = max(len(c) for c in curves)
        padded = np.array([c + [c[-1]] * (max_len - len(c)) for c in curves])

        best_seed = int(np.argmin(padded[:, -1]))
        return {
            "mean_curve": padded.mean(axis=0).tolist(),
            "std_curve": padded.std(axis=0).tolist(),
            "final_mean": float(padded[:, -1].mean()),
            "final_std": float(padded[:, -1].std()),
            "per_seed_final": padded[:, -1].tolist(),
            "best_params": results[best_seed][0],
            "num_seeds": self.cfg.num_seeds,
        }
