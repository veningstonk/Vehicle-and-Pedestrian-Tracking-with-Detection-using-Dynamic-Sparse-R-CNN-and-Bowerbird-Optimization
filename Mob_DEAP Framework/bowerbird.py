"""
optimizers/bowerbird.py  —  Satin Bowerbird Optimizer  (§3.3)
optimizers/epo.py        —  Adaptive Emperor Penguin Optimizer  (§3.4.3)

Both optimisers fully documented with all hyperparameters explicit,
satisfying Reviewer #3 Comment 6 reproducibility requirements.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from config import BOConfig, EPOConfig


# ═══════════════════════════════════════════════════════════════════════════
# Bowerbird Optimizer  (Samareh Moosavi & Khatibi Bardsiri, 2017)
# ═══════════════════════════════════════════════════════════════════════════
class BowerbirdOptimizer:
    """
    Optimises the proposal score weight vector w ∈ ℝ^{100} for DSR-CNN.
    Fitness function: F(w) = (1/N) Σ_n (s̃_n − g_n)²  [Eq. 3]

    Two phases per individual per iteration:
      Attraction (exploitation): move toward global best bower
      Decoration (exploration) : random displacement

    Initialisation uses a logistic chaotic map (Wangkhamhan, 2021)
    for higher population diversity.
    """

    def __init__(self, cfg: BOConfig = None):
        self.cfg = cfg or BOConfig()

    # ── Initialisation ────────────────────────────────────────────────────
    def _init_population(self, dim: int, seed: int) -> np.ndarray:
        np.random.seed(seed)
        if self.cfg.chaotic_init:
            r   = self.cfg.chaotic_r
            pop = np.zeros((self.cfg.population_size, dim))
            x   = np.random.uniform(0.01, 0.99, dim)
            for i in range(self.cfg.population_size):
                pop[i] = x
                x = r * x * (1 - x)
            return np.clip(pop, 0.0, 1.0)
        return np.random.uniform(0.0, 1.0, (self.cfg.population_size, dim))

    # ── Fitness ───────────────────────────────────────────────────────────
    @staticmethod
    def _fitness(w: np.ndarray,
                 proposals : np.ndarray,
                 gt_scores : np.ndarray) -> float:
        """MSE between weighted proposal scores and ground-truth."""
        pred = (proposals * w).sum(axis=-1)
        return float(np.mean((pred - gt_scores) ** 2))

    # ── Single-seed optimisation ──────────────────────────────────────────
    def optimise(self,
                 proposals : np.ndarray,   # (N, dim)
                 gt_scores : np.ndarray,   # (N,)
                 seed      : int = 0
                 ) -> Tuple[np.ndarray, List[float]]:
        """
        Returns:
            best_weights : (dim,) optimal weight vector
            curve        : convergence curve (best fitness per iteration)
        """
        dim = proposals.shape[1]
        pop = self._init_population(dim, seed)
        fit = np.array([self._fitness(p, proposals, gt_scores) for p in pop])

        best_idx = int(np.argmin(fit))
        best_w, best_f = pop[best_idx].copy(), fit[best_idx]
        curve: List[float] = []

        for _ in range(self.cfg.max_iterations):
            for i in range(self.cfg.population_size):
                r = np.random.rand(dim)

                # Attraction phase (exploitation)
                cA = np.clip(pop[i] + self.cfg.alpha * (best_w - pop[i]) * r,
                             0.0, 1.0)
                # Decoration phase (exploration)
                cB = np.clip(pop[i] + self.cfg.beta * np.random.randn(dim),
                             0.0, 1.0)

                c = cA if np.random.rand() > 0.5 else cB

                # Optional Gaussian mutation
                if np.random.rand() < self.cfg.mutation_prob:
                    c = np.clip(c + np.random.randn(dim) * 0.01, 0.0, 1.0)

                f = self._fitness(c, proposals, gt_scores)
                if f < best_f:
                    best_f, best_w = f, c.copy()
                pop[i] = c

            curve.append(best_f)

        return best_w, curve

    # ── Multi-seed convergence analysis ───────────────────────────────────
    def multi_seed_convergence(self,
                                proposals : np.ndarray,
                                gt_scores : np.ndarray
                                ) -> Dict:
        """
        Runs optimisation over cfg.num_seeds seeds.
        Returns mean ± std convergence statistics.
        Required by Reviewer #3 Comment 6.
        """
        curves, weights = [], []
        for s in range(self.cfg.num_seeds):
            w, curve = self.optimise(proposals, gt_scores, seed=s)
            curves.append(curve)
            weights.append(w)

        arr = np.array(curves)
        best_s = int(np.argmin(arr[:, -1]))
        return {
            "mean_curve"   : arr.mean(axis=0).tolist(),
            "std_curve"    : arr.std(axis=0).tolist(),
            "final_mean"   : float(arr[:, -1].mean()),
            "final_std"    : float(arr[:, -1].std()),
            "best_weights" : weights[best_s],
            "num_seeds"    : self.cfg.num_seeds,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive Emperor Penguin Optimizer  (§3.4.3)
# ═══════════════════════════════════════════════════════════════════════════
class AdaptiveEPO:
    """
    Optimises {lr, dropout, λ_DA-EVA} for MobDEAP by minimising
    validation loss.  Fuzzy-adaptive social forces improve convergence
    stability relative to the baseline EPO.

    Hyperparameters (all explicit per Reviewer #3 Comment 6):
        population_size   = 30
        max_iterations    = 100
        temperature_init  = 1.0,  decay = 0.95
        social_force_f    = 0.5  (exploration, decreases linearly)
        social_force_l    = 1.5  (exploitation, increases linearly)
        fuzzy_adapt       = True
        early_stop_delta  = 1e-5
    """

    def __init__(self, cfg: EPOConfig = None):
        self.cfg = cfg or EPOConfig()

    def _init_population(self, bounds: Dict) -> Tuple[np.ndarray, List]:
        keys = list(bounds.keys())
        pop  = np.array([
            [np.random.uniform(*bounds[k]) for k in keys]
            for _ in range(self.cfg.population_size)
        ])
        return pop, keys

    def _fuzzy_forces(self, iteration: int) -> Tuple[float, float]:
        progress = iteration / max(self.cfg.max_iterations, 1)
        f = self.cfg.social_force_f * (1 - progress)   # exploration reduces
        l = self.cfg.social_force_l * (1 + progress)   # exploitation increases
        return f, l

    def optimise(self,
                 objective_fn : Callable,
                 bounds       : Optional[Dict] = None,
                 seed         : int = 0
                 ) -> Tuple[Dict, List[float]]:
        """
        Args:
            objective_fn: callable(params_dict: Dict) -> float (loss, lower=better)
            bounds:       search space dict; defaults to cfg.search_space
            seed:         random seed for this run
        Returns:
            best_params : Dict with optimised hyperparameters
            curve       : convergence curve
        """
        np.random.seed(seed)
        if bounds is None:
            bounds = self.cfg.search_space
        pop, keys = self._init_population(bounds)

        fitness = np.array([objective_fn(dict(zip(keys, p))) for p in pop])
        best_idx = int(np.argmin(fitness))
        best_p, best_f = pop[best_idx].copy(), fitness[best_idx]
        temp = self.cfg.temperature_init
        curve: List[float] = []

        for it in range(self.cfg.max_iterations):
            f, l = (self._fuzzy_forces(it) if self.cfg.fuzzy_adapt
                    else (self.cfg.social_force_f, self.cfg.social_force_l))

            for i in range(self.cfg.population_size):
                delta = pop[i] - best_p
                new_p = (pop[i]
                         - f * np.random.rand() * delta * temp
                         + l * np.random.rand() * (best_p - pop[i]))
                # Clip to bounds
                for j, k in enumerate(keys):
                    lo, hi = bounds[k]
                    new_p[j] = np.clip(new_p[j], lo, hi)

                fv = objective_fn(dict(zip(keys, new_p)))
                if fv < best_f:
                    best_f, best_p = fv, new_p.copy()
                pop[i] = new_p

            temp *= self.cfg.temperature_decay
            curve.append(best_f)

            # Early stopping
            if (len(curve) > 5
                    and abs(curve[-5] - curve[-1]) < self.cfg.early_stop_delta):
                break

        return dict(zip(keys, best_p)), curve

    def multi_seed_convergence(self,
                                objective_fn : Callable,
                                bounds       : Optional[Dict] = None
                                ) -> Dict:
        results = [self.optimise(objective_fn, bounds, seed=s)
                   for s in range(self.cfg.num_seeds)]
        curves = [r[1] for r in results]
        ml = max(len(c) for c in curves)
        padded = np.array([c + [c[-1]] * (ml - len(c)) for c in curves])
        best_s = int(np.argmin(padded[:, -1]))
        return {
            "mean_curve" : padded.mean(axis=0).tolist(),
            "std_curve"  : padded.std(axis=0).tolist(),
            "final_mean" : float(padded[:, -1].mean()),
            "final_std"  : float(padded[:, -1].std()),
            "best_params": results[best_s][0],
        }
