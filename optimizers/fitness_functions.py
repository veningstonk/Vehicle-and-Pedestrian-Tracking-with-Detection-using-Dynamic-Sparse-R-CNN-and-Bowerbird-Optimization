"""
experiments/fitness_functions.py

The two optimisation targets compared in Section 3.5 / Table 16:

  1. proposal_score_fitness  -- BO's 100-D target (manuscript Eq. 3)
  2. hyperparameter_val_loss -- EPO's 3-D target (surrogate validation loss)

Both functions are deterministic given a fixed random seed for the
synthetic data, so results are exactly reproducible across machines.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


# ============================================================================
# BO target: proposal score weighting (Eq. 3)
# ============================================================================
NUM_PROPOSALS = 100   # matches DSRCNNConfig.num_proposals
NUM_IMAGES = 300       # mini-batch images used to evaluate fitness
DATA_SEED = 42


def make_proposal_data(num_images: int = NUM_IMAGES,
                        num_proposals: int = NUM_PROPOSALS,
                        seed: int = DATA_SEED):
    """Generate synthetic (raw_scores, gt_labels) consistent with a
    realistic object-detection proposal distribution.

    raw_scores: Beta(2, 5) -- right-skewed, mimicking typical detector
                confidence score distributions (most proposals score low).
    gt_labels:  Bernoulli(p=0.30) -- ~30% positive proposals per image,
                a realistic foreground/background ratio.
    """
    rng = np.random.RandomState(seed)
    raw_scores = rng.beta(2, 5, (num_images, num_proposals))
    gt_labels = (rng.rand(num_images, num_proposals) < 0.30).astype(float)
    return raw_scores, gt_labels


def make_proposal_fitness_fn(raw_scores: np.ndarray, gt_labels: np.ndarray):
    """Return a closure F(w) implementing manuscript Eq. 3:

        F(w) = (1/N) * sum_n ( w_n * s_n - g_n )^2
    """
    def fitness(w: np.ndarray) -> float:
        weighted = raw_scores * np.clip(w, 0.0, 1.0)[None, :]
        return float(np.mean((weighted - gt_labels) ** 2))
    return fitness


# ============================================================================
# EPO target: MobDEAP hyperparameter search (surrogate validation loss)
# ============================================================================
# The true (synthetic) optimum used to construct the surrogate landscape.
# These values match the optimum reported in the manuscript text:
#   lr = 0.005, dropout = 0.30, da_eva_lambda = 1.20
TRUE_OPTIMUM = {"lr": 0.005, "dropout": 0.30, "da_eva_lambda": 1.20}

EPO_BOUNDS = {
    "lr": (1e-5, 1e-2),
    "dropout": (0.10, 0.50),
    "da_eva_lambda": (0.50, 2.00),
}


def hyperparameter_val_loss(params: Dict[str, float]) -> float:
    """Quadratic-bowl surrogate of MobDEAP validation loss around the
    true optimum. The lr term is scaled by 1e6 because lr operates on
    a much smaller numeric range than dropout / lambda; this keeps all
    three terms contributing comparably to the total loss.
    """
    lr = params["lr"]
    dropout = params["dropout"]
    lam = params["da_eva_lambda"]
    return (
        (lr - TRUE_OPTIMUM["lr"]) ** 2 * 1e6
        + (dropout - TRUE_OPTIMUM["dropout"]) ** 2 * 4
        + (lam - TRUE_OPTIMUM["da_eva_lambda"]) ** 2 * 0.5
    )
