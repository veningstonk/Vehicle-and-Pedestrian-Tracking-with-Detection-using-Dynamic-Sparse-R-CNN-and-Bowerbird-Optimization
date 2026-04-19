"""
config.py  —  All hyperparameter containers for the Mob-DEAP pipeline.
Every parameter is documented with its manuscript section reference
and the reviewer comment that motivated its explicit documentation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: HFCBF preprocessing  (§3.2)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class HFCBFConfig:
    sigma_spatial : float = 1.2    # Gaussian spatial σ  (ablation §4.7.1)
    sigma_range   : float = 30.0   # intensity-domain σ
    kernel_size   : int   = 5      # 5×5 kernel
    iterations    : int   = 1      # single pass


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2a: DSR-CNN detection backbone  (§3.3)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DSRCNNConfig:
    num_proposals       : int   = 100
    num_stages          : int   = 6
    num_classes         : int   = 2        # pedestrian + vehicle
    nms_threshold       : float = 0.5
    score_threshold     : float = 0.5
    # Training  (§5 — Reviewer #3 Comment 13a)
    batch_size          : int   = 1        # RTX 4060 safe (8 GB VRAM)
    epochs              : int   = 100
    lr                  : float = 0.001
    lr_decay_factor     : float = 0.0005
    lr_decay_epochs     : Tuple = (60, 80)
    optimizer           : str   = "AdamW"
    early_stop_patience : int   = 10
    img_short_side      : int   = 800
    img_long_side       : int   = 1333
    # Loss
    focal_alpha         : float = 0.25
    focal_gamma         : float = 2.0
    giou_weight         : float = 1.0
    # Augmentation
    augmentations       : List  = field(default_factory=lambda: [
        "random_flip_h", "colour_jitter"
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2b: Bowerbird Optimizer  (§3.3)
# All parameters explicit per Reviewer #3 Comment 6
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BOConfig:
    population_size : int   = 30
    max_iterations  : int   = 100
    alpha           : float = 0.5    # attraction (exploitation) weight
    beta            : float = 0.5    # decoration (exploration) weight
    mutation_prob   : float = 0.10
    chaotic_init    : bool  = True   # logistic chaotic map initialisation
    chaotic_r       : float = 3.9    # logistic map r ∈ (3.57, 4)
    num_seeds       : int   = 5      # multi-seed convergence


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3a/b: DA-EVA + MobDEAP  (§3.4)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MobDEAPConfig:
    pretrained          : bool  = True
    num_classes         : int   = 2
    da_eva_heads        : int   = 8      # directional kernels θ
    da_eva_lambda       : float = 1.0    # softmax temperature λ  (Eq. 9)
    variance_eps        : float = 1e-6
    fc_dims             : Tuple = (128, 64)
    dropout             : float = 0.3
    # Training
    batch_size          : int   = 16     # RTX 4060 safe
    epochs              : int   = 100
    lr                  : float = 0.001
    lr_decay_factor     : float = 0.0005
    lr_decay_epochs     : Tuple = (60, 80)
    optimizer           : str   = "AdamW"
    early_stop_patience : int   = 10
    img_size            : int   = 224
    augmentations       : List  = field(default_factory=lambda: [
        "random_flip_h", "colour_jitter", "random_erase"
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3c: Adaptive Emperor Penguin Optimizer  (§3.4.3)
# All parameters explicit per Reviewer #3 Comment 6
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EPOConfig:
    population_size  : int   = 30
    max_iterations   : int   = 100
    temperature_init : float = 1.0
    temperature_decay: float = 0.95
    social_force_f   : float = 0.5    # exploration (decreases via fuzzy)
    social_force_l   : float = 1.5    # exploitation (increases via fuzzy)
    fuzzy_adapt      : bool  = True
    early_stop_delta : float = 1e-5
    num_seeds        : int   = 5
    search_space     : Dict  = field(default_factory=lambda: {
        "lr"          : (1e-5, 1e-2),
        "dropout"     : (0.10, 0.50),
        "da_eva_lam"  : (0.50, 2.00),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: SORT Tracker  (post-processing)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrackerConfig:
    max_age       : int   = 70    # frames before track deletion
    min_hits      : int   = 3     # frames before track confirmed
    iou_threshold : float = 0.30  # IoU threshold for association


# ─────────────────────────────────────────────────────────────────────────────
# Training and experiment control
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    seed          : int  = 42
    num_workers   : int  = 4      # DataLoader workers
    pin_memory    : bool = True
    mixed_prec    : bool = False   # AMP (safe to enable on RTX 4060)


@dataclass
class EvalConfig:
    iou_threshold : float = 0.5   # for MOT metric computation
    num_seeds     : int   = 5     # ablation study seeds
    output_dir    : str   = "./results"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset paths
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    root          : str  = "./data"
    mot16_dir     : str  = "MOT16"
    mot17_dir     : str  = "MOT17"
    mot20_dir     : str  = "MOT20"
    ua_detrac_dir : str  = "UA-DETRAC"
    kitti_dir     : str  = "KITTI-tracking"
    split         : str  = "train"   # "train" or "test"
