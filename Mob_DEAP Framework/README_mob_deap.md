# Mob-DEAP: Enhanced Vehicle and Pedestrian Tracking Pipeline

**Manuscript:** MEAS-D-25-13753 — *Measurement* (Elsevier)  
**Title:** Enhanced Vehicle and Pedestrian Tracking with Detection using Dynamic Sparse R-CNN and Bowerbird Optimization  
**Authors:** Selvan C · Veningston K · Seifedine Kadry

---

## Overview

This repository contains the complete reproduction pipeline for the above manuscript. It implements all four stages of the Mob-DEAP framework:

1. **HFCBF** — Hybrid Fast Conventional Bilateral Filter preprocessing
2. **DSR-CNN-BO** — Dynamic Sparse R-CNN with Bowerbird Optimization for object detection
3. **Mob-DEAP** — MobileNetV2 with Directional Adaptive Emperor Variance Attention (DA-EVA) for classification
4. **SORT Tracker** — Kalman-filter multi-object tracking for identity assignment

The pipeline evaluates on five benchmark datasets: **MOT16**, **MOT17**, **MOT20**, **UA-DETRAC**, and **KITTI Tracking**, and reports MOT-standard metrics (MOTA, IDF1, HOTA) with full condition-wise and ablation analysis.

---

## Hardware Requirements

| Component | Specification |
|---|---|
| CPU | Intel Core i7-14700K (or equivalent) |
| GPU | NVIDIA GeForce RTX 4060 (8 GB VRAM) |
| RAM | 16 GB minimum, 32 GB recommended |
| Disk | 50 GB free (datasets + checkpoints) |
| CUDA | 11.8 minimum, 12.1 recommended |
| OS | Windows 10/11 or Ubuntu 20.04+ |

> The batch sizes in `config.py` are already tuned for the RTX 4060 (8 GB VRAM):
> `DSRCNNConfig.batch_size = 1` and `MobDEAPConfig.batch_size = 16`.

---

## Installation

### Step 1 — Install PyTorch with CUDA 12.1

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> Replace `cu121` with your CUDA version (`cu118`, `cu124`, etc.)  
> For CPU-only: `pip install torch torchvision`  
> Verify GPU: `python -c "import torch; print(torch.cuda.get_device_name(0))"`

### Step 2 — Install all remaining dependencies

```powershell
pip install -r requirements.txt
```

> If `lap` fails to compile on Windows, use the pre-built alternative:
> `pip install lapjv`

### Step 3 — Verify installation (no datasets needed)

```powershell
python main.py --mode smoke
```

Expected output in ~30 seconds:

```
HFCBF        OK  — X.XX ms/frame
MobDEAP      OK  — 2.41M parameters
BO           OK  — final MSE X.XXXXX ± X.XXXXX
EPO          OK  — best params: {lr: ..., dropout: ..., da_eva_lam: ...}
SORTTracker  OK  — 5 active tracks
...
ALL TESTS PASSED
```

---

## Project Structure

```
mob_deap_pipeline/
│
├── main.py                    Entry point — all run modes via CLI
├── config.py                  All hyperparameters as typed dataclasses
├── requirements.txt           Python dependencies
│
├── models/
│   ├── hfcbf.py               Stage 1 — HFCBF preprocessing (§3.2)
│   ├── attention.py           DA-EVA attention module, Eq. 8–9 (§3.4.2)
│   └── mob_deap.py            MobDEAP classifier + DSR-CNN wrapper (§3.3–3.4)
│
├── optimizers/
│   └── bowerbird.py           Bowerbird Optimizer (§3.3) + Adaptive EPO (§3.4.3)
│
├── tracking/
│   └── sort_tracker.py        Kalman-filter SORT tracker — Stage 4
│
├── data/
│   ├── mot_dataset.py         MOT16 / MOT17 / MOT20 sequence loaders
│   └── ua_detrac.py           UA-DETRAC (XML parser) + KITTI Tracking loaders
│
├── evaluation/
│   ├── mot_metrics.py         MOTA, IDF1, HOTA computation via motmetrics
│   └── ablation.py            C0–C5 independent ablation + Wilcoxon tests
│
├── training/
│   └── trainer.py             Focal Loss, GIoU Loss, MobDEAP training loop
│
└── utils/
    ├── seed.py                Global reproducibility seeding
    └── profiler.py            End-to-end latency profiler (per-stage + total)
```

Total: **15 Python files, ~2,600 lines of code**

---

## Dataset Setup

Download and extract datasets into a root folder (default: `./data/`):

| Dataset | URL | Purpose |
|---|---|---|
| MOT16 | https://motchallenge.net/data/MOT16.zip | Pedestrian tracking |
| MOT17 | https://motchallenge.net/data/MOT17.zip | Pedestrian tracking |
| MOT20 | https://motchallenge.net/data/MOT20.zip | Crowded scenes |
| UA-DETRAC | https://detrac-db.rit.albany.edu | Vehicle tracking (register first) |
| KITTI Tracking | https://www.cvlibs.net/datasets/kitti/eval_tracking.php | Multi-class vehicle tracking |

### Expected directory layout

```
data/
├── MOT16/
│   ├── train/
│   │   ├── MOT16-02/
│   │   │   ├── det/det.txt
│   │   │   ├── gt/gt.txt
│   │   │   └── img1/000001.jpg ...
│   │   └── MOT16-04/ ...
│   └── test/
│       └── MOT16-01/ ...
│
├── MOT17/
│   └── train/ ... test/ ...
│
├── MOT20/
│   └── train/ ... test/ ...
│
├── UA-DETRAC/
│   ├── DETRAC-train-data/
│   │   └── MVI_20011/img/*.jpg
│   ├── DETRAC-train-annotations-XML/
│   │   └── MVI_20011.xml
│   └── DETRAC-test-data/ ...
│
└── KITTI-tracking/
    ├── training/
    │   ├── image_02/0000/000000.png ...
    │   └── label_02/0000.txt
    └── testing/
        └── image_02/ ...
```

---

## Running the Pipeline

### Smoke test (no datasets required)

```powershell
python main.py --mode smoke
```

Verifies all components (HFCBF, MobDEAP, BO, EPO, tracker, ablation, profiler) using mock data.

---

### Full evaluation on a single dataset

```powershell
# MOT17 (most common benchmark)
python main.py --mode eval --dataset MOT17 --split train --device cuda --output ./results

# UA-DETRAC (vehicle benchmark)
python main.py --mode eval --dataset UA-DETRAC --split train --device cuda --output ./results

# KITTI Tracking
python main.py --mode eval --dataset KITTI --split training --device cuda --output ./results
```

---

### Independent ablation study (C0–C5)

```powershell
python main.py --mode ablation --dataset MOT17 --num_seeds 5 --output ./results
```

Runs six configurations (C0 = baseline, C5 = full model) across 5 seeds each, and reports `mean ± std` with Wilcoxon Signed-Rank p-values (C5 vs each).

| Config | HFCBF | BO | EPO + DA-EVA | Purpose |
|---|---|---|---|---|
| C0 | ✗ | ✗ | ✗ | Baseline — no novel components |
| C1 | ✓ | ✗ | ✗ | Preprocessing contribution only |
| C2 | ✗ | ✓ | ✗ | Detection optimizer only |
| C3 | ✗ | ✗ | ✓ | Classifier optimizer + attention only |
| C4 | ✓ | ✓ | ✗ | HFCBF + BO interaction |
| C5 | ✓ | ✓ | ✓ | Full model (reference) |

---

### All datasets sequentially

```powershell
python main.py --mode all --dataset all --device cuda --output ./results
```

---

### With a saved checkpoint and BO weights

```powershell
python main.py --mode eval --dataset MOT17 --split train \
    --checkpoint ./checkpoints/mobdeap_mot17.pth \
    --bo_weights  ./checkpoints/bo_weights.npy \
    --device cuda --output ./results
```

---

## Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | `smoke` | `smoke` / `eval` / `ablation` / `train` / `all` |
| `--dataset` | `MOT17` | `MOT16` / `MOT17` / `MOT20` / `UA-DETRAC` / `KITTI` / `all` |
| `--split` | `train` | `train` or `test` (MOT); `training` or `testing` (KITTI) |
| `--data_root` | `./data` | Root directory containing dataset sub-folders |
| `--device` | `cuda` | `cuda` / `cpu` / `cuda:0` |
| `--num_seeds` | `5` | Seeds for ablation study |
| `--output` | `./results` | Output directory for CSV files and JSON |
| `--checkpoint` | `None` | Path to saved MobDEAP `.pth` checkpoint |
| `--bo_weights` | `None` | Path to saved BO weight vector (`.npy`) |

---

## Hyperparameters

All hyperparameters live in `config.py` as typed dataclasses. Key values from the manuscript (§3, §5):

### HFCBF (§3.2)
| Parameter | Value | Description |
|---|---|---|
| `sigma_spatial` | 1.2 | Gaussian spatial σ |
| `sigma_range` | 30.0 | Intensity-domain σ |
| `kernel_size` | 5 | 5×5 kernel |

### Bowerbird Optimizer (§3.3)
| Parameter | Value | Description |
|---|---|---|
| `population_size` | 30 | Number of candidate solutions |
| `max_iterations` | 100 | Stopping criterion |
| `alpha` | 0.5 | Attraction weight (exploitation) |
| `beta` | 0.5 | Decoration weight (exploration) |
| `mutation_prob` | 0.10 | Gaussian mutation probability |
| `chaotic_init` | True | Logistic chaotic map initialisation |
| `chaotic_r` | 3.9 | Logistic map r ∈ (3.57, 4) |
| `num_seeds` | 5 | Seeds for convergence analysis |

### DA-EVA Attention (§3.4.2)
| Parameter | Value | Description |
|---|---|---|
| `da_eva_heads` | 8 | Number of directional kernels K_θ |
| `da_eva_lambda` | 1.0 | Softmax temperature λ (Eq. 9) |
| Inserted after | Block 13 | MobileNetV2 features[:14] → 96 channels |

### Adaptive EPO (§3.4.3)
| Parameter | Value | Description |
|---|---|---|
| `population_size` | 30 | Number of emperor penguins |
| `max_iterations` | 100 | Outer loop bound |
| `temperature_init` | 1.0 | Initial social-force temperature |
| `temperature_decay` | 0.95 | Geometric decay per iteration |
| `social_force_f` | 0.5 | Exploration factor |
| `social_force_l` | 1.5 | Exploitation factor |
| `fuzzy_adapt` | True | Linear fuzzy adaptation enabled |
| Search: `lr` | [1e-5, 1e-2] | MobDEAP learning rate range |
| Search: `dropout` | [0.10, 0.50] | Dropout range |
| Search: `da_eva_lam` | [0.50, 2.00] | λ range |

### Training (§5)
| Parameter | Value | Description |
|---|---|---|
| Optimizer | AdamW | Both stages |
| Learning rate | 0.001 | Step decay at epochs 60, 80 |
| Batch size (DSR-CNN) | 1 | RTX 4060 safe |
| Batch size (MobDEAP) | 16 | RTX 4060 safe |
| Epochs | 100 | With early stopping (patience = 10) |
| Loss — classification | Focal (α=0.25, γ=2.0) | |
| Loss — bounding box | GIoU | |
| Global seed | 42 | |

---

## Output Files

After a run, results are saved in `--output`:

```
results/
├── hyperparameters.json          All BO/EPO/training hyperparameters
├── ablation/
│   └── MOT17/
│       └── ablation_independent.csv   C0–C5 table with Wilcoxon p-values
├── MOT17/
│   ├── mot_results.csv               MOTA, IDF1, num_switches, FPS per sequence
│   └── condition_wise.csv            MOTA/IDF1 by day/night/crowded condition
├── UA-DETRAC/
│   └── mot_results.csv
└── KITTI/
    └── mot_results.csv
```

---

## Metrics

| Metric | Formula | Better |
|---|---|---|
| MOTA | `1 − (FP + FN + IDs) / GT` | Higher ↑ |
| IDF1 | `2·IDTP / (2·IDTP + IDFP + IDFN)` | Higher ↑ |
| HOTA | `√(DetA × AssA)` | Higher ↑ |
| Num-IDs | Total identity switches | Lower ↓ |
| FPS (E2E) | `1000 / total_pipeline_ms` | Higher ↑ |

> **Important:** FPS is always measured end-to-end (HFCBF + detection + classification + tracking). Module-level backbone FPS is not reported.

---

## The One Function to Implement

To get real results, open `main.py` and replace `_train_eval_stub` in `run_ablation_mode` with a function that trains MobDEAP on your data and returns actual metrics:

```python
def _train_eval_stub(flags: dict, seed: int) -> dict:
    # 1. Build MobDEAP with flags (hfcbf, bo, epo, da_eva booleans)
    # 2. Train for cfg.epochs on your DataLoader
    # 3. Run MOTEvaluator on the validation sequences
    # 4. Return:
    return {
        "mota"   : float,   # e.g. 66.2
        "idf1"   : float,   # e.g. 74.8
        "hota"   : float,   # e.g. 61.4
        "num_ids": int,     # e.g. 1847
    }
```

The six ablation configs, five-seed loop, Wilcoxon tests, and CSV output all run automatically around this function.

---

## Common Errors and Fixes

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'motmetrics'` | `pip install motmetrics` |
| `ModuleNotFoundError: No module named 'lap'` | `pip install lapjv` |
| `CUDA out of memory` | Reduce `batch_size` in `config.py`; add `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256` |
| `HTTPError 403` when loading MobileNetV2 weights | Download manually: `python -c "import torchvision.models as m; m.mobilenet_v2(weights=m.MobileNet_V2_Weights.IMAGENET1K_V1)"` then re-run |
| `FileNotFoundError: data/MOT17` | Pass `--data_root /full/path/to/your/data` |
| Wilcoxon p = 0.25 (not < 0.05) | Increase `--num_seeds 10`; with 5 seeds, minimum p is 0.0625 |
| PowerShell: `PYTORCH_CUDA_ALLOC_CONF` not recognized | Use: `$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:256"` |

---

## Citation

If you use this code, please cite:

```
Selvan C, Veningston K, Kadry S (2025).
Enhanced Vehicle and Pedestrian Tracking with Detection using
Dynamic Sparse R-CNN and Bowerbird Optimization.
Measurement, MEAS-D-25-13753 (under revision).
```

---

## References

- Samareh Moosavi & Khatibi Bardsiri (2017). Satin Bowerbird Optimizer.
- Wangkhamhan (2021). Chaotic logistic map initialisation.
- Hong et al. (2022). Dynamic Sparse R-CNN.
- Luiten et al. (2021). HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking.
- Bewley et al. (2016). Simple Online and Realtime Tracking (SORT).
- Wen et al. (2020). UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking.
- Geiger et al. (2012). Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite.
- Milan et al. (2016). MOT16: A Benchmark for Multi-Object Tracking.
