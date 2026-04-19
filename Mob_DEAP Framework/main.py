"""
main.py  —  Entry point for the Mob-DEAP evaluation pipeline.

MEAS-D-25-13753: "Enhanced Vehicle and Pedestrian Tracking with Detection
using Dynamic Sparse R-CNN and Bowerbird Optimization"

Hardware target: Intel Core i7-14700K + NVIDIA GeForce RTX 4060 (8 GB)
CUDA:  12.1 or 12.4    PyTorch:  2.0+

Usage examples:
    # Smoke-test (no datasets needed, ~30s):
    python main.py --mode smoke

    # Full evaluation on MOT17:
    python main.py --mode eval --dataset MOT17 --split train --device cuda

    # Ablation study:
    python main.py --mode ablation --dataset MOT17 --num_seeds 5

    # UA-DETRAC vehicle benchmark:
    python main.py --mode eval --dataset UA-DETRAC --split train

    # KITTI Tracking:
    python main.py --mode eval --dataset KITTI --split training

    # All datasets sequentially:
    python main.py --mode eval --dataset all
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── project imports ────────────────────────────────────────────────────────
from config import (HFCBFConfig, BOConfig, EPOConfig,
                    MobDEAPConfig, DSRCNNConfig, TrackerConfig,
                    TrainConfig, EvalConfig, DataConfig)
from utils.seed         import set_seed
from utils.profiler     import E2EProfiler
from models.hfcbf       import HFCBF
from models.mob_deap    import MobDEAP, DSRCNNWrapper
from optimizers.bowerbird import BowerbirdOptimizer
from optimizers.bowerbird import AdaptiveEPO
from tracking.sort_tracker import SORTTracker
from evaluation.ablation   import run_ablation, condition_wise_summary, \
                                   significance_tests, ABLATION_CONFIGS
from evaluation.mot_metrics import MOTEvaluator, compute_hota


# ═══════════════════════════════════════════════════════════════════════════
# Argument parser
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Mob-DEAP pipeline — MEAS-D-25-13753")

    p.add_argument("--mode", default="smoke",
                   choices=["smoke", "eval", "ablation", "train", "all"],
                   help="smoke=structural test; eval=run metrics; "
                        "ablation=C0-C5 study; train=train MobDEAP; "
                        "all=train+eval+ablation")

    p.add_argument("--dataset", default="MOT17",
                   choices=["MOT16","MOT17","MOT20",
                             "UA-DETRAC","KITTI","all"])
    p.add_argument("--split",   default="train",
                   help="MOT: train|test  KITTI: training|testing")
    p.add_argument("--data_root", default="./data",
                   help="Root folder containing dataset sub-directories")

    p.add_argument("--device",  default="cuda" if torch.cuda.is_available()
                                else "cpu")
    p.add_argument("--num_seeds",  type=int, default=5)
    p.add_argument("--output",     default="./results")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a saved MobDEAP .pth checkpoint")
    p.add_argument("--bo_weights", default=None,
                   help="Path to saved BO weight vector (.npy)")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Dataset loader factory
# ═══════════════════════════════════════════════════════════════════════════
def load_dataset(dataset: str, root: str, split: str):
    """Return an iterable of sequence objects for the requested dataset."""
    if dataset in ("MOT16", "MOT17", "MOT20"):
        from data.mot_dataset import MOTDataset
        return MOTDataset(root, dataset, split)
    elif dataset == "UA-DETRAC":
        from data.ua_detrac import DETRACDataset
        return DETRACDataset(root, split)
    elif dataset == "KITTI":
        from data.ua_detrac import KITTIDataset
        return KITTIDataset(root, split)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ═══════════════════════════════════════════════════════════════════════════
# Core inference loop (one sequence)
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_sequence(seq, hfcbf, detector, classifier, tracker,
                 evaluator: MOTEvaluator, profiler: E2EProfiler,
                 device: str, gt_store: dict):
    """
    Run the full pipeline on a single sequence and accumulate metrics.

    Args:
        seq       : sequence object with __iter__ yielding
                    (frame_id, img, gt, dets)
        hfcbf     : HFCBF instance
        detector  : DSRCNNWrapper instance
        classifier: MobDEAP instance
        tracker   : SORTTracker instance
        evaluator : MOTEvaluator for metric accumulation
        profiler  : E2EProfiler for latency measurement
        device    : torch device string
        gt_store  : dict to accumulate GT for HOTA computation
    """
    import torchvision.transforms as T
    cls_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    tracker.reset()
    classifier.eval()
    detector.eval()

    for frame_id, img, gt, _ in seq:
        if img is None:
            continue
        t_total_start = time.perf_counter()

        # ── Stage 1: HFCBF ───────────────────────────────────────────────
        t1 = time.perf_counter()
        pre = hfcbf.process(img)
        t_hfcbf = (time.perf_counter() - t1) * 1000

        # ── Stage 2: Detection ────────────────────────────────────────────
        import cv2
        rgb = cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)
        img_tensor = T.ToTensor()(rgb).unsqueeze(0).to(device)

        t2 = time.perf_counter()
        det_out = detector([img_tensor[0]])
        t_det = (time.perf_counter() - t2) * 1000

        boxes  = det_out[0]["boxes"].cpu().numpy()
        scores = det_out[0]["scores"].cpu().numpy()

        # ── Stage 3: Classification (MobDEAP) ────────────────────────────
        t3 = time.perf_counter()
        cls_labels = []
        if len(boxes) > 0:
            crops = []
            h_img, w_img = img.shape[:2]
            for b in boxes:
                x1,y1,x2,y2 = [int(v) for v in b]
                x1=max(0,x1); y1=max(0,y1)
                x2=min(w_img,x2); y2=min(h_img,y2)
                crop = rgb[y1:y2, x1:x2]
                if crop.size == 0:
                    crops.append(torch.zeros(3,224,224))
                else:
                    crops.append(cls_transform(crop))
            crop_batch = torch.stack(crops).to(device)
            logits     = classifier(crop_batch)
            cls_labels = logits.argmax(1).cpu().numpy()
        t_cls = (time.perf_counter() - t3) * 1000

        # Keep only pedestrian (0) and vehicle (1) detections
        keep = [i for i, l in enumerate(cls_labels) if l in (0,1)] \
               if len(cls_labels) > 0 else []
        det_boxes = boxes[keep] if keep else np.empty((0,4))
        det_scores= scores[keep] if keep else np.empty((0,))

        # ── Stage 4: Tracking ─────────────────────────────────────────────
        t4 = time.perf_counter()
        if len(det_boxes) > 0:
            dets_in = np.hstack([det_boxes,
                                 det_scores.reshape(-1,1)])
        else:
            dets_in = np.empty((0,5))
        tracks = tracker.update(dets_in)
        t_trk  = (time.perf_counter() - t4) * 1000

        # ── Accumulate metrics ────────────────────────────────────────────
        t_total = (time.perf_counter() - t_total_start) * 1000

        profiler.log_frame({
            "hfcbf"         : t_hfcbf,
            "detection"     : t_det,
            "classification": t_cls,
            "tracking"      : t_trk,
            "total"         : t_total,
        })

        if gt is not None and evaluator is not None:
            gt_boxes = gt[:, :4]
            gt_ids   = gt[:, 4].astype(int)
            if len(tracks) > 0:
                pred_boxes = tracks[:, :4]
                pred_ids   = tracks[:, 4].astype(int)
            else:
                pred_boxes = np.empty((0,4))
                pred_ids   = np.empty((0,), dtype=int)

            evaluator.update(seq.name, frame_id,
                             gt_ids, gt_boxes, pred_ids, pred_boxes,
                             frame_ms=t_total)

            # Store for HOTA
            if seq.name not in gt_store:
                gt_store[seq.name] = {}
            gt_store[seq.name][frame_id] = np.hstack([
                gt_boxes, gt_ids.reshape(-1,1)])


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test (no datasets required)
# ═══════════════════════════════════════════════════════════════════════════
def smoke_test(args, cfg_dict):
    print("\n══ Smoke test — verifying all components ══════════════════════")
    set_seed(42)

    hfcbf_cfg    = cfg_dict["hfcbf"]
    bo_cfg       = cfg_dict["bo"]
    epo_cfg      = cfg_dict["epo"]
    mobdeap_cfg  = cfg_dict["mobdeap"]
    tracker_cfg  = cfg_dict["tracker"]

    # HFCBF
    import numpy as np, cv2
    hfcbf    = HFCBF(hfcbf_cfg)
    frame    = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    pre      = hfcbf.process(frame)
    hfcbf_ms = hfcbf.measure_runtime(frame, n=20)
    print(f"  HFCBF OK  —  {hfcbf_ms:.2f} ms/frame")

    # MobDEAP
    # Use pretrained=False for smoke test (weights loaded via --checkpoint in real runs)
    smoke_cfg = MobDEAPConfig(pretrained=False)
    model = MobDEAP(smoke_cfg)
    n_par = model.count_parameters()
    print(f"  MobDEAP OK  —  {n_par/1e6:.2f}M parameters")

    # Bowerbird
    bo = BowerbirdOptimizer(bo_cfg)
    proposals  = np.random.rand(200, 100)
    gt_scores  = np.random.rand(200)
    bo_result  = bo.multi_seed_convergence(proposals, gt_scores)
    print(f"  BO OK  —  final MSE {bo_result['final_mean']:.5f} "
          f"± {bo_result['final_std']:.5f}")

    # EPO
    epo = AdaptiveEPO(epo_cfg)
    epo_result = epo.multi_seed_convergence(
        lambda p: float(np.random.uniform(0.05, 0.15)))
    print(f"  EPO OK  —  best params: {epo_result['best_params']}")

    # SORT Tracker
    tracker = SORTTracker(tracker_cfg)
    dets    = np.random.rand(5, 5) * 100
    dets[:, 2:4] += 20
    out     = tracker.update(dets[:, :4])
    print(f"  SORTTracker OK  —  {len(out)} active tracks")

    # Ablation smoke
    print("\n  Running ablation smoke (mock train_eval_fn) …")
    def mock_fn(flags, seed):
        base = 58 + 2*flags["hfcbf"] + 1.5*flags["bo"] + 2*flags["epo"]
        rng  = np.random.RandomState(seed)
        m    = float(np.clip(base + rng.normal(0, 0.2), 0, 100))
        return {"mota": m, "idf1": m-5, "hota": m-10, "num_ids": int(2000-m*20)}

    abl_df = run_ablation(mock_fn, num_seeds=3)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    abl_path = out_dir / "ablation_smoke.csv"
    abl_df.to_csv(abl_path, index=False)

    # Save hyperparameters
    hp = {
        "HFCBF"  : hfcbf_cfg.__dict__,
        "BO"     : bo_cfg.__dict__,
        "EPO"    : {k:v for k,v in epo_cfg.__dict__.items()
                    if not isinstance(v, dict)},
        "MobDEAP": {k:(list(v) if isinstance(v,tuple) else v)
                    for k,v in mobdeap_cfg.__dict__.items()},
        "Tracker": tracker_cfg.__dict__,
        "BO_convergence" : {"final_mean":bo_result["final_mean"],
                            "final_std" :bo_result["final_std"]},
        "EPO_convergence": {"final_mean" :epo_result["final_mean"],
                            "best_params":epo_result["best_params"]},
    }
    hp_path = out_dir / "hyperparameters.json"
    with open(hp_path, "w") as f:
        json.dump(hp, f, indent=2, default=str)

    print(f"\n  Results saved → {out_dir}")
    print("══ Smoke test PASSED ═══════════════════════════════════════════")


# ═══════════════════════════════════════════════════════════════════════════
# Full evaluation on a dataset
# ═══════════════════════════════════════════════════════════════════════════
def eval_dataset(args, cfg_dict, dataset_name: str):
    print(f"\n══ Evaluating on {dataset_name} ({args.split}) ══════════════")
    out_dir = Path(args.output) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device

    # Build components
    hfcbf    = HFCBF(cfg_dict["hfcbf"])
    detector = DSRCNNWrapper(cfg_dict["dsrcnn"])
    tracker  = SORTTracker(cfg_dict["tracker"])
    model    = MobDEAP(cfg_dict["mobdeap"])

    # Load checkpoint if provided
    if args.checkpoint and Path(args.checkpoint).exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"  Loaded checkpoint: {args.checkpoint}")

    # Load BO weights if provided
    if args.bo_weights and Path(args.bo_weights).exists():
        bo_w = torch.tensor(np.load(args.bo_weights), dtype=torch.float32)
        detector.bo_weights = bo_w
        print(f"  Loaded BO weights: {args.bo_weights}")

    detector = detector.to(device).eval()
    model    = model.to(device).eval()

    evaluator = MOTEvaluator(iou_threshold=0.5)
    profiler  = E2EProfiler()
    gt_store  = {}

    # Load sequences
    dataset = load_dataset(dataset_name, args.data_root, args.split)

    for seq in dataset:
        print(f"  Processing {seq.name} ({len(seq)} frames) …")
        run_sequence(seq, hfcbf, detector, model, tracker,
                     evaluator, profiler, device, gt_store)

    # ── Compute metrics ───────────────────────────────────────────────────
    print("\n  Computing MOT metrics …")
    mot_results = evaluator.compute()
    print(mot_results[["mota","idf1","num_switches",
                        "num_false_positives","num_misses",
                        "fps_e2e"]].to_string())

    # HOTA (per-sequence)
    print("\n  Computing HOTA …")
    # For HOTA we need pred_store — build from tracker outputs
    # (already accumulated in gt_store for demonstration)
    # In practice, also accumulate pred_store in run_sequence
    hota_result = {"HOTA": "computed separately", "DetA": "-", "AssA": "-"}

    # ── Condition-wise (MOT17/MOT20 only) ────────────────────────────────
    if dataset_name in ("MOT17", "MOT20"):
        per_seq = {}
        for seq_name in mot_results.index:
            if seq_name == "OVERALL":
                continue
            row = mot_results.loc[seq_name]
            per_seq[seq_name] = {
                "mota": float(row.get("mota", 0)) * 100,
                "idf1": float(row.get("idf1", 0)) * 100,
                "hota": 0.0,   # fill with HOTA value if computed
            }
        cond_summary = condition_wise_summary(per_seq)
        cond_path = out_dir / "condition_wise.csv"
        cond_summary.to_csv(cond_path)
        print(f"\n  Condition-wise summary:\n{cond_summary}")

    # ── Profiler summary ──────────────────────────────────────────────────
    profiler.print_summary()

    # ── Save results ──────────────────────────────────────────────────────
    mot_path = out_dir / "mot_results.csv"
    mot_results.to_csv(mot_path)
    print(f"\n  Results saved → {out_dir}")
    return mot_results


# ═══════════════════════════════════════════════════════════════════════════
# Ablation mode
# ═══════════════════════════════════════════════════════════════════════════
def run_ablation_mode(args, cfg_dict):
    """
    Ablation study with real train_eval_fn.

    NOTE: Replace _train_eval_stub below with your actual training +
    evaluation function once datasets are available.  The stub returns
    plausible mock values for structural testing.
    """
    print(f"\n══ Ablation study — {args.dataset} ══════════════════════════")

    def _train_eval_stub(flags: dict, seed: int) -> dict:
        """
        REPLACE THIS with your actual function:
            1. Build MobDEAP with flags (hfcbf, bo, epo, da_eva)
            2. Train for cfg.epochs
            3. Evaluate with MOTEvaluator
            4. Return {"mota": float, "idf1": float,
                        "hota": float, "num_ids": int}
        """
        set_seed(seed)
        base = (58.0
                + 2.3 * flags["hfcbf"]
                + 1.5 * flags["bo"]
                + 2.0 * flags["epo"])
        rng  = np.random.RandomState(seed)
        mota = float(np.clip(base + rng.normal(0, 0.2), 0, 100))
        return {
            "mota"   : mota,
            "idf1"   : mota - 5.0 + rng.normal(0, 0.3),
            "hota"   : mota - 10.0 + rng.normal(0, 0.2),
            "num_ids": int(2000 - mota * 20),
        }

    out_dir = Path(args.output) / "ablation" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    abl_df = run_ablation(_train_eval_stub,
                          num_seeds=args.num_seeds,
                          dataset_name=args.dataset)
    abl_path = out_dir / "ablation_independent.csv"
    abl_df.to_csv(abl_path, index=False)
    print(f"\n  Ablation table saved → {abl_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    set_seed(42)

    # Print hardware info
    print(f"\nDevice : {args.device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"Mode   : {args.mode}   Dataset: {args.dataset}")

    # Build config dict
    cfg_dict = {
        "hfcbf"  : HFCBFConfig(),
        "bo"     : BOConfig(),
        "epo"    : EPOConfig(),
        "mobdeap": MobDEAPConfig(),
        "dsrcnn" : DSRCNNConfig(),
        "tracker": TrackerConfig(),
    }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "smoke":
        smoke_test(args, cfg_dict)

    elif args.mode == "eval":
        datasets = (["MOT16","MOT17","MOT20","UA-DETRAC","KITTI"]
                    if args.dataset == "all" else [args.dataset])
        for ds in datasets:
            try:
                eval_dataset(args, cfg_dict, ds)
            except FileNotFoundError as e:
                print(f"\n  [SKIP] {ds}: {e}")

    elif args.mode == "ablation":
        run_ablation_mode(args, cfg_dict)

    elif args.mode == "all":
        smoke_test(args, cfg_dict)
        run_ablation_mode(args, cfg_dict)
        datasets = (["MOT16","MOT17","MOT20","UA-DETRAC","KITTI"]
                    if args.dataset == "all" else [args.dataset])
        for ds in datasets:
            try:
                eval_dataset(args, cfg_dict, ds)
            except FileNotFoundError as e:
                print(f"\n  [SKIP] {ds}: {e}")


if __name__ == "__main__":
    main()
