"""
evaluation/mot_metrics.py  —  MOT-standard metric computation.

Primary metrics (replacing bare accuracy %):
    MOTA   = 1 − (FP + FN + IDs) / GT
    IDF1   = 2·IDTP / (2·IDTP + IDFP + IDFN)
    HOTA   = √(DetA × AssA)   (Luiten et al., 2021)
    Num-IDs = total identity switches (lower ↓ is better)
    FPS    = end-to-end frames per second (NOT module-level)

Uses motmetrics for MOTA and IDF1 computation.
HOTA is computed independently per Luiten et al. (2021).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torchvision.ops as ops
import pandas as pd
from typing import Dict, List, Optional
from collections import defaultdict

try:
    import motmetrics as mm
    MM_AVAILABLE = True
except ImportError:
    MM_AVAILABLE = False
    print("[WARNING] motmetrics not installed. "
          "MOTA/IDF1 computation unavailable. "
          "Install with: pip install motmetrics")


# ─────────────────────────────────────────────────────────────────────────────
# MOT Evaluator
# ─────────────────────────────────────────────────────────────────────────────
class MOTEvaluator:
    """
    Accumulates per-frame tracking results and computes MOT metrics.

    Usage:
        evaluator = MOTEvaluator()
        for frame_id, gt, pred in ...:
            evaluator.update(seq_name, frame_id,
                             gt_ids, gt_boxes, pred_ids, pred_boxes)
        results = evaluator.compute()
    """

    def __init__(self, iou_threshold: float = 0.5):
        if not MM_AVAILABLE:
            raise ImportError("Install motmetrics: pip install motmetrics")
        self.iou_threshold = iou_threshold
        self._accs: Dict[str, mm.MOTAccumulator] = {}
        self._fps : Dict[str, List[float]] = defaultdict(list)

    def _dist_matrix(self,
                     boxes_gt  : np.ndarray,
                     boxes_pred: np.ndarray) -> np.ndarray:
        """IoU distance matrix (1 - IoU)."""
        if len(boxes_gt) == 0 or len(boxes_pred) == 0:
            return np.empty((len(boxes_gt), len(boxes_pred)))
        iou = ops.box_iou(
            torch.as_tensor(boxes_gt,   dtype=torch.float32),
            torch.as_tensor(boxes_pred, dtype=torch.float32),
        ).numpy()
        return 1.0 - iou

    def update(self,
               seq_name  : str,
               frame_id  : int,
               gt_ids    : np.ndarray,
               gt_boxes  : np.ndarray,
               pred_ids  : np.ndarray,
               pred_boxes: np.ndarray,
               frame_ms  : Optional[float] = None) -> None:
        """
        Accumulate one frame.

        Args:
            seq_name  : sequence identifier
            frame_id  : 1-indexed frame number
            gt_ids    : (N,) ground-truth track IDs
            gt_boxes  : (N,4) [x1,y1,x2,y2]
            pred_ids  : (M,) predicted track IDs
            pred_boxes: (M,4) [x1,y1,x2,y2]
            frame_ms  : optional end-to-end latency for this frame (ms)
        """
        if seq_name not in self._accs:
            self._accs[seq_name] = mm.MOTAccumulator(auto_id=True)
        dist = self._dist_matrix(gt_boxes, pred_boxes)
        self._accs[seq_name].update(
            gt_ids.tolist()   if len(gt_ids)   else [],
            pred_ids.tolist() if len(pred_ids) else [],
            dist,
        )
        if frame_ms is not None:
            self._fps[seq_name].append(frame_ms)

    def compute(self) -> pd.DataFrame:
        """
        Compute MOTA, IDF1, and auxiliary metrics for all sequences.
        Returns a DataFrame with one row per sequence + 'OVERALL' row.
        """
        mh = mm.metrics.create()
        metrics = [
            "num_frames", "mota", "idf1", "motp",
            "num_switches", "num_false_positives",
            "num_misses", "mostly_tracked", "mostly_lost",
            "precision", "recall",
        ]
        summary = mh.compute_many(
            list(self._accs.values()),
            metrics=metrics,
            names=list(self._accs.keys()),
            generate_overall=True,
        )
        # Add end-to-end FPS column
        fps_col = {}
        for seq in list(self._accs.keys()) + ["OVERALL"]:
            if seq == "OVERALL":
                all_ms = [m for ms in self._fps.values() for m in ms]
                fps_col[seq] = round(1000.0 / np.mean(all_ms), 2) \
                    if all_ms else float("nan")
            else:
                ms = self._fps.get(seq, [])
                fps_col[seq] = round(1000.0 / np.mean(ms), 2) \
                    if ms else float("nan")
        summary["fps_e2e"] = pd.Series(fps_col)
        return summary

    def reset(self) -> None:
        self._accs.clear()
        self._fps.clear()


# ─────────────────────────────────────────────────────────────────────────────
# HOTA computation  (Luiten et al., 2021)
# ─────────────────────────────────────────────────────────────────────────────
def compute_hota(gt_data  : Dict[str, Dict[int, np.ndarray]],
                 pred_data: Dict[str, Dict[int, np.ndarray]],
                 alpha_range: np.ndarray = np.arange(0.05, 0.96, 0.05)
                 ) -> Dict[str, float]:
    """
    Compute HOTA, DetA, AssA averaged over IoU threshold α.

    Args:
        gt_data   : {seq: {frame_id: array[[x1,y1,x2,y2,track_id]]}}
        pred_data : {seq: {frame_id: array[[x1,y1,x2,y2,track_id]]}}
        alpha_range: IoU thresholds to average over

    Returns:
        {"HOTA": float, "DetA": float, "AssA": float}
    """
    det_a_list, ass_a_list = [], []

    for alpha in alpha_range:
        tp_det = fp_det = fn_det = 0
        tp_ass = fp_ass = fn_ass = 0

        for seq, frames_gt in gt_data.items():
            frames_pr = pred_data.get(seq, {})
            for fid, gt_boxes_full in frames_gt.items():
                gt_boxes = gt_boxes_full[:, :4]
                gt_ids   = gt_boxes_full[:, 4].astype(int)
                pr_full  = frames_pr.get(fid, np.empty((0, 5)))
                pr_boxes = pr_full[:, :4]
                pr_ids   = pr_full[:, 4].astype(int)

                if len(gt_boxes) == 0:
                    fp_det += len(pr_boxes)
                    continue
                if len(pr_boxes) == 0:
                    fn_det += len(gt_boxes)
                    continue

                iou = ops.box_iou(
                    torch.as_tensor(gt_boxes, dtype=torch.float32),
                    torch.as_tensor(pr_boxes, dtype=torch.float32),
                ).numpy()

                matched_gt  = set()
                matched_pr  = set()
                for gi in range(len(gt_boxes)):
                    best_pr = int(np.argmax(iou[gi]))
                    if iou[gi, best_pr] >= alpha and best_pr not in matched_pr:
                        matched_gt.add(gi)
                        matched_pr.add(best_pr)
                        tp_det += 1
                        # Association: same identity?
                        if gt_ids[gi] == pr_ids[best_pr]:
                            tp_ass += 1
                        else:
                            fp_ass += 1
                            fn_ass += 1
                fp_det += len(pr_boxes) - len(matched_pr)
                fn_det += len(gt_boxes) - len(matched_gt)

        deta = tp_det / max(tp_det + fp_det + fn_det, 1)
        assa = tp_ass / max(tp_ass + fp_ass + fn_ass, 1)
        det_a_list.append(deta)
        ass_a_list.append(assa)

    DetA = float(np.mean(det_a_list))
    AssA = float(np.mean(ass_a_list))
    HOTA = float(np.sqrt(DetA * AssA))
    return {"HOTA": HOTA, "DetA": DetA, "AssA": AssA}
