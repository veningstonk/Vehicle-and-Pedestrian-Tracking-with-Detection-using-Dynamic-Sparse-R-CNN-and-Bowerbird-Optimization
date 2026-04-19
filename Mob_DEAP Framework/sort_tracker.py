"""
tracking/sort_tracker.py  —  SORT-based multi-object tracker  (Stage 4)

Implements:
  - Kalman filter for motion prediction (constant velocity model)
  - IoU-based Hungarian assignment via scipy.optimize.linear_sum_assignment
  - Track management: birth, confirmation, deletion

This provides the identity assignment stage that produces IDS, IDF1, and
HOTA metrics.  The tracker is dataset-agnostic.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from typing import List, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import TrackerConfig


# ─────────────────────────────────────────────────────────────────────────────
# Kalman-filter track
# ─────────────────────────────────────────────────────────────────────────────
class KalmanTrack:
    """Single object track using a Kalman filter with constant-velocity model.

    State vector: [cx, cy, s, r, ẋ, ẏ, ṡ]
      cx, cy : bounding box centre
      s      : box area (scale)
      r      : aspect ratio (width/height) — assumed constant
      ẋ, ẏ, ṡ: velocities
    """

    _id_counter = 0

    def __init__(self, bbox: np.ndarray):
        """bbox: [x1, y1, x2, y2]"""
        KalmanTrack._id_counter += 1
        self.id       = KalmanTrack._id_counter
        self.hits     = 1
        self.no_match = 0
        self.age      = 1
        self._kf      = self._build_kf(bbox)

    @staticmethod
    def _build_kf(bbox: np.ndarray) -> KalmanFilter:
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([           # state transition matrix
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=float)
        kf.H = np.array([           # measurement matrix
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=float)
        kf.R[2:, 2:] *= 10.0
        kf.P[4:, 4:] *= 1000.0
        kf.P         *= 10.0
        kf.Q[-1,-1]  *= 0.01
        kf.Q[4:, 4:] *= 0.01
        kf.x[:4]      = KalmanTrack._xyxy_to_z(bbox)
        return kf

    @staticmethod
    def _xyxy_to_z(bbox: np.ndarray) -> np.ndarray:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2
        cy = bbox[1] + h / 2
        s  = w * h
        r  = w / float(h + 1e-6)
        return np.array([[cx], [cy], [s], [r]])

    @staticmethod
    def _z_to_xyxy(z: np.ndarray) -> np.ndarray:
        w = np.sqrt(z[2] * z[3])
        h = z[2] / (z[3] + 1e-6)
        x1 = z[0] - w / 2
        y1 = z[1] - h / 2
        return np.array([x1, y1, x1 + w, y1 + h])

    def predict(self) -> np.ndarray:
        if (self._kf.x[6] + self._kf.x[2]) <= 0:
            self._kf.x[6] = 0
        self._kf.predict()
        self.age += 1
        if self.no_match > 0:
            self.hits = 0
        self.no_match += 1
        return self._z_to_xyxy(self._kf.x[:4].flatten())

    def update(self, bbox: np.ndarray) -> None:
        self.no_match = 0
        self.hits    += 1
        self._kf.update(self._xyxy_to_z(bbox))

    @property
    def bbox(self) -> np.ndarray:
        return self._z_to_xyxy(self._kf.x[:4].flatten())


# ─────────────────────────────────────────────────────────────────────────────
# IoU utilities
# ─────────────────────────────────────────────────────────────────────────────
def _iou_matrix(bb_det: np.ndarray, bb_trk: np.ndarray) -> np.ndarray:
    """Compute IoU matrix (N_det × N_trk)."""
    if len(bb_det) == 0 or len(bb_trk) == 0:
        return np.zeros((len(bb_det), len(bb_trk)))
    x1 = np.maximum(bb_det[:, 0:1], bb_trk[:, 0])
    y1 = np.maximum(bb_det[:, 1:2], bb_trk[:, 1])
    x2 = np.minimum(bb_det[:, 2:3], bb_trk[:, 2])
    y2 = np.minimum(bb_det[:, 3:4], bb_trk[:, 3])
    inter  = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_d = ((bb_det[:, 2]-bb_det[:, 0]) * (bb_det[:, 3]-bb_det[:, 1]))[:, None]
    area_t = ((bb_trk[:, 2]-bb_trk[:, 0]) * (bb_trk[:, 3]-bb_trk[:, 1]))[None, :]
    union  = area_d + area_t - inter
    return inter / (union + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# SORT Tracker
# ─────────────────────────────────────────────────────────────────────────────
class SORTTracker:
    """
    Simple Online and Realtime Tracker (Bewley et al., 2016) with
    Kalman filter prediction and IoU-based Hungarian assignment.

    Args:
        cfg: TrackerConfig
    """

    def __init__(self, cfg: TrackerConfig = None):
        self.cfg    = cfg or TrackerConfig()
        self.tracks : List[KalmanTrack] = []

    def reset(self) -> None:
        self.tracks = []
        KalmanTrack._id_counter = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Args:
            detections: (N, 5) array [x1, y1, x2, y2, score]
                        or (N, 4) [x1, y1, x2, y2]
        Returns:
            (M, 5) array [x1, y1, x2, y2, track_id]  (confirmed tracks only)
        """
        if detections.ndim == 1:
            detections = detections[None]
        if detections.shape[1] == 5:
            detections = detections[:, :4]

        # ── Predict existing tracks ───────────────────────────────────────
        pred_boxes = np.array([t.predict() for t in self.tracks]) \
            if self.tracks else np.empty((0, 4))

        # ── Hungarian assignment ──────────────────────────────────────────
        matched, unmatched_det, unmatched_trk = self._associate(
            detections, pred_boxes)

        # ── Update matched ────────────────────────────────────────────────
        for d_idx, t_idx in matched:
            self.tracks[t_idx].update(detections[d_idx])

        # ── Create new tracks for unmatched detections ────────────────────
        for d_idx in unmatched_det:
            self.tracks.append(KalmanTrack(detections[d_idx]))

        # ── Delete lost tracks ────────────────────────────────────────────
        self.tracks = [t for t in self.tracks
                       if t.no_match <= self.cfg.max_age]

        # ── Return confirmed tracks ───────────────────────────────────────
        result = []
        for t in self.tracks:
            if t.hits >= self.cfg.min_hits or t.age <= self.cfg.min_hits:
                b = t.bbox
                result.append([b[0], b[1], b[2], b[3], t.id])
        return np.array(result) if result else np.empty((0, 5))

    def _associate(self,
                   dets : np.ndarray,
                   trks : np.ndarray
                   ) -> Tuple[List, List, List]:
        if len(trks) == 0:
            return [], list(range(len(dets))), []
        if len(dets) == 0:
            return [], [], list(range(len(trks)))

        iou = _iou_matrix(dets, trks)
        row, col = linear_sum_assignment(-iou)

        matched, unmatched_d, unmatched_t = [], [], []
        for d in range(len(dets)):
            if d not in row:
                unmatched_d.append(d)
        for t in range(len(trks)):
            if t not in col:
                unmatched_t.append(t)
        for r, c in zip(row, col):
            if iou[r, c] < self.cfg.iou_threshold:
                unmatched_d.append(r)
                unmatched_t.append(c)
            else:
                matched.append((r, c))

        return matched, unmatched_d, unmatched_t
