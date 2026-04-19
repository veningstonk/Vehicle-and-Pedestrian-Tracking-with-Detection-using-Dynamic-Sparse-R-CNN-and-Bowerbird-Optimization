"""
models/hfcbf.py  —  Hybrid Fast Conventional Bilateral Filter  (§3.2)

Two-pass approach:
  (a) Fast bilateral: Gaussian-blur seed + range-threshold mask.
  (b) Conventional cv2.bilateralFilter pass to recover accurate edges.

Parameters: σ_spatial=1.2, σ_range=30, 5×5 kernel  (from ablation §4.7.1).
"""
import numpy as np
import cv2
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import HFCBFConfig


class HFCBF:

    def __init__(self, cfg: HFCBFConfig = None):
        self.cfg    = cfg or HFCBFConfig()
        self._ksize = (self.cfg.kernel_size, self.cfg.kernel_size)
        self._d     = 2 * self.cfg.kernel_size + 1

    # ── private passes ───────────────────────────────────────────────────
    def _fast_bilateral(self, frame: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, self._ksize, self.cfg.sigma_spatial)
        diff    = np.abs(frame.astype(np.float32) -
                         blurred.astype(np.float32))
        mask    = diff < self.cfg.sigma_range
        return np.where(mask, blurred, frame).astype(np.uint8)

    def _conventional_bilateral(self, frame: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(
            frame, self._d,
            self.cfg.sigma_range,
            self.cfg.sigma_spatial * 10   # scale to pixel units
        )

    # ── public API ───────────────────────────────────────────────────────
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Args:
            frame: H×W×3 uint8 BGR image.
        Returns:
            Preprocessed frame, same shape and dtype.
        """
        out = self._fast_bilateral(frame)
        for _ in range(self.cfg.iterations):
            out = self._conventional_bilateral(out)
        return out

    def process_batch(self, frames):
        return [self.process(f) for f in frames]

    def measure_runtime(self, frame: np.ndarray, n: int = 100) -> float:
        """Returns mean ms/frame (warmup excluded)."""
        self.process(frame)           # warmup
        t0 = time.perf_counter()
        for _ in range(n):
            self.process(frame)
        return (time.perf_counter() - t0) / n * 1000
