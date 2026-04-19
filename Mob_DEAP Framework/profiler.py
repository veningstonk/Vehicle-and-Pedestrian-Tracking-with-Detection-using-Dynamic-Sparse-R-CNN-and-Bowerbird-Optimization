"""
utils/profiler.py  —  End-to-end wall-clock latency profiler.

Only end-to-end FPS is reported as the primary speed metric.
Module-level backbone FPS (e.g. 82 FPS on RTX 4090) is NOT
reported, following the correction demanded by Reviewer #3 (pt 7-8).
"""
import time
from collections import defaultdict
from typing import Dict, List
import numpy as np


class E2EProfiler:
    """
    Measures wall-clock time for each stage and total pipeline.

    Usage:
        profiler = E2EProfiler()
        with profiler.stage("hfcbf"):
            out = hfcbf.process(frame)
        with profiler.stage("detection"):
            dets = detector(out)
        summary = profiler.summary()
    """

    STAGES = ["hfcbf", "detection", "classification", "tracking", "total"]

    def __init__(self):
        self._times: Dict[str, List[float]] = defaultdict(list)
        self._t0: Dict[str, float] = {}

    class _ContextTimer:
        def __init__(self, profiler, name):
            self._p   = profiler
            self._name= name
        def __enter__(self):
            self._p._t0[self._name] = time.perf_counter()
            return self
        def __exit__(self, *_):
            ms = (time.perf_counter() - self._p._t0[self._name]) * 1000
            self._p._times[self._name].append(ms)

    def stage(self, name: str) -> "_ContextTimer":
        return self._ContextTimer(self, name)

    def log_frame(self, stage_times: Dict[str, float]) -> None:
        """Manually log pre-measured stage times (ms)."""
        for k, v in stage_times.items():
            self._times[k].append(v)

    def summary(self) -> Dict:
        out = {}
        for stage, times in self._times.items():
            if not times:
                continue
            mean_ms = np.mean(times)
            out[stage] = {
                "mean_ms" : round(float(mean_ms), 2),
                "std_ms"  : round(float(np.std(times)), 2),
                "min_ms"  : round(float(np.min(times)), 2),
                "max_ms"  : round(float(np.max(times)), 2),
                "eff_fps" : round(1000.0 / mean_ms, 2),
                "n_frames": len(times),
            }
        return out

    def print_summary(self) -> None:
        s = self.summary()
        print("\n── End-to-end latency profile ──────────────────────────")
        print(f"{'Stage':<18} {'Mean (ms)':>10} {'Std':>8} {'FPS':>8}")
        print("─" * 46)
        for stage in self.STAGES:
            if stage not in s:
                continue
            d = s[stage]
            marker = " ←" if stage == "total" else ""
            print(f"{stage:<18} {d['mean_ms']:>10.2f} "
                  f"{d['std_ms']:>8.2f} {d['eff_fps']:>8.2f}{marker}")
        print("─" * 46)
        if "total" in s:
            print(f"  End-to-end FPS = {s['total']['eff_fps']:.2f}  "
                  f"(primary speed metric)")
        print()
