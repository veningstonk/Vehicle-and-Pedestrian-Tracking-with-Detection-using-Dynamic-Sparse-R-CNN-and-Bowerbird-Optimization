"""
data/mot_dataset.py  —  MOT16 / MOT17 / MOT20 dataset loader.

Expected directory layout:
    data/MOT17/
        train/
            MOT17-02-SDP/
                det/det.txt
                gt/gt.txt
                img1/000001.jpg ...
            MOT17-04-SDP/ ...
        test/
            MOT17-01-SDP/ ...

GT format (MOT):
    frame, id, x, y, w, h, conf, class, visibility
Detection format:
    frame, -1, x, y, w, h, conf, -1, -1, -1
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Sequence reader (single MOT sequence)
# ─────────────────────────────────────────────────────────────────────────────
class MOTSequence:
    """
    Reads a single MOT sequence (images + ground truth).

    Args:
        seq_dir : path to the sequence folder (e.g. MOT17-02-SDP)
        split   : "train" or "test" (test has no gt)
    """

    def __init__(self, seq_dir: str, split: str = "train"):
        self.seq_dir = Path(seq_dir)
        self.name    = self.seq_dir.name
        self.split   = split
        self.img_dir = self.seq_dir / "img1"

        # Sort frames
        self.frames = sorted(self.img_dir.glob("*.jpg")) + \
                      sorted(self.img_dir.glob("*.png"))
        self.n_frames = len(self.frames)

        # Load ground truth (train only)
        self.gt: Dict[int, np.ndarray] = {}
        gt_path = self.seq_dir / "gt" / "gt.txt"
        if gt_path.exists() and split == "train":
            self.gt = self._load_mot_gt(gt_path)

        # Load detections if available
        self.dets: Dict[int, np.ndarray] = {}
        det_path = self.seq_dir / "det" / "det.txt"
        if det_path.exists():
            self.dets = self._load_mot_det(det_path)

    def _load_mot_gt(self, path: Path) -> Dict[int, np.ndarray]:
        """Returns {frame_id: array[[x1,y1,x2,y2,track_id,class_id]]}"""
        data = np.loadtxt(str(path), delimiter=",")
        gt: Dict[int, List] = {}
        for row in data:
            fid = int(row[0])
            tid = int(row[1])
            x, y, w, h = row[2], row[3], row[4], row[5]
            cls = int(row[7]) if len(row) > 7 else 1
            vis = row[8] if len(row) > 8 else 1.0
            if vis < 0.1 or cls not in (1, 2):   # ignore pedestrian class 1 only in standard
                continue
            gt.setdefault(fid, []).append(
                [x, y, x + w, y + h, tid, cls])
        return {fid: np.array(v) for fid, v in gt.items()}

    def _load_mot_det(self, path: Path) -> Dict[int, np.ndarray]:
        """Returns {frame_id: array[[x1,y1,x2,y2,score]]}"""
        data = np.loadtxt(str(path), delimiter=",")
        dets: Dict[int, List] = {}
        for row in data:
            fid = int(row[0])
            x, y, w, h = row[2], row[3], row[4], row[5]
            score = float(row[6])
            dets.setdefault(fid, []).append([x, y, x + w, y + h, score])
        return {fid: np.array(v) for fid, v in dets.items()}

    def get_frame(self, frame_idx: int) -> np.ndarray:
        """Returns BGR uint8 frame."""
        return cv2.imread(str(self.frames[frame_idx]))

    def get_gt(self, frame_id: int) -> Optional[np.ndarray]:
        """Returns GT boxes for frame_id (1-indexed) or None."""
        return self.gt.get(frame_id)

    def __len__(self) -> int:
        return self.n_frames

    def __iter__(self):
        for i, f in enumerate(self.frames):
            frame_id = i + 1
            img  = cv2.imread(str(f))
            gt   = self.gt.get(frame_id)
            dets = self.dets.get(frame_id)
            yield frame_id, img, gt, dets


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrapper (all sequences in a split)
# ─────────────────────────────────────────────────────────────────────────────
class MOTDataset:
    """
    Loads all sequences for a given dataset and split.

    Args:
        root       : root data directory
        dataset    : "MOT16", "MOT17", or "MOT20"
        split      : "train" or "test"
    """

    # Condition tags for condition-wise evaluation (§4, Table C)
    SEQUENCE_CONDITIONS = {
        "MOT17-02": "day_static",    "MOT17-04": "day_static",
        "MOT17-09": "day_static",    "MOT17-05": "day_moving",
        "MOT17-10": "day_moving",    "MOT17-11": "day_moving",
        "MOT17-13": "night_low_light",
        "MOT20-01": "crowded",       "MOT20-02": "crowded",
        "MOT20-03": "crowded",       "MOT20-05": "crowded",
    }

    def __init__(self, root: str, dataset: str = "MOT17",
                 split: str = "train"):
        self.root    = Path(root)
        self.dataset = dataset
        self.split   = split

        split_dir = self.root / dataset / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {split_dir}\n"
                f"Download from motchallenge.net and extract to {self.root}/{dataset}/")

        self.sequences = [
            MOTSequence(str(d), split)
            for d in sorted(split_dir.iterdir())
            if d.is_dir()
        ]
        print(f"[MOTDataset] {dataset}/{split}: "
              f"{len(self.sequences)} sequences loaded.")

    def __len__(self) -> int:
        return len(self.sequences)

    def __iter__(self):
        return iter(self.sequences)

    def condition_of(self, seq_name: str) -> str:
        """Return condition tag for a sequence name."""
        base = "-".join(seq_name.split("-")[:2])
        return self.SEQUENCE_CONDITIONS.get(base, "normal")


# ─────────────────────────────────────────────────────────────────────────────
# Classification crop dataset (for MobDEAP training)
# ─────────────────────────────────────────────────────────────────────────────
class MOTCropDataset(Dataset):
    """
    Extracts cropped pedestrian / vehicle patches from MOT sequences
    for MobDEAP classifier training.

    Args:
        sequences  : list of MOTSequence objects
        img_size   : resize target (default 224 for MobileNetV2)
        transform  : optional torchvision transform
    """

    def __init__(self, sequences: List[MOTSequence],
                 img_size: int = 224, transform=None):
        import torchvision.transforms as T
        self.img_size  = img_size
        self.transform = transform or T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])
        self.samples: List[Tuple] = []
        for seq in sequences:
            for frame_id, img, gt, _ in seq:
                if img is None or gt is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w    = img_rgb.shape[:2]
                for ann in gt:
                    x1, y1, x2, y2 = [int(v) for v in ann[:4]]
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(w, x2); y2 = min(h, y2)
                    if (x2 - x1) < 10 or (y2 - y1) < 10:
                        continue
                    crop  = img_rgb[y1:y2, x1:x2]
                    label = int(ann[5]) - 1   # 0=pedestrian, 1=vehicle
                    self.samples.append((crop, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        crop, label = self.samples[idx]
        return self.transform(crop), label
