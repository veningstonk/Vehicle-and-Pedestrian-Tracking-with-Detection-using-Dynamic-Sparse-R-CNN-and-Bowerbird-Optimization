"""
data/ua_detrac.py  —  UA-DETRAC dataset loader  (Wen et al., 2020)
data/kitti.py      —  KITTI Tracking dataset loader  (Geiger et al., 2012)

UA-DETRAC layout:
    data/UA-DETRAC/
        DETRAC-train-data/
            MVI_20011/ img/ *.jpg
        DETRAC-train-annotations-XML/
            MVI_20011.xml
        DETRAC-test-data/ ...

KITTI Tracking layout:
    data/KITTI-tracking/
        training/
            image_02/0000/000000.png ...
            label_02/0000.txt
        testing/
            image_02/0000/ ...
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# UA-DETRAC
# ═══════════════════════════════════════════════════════════════════════════
class DETRACSequence:
    """
    Single UA-DETRAC sequence with XML annotation parsing.

    GT format (XML):
        <sequence name="MVI_20011">
          <frame density="..." num="1">
            <target_list>
              <target id="1"><box left="..." top="..." width="..." height="..."/></target>
              ...

    Args:
        seq_dir    : path to sequence image directory (e.g. MVI_20011/)
        xml_path   : path to corresponding XML annotation file
    """

    def __init__(self, seq_dir: str, xml_path: Optional[str] = None):
        self.seq_dir = Path(seq_dir)
        self.name    = self.seq_dir.name

        self.frames = sorted(
            list(self.seq_dir.glob("*.jpg")) +
            list(self.seq_dir.glob("img/*.jpg"))
        )
        self.n_frames = len(self.frames)

        self.gt: Dict[int, np.ndarray] = {}
        if xml_path and Path(xml_path).exists():
            self.gt = self._parse_xml(xml_path)

    @staticmethod
    def _parse_xml(xml_path: str) -> Dict[int, np.ndarray]:
        from lxml import etree
        tree = etree.parse(xml_path)
        gt: Dict[int, List] = {}
        for frame in tree.findall(".//frame"):
            fid = int(frame.get("num", 0))
            for target in frame.findall(".//target"):
                tid  = int(target.get("id", 0))
                box  = target.find("box")
                if box is None:
                    continue
                left = float(box.get("left",   0))
                top  = float(box.get("top",    0))
                w    = float(box.get("width",  0))
                h    = float(box.get("height", 0))
                gt.setdefault(fid, []).append(
                    [left, top, left + w, top + h, tid, 1])  # class 1=vehicle
        return {fid: np.array(v) for fid, v in gt.items()}

    def __len__(self) -> int:
        return self.n_frames

    def __iter__(self):
        for i, f in enumerate(self.frames):
            frame_id = i + 1
            img      = cv2.imread(str(f))
            gt       = self.gt.get(frame_id)
            yield frame_id, img, gt, None


class DETRACDataset:
    """
    Loads all UA-DETRAC sequences for a given split.

    Args:
        root  : data root (should contain DETRAC-train-data/ and
                DETRAC-train-annotations-XML/ subdirectories)
        split : "train" or "test"
    """

    def __init__(self, root: str, split: str = "train"):
        self.root  = Path(root)
        self.split = split

        img_root  = self.root / f"DETRAC-{split}-data"
        xml_root  = self.root / f"DETRAC-{split}-annotations-XML"

        if not img_root.exists():
            raise FileNotFoundError(
                f"UA-DETRAC images not found at {img_root}\n"
                f"Download from https://detrac-db.rit.albany.edu")

        self.sequences = []
        for seq_dir in sorted(img_root.iterdir()):
            if not seq_dir.is_dir():
                continue
            xml_path = xml_root / f"{seq_dir.name}.xml" \
                if xml_root.exists() else None
            self.sequences.append(
                DETRACSequence(str(seq_dir), str(xml_path)
                               if xml_path and xml_path.exists() else None))

        print(f"[DETRACDataset] {split}: {len(self.sequences)} sequences.")

    def __len__(self) -> int:
        return len(self.sequences)

    def __iter__(self):
        return iter(self.sequences)


# ═══════════════════════════════════════════════════════════════════════════
# KITTI Tracking
# ═══════════════════════════════════════════════════════════════════════════
# KITTI class mapping
KITTI_CLASSES = {
    "Car": 0, "Van": 1, "Truck": 2, "Pedestrian": 3,
    "Person_sitting": 3, "Cyclist": 4, "Tram": 5, "Misc": 6,
}

class KITTISequence:
    """
    Single KITTI tracking sequence.

    Label file format (training only):
        frame  track_id  class  truncated  occluded  alpha
        left  top  right  bottom  height  width  length
        x  y  z  rotation_y  [score]
    """

    def __init__(self, seq_id: str, split_dir: Path):
        self.seq_id  = seq_id
        self.name    = seq_id
        self.img_dir = split_dir / "image_02" / seq_id

        self.frames = sorted(self.img_dir.glob("*.png")) + \
                      sorted(self.img_dir.glob("*.jpg"))
        self.n_frames = len(self.frames)

        self.gt: Dict[int, np.ndarray] = {}
        label_file = split_dir / "label_02" / f"{seq_id}.txt"
        if label_file.exists():
            self.gt = self._parse_kitti_labels(label_file)

    @staticmethod
    def _parse_kitti_labels(path: Path) -> Dict[int, np.ndarray]:
        gt: Dict[int, List] = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 17:
                    continue
                fid   = int(parts[0])
                tid   = int(parts[1])
                cls   = parts[2]
                if cls == "DontCare":
                    continue
                left  = float(parts[6])
                top   = float(parts[7])
                right = float(parts[8])
                bot   = float(parts[9])
                class_id = KITTI_CLASSES.get(cls, -1)
                if class_id < 0:
                    continue
                gt.setdefault(fid, []).append(
                    [left, top, right, bot, tid, class_id])
        return {fid: np.array(v) for fid, v in gt.items()}

    def __len__(self) -> int:
        return self.n_frames

    def __iter__(self):
        for i, f in enumerate(self.frames):
            frame_id = i
            img  = cv2.imread(str(f))
            gt   = self.gt.get(frame_id)
            yield frame_id, img, gt, None


class KITTIDataset:
    """
    Loads all KITTI Tracking sequences for a given split.

    Args:
        root  : data root  (should contain training/ or testing/ subdirectory)
        split : "training" or "testing"
    """

    CLASS_NAMES = ["Car", "Van", "Truck", "Pedestrian", "Cyclist", "Tram"]

    def __init__(self, root: str, split: str = "training"):
        self.root      = Path(root)
        self.split     = split
        split_dir      = self.root / split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"KITTI tracking data not found at {split_dir}\n"
                f"Download from https://www.cvlibs.net/datasets/kitti/eval_tracking.php")

        img_dir = split_dir / "image_02"
        self.sequences = [
            KITTISequence(str(d.name), split_dir)
            for d in sorted(img_dir.iterdir())
            if d.is_dir()
        ]
        print(f"[KITTIDataset] {split}: {len(self.sequences)} sequences.")

    def __len__(self) -> int:
        return len(self.sequences)

    def __iter__(self):
        return iter(self.sequences)
