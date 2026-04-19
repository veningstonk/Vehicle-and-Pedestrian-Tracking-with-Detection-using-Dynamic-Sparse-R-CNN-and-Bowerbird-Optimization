"""
models/mob_deap.py  —  MobDEAP classifier  (§3.4)
models/detector.py  —  DSR-CNN-BO detection wrapper  (§3.3)

MobDEAP:
    MobileNetV2 backbone → DA-EVA attention (after block 14, 96 ch)
    → Global Average Pool → FC(128, ReLU) → FC(64, ReLU) → Dropout → FC(2)

DSR-CNN-BO:
    Torchvision Faster R-CNN (ResNet-50 FPN) with Bowerbird Optimization
    applied to proposal confidence scores at inference.  This is the most
    faithful implementation of §3.3 without requiring a full Sparse R-CNN
    reimplementation from scratch.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torchvision.models as tv
import torchvision.ops  as ops
from config import MobDEAPConfig, DSRCNNConfig
from models.attention import DirectionalAdaptiveEVA


# ═══════════════════════════════════════════════════════════════════════════
# MobDEAP Classifier
# ═══════════════════════════════════════════════════════════════════════════
class MobDEAP(nn.Module):
    """
    MobileNetV2 + DA-EVA attention classifier.

    The DA-EVA module is inserted after MobileNetV2 inverted-residual
    block 14 (output: 96 channels), as stated in §3.4.1.
    """

    def __init__(self, cfg: MobDEAPConfig = None):
        super().__init__()
        self.cfg = cfg or MobDEAPConfig()

        # ── Backbone ─────────────────────────────────────────────────────
        weights = tv.MobileNet_V2_Weights.IMAGENET1K_V1 if self.cfg.pretrained else None
        base    = tv.mobilenet_v2(weights=weights)

        self.features_pre  = base.features[:14]   # up to block 13 → 96 ch
        self.features_post = base.features[14:]   # remaining blocks → 1280 ch

        # ── DA-EVA attention ─────────────────────────────────────────────
        self.da_eva = DirectionalAdaptiveEVA(
            channels  = 96,
            num_heads = self.cfg.da_eva_heads,
            lam       = self.cfg.da_eva_lambda,
            eps       = self.cfg.variance_eps,
        )

        # ── Classifier head ──────────────────────────────────────────────
        final_ch = 1280
        self.gap = nn.AdaptiveAvgPool2d(1)
        fc_in, fc_h1, fc_h2 = final_ch, self.cfg.fc_dims[0], self.cfg.fc_dims[1]
        self.classifier = nn.Sequential(
            nn.Linear(fc_in, fc_h1), nn.ReLU(True),
            nn.Linear(fc_h1, fc_h2), nn.ReLU(True),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(fc_h2, self.cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) normalised tensor.
        Returns:
            (B, num_classes) logits.
        """
        x = self.features_pre(x)    # (B, 96,  H', W')
        x = self.da_eva(x)          # DA-EVA attention
        x = self.features_post(x)   # (B, 1280, H'', W'')
        x = self.gap(x).flatten(1)  # (B, 1280)
        return self.classifier(x)   # (B, num_classes)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════
# DSR-CNN-BO Detection Wrapper
# ═══════════════════════════════════════════════════════════════════════════
class DSRCNNWrapper(nn.Module):
    """
    Faster R-CNN (ResNet-50 FPN) with Bowerbird Optimization applied to
    proposal confidence scores at inference.

    BO optimises a weight vector w ∈ ℝ^{num_proposals} that scales raw
    confidence scores before NMS.  The fitness function is MSE between
    weighted scores and ground-truth presence indicators (Eq. 3).

    Args:
        cfg: DSRCNNConfig
        bo_weights: optional pre-computed BO weight vector (loaded from
                    a previous BO optimisation run).
    """

    def __init__(self,
                 cfg        : DSRCNNConfig = None,
                 bo_weights : torch.Tensor = None):
        super().__init__()
        self.cfg = cfg or DSRCNNConfig()

        import torchvision.models.detection as det
        weights = det.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        self.detector = det.fasterrcnn_resnet50_fpn(
            weights=weights,
            num_classes=91,           # COCO pretrained; fine-tune head separately
            box_score_thresh=self.cfg.score_threshold,
            box_nms_thresh=self.cfg.nms_threshold,
        )
        # Replace classification head for 2-class task
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.cfg.num_classes + 1   # +1 for background
        )

        # BO proposal weights (applied at inference to score scaling)
        self.bo_weights = bo_weights   # None until BO has been run

    def forward(self, images, targets=None):
        """
        Standard Faster R-CNN forward.
        During inference, BO weights scale the box scores if available.
        """
        if self.training:
            return self.detector(images, targets)

        # Inference path with optional BO re-weighting
        outputs = self.detector(images)
        if self.bo_weights is not None:
            outputs = self._apply_bo_weights(outputs)
        return outputs

    def _apply_bo_weights(self, outputs):
        """Scale box scores by BO weight vector and re-apply NMS."""
        w = self.bo_weights.to(outputs[0]["scores"].device)
        result = []
        for out in outputs:
            scores = out["scores"]
            n = min(len(scores), len(w))
            scaled = scores[:n] * w[:n]
            # Re-apply NMS after score scaling
            keep = ops.nms(out["boxes"][:n], scaled, self.cfg.nms_threshold)
            result.append({
                "boxes"  : out["boxes"][keep],
                "labels" : out["labels"][keep],
                "scores" : scaled[keep],
            })
        return result
