"""
training/losses.py  —  Focal Loss + GIoU Loss  (§5)
training/trainer.py —  Training loop for MobDEAP classifier
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import numpy as np
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Focal Loss  (Lin et al., 2017)  — α=0.25, γ=2.0  (§5)
# ═══════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    """
    Focal Loss for classification (handles class imbalance).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: balancing factor (default 0.25, §5)
        gamma: focusing parameter (default 2.0, §5)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self,
                inputs : torch.Tensor,   # (B, C) logits
                targets: torch.Tensor    # (B,) class indices
                ) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t     = torch.exp(-ce_loss)
        alpha_t = torch.full_like(targets, 1 - self.alpha, dtype=torch.float32)
        alpha_t[targets == 1] = self.alpha
        focal   = alpha_t * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


# ═══════════════════════════════════════════════════════════════════════════
# GIoU Loss  (Rezatofighi et al., 2019)  — for bounding box regression
# ═══════════════════════════════════════════════════════════════════════════
class GIoULoss(nn.Module):
    """
    Generalised IoU Loss.
    GIoU = IoU - |C \\ (A union B)| / |C|
    ℒ_GIoU = 1 − GIoU

    Provides valid gradients even when boxes do not overlap (IoU=0).
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self,
                pred  : torch.Tensor,   # (N, 4) [x1,y1,x2,y2]
                target: torch.Tensor    # (N, 4) [x1,y1,x2,y2]
                ) -> torch.Tensor:
        giou = ops.generalized_box_iou(pred, target).diag()
        loss = 1.0 - giou
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ═══════════════════════════════════════════════════════════════════════════
# MobDEAP Trainer
# ═══════════════════════════════════════════════════════════════════════════
class MobDEAPTrainer:
    r"""
    Training loop for the MobDEAP classifier.

    Integrates:
      - Focal Loss for classification
      - AdamW optimiser with step-decay learning rate (Sec 5)
      - Early stopping with patience (Sec 5)
      - 5-fold cross-validation (called externally)
      - Optional EPO hyperparameter search before training

    Args:
        model    : MobDEAP instance
        cfg      : MobDEAPConfig
        device   : torch device string
    """

    def __init__(self, model, cfg, device: str = "cuda"):
        self.model  = model.to(device)
        self.cfg    = cfg
        self.device = device
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)

    def _make_optimizer(self, lr: Optional[float] = None) -> torch.optim.Optimizer:
        lr = lr or self.cfg.lr
        return torch.optim.AdamW(
            self.model.parameters(), lr=lr,
            weight_decay=1e-4
        )

    def _make_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(self.cfg.lr_decay_epochs),
            gamma=1 - self.cfg.lr_decay_factor,
        )

    def train_epoch(self, loader, optimizer) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        for imgs, labels in loader:
            imgs   = imgs.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            logits = self.model(imgs)
            loss   = self.criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(imgs)
            n          += len(imgs)
        return total_loss / max(n, 1)

    @torch.no_grad()
    def eval_epoch(self, loader) -> Tuple[float, float]:
        """Returns (val_loss, accuracy)."""
        self.model.eval()
        total_loss, correct, n = 0.0, 0, 0
        for imgs, labels in loader:
            imgs   = imgs.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(imgs)
            loss   = self.criterion(logits, labels)
            total_loss += loss.item() * len(imgs)
            correct    += (logits.argmax(1) == labels).sum().item()
            n          += len(imgs)
        return total_loss / max(n, 1), correct / max(n, 1)

    def fit(self, train_loader, val_loader,
            lr: Optional[float] = None) -> dict:
        """
        Full training run with early stopping.

        Returns:
            dict with "best_val_loss", "best_accuracy", "epochs_trained"
        """
        optimizer = self._make_optimizer(lr)
        scheduler = self._make_scheduler(optimizer)
        best_loss, patience_cnt = float("inf"), 0
        best_acc  = 0.0
        history   = []

        for epoch in range(1, self.cfg.epochs + 1):
            tr_loss = self.train_epoch(train_loader, optimizer)
            vl_loss, vl_acc = self.eval_epoch(val_loader)
            scheduler.step()
            history.append({"epoch": epoch, "tr_loss": tr_loss,
                            "vl_loss": vl_loss, "vl_acc": vl_acc})

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{self.cfg.epochs}  "
                      f"train={tr_loss:.4f}  val={vl_loss:.4f}  "
                      f"acc={vl_acc*100:.2f}%")

            if vl_loss < best_loss:
                best_loss  = vl_loss
                best_acc   = vl_acc
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.cfg.early_stop_patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        return {
            "best_val_loss" : best_loss,
            "best_accuracy" : best_acc,
            "epochs_trained": epoch,
            "history"       : history,
        }
