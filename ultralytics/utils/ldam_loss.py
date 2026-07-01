"""
LDAM-DRW: Label-Distribution-Aware Margin loss with Deferred Re-Weighting.
Reference: Cao et al. NeurIPS 2019 (arxiv.org/abs/1906.07413).
Adapted for YOLO26's classification loss on GRAZPEDWRI-DX.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────
# CONFIGURATION FOR GRAZPEDWRI-DX (Ju et al. patient-level split, train_aug)
# ─────────────────────────────────────────────────────────────────────────
# Per-class instance counts from your EDA
CLASS_COUNTS = {
    0: 368,     # boneanomaly
    1: 52,      # bonelesion
    2: 16,      # foreignbody
    3: 25224,   # fracture
    4: 1134,    # metal
    5: 4818,    # periostealreaction
    6: 816,     # pronatorsign
    7: 632,     # softtissue
    8: 33176,   # text
}

# LDAM hyperparameters
MAX_MARGIN = 0.5              # C in LDAM paper — caps the largest margin
# DRW hyperparameters
DRW_START_EPOCH = 30          # begin per-class weighting after this epoch
DRW_BETA = 0.9999             # class-balanced weighting parameter


def compute_ldam_margins(num_classes=9):
    """Compute per-class margins: m_c = C / n_c^(1/4).
    Rare classes get larger margins."""
    counts = np.array([CLASS_COUNTS[c] for c in range(num_classes)], dtype=np.float64)
    m = 1.0 / np.power(counts, 0.25)
    m = m * (MAX_MARGIN / m.max())  # normalize so max margin = MAX_MARGIN
    return torch.tensor(m, dtype=torch.float32)


def compute_drw_weights(num_classes=9, beta=DRW_BETA):
    """Class-balanced effective-number weights, applied after DRW_START_EPOCH."""
    counts = np.array([CLASS_COUNTS[c] for c in range(num_classes)], dtype=np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * num_classes  # normalize to sum = num_classes
    return torch.tensor(weights, dtype=torch.float32)


class LDAMLoss(nn.Module):
    """
    LDAM loss with DRW schedule.
    Drop-in replacement for BCEWithLogitsLoss inside YOLO26's v8DetectionLoss.

    Forward:
        pred_logits: (N, num_classes)  — pre-sigmoid class logits
        target:      (N, num_classes)  — soft targets from TAL assignment
    """

    def __init__(self, num_classes=9):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer('margins', compute_ldam_margins(num_classes))
        self.register_buffer('drw_weights', compute_drw_weights(num_classes))
        self.current_epoch = 0  # updated externally each epoch

    def set_epoch(self, epoch):
        """Called by the training loop each epoch to control DRW schedule."""
        self.current_epoch = epoch

    def forward(self, pred_logits, target):
        # target is soft-labeled from TAL; find dominant class per sample
        # For YOLO detection, target has shape (N, num_classes) with soft weights
        # Compute margin adjustment: subtract margin from the target class logit
        # This is equivalent to: logit_c_adjusted = logit_c - margin_c if class c is positive

        # Broadcast margins across batch
        margins_broadcast = self.margins.to(pred_logits.device).view(1, -1)

        # Margin subtraction happens ONLY for positive targets (weighted by target strength)
        # This is the LDAM formulation for multi-label / soft-target detection
        adjusted_logits = pred_logits - target * margins_broadcast

        # Deferred re-weighting: use uniform weights before DRW_START_EPOCH, then class weights
        if self.current_epoch < DRW_START_EPOCH:
            loss = F.binary_cross_entropy_with_logits(adjusted_logits, target, reduction='none')
        else:
            drw = self.drw_weights.to(pred_logits.device).view(1, -1)
            loss = F.binary_cross_entropy_with_logits(
                adjusted_logits, target, reduction='none'
            ) * drw

        return loss