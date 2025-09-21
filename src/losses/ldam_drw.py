"""Implementation of LDAM-DRW loss for tail-aware training."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LDAMDRWLoss(nn.Module):
    """LDAM loss with deferred re-weighting (DRW).

    Reference: "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss" (Cao et al., NeurIPS 2019).
    """

    def __init__(
        self,
        class_freq: torch.Tensor,
        max_m: float = 0.5,
        s: float = 30.0,
        drw_start_epoch: int = 160,
        weight_power: float = 0.5,
    ) -> None:
        super().__init__()
        if class_freq.ndim != 1:
            raise ValueError("class_freq must be 1-D tensor")

        freq = class_freq.clone().float()
        freq = torch.clamp(freq, min=1.0)
        m_list = 1.0 / (freq ** 0.25)
        m_list = m_list * (max_m / m_list.max())
        self.register_buffer("m_list", m_list)
        self.max_m = max_m
        self.s = s
        self.drw_start_epoch = drw_start_epoch
        self.weight_power = weight_power
        weights = (freq ** (-weight_power))
        self.register_buffer("cls_weight", weights / weights.mean())

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        epoch: Optional[int] = None,
    ) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("Expected logits shape [B, C]")

        margins = self.m_list[targets].unsqueeze(1)
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.unsqueeze(1), True)
        logits_m = logits - index.float() * margins
        scaled_logits = self.s * logits_m

        if epoch is not None and epoch >= self.drw_start_epoch:
            weight = self.cls_weight.to(logits.device)
        else:
            weight = None

        return F.cross_entropy(scaled_logits, targets, weight=weight)


__all__ = ["LDAMDRWLoss"]
