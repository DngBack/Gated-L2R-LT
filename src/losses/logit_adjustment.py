"""Logit-adjusted cross entropy loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitAdjustedLoss(nn.Module):
    def __init__(self, class_freq: torch.Tensor, tau: float = 1.0) -> None:
        super().__init__()
        if class_freq.ndim != 1:
            raise ValueError("class_freq must be 1-D tensor")
        freq = torch.clamp(class_freq.clone().float(), min=1.0)
        prior = torch.log(freq / freq.sum())
        self.register_buffer("bias", -tau * prior)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("Expected logits shape [B, C]")
        logits = logits + self.bias.to(logits.device)
        return F.cross_entropy(logits, targets)


__all__ = ["LogitAdjustedLoss"]
