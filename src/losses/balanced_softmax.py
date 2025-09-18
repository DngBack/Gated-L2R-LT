"""Balanced Softmax cross-entropy loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax loss that corrects class priors."""

    def __init__(self, class_freq: torch.Tensor) -> None:
        super().__init__()
        if class_freq.ndim != 1:
            raise ValueError("class_freq must be 1-D tensor")
        freq = class_freq.clone().float()
        freq = torch.clamp(freq, min=1.0)
        self.register_buffer("log_prior", torch.log(freq / freq.sum()))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("Expected logits shape [B, C]")
        logits = logits + self.log_prior.to(logits.device)
        return F.cross_entropy(logits, targets)


__all__ = ["BalancedSoftmaxLoss"]
