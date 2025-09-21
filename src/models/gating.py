"""Gating network with abstention head."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts + 1),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(features)
        weights = F.softmax(logits, dim=-1)
        expert_weights = weights[..., : self.num_experts]
        abstain_weight = weights[..., self.num_experts]
        return expert_weights, abstain_weight


def mixture_probabilities(weights: torch.Tensor, expert_probs: Sequence[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(expert_probs, dim=-1)
    weights = weights.unsqueeze(1)
    return torch.sum(stacked * weights, dim=-1)


@dataclass
class PluginParameters:
    alpha: torch.Tensor  # shape [K]
    mu: torch.Tensor  # shape [K]
    cost: float
    class_to_group: torch.Tensor  # shape [C]

    def to(self, device: torch.device) -> "PluginParameters":
        self.alpha = self.alpha.to(device)
        self.mu = self.mu.to(device)
        self.class_to_group = self.class_to_group.to(device)
        return self


def plugin_classifier(p: torch.Tensor, params: PluginParameters) -> Tuple[torch.Tensor, torch.Tensor]:
    # Ensure class_to_group is on the same device as other tensors
    class_to_group = params.class_to_group.to(params.alpha.device)
    alpha_inv = 1.0 / params.alpha[class_to_group]
    mu = params.mu[class_to_group]

    lhs = p * alpha_inv.unsqueeze(0)
    lhs_max, y_hat = lhs.max(dim=1)
    rhs = torch.sum(p * (alpha_inv.unsqueeze(0) - mu.unsqueeze(0)), dim=1) - params.cost
    reject = (lhs_max < rhs).long()
    return y_hat, reject


__all__ = ["GatingNetwork", "mixture_probabilities", "PluginParameters", "plugin_classifier"]
