"""Improved Gating network with separate abstain-expert following L2R-LT theory."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedGatingNetwork(nn.Module):
    """Gating network with separate abstain-expert as per L2R-LT theory."""
    
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.num_experts = num_experts
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        # Expert routing head
        self.expert_head = nn.Linear(hidden_dim, num_experts)
        
        # Abstain head - separate as per theory
        self.abstain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass following L2R-LT theory.
        
        Returns:
            expert_weights: [B, E] - weights for experts (normalized)
            abstain_logits: [B, 1] - raw logits for abstention decision
        """
        shared = self.backbone(features)
        
        # Expert weights (normalized across experts only)
        expert_logits = self.expert_head(shared)
        expert_weights = F.softmax(expert_logits, dim=-1)
        
        # Abstain logits (NOT normalized - will be used with threshold)
        abstain_logits = self.abstain_head(shared).squeeze(-1)
        
        return expert_weights, abstain_logits


def mixture_probabilities(weights: torch.Tensor, expert_probs: Sequence[torch.Tensor]) -> torch.Tensor:
    """Compute mixture probabilities from expert weights and predictions."""
    stacked = torch.stack(list(expert_probs), dim=-1)  # [B, C, E]
    weights = weights.unsqueeze(1)  # [B, 1, E]
    return torch.sum(stacked * weights, dim=-1)  # [B, C]


@dataclass
class ImprovedPluginParameters:
    """Plugin parameters following L2R-LT theory exactly."""
    alpha: torch.Tensor  # shape [K] - group acceptance rates
    mu: torch.Tensor     # shape [K] - Lagrange multipliers
    cost: float          # abstention cost c
    class_to_group: torch.Tensor  # shape [C] - class to group mapping
    abstain_threshold: float = 0.0  # threshold for abstain_logits

    def to(self, device: torch.device) -> "ImprovedPluginParameters":
        self.alpha = self.alpha.to(device)
        self.mu = self.mu.to(device)
        self.class_to_group = self.class_to_group.to(device)
        return self


def improved_plugin_classifier(
    p: torch.Tensor, 
    abstain_logits: torch.Tensor,
    params: ImprovedPluginParameters
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Plugin classifier following L2R-LT theory exactly.
    
    Implementation of the Bayes-optimal rule:
    max_y α̃[y] p_y(x) ≶ Σ_y' (α̃[y'] - μ̃[y']) p_y'(x) - c
    
    Combined with abstain-expert output for final decision.
    """
    # Ensure tensors are on same device
    class_to_group = params.class_to_group.to(params.alpha.device)
    
    # Get group-specific parameters for each class
    alpha_inv = 1.0 / params.alpha[class_to_group]  # [C]
    mu = params.mu[class_to_group]  # [C]
    
    # L2R-LT rule: max_y α^(-1)[y] p_y(x)
    lhs = p * alpha_inv.unsqueeze(0)  # [B, C]
    lhs_max, y_hat = lhs.max(dim=1)  # [B]
    
    # R2R-LT rule: Σ_y' (α^(-1)[y'] - μ[y']) p_y'(x) - c
    rhs = torch.sum(p * (alpha_inv.unsqueeze(0) - mu.unsqueeze(0)), dim=1) - params.cost  # [B]
    
    # Combine L2R-LT rule with abstain-expert
    # Accept if both conditions are met:
    # 1. L2R-LT rule says accept: lhs_max >= rhs
    # 2. Abstain-expert says accept: abstain_logits <= threshold
    l2r_accept = (lhs_max >= rhs)
    abstain_accept = (abstain_logits <= params.abstain_threshold)
    
    # Final decision: accept only if both agree
    reject = ~(l2r_accept & abstain_accept)
    
    return y_hat, reject.long()


def compute_balanced_risk_loss(
    probs: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    reject: torch.Tensor,
    expert_weights: torch.Tensor,
    abstain_logits: torch.Tensor,
    params: ImprovedPluginParameters,
    group_weights: torch.Tensor | None = None,
    lambda_div: float = 1e-3,
    lambda_abstain: float = 1.0,
) -> torch.Tensor:
    """Compute balanced risk loss following L2R-LT theory exactly.
    
    Loss = (1/K) Σ_k (1/α_k) E[ℓ(h(x),y) 1{r=0, y∈G_k}] 
           + c E[1{r=1}] 
           + Σ_k μ_k (K Pr(r=0, y∈G_k) - α_k)
           + regularizers
    """
    num_groups = params.alpha.shape[0]
    
    # Base classification loss (cross-entropy)
    ce_loss = F.nll_loss(
        torch.log(torch.clamp(probs, min=1e-8)), 
        labels, 
        reduction="none"
    )
    
    # Group-specific terms
    balanced_loss = torch.tensor(0.0, device=probs.device)
    lagrange_penalty = torch.tensor(0.0, device=probs.device)
    
    for k in range(num_groups):
        # Mask for group k
        group_mask = (groups == k)
        if group_mask.sum() == 0:
            continue
            
        # Acceptance mask for group k
        accept_mask = (reject == 0) & group_mask
        
        # Group weight (for worst-group training)
        if group_weights is not None:
            group_weight = group_weights[k]
        else:
            group_weight = torch.tensor(1.0, device=probs.device)
        
        # Balanced risk term: (1/α_k) E[ℓ 1{r=0, y∈G_k}]
        if accept_mask.sum() > 0:
            group_ce = ce_loss[accept_mask].mean()
            alpha_k = params.alpha[k].clamp(min=1e-8)
            balanced_loss += group_weight * group_ce / alpha_k
        
        # Lagrange penalty: μ_k (K Pr(r=0, y∈G_k) - α_k)
        group_acceptance_rate = accept_mask.float().mean()
        constraint_violation = num_groups * group_acceptance_rate - params.alpha[k]
        lagrange_penalty += params.mu[k] * constraint_violation
    
    balanced_loss = balanced_loss / num_groups
    
    # Abstention cost: c Pr(r=1)
    abstain_cost = params.cost * reject.float().mean()
    
    # Abstain-expert loss (encourage learning good abstention)
    # Use BCE between abstain_logits and optimal abstain decisions
    optimal_abstain = reject.float()
    abstain_loss = F.binary_cross_entropy_with_logits(
        abstain_logits, optimal_abstain, reduction="mean"
    )
    
    # Diversity regularizer (encourage expert specialization)
    expert_usage = expert_weights.mean(dim=0)
    diversity_loss = -(expert_usage * torch.log(expert_usage.clamp(min=1e-8))).sum()
    
    # Total loss
    total_loss = (
        balanced_loss 
        + abstain_cost 
        + lagrange_penalty 
        + lambda_abstain * abstain_loss
        + lambda_div * diversity_loss
    )
    
    return total_loss


__all__ = [
    "ImprovedGatingNetwork", 
    "ImprovedPluginParameters", 
    "improved_plugin_classifier",
    "compute_balanced_risk_loss",
    "mixture_probabilities"
]