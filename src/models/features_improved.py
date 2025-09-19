"""Improved feature engineering following L2R-LT theory."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def build_improved_gating_features(
    probabilities: Sequence[torch.Tensor], 
    class_frequencies: torch.Tensor,
    group_info=None
) -> torch.Tensor:
    """Construct enhanced feature tensor for gating network following L2R-LT theory.

    Features include:
    1. Expert probability vectors
    2. Log-probabilities (stabilized)
    3. Entropy per expert
    4. Confidence measures (max prob, margin)
    5. Class frequency features (prior knowledge)
    6. Disagreement/uncertainty measures
    7. Group-specific features
    """
    device = probabilities[0].device
    batch_size = probabilities[0].shape[0]
    
    feats = []
    
    # 1. Raw probabilities and log-probabilities
    for prob in probabilities:
        feats.append(prob)  # [B, C]
        feats.append(torch.log(torch.clamp(prob, min=1e-8)))  # [B, C]
    
    # 2. Per-expert confidence measures
    for prob in probabilities:
        # Max probability (confidence)
        max_prob, _ = prob.max(dim=1, keepdim=True)  # [B, 1]
        feats.append(max_prob)
        
        # Margin (difference between top-2 predictions)
        top2, _ = prob.topk(2, dim=1)  # [B, 2]
        margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)  # [B, 1]
        feats.append(margin)
        
        # Entropy
        entropy = -(prob * torch.log(torch.clamp(prob, min=1e-8))).sum(dim=1, keepdim=True)  # [B, 1]
        feats.append(entropy)
    
    # 3. Inter-expert disagreement measures
    if len(probabilities) > 1:
        # Pairwise KL divergences between experts
        kl_divs = []
        for i in range(len(probabilities)):
            for j in range(i + 1, len(probabilities)):
                kl = F.kl_div(
                    torch.log(torch.clamp(probabilities[i], min=1e-8)),
                    probabilities[j],
                    reduction='none'
                ).sum(dim=1, keepdim=True)  # [B, 1]
                kl_divs.append(kl)
        
        if kl_divs:
            feats.extend(kl_divs)
        
        # Variance across experts (measure of disagreement)
        stacked_probs = torch.stack(list(probabilities), dim=-1)  # [B, C, E]
        expert_variance = torch.var(stacked_probs, dim=-1).sum(dim=1, keepdim=True)  # [B, 1]
        feats.append(expert_variance)
        
        # Average probability across experts
        avg_prob = stacked_probs.mean(dim=-1)  # [B, C]
        feats.append(avg_prob)
    
    # 4. Class frequency features (prior knowledge for long-tail)
    class_freq_normalized = class_frequencies / class_frequencies.sum()
    class_freq_features = class_freq_normalized.unsqueeze(0).expand(batch_size, -1)  # [B, C]
    feats.append(class_freq_features)
    
    # Log class frequencies (imbalance signal)
    log_class_freq = torch.log(torch.clamp(class_freq_normalized, min=1e-8))
    log_freq_features = log_class_freq.unsqueeze(0).expand(batch_size, -1)  # [B, C]
    feats.append(log_freq_features)
    
    # 5. Group-specific features (if available)
    if group_info is not None:
        # Create group indicators for each sample based on predicted class
        for prob in probabilities:
            pred_class = prob.argmax(dim=1)  # [B]
            pred_group = group_info.class_to_group[pred_class]  # [B]
            
            # One-hot encoding of predicted group
            group_onehot = F.one_hot(
                torch.tensor(pred_group, device=device), 
                num_classes=group_info.num_groups()
            ).float()  # [B, K]
            feats.append(group_onehot)
    
    # 6. Temperature-scaled probabilities (calibration signal)
    temperatures = [0.5, 1.0, 2.0]  # Different temperature scales
    for temp in temperatures:
        for prob in probabilities:
            # Re-compute softmax with temperature
            logits = torch.log(torch.clamp(prob, min=1e-8))  # Approximate logits
            temp_prob = F.softmax(logits / temp, dim=1)  # [B, C]
            feats.append(temp_prob)
    
    # Concatenate all features
    return torch.cat(feats, dim=1)


def build_abstain_specific_features(
    probabilities: Sequence[torch.Tensor],
    class_frequencies: torch.Tensor,
    abstain_cost: float,
    group_info=None
) -> torch.Tensor:
    """Build features specifically for abstain-expert following L2R-LT theory."""
    
    device = probabilities[0].device
    batch_size = probabilities[0].shape[0]
    
    feats = []
    
    # 1. Uncertainty measures (key for abstention)
    for prob in probabilities:
        # Predictive entropy
        entropy = -(prob * torch.log(torch.clamp(prob, min=1e-8))).sum(dim=1, keepdim=True)
        feats.append(entropy)
        
        # Max probability (inverse of uncertainty)
        max_prob, _ = prob.max(dim=1, keepdim=True)
        feats.append(1.0 - max_prob)  # Uncertainty = 1 - confidence
    
    # 2. Cost-benefit signals
    # Abstention cost as feature
    cost_feature = torch.full((batch_size, 1), abstain_cost, device=device)
    feats.append(cost_feature)
    
    # 3. Long-tail signals
    if group_info is not None:
        for prob in probabilities:
            pred_class = prob.argmax(dim=1)
            pred_group = torch.tensor(group_info.class_to_group[pred_class], device=device)
            
            # Tail indicator (1 if tail group, 0 if head)
            tail_indicator = (pred_group > 0).float().unsqueeze(1)  # Assume group 0 is head
            feats.append(tail_indicator)
    
    # 4. Class frequency of predicted class (rarity signal)
    for prob in probabilities:
        pred_class = prob.argmax(dim=1)  # [B]
        class_freq = class_frequencies[pred_class].unsqueeze(1)  # [B, 1]
        # Use inverse frequency as rarity signal
        rarity = 1.0 / (class_freq + 1e-8)
        feats.append(rarity)
    
    return torch.cat(feats, dim=1)


__all__ = [
    "build_improved_gating_features",
    "build_abstain_specific_features"
]