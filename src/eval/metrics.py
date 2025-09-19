"""Evaluation utilities for selective classification."""
from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from ..models.experts import Expert
from ..models.features import build_gating_features
from ..models.gating import (
    GatingNetwork,
    PluginParameters,
    mixture_probabilities,
    plugin_classifier,
)


def balanced_error(
    errors: torch.Tensor,
    accepts: torch.Tensor,
    groups: torch.Tensor,
    num_groups: int,
) -> float:
    """Compute balanced risk conditioned on acceptance for each group."""

    per_group: List[float] = []
    for gid in range(num_groups):
        mask = groups == gid
        if mask.sum() == 0:
            continue
        group_accepts = accepts[mask]
        if group_accepts.sum() == 0:
            continue
        group_errors = errors[mask][group_accepts.bool()].float()
        per_group.append(float(group_errors.mean().item()))
    if not per_group:
        return 0.0
    return float(sum(per_group) / len(per_group))


def worst_group_error(
    errors: torch.Tensor,
    accepts: torch.Tensor,
    groups: torch.Tensor,
    num_groups: int,
) -> float:
    """Compute worst-group selective risk."""

    per_group: List[float] = []
    for gid in range(num_groups):
        mask = groups == gid
        if mask.sum() == 0:
            continue
        group_accepts = accepts[mask]
        if group_accepts.sum() == 0:
            continue
        group_errors = errors[mask][group_accepts.bool()].float()
        per_group.append(float(group_errors.mean().item()))
    if not per_group:
        return 0.0
    return float(max(per_group))


def compute_metrics(
    gating: GatingNetwork,
    experts: List[Expert],
    dataloader: DataLoader,
    params: PluginParameters,
) -> Dict[str, object]:
    """Evaluate coverage and selective risks (with group breakdowns)."""

    device = next(gating.parameters()).device
    num_groups = int(params.alpha.shape[0])

    all_errors = []
    all_groups = []
    all_rejects = []

    gating.eval()
    for expert in experts:
        expert.model.eval()

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["x"].to(device)
            labels = batch["y"].to(device)
            groups = batch["group"].detach().clone().to(device)

            expert_probs = [expert.predict_proba(inputs) for expert in experts]
            features = build_gating_features(expert_probs)
            weights, _ = gating(features)
            probs = mixture_probabilities(weights, expert_probs)
            y_hat, reject = plugin_classifier(probs, params)
            err = ((y_hat != labels) & (reject == 0)).long()
            all_errors.append(err.cpu())
            all_groups.append(groups.cpu())
            all_rejects.append(reject.cpu())

    if not all_errors:
        return {
            "coverage": 0.0,
            "selective_risk": 0.0,
            "balanced_error": 0.0,
            "worst_group_error": 0.0,
            "per_group_coverage": [],
            "per_group_error": [],
            "per_group_counts": [],
            "per_group_accepted": [],
            "min_group_coverage": 0.0,
            "num_samples": 0,
            "num_accepted": 0,
        }

    errors = torch.cat(all_errors)
    groups = torch.cat(all_groups)
    rejects = torch.cat(all_rejects)
    accepts = (rejects == 0)

    num_samples = int(groups.numel())
    num_accepted = int(accepts.sum().item())

    overall_coverage = float(accepts.float().mean().item())
    overall_selective_risk = (
        float(errors.float().sum().item() / max(1, num_accepted))
        if num_samples > 0
        else 0.0
    )

    per_group_coverage: List[float] = []
    per_group_error: List[float] = []
    per_group_counts: List[int] = []
    per_group_accepted: List[int] = []

    for gid in range(num_groups):
        mask = groups == gid
        total = int(mask.sum().item())
        per_group_counts.append(total)
        if total == 0:
            per_group_coverage.append(0.0)
            per_group_error.append(0.0)
            per_group_accepted.append(0)
            continue
        group_accepts = accepts[mask]
        accepted_count = int(group_accepts.sum().item())
        per_group_accepted.append(accepted_count)
        coverage = float(group_accepts.float().mean().item())
        per_group_coverage.append(coverage)
        if accepted_count > 0:
            group_errors = errors[mask][group_accepts.bool()].float()
            per_group_error.append(float(group_errors.mean().item()))
        else:
            per_group_error.append(0.0)

    bal_err = balanced_error(errors, accepts, groups, num_groups)
    wg_err = worst_group_error(errors, accepts, groups, num_groups)
    min_group_cov = min(per_group_coverage) if per_group_coverage else 0.0

    return {
        "coverage": overall_coverage,
        "selective_risk": overall_selective_risk,
        "balanced_error": bal_err,
        "worst_group_error": wg_err,
        "per_group_coverage": per_group_coverage,
        "per_group_error": per_group_error,
        "per_group_counts": per_group_counts,
        "per_group_accepted": per_group_accepted,
        "min_group_coverage": min_group_cov,
        "num_samples": num_samples,
        "num_accepted": num_accepted,
    }


__all__ = ["compute_metrics", "balanced_error", "worst_group_error"]
