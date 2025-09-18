"""Risk-coverage evaluation utilities."""
from __future__ import annotations

from typing import Iterable, List, Tuple

import torch

from ..models.experts import Expert
from ..models.gating import GatingNetwork, PluginParameters
from .metrics import compute_metrics


def risk_coverage_curve(
    gating: GatingNetwork,
    experts: List[Expert],
    dataloader,
    base_params: PluginParameters,
    costs: Iterable[float],
    metric: str = "balanced_error",
) -> Tuple[List[float], List[float], List[dict]]:
    coverages: List[float] = []
    risks: List[float] = []
    per_cost_metrics: List[dict] = []
    for cost in costs:
        params = PluginParameters(
            alpha=base_params.alpha.clone(),
            mu=base_params.mu.clone(),
            cost=float(cost),
            class_to_group=base_params.class_to_group.clone(),
        )
        metrics = compute_metrics(gating, experts, dataloader, params)
        coverages.append(metrics["coverage"])
        risks.append(metrics[metric])
        per_cost_metrics.append(metrics)
    return coverages, risks, per_cost_metrics


def trapezoidal_area(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    area = 0.0
    for i in range(1, len(xs)):
        dx = abs(xs[i] - xs[i - 1])
        area += dx * (ys[i] + ys[i - 1]) / 2.0
    return area


__all__ = ["risk_coverage_curve", "trapezoidal_area"]
