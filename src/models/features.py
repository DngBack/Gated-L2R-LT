"""Feature engineering utilities for gating inputs."""
from __future__ import annotations

from typing import Sequence

import torch


def build_gating_features(probabilities: Sequence[torch.Tensor]) -> torch.Tensor:
    """Construct a feature tensor for the gating network.

    The features include each expert's probability vector, log-probabilities
    (stabilised) and entropy, concatenated along the feature dimension.
    """

    feats = []
    for prob in probabilities:
        feats.append(prob)
        feats.append(torch.log(torch.clamp(prob, min=1e-8)))
        entropy = -(prob * torch.log(torch.clamp(prob, min=1e-8))).sum(dim=1, keepdim=True)
        feats.append(entropy)
    return torch.cat(feats, dim=1)


__all__ = ["build_gating_features"]
