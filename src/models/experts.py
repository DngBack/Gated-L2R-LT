"""Model definitions for expert networks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


_AVAILABLE_MODELS = {
    "resnet18": lambda num_classes, pretrained: _create_resnet(models.resnet18, num_classes, pretrained),
    "resnet32": lambda num_classes, pretrained: _create_resnet_cifar(num_classes),
}


def _create_resnet(factory, num_classes: int, pretrained: bool) -> nn.Module:
    model = factory(pretrained=pretrained)
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model


def _create_resnet_cifar(num_classes: int) -> nn.Module:
    from torchvision.models.resnet import BasicBlock

    class ResNet32(models.ResNet):
        def __init__(self, num_classes: int) -> None:
            super().__init__(BasicBlock, [5, 5, 5], num_classes=num_classes)

    return ResNet32(num_classes)


@dataclass
class Expert:
    name: str
    model: nn.Module
    loss_fn: nn.Module

    def to(self, device: torch.device) -> "Expert":
        self.model.to(device)
        self.loss_fn.to(device)
        return self

    def state_dict(self) -> Dict[str, torch.Tensor]:  # pragma: no cover - passthrough
        return {"model": self.model.state_dict()}

    @torch.no_grad()
    def predict_proba(self, inputs: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        logits = self.model(inputs)
        return F.softmax(logits, dim=-1)


def create_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    name = name.lower()
    if name not in _AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {name}")
    return _AVAILABLE_MODELS[name](num_classes, pretrained)


__all__ = ["Expert", "create_model"]
