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
    import torch
    import torch.nn as nn

    class ResNet32(nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.inplanes = 16
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace=True)
            
            # ResNet32 has 3 stages with [5, 5, 5] blocks each
            self.layer1 = self._make_layer(BasicBlock, 16, 5, stride=1)
            self.layer2 = self._make_layer(BasicBlock, 32, 5, stride=2)
            self.layer3 = self._make_layer(BasicBlock, 64, 5, stride=2)
            
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)
            
            # Initialize weights
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x

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
