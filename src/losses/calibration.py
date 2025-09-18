"""Temperature scaling for probability calibration."""
from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.exp(self.log_temperature)
        return logits / temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 1000) -> None:
        self.eval()
        optimizer = optim.LBFGS([self.log_temperature])

        labels = labels.to(logits.device)

        def _eval() -> torch.Tensor:
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(self.forward(logits), labels)
            loss.backward()
            return loss

        for _ in range(max_iter):
            optimizer.step(_eval)

    @torch.no_grad()
    def transform_dataset(self, logits: torch.Tensor) -> torch.Tensor:
        return self.forward(logits)


def apply_temperature_scaling(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    scaler = TemperatureScaler()
    scaler.fit(logits, labels)
    calibrated = scaler.transform_dataset(logits)
    temperature = torch.exp(scaler.log_temperature).item()
    return calibrated, temperature


__all__ = ["TemperatureScaler", "apply_temperature_scaling"]
