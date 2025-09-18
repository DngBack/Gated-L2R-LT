"""Utility helpers for deterministic training."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: Optional[int]) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""

    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
