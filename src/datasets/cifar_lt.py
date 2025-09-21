"""Utilities for constructing long-tailed variants of CIFAR datasets.

This module provides helper functions to build training/validation/test
splits with user-specified imbalance factors and to keep track of group
assignments (head/tail) that are needed for balanced and worst-group
metrics.
"""
from __future__ import annotations

import dataclasses
import math
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from ..utils.seed import seed_everything


@dataclasses.dataclass
class DatasetConfig:
    """Configuration for building CIFAR long-tailed datasets."""

    dataset: str
    root: str
    train: bool
    download: bool
    imbalance_factor: float
    max_images_per_class: int
    num_classes: int
    val_fraction: float = 0.1
    random_seed: int = 42
    transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None


@dataclasses.dataclass
class GroupInfo:
    """Stores metadata describing group splits."""

    class_to_group: np.ndarray
    group_to_classes: List[np.ndarray]
    group_names: List[str]

    def num_groups(self) -> int:
        return len(self.group_names)


def make_long_tailed_counts(
    num_classes: int,
    max_images_per_class: int,
    imbalance_factor: float,
) -> np.ndarray:
    """Create per-class sample counts following an exponential decay.

    Args:
        num_classes: number of classes.
        max_images_per_class: the number of training examples for the head class.
        imbalance_factor: ratio between the number of head vs tail examples.

    Returns:
        An array of length ``num_classes`` containing integer counts.
    """

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if imbalance_factor <= 0:
        raise ValueError("imbalance_factor must be positive")

    counts = []
    for cls_idx in range(num_classes):
        fraction = cls_idx / max(1, num_classes - 1)
        count = max_images_per_class * (imbalance_factor ** (-fraction))
        counts.append(int(round(count)))

    counts = np.array(counts, dtype=int)
    counts[counts < 1] = 1
    return counts


def split_head_tail_groups(num_classes: int, head_ratio: float = 0.3) -> GroupInfo:
    """Split the classes into head/tail groups."""

    if not (0 < head_ratio <= 1):
        raise ValueError("head_ratio must be in (0, 1]")

    num_head = max(1, int(math.floor(num_classes * head_ratio)))
    class_to_group = np.zeros(num_classes, dtype=np.int64)
    class_to_group[num_head:] = 1
    group_to_classes = [np.arange(0, num_head), np.arange(num_head, num_classes)]
    return GroupInfo(
        class_to_group=class_to_group,
        group_to_classes=group_to_classes,
        group_names=["head", "tail"],
    )


class CIFARLongTail(Dataset):
    """Long-tailed subset of CIFAR10/100 with cached group assignments."""

    def __init__(
        self,
        base_dataset: Dataset,
        counts: Sequence[int],
        class_to_group: np.ndarray,
        random_seed: int = 42,
    ) -> None:
        self.base_dataset = base_dataset
        self.random_seed = random_seed
        self.num_classes = len(counts)
        self.class_to_group = class_to_group

        targets = np.array(getattr(base_dataset, "targets"))
        rng = np.random.default_rng(random_seed)

        selected_indices: List[int] = []
        for cls, count in enumerate(counts):
            cls_indices = np.flatnonzero(targets == cls)
            if len(cls_indices) < count:
                raise ValueError(
                    f"Not enough samples ({len(cls_indices)}) for class {cls} to satisfy count {count}."
                )
            rng.shuffle(cls_indices)
            selected_indices.extend(cls_indices[:count].tolist())

        self.indices = np.array(selected_indices, dtype=np.int64)
        rng.shuffle(self.indices)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.indices)

    def __getitem__(self, index: int):  # pragma: no cover - passthrough to base dataset
        base_idx = int(self.indices[index])
        image, label = self.base_dataset[base_idx]
        group = int(self.class_to_group[label])
        return {"x": image, "y": label, "group": group, "index": base_idx}


def _build_base_dataset(cfg: DatasetConfig) -> Dataset:
    transform = cfg.transform
    target_transform = cfg.target_transform
    dataset_name = cfg.dataset.lower()

    if dataset_name == "cifar10":
        dataset_class = datasets.CIFAR10
    elif dataset_name == "cifar100":
        dataset_class = datasets.CIFAR100
    else:  # pragma: no cover - not used in tests but kept for extensibility
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")

    return dataset_class(
        root=cfg.root,
        train=cfg.train,
        download=cfg.download,
        transform=transform,
        target_transform=target_transform,
    )


def build_cifar_lt_datasets(
    dataset: str,
    root: str,
    imbalance_factor: float,
    max_images_per_class: int,
    num_classes: int,
    val_fraction: float,
    seed: int,
    transform_train: Optional[Callable],
    transform_test: Optional[Callable],
    download: bool = True,
    head_ratio: float = 0.3,
) -> Tuple[Dataset, Dataset, Dataset, GroupInfo]:
    """Construct train/val/test splits for long-tailed CIFAR."""

    seed_everything(seed)

    group_info = split_head_tail_groups(num_classes=num_classes, head_ratio=head_ratio)

    train_cfg = DatasetConfig(
        dataset=dataset,
        root=root,
        train=True,
        download=download,
        imbalance_factor=imbalance_factor,
        max_images_per_class=max_images_per_class,
        num_classes=num_classes,
        val_fraction=val_fraction,
        random_seed=seed,
        transform=transform_train,
    )

    base_train = _build_base_dataset(train_cfg)
    counts = make_long_tailed_counts(
        num_classes=num_classes,
        max_images_per_class=max_images_per_class,
        imbalance_factor=imbalance_factor,
    )
    lt_train = CIFARLongTail(base_train, counts, group_info.class_to_group, random_seed=seed)

    # Split into train/val subsets
    num_val = int(round(len(lt_train) * val_fraction))
    num_train = len(lt_train) - num_val
    if num_val <= 0 or num_train <= 0:
        raise ValueError("val_fraction leads to empty train/val splits")

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        lt_train,
        lengths=[num_train, num_val],
        generator=generator,
    )

    # Build test dataset (balanced)
    test_cfg = DatasetConfig(
        dataset=dataset,
        root=root,
        train=False,
        download=download,
        imbalance_factor=imbalance_factor,
        max_images_per_class=max_images_per_class,
        num_classes=num_classes,
        random_seed=seed,
        transform=transform_test,
    )
    base_test = _build_base_dataset(test_cfg)
    # For test set, use all available samples (balanced)
    # CIFAR-100 test set has 100 samples per class, CIFAR-10 has 1000 per class
    if dataset.lower() == "cifar10":
        test_samples_per_class = 1000
    else:  # cifar100
        test_samples_per_class = 100
    test_counts = [test_samples_per_class] * num_classes
    test_dataset = CIFARLongTail(base_test, test_counts, group_info.class_to_group, random_seed=seed)

    return train_subset, val_subset, test_dataset, group_info


def build_dataloaders(
    dataset: str,
    root: str,
    imbalance_factor: float,
    max_images_per_class: int,
    num_classes: int,
    val_fraction: float,
    seed: int,
    batch_size: int,
    test_batch_size: int,
    num_workers: int,
    transform_train: Optional[Callable] = None,
    transform_test: Optional[Callable] = None,
    download: bool = True,
    head_ratio: float = 0.3,
) -> Tuple[DataLoader, DataLoader, DataLoader, GroupInfo]:
    """Factory method returning dataloaders and group metadata."""

    train_dataset, val_dataset, test_dataset, group_info = build_cifar_lt_datasets(
        dataset=dataset,
        root=root,
        imbalance_factor=imbalance_factor,
        max_images_per_class=max_images_per_class,
        num_classes=num_classes,
        val_fraction=val_fraction,
        seed=seed,
        transform_train=transform_train,
        transform_test=transform_test,
        download=download,
        head_ratio=head_ratio,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, group_info


def default_transforms(dataset: str) -> Tuple[Callable, Callable]:
    """Return default augmentation and evaluation transforms."""

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if dataset.lower() == "cifar10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    return train_transform, test_transform


__all__ = [
    "CIFARLongTail",
    "DatasetConfig",
    "GroupInfo",
    "build_cifar_lt_datasets",
    "build_dataloaders",
    "default_transforms",
    "make_long_tailed_counts",
    "split_head_tail_groups",
]
