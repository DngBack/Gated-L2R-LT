"""Training script for expert models."""
from __future__ import annotations

import argparse
import os
from typing import Dict

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..datasets.cifar_lt import build_dataloaders, default_transforms
from ..losses.balanced_softmax import BalancedSoftmaxLoss
from ..losses.ldam_drw import LDAMDRWLoss
from ..losses.logit_adjustment import LogitAdjustedLoss
from ..models.experts import Expert, create_model
from ..utils.config import add_common_args, load_config
from ..utils.logger import MetricLogger
from ..utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a long-tailed expert")
    parser.add_argument("--expert", type=str, required=True, help="Expert identifier: head/tail/balanced")
    add_common_args(parser)
    return parser.parse_args()


def compute_class_frequency(loader: DataLoader, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes)
    dataset = loader.dataset
    if hasattr(dataset, "__getitem__"):
        for idx in range(len(dataset)):
            sample = dataset[idx]
            if isinstance(sample, dict):
                label = int(sample["y"])  # type: ignore[arg-type]
            else:
                label = int(sample[1])
            counts[label] += 1
        return counts
    for batch in loader:
        labels = batch["y"]
        counts.scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
    return counts


def build_loss(loss_cfg: Dict, class_freq: torch.Tensor) -> torch.nn.Module:
    name = loss_cfg["loss"].lower()
    if name == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    if name == "balanced_softmax":
        return BalancedSoftmaxLoss(class_freq)
    if name == "logit_adjustment":
        tau = loss_cfg.get("tau", 1.0)
        return LogitAdjustedLoss(class_freq, tau=tau)
    if name == "ldam_drw":
        max_m = loss_cfg.get("max_m", 0.5)
        weight_power = loss_cfg.get("weight_power", 0.5)
        return LDAMDRWLoss(class_freq, max_m=max_m, weight_power=weight_power)
    raise ValueError(f"Unsupported loss: {name}")


def train_one_epoch(
    expert: Expert,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> float:
    device = next(expert.model.parameters()).device
    expert.model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        inputs = batch["x"].to(device)
        labels = batch["y"].to(device)
        logits = expert.model(inputs)
        if isinstance(expert.loss_fn, LDAMDRWLoss):
            loss = expert.loss_fn(logits, labels, epoch=epoch)
        else:
            loss = expert.loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(1, num_batches)


def evaluate_accuracy(expert: Expert, loader: DataLoader) -> float:
    device = next(expert.model.parameters()).device
    expert.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["x"].to(device)
            labels = batch["y"].to(device)
            logits = expert.model(inputs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(1, total)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg)
    if args.seed is not None:
        cfg["seed"] = args.seed
    output_dir = args.output or cfg["logging"]["output_dir"]
    output_dir = os.path.join(output_dir, f"expert_{args.expert}")

    seed_everything(cfg.get("seed", 42))

    train_transform, test_transform = default_transforms(cfg["dataset"])
    train_loader, val_loader, _, _ = build_dataloaders(
        dataset=cfg["dataset"],
        root=cfg["root"],
        imbalance_factor=cfg["imbalance_factor"],
        max_images_per_class=cfg["max_images_per_class"],
        num_classes=cfg["num_classes"],
        val_fraction=cfg["val_fraction"],
        seed=cfg["seed"],
        batch_size=cfg["batch_size"],
        test_batch_size=cfg["validation"]["batch_size"],
        num_workers=cfg["num_workers"],
        transform_train=train_transform,
        transform_test=test_transform,
    )

    class_freq = compute_class_frequency(train_loader, cfg["num_classes"])
    model = create_model(cfg["model"]["name"], cfg["num_classes"], cfg["model"].get("pretrained", False))
    loss_fn = build_loss(cfg["experts"][args.expert], class_freq)
    expert = Expert(name=args.expert, model=model, loss_fn=loss_fn).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    params = [p for p in expert.model.parameters() if p.requires_grad]
    optimizer = SGD(
        params,
        lr=cfg["train"]["lr"],
        momentum=cfg["train"].get("momentum", 0.9),
        weight_decay=cfg["train"].get("weight_decay", 5e-4),
        nesterov=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    logger = MetricLogger(output_dir)
    best_acc = 0.0
    best_path = os.path.join(output_dir, "best.pt")
    os.makedirs(output_dir, exist_ok=True)

    device = next(expert.model.parameters()).device

    for epoch in range(cfg["train"]["epochs"]):
        loss = train_one_epoch(expert, train_loader, optimizer, epoch)
        scheduler.step()
        val_acc = evaluate_accuracy(expert, val_loader)
        logger.log(epoch, train_loss=loss, val_acc=val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": expert.model.state_dict()}, best_path)

    logger.close()
    print(f"Training finished. Best val acc={best_acc:.4f}. Saved to {best_path}")


if __name__ == "__main__":
    main()
