"""Worst-group optimisation via exponentiated gradient."""
from __future__ import annotations

import argparse
import os
from typing import List

import torch
from torch.optim import Adam

from ..datasets.cifar_lt import build_dataloaders, default_transforms
from ..models.features import build_gating_features
from ..models.gating import GatingNetwork, PluginParameters, mixture_probabilities, plugin_classifier
from ..train.train_gating_bal import (
    compute_validation_error,
    gating_loss,
    grid_search_mu,
    load_experts,
)
from ..utils.config import add_common_args, load_config
from ..utils.logger import MetricLogger
from ..utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Worst-group training with EG")
    parser.add_argument("--experts", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per EG step")
    add_common_args(parser)
    return parser.parse_args()


def estimate_group_errors(
    gating: GatingNetwork,
    experts,
    loader,
    params: PluginParameters,
) -> torch.Tensor:
    device = next(gating.parameters()).device
    num_groups = params.alpha.shape[0]
    err = torch.zeros(num_groups, device=device)
    denom = torch.zeros(num_groups, device=device)
    gating.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = batch["x"].to(device)
            labels = batch["y"].to(device)
            groups = torch.tensor(batch["group"], device=device)
            expert_probs = [expert.predict_proba(inputs) for expert in experts]
            features = build_gating_features(expert_probs)
            weights, _ = gating(features)
            probs = mixture_probabilities(weights, expert_probs)
            y_hat, reject = plugin_classifier(probs, params)
            for gid in range(num_groups):
                mask = groups == gid
                accepted = ((reject == 0) & mask)
                mis = ((y_hat != labels) & accepted).sum()
                count = accepted.sum()
                err[gid] += mis.float()
                denom[gid] += count.float()
    denom = torch.clamp(denom, min=1.0)
    return err / denom


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg)
    if args.seed is not None:
        cfg["seed"] = args.seed
    output_dir = args.output or cfg["logging"]["output_dir"]
    output_dir = os.path.join(output_dir, "worst_group")
    os.makedirs(output_dir, exist_ok=True)

    seed_everything(cfg.get("seed", 42))

    expert_names = [name.strip() for name in args.experts.split(",") if name.strip()]
    if not expert_names:
        raise ValueError("No experts specified")

    train_transform, test_transform = default_transforms(cfg["dataset"])
    train_loader, val_loader, _, group_info = build_dataloaders(
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

    class_freq = torch.zeros(cfg["num_classes"], dtype=torch.float32)
    for batch in train_loader:
        labels = batch["y"]
        ones = torch.ones_like(labels, dtype=torch.float32)
        class_freq.scatter_add_(0, labels, ones)

    experts = load_experts(cfg, expert_names, class_freq)

    feature_dim = len(experts) * (2 * cfg["num_classes"] + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gating = GatingNetwork(feature_dim, len(experts)).to(device)
    optimizer = Adam(gating.parameters(), lr=1e-3, weight_decay=1e-4)

    alpha = torch.full((group_info.num_groups(),), cfg["abstain"].get("alpha_init", 1.0), device=device)
    mu = torch.zeros_like(alpha)
    params = PluginParameters(
        alpha=alpha,
        mu=mu,
        cost=cfg["abstain"]["cost"],
        class_to_group=torch.tensor(group_info.class_to_group, device=device),
    )

    beta = torch.ones_like(alpha) / alpha.numel()
    logger = MetricLogger(output_dir)
    smoothing = cfg["abstain"].get("smoothing", 0.7)
    mu_grid = cfg["abstain"].get("mu_grid", [0.0])
    eg_lr = cfg["worst_group"].get("eg_lr", 1.0)

    for step in range(cfg["worst_group"].get("eg_steps", 25)):
        gating.train()
        last_loss = torch.tensor(0.0, device=device)
        for epoch in range(args.epochs):
            for batch in train_loader:
                inputs = batch["x"].to(device)
                labels = batch["y"].to(device)
                expert_probs = [expert.predict_proba(inputs) for expert in experts]
                features = build_gating_features(expert_probs)
                weights, _ = gating(features)
                probs = mixture_probabilities(weights, expert_probs)
                _, reject = plugin_classifier(probs, params)
                loss = gating_loss(probs, labels, reject, weights, params, group_weights=beta)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = loss.detach()

        new_alpha = estimate_group_acceptance(gating, experts, val_loader, params)
        params.alpha = smoothing * params.alpha + (1 - smoothing) * new_alpha.to(device)
        params.mu = grid_search_mu(gating, experts, val_loader, params, mu_grid)

        metrics = compute_validation_error(gating, experts, val_loader, params)
        group_err = estimate_group_errors(gating, experts, val_loader, params)
        beta = beta * torch.exp(eg_lr * group_err)
        beta = beta / beta.sum()
        logger.log(step, loss=float(last_loss.item()), **metrics)

    torch.save(
        {
            "gating": gating.state_dict(),
            "alpha": params.alpha.cpu(),
            "mu": params.mu.cpu(),
            "beta": beta.cpu(),
        },
        os.path.join(output_dir, "worst_group.pt"),
    )
    logger.close()


def estimate_group_acceptance(gating, experts, loader, params):
    device = next(gating.parameters()).device
    counts = torch.zeros(params.alpha.shape[0], device=device)
    total = torch.zeros_like(counts)
    gating.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = batch["x"].to(device)
            groups = torch.tensor(batch["group"], device=device)
            expert_probs = [expert.predict_proba(inputs) for expert in experts]
            features = build_gating_features(expert_probs)
            weights, _ = gating(features)
            probs = mixture_probabilities(weights, expert_probs)
            _, reject = plugin_classifier(probs, params)
            for gid in range(params.alpha.shape[0]):
                mask = groups == gid
                total[gid] += mask.sum()
                counts[gid] += ((reject == 0) & mask).sum()
    total_samples = total.sum().clamp_min(1.0)
    acceptance = counts / total_samples
    return params.alpha.shape[0] * acceptance


if __name__ == "__main__":
    main()
