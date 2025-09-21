"""Balanced-risk training for gating network with plug-in parameters."""
from __future__ import annotations

import argparse
import os
from typing import List

import torch
import torch.nn.functional as F
from torch.optim import Adam

from ..datasets.cifar_lt import build_dataloaders, default_transforms
from ..models.experts import Expert, create_model
from ..models.features import build_gating_features
from ..models.gating import GatingNetwork, PluginParameters, mixture_probabilities, plugin_classifier
from ..utils.config import add_common_args, load_config
from ..utils.logger import MetricLogger
from ..utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train gating network with plug-in rule")
    parser.add_argument("--experts", type=str, required=True, help="Comma separated expert identifiers")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per alpha update iteration")
    add_common_args(parser)
    return parser.parse_args()


def load_experts(cfg, expert_names: List[str], class_freq: torch.Tensor) -> List[Expert]:
    experts: List[Expert] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in expert_names:
        model = create_model(cfg["model"]["name"], cfg["num_classes"], cfg["model"].get("pretrained", False))
        ckpt_path = os.path.join(cfg["logging"]["output_dir"], f"expert_{name}", "best.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint for expert '{name}' not found at {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        model.to(device)
        model.eval()
        loss_cfg = cfg["experts"][name]
        loss_type = loss_cfg["loss"].lower()
        if loss_type == "cross_entropy":
            loss_fn = torch.nn.CrossEntropyLoss()
        elif loss_type == "balanced_softmax":
            from ..losses.balanced_softmax import BalancedSoftmaxLoss

            loss_fn = BalancedSoftmaxLoss(class_freq)
        elif loss_type == "logit_adjustment":
            from ..losses.logit_adjustment import LogitAdjustedLoss

            loss_fn = LogitAdjustedLoss(class_freq, tau=loss_cfg.get("tau", 1.0))
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        experts.append(Expert(name=name, model=model, loss_fn=loss_fn))
    return experts


def gating_loss(
    probs: torch.Tensor,
    labels: torch.Tensor,
    reject: torch.Tensor,
    weights: torch.Tensor,
    params: PluginParameters,
    lambda_div: float = 1e-3,
    group_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    class_to_group = params.class_to_group
    inv_alpha = 1.0 / params.alpha[class_to_group[labels]]
    if group_weights is None:
        group_weight = torch.ones_like(inv_alpha)
    else:
        group_weight = group_weights[class_to_group[labels]]
    ce = F.nll_loss(torch.log(torch.clamp(probs, min=1e-8)), labels, reduction="none")
    accept_mask = (reject == 0).float()
    loss_accept = (group_weight * inv_alpha * ce * accept_mask).mean()
    loss_reject = params.cost * (1.0 - accept_mask).mean()
    usage = weights.mean(dim=0)
    diversity = (usage * torch.log(torch.clamp(usage, min=1e-8))).sum()
    return loss_accept + loss_reject + lambda_div * diversity


def estimate_alpha(
    gating: GatingNetwork,
    experts: List[Expert],
    loader,
    params: PluginParameters,
) -> torch.Tensor:
    device = next(gating.parameters()).device
    counts = torch.zeros(params.alpha.shape[0], device=device)
    total = torch.zeros_like(counts)
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
            _, reject = plugin_classifier(probs, params)
            for gid in range(params.alpha.shape[0]):
                mask = groups == gid
                total[gid] += mask.sum()
                counts[gid] += ((reject == 0) & mask).sum()
    total_samples = total.sum().clamp_min(1.0)
    acceptance = counts / total_samples
    return params.alpha.shape[0] * acceptance


def grid_search_mu(
    gating: GatingNetwork,
    experts: List[Expert],
    loader,
    params: PluginParameters,
    mu_grid,
) -> torch.Tensor:
    if params.alpha.shape[0] != 2:
        return params.mu
    best_mu = params.mu.clone()
    best_metric = float("inf")
    base_mu = params.mu.clone()
    for value in mu_grid:
        candidate = base_mu.clone()
        candidate[0] = value
        candidate[1] = 0.0
        candidate_params = PluginParameters(
            alpha=params.alpha.clone(),
            mu=candidate.clone(),
            cost=params.cost,
            class_to_group=params.class_to_group.clone(),
        )
        metrics = compute_validation_error(gating, experts, loader, candidate_params)
        if metrics["balanced_error"] < best_metric:
            best_metric = metrics["balanced_error"]
            best_mu = candidate
    return best_mu


def compute_validation_error(gating, experts, loader, params):
    from ..eval.metrics import compute_metrics

    return compute_metrics(gating, experts, loader, params)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg)
    if args.seed is not None:
        cfg["seed"] = args.seed
    output_dir = args.output or cfg["logging"]["output_dir"]
    output_dir = os.path.join(output_dir, "gating")
    os.makedirs(output_dir, exist_ok=True)

    expert_names = [name.strip() for name in args.experts.split(",") if name.strip()]
    if not expert_names:
        raise ValueError("No experts specified")

    seed_everything(cfg.get("seed", 42))

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
    gating = GatingNetwork(feature_dim, len(experts)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = Adam(gating.parameters(), lr=1e-3, weight_decay=1e-4)

    alpha = torch.full((group_info.num_groups(),), cfg["abstain"].get("alpha_init", 1.0))
    mu = torch.zeros_like(alpha)
    params = PluginParameters(
        alpha=alpha,
        mu=mu,
        cost=cfg["abstain"]["cost"],
        class_to_group=torch.tensor(group_info.class_to_group),
    ).to(next(gating.parameters()).device)

    logger = MetricLogger(output_dir)

    smoothing = cfg["abstain"].get("smoothing", 0.7)
    mu_grid = cfg["abstain"].get("mu_grid", [0.0])

    for update in range(cfg["abstain"].get("max_alpha_updates", 10)):
        gating.train()
        last_loss = torch.tensor(0.0, device=params.alpha.device)
        for epoch in range(args.epochs):
            for batch in train_loader:
                inputs = batch["x"].to(params.alpha.device)
                labels = batch["y"].to(params.alpha.device)
                expert_probs = [expert.predict_proba(inputs) for expert in experts]
                features = build_gating_features(expert_probs)
                weights, _ = gating(features)
                probs = mixture_probabilities(weights, expert_probs)
                _, reject = plugin_classifier(probs, params)
                loss = gating_loss(probs, labels, reject, weights, params)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = loss.detach()

        new_alpha = estimate_alpha(gating, experts, val_loader, params)
        params.alpha = smoothing * params.alpha + (1 - smoothing) * new_alpha
        params.mu = grid_search_mu(gating, experts, val_loader, params, mu_grid)

        metrics = compute_validation_error(gating, experts, val_loader, params)
        logger.log(update, loss=float(last_loss.item()), **metrics)

    torch.save({"gating": gating.state_dict(), "alpha": params.alpha.cpu(), "mu": params.mu.cpu()}, os.path.join(output_dir, "gating.pt"))
    logger.close()


if __name__ == "__main__":
    main()
