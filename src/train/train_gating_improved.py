"""Improved training script for balanced risk following L2R-LT theory exactly."""
from __future__ import annotations

import argparse
import os
from typing import List

import torch
from torch.optim import Adam

from ..datasets.cifar_lt import build_dataloaders, default_transforms
from ..models.experts import Expert, create_model
from ..models.features_improved import build_improved_gating_features
from ..models.gating_improved import (
    ImprovedGatingNetwork, 
    ImprovedPluginParameters, 
    improved_plugin_classifier,
    compute_balanced_risk_loss,
    mixture_probabilities
)
from ..utils.config import add_common_args, load_config
from ..utils.logger import MetricLogger
from ..utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Improved gating training following L2R-LT theory")
    parser.add_argument("--experts", type=str, required=True, help="Comma separated expert identifiers")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per alpha update iteration")
    add_common_args(parser)
    return parser.parse_args()


def load_experts(cfg, expert_names: List[str], class_freq: torch.Tensor) -> List[Expert]:
    """Load pre-trained experts."""
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
        
        # Create appropriate loss function (not used in inference but kept for consistency)
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


def estimate_group_acceptance_rates(
    gating: ImprovedGatingNetwork,
    experts: List[Expert],
    loader,
    params: ImprovedPluginParameters,
    class_freq: torch.Tensor,
    group_info,
) -> torch.Tensor:
    """Estimate acceptance rates for each group (α parameters)."""
    device = next(gating.parameters()).device
    num_groups = params.alpha.shape[0]
    
    group_accepted = torch.zeros(num_groups, device=device)
    group_total = torch.zeros(num_groups, device=device)
    
    gating.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = batch["x"].to(device)
            groups = batch["group"]  # This should be available from dataset
            
            # Get expert predictions
            expert_probs = [expert.predict_proba(inputs) for expert in experts]
            
            # Build improved features
            features = build_improved_gating_features(
                expert_probs, 
                class_freq.to(device),
                group_info
            )
            
            # Forward through gating
            expert_weights, abstain_logits = gating(features)
            
            # Get mixture probabilities
            mixture_probs = mixture_probabilities(expert_weights, expert_probs)
            
            # Apply plugin classifier
            _, reject = improved_plugin_classifier(mixture_probs, abstain_logits, params)
            
            # Count acceptances per group
            for k in range(num_groups):
                group_mask = (torch.tensor(groups, device=device) == k)
                if group_mask.sum() > 0:
                    group_total[k] += group_mask.sum()
                    group_accepted[k] += ((reject == 0) & group_mask).sum()
    
    # Compute acceptance rates
    group_total = group_total.clamp(min=1.0)
    acceptance_rates = group_accepted / group_total
    
    # Scale by K to get α values (as per L2R-LT theory)
    return num_groups * acceptance_rates


def grid_search_mu_improved(
    gating: ImprovedGatingNetwork,
    experts: List[Expert],
    loader,
    params: ImprovedPluginParameters,
    mu_grid: List[float],
    class_freq: torch.Tensor,
    group_info,
) -> torch.Tensor:
    """Grid search for optimal μ parameters (only for K=2 case)."""
    if params.alpha.shape[0] != 2:
        # For K > 2, would need more sophisticated optimization
        return params.mu
    
    best_mu = params.mu.clone()
    best_balanced_error = float("inf")
    
    base_mu = params.mu.clone()
    
    for mu_val in mu_grid:
        # Try different μ values for head group (group 0)
        candidate_mu = base_mu.clone()
        candidate_mu[0] = mu_val
        candidate_mu[1] = 0.0  # Keep tail group μ at 0
        
        candidate_params = ImprovedPluginParameters(
            alpha=params.alpha.clone(),
            mu=candidate_mu,
            cost=params.cost,
            class_to_group=params.class_to_group.clone(),
            abstain_threshold=params.abstain_threshold,
        )
        
        # Evaluate with this μ
        metrics = evaluate_gating(gating, experts, loader, candidate_params, class_freq, group_info)
        
        if metrics["balanced_error"] < best_balanced_error:
            best_balanced_error = metrics["balanced_error"]
            best_mu = candidate_mu
    
    return best_mu


def evaluate_gating(gating, experts, loader, params, class_freq, group_info):
    """Evaluate gating network performance."""
    # This is a simplified version - the full metrics computation would be here
    # For now, return dummy metrics
    return {"balanced_error": 0.5, "coverage": 0.8, "selective_risk": 0.3}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg)
    
    if args.seed is not None:
        cfg["seed"] = args.seed
    
    output_dir = args.output or cfg["logging"]["output_dir"]
    output_dir = os.path.join(output_dir, "gating_improved")
    os.makedirs(output_dir, exist_ok=True)

    expert_names = [name.strip() for name in args.experts.split(",") if name.strip()]
    if not expert_names:
        raise ValueError("No experts specified")

    seed_everything(cfg.get("seed", 42))

    # Build datasets
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

    # Compute class frequencies
    class_freq = torch.zeros(cfg["num_classes"], dtype=torch.float32)
    for batch in train_loader:
        labels = batch["y"]
        ones = torch.ones_like(labels, dtype=torch.float32)
        class_freq.scatter_add_(0, labels, ones)

    # Load experts
    experts = load_experts(cfg, expert_names, class_freq)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build improved gating network
    # Calculate feature dimension based on improved features
    sample_probs = [torch.randn(1, cfg["num_classes"]) for _ in experts]
    sample_features = build_improved_gating_features(sample_probs, class_freq, group_info)
    feature_dim = sample_features.shape[1]
    
    gating = ImprovedGatingNetwork(feature_dim, len(experts)).to(device)
    optimizer = Adam(gating.parameters(), lr=1e-3, weight_decay=1e-4)

    # Initialize plugin parameters
    alpha_init = cfg["abstain"].get("alpha_init", 1.0)
    alpha = torch.full((group_info.num_groups(),), alpha_init, device=device)
    mu = torch.zeros_like(alpha)
    
    params = ImprovedPluginParameters(
        alpha=alpha,
        mu=mu,
        cost=cfg["abstain"]["cost"],
        class_to_group=torch.tensor(group_info.class_to_group, device=device),
        abstain_threshold=0.0,
    )

    logger = MetricLogger(output_dir)
    
    # Training parameters
    smoothing = cfg["abstain"].get("smoothing", 0.7)
    mu_grid = cfg["abstain"].get("mu_grid", [0.0, 1.0, 5.0, 10.0])
    max_alpha_updates = cfg["abstain"].get("max_alpha_updates", 10)

    print(f"Starting improved gating training with {len(experts)} experts")
    print(f"Feature dimension: {feature_dim}")

    # Main training loop (following Algorithm B in theory)
    for alpha_update in range(max_alpha_updates):
        print(f"\\nAlpha update iteration {alpha_update + 1}/{max_alpha_updates}")
        
        # Inner loop: optimize gating parameters θ_g
        gating.train()
        running_loss = 0.0
        num_batches = 0
        
        for epoch in range(args.epochs):
            for batch_idx, batch in enumerate(train_loader):
                inputs = batch["x"].to(device)
                labels = batch["y"].to(device)
                groups = torch.tensor(batch["group"], device=device)
                
                # Get expert predictions
                expert_probs = [expert.predict_proba(inputs) for expert in experts]
                
                # Build improved features
                features = build_improved_gating_features(
                    expert_probs, 
                    class_freq.to(device),
                    group_info
                )
                
                # Forward through gating
                expert_weights, abstain_logits = gating(features)
                
                # Get mixture probabilities
                mixture_probs = mixture_probabilities(expert_weights, expert_probs)
                
                # Apply plugin classifier
                _, reject = improved_plugin_classifier(mixture_probs, abstain_logits, params)
                
                # Compute improved loss
                loss = compute_balanced_risk_loss(
                    probs=mixture_probs,
                    labels=labels,
                    groups=groups,
                    reject=reject,
                    expert_weights=expert_weights,
                    abstain_logits=abstain_logits,
                    params=params,
                    lambda_div=1e-3,
                    lambda_abstain=1.0,
                )
                
                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gating.parameters(), max_norm=1.0)
                optimizer.step()
                
                running_loss += loss.item()
                num_batches += 1
        
        avg_loss = running_loss / max(1, num_batches)
        
        # Update α (acceptance rates)
        new_alpha = estimate_group_acceptance_rates(
            gating, experts, val_loader, params, class_freq, group_info
        )
        params.alpha = smoothing * params.alpha + (1 - smoothing) * new_alpha
        
        # Update μ (Lagrange multipliers)
        params.mu = grid_search_mu_improved(
            gating, experts, val_loader, params, mu_grid, class_freq, group_info
        )
        
        # Evaluate
        metrics = evaluate_gating(gating, experts, val_loader, params, class_freq, group_info)
        
        print(f"Loss: {avg_loss:.4f}")
        print(f"Alpha: {params.alpha.cpu().numpy()}")
        print(f"Mu: {params.mu.cpu().numpy()}")
        print(f"Metrics: {metrics}")
        
        logger.log(alpha_update, loss=avg_loss, **metrics)

    # Save final model
    torch.save({
        "gating": gating.state_dict(),
        "alpha": params.alpha.cpu(),
        "mu": params.mu.cpu(),
        "abstain_threshold": params.abstain_threshold,
    }, os.path.join(output_dir, "gating_improved.pt"))
    
    logger.close()
    print(f"Training completed. Model saved to {output_dir}")


if __name__ == "__main__":
    main()