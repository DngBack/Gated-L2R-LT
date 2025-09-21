"""Simple debug script for gating network analysis."""
from __future__ import annotations

import argparse
import os
import json

import torch
import numpy as np

from ..datasets.cifar_lt import build_dataloaders, default_transforms
from ..models.experts import Expert, create_model
from ..models.features import build_gating_features
from ..models.gating import GatingNetwork, PluginParameters, mixture_probabilities, plugin_classifier
from ..utils.config import add_common_args, load_config
from ..utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple debug for gating network")
    parser.add_argument("--experts", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    add_common_args(parser)
    return parser.parse_args()


def load_experts_simple(cfg, expert_names, class_freq):
    """Load experts."""
    experts = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading experts...")
    for name in expert_names:
        model = create_model(cfg["model"]["name"], cfg["num_classes"], cfg["model"].get("pretrained", False))
        ckpt_path = os.path.join(cfg["logging"]["output_dir"], f"expert_{name}", "best.pt")
        
        if not os.path.exists(ckpt_path):
            print(f"ERROR: Missing checkpoint {ckpt_path}")
            continue
        
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        model.to(device)
        model.eval()
        
        # Simple loss function
        loss_fn = torch.nn.CrossEntropyLoss()
        experts.append(Expert(name=name, model=model, loss_fn=loss_fn))
        print(f"  ✓ Loaded {name}")
    
    return experts


def debug_expert_predictions(experts, loader, n_samples=200):
    """Debug expert predictions."""
    print("\\nAnalyzing expert predictions...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    
    for expert in experts:
        correct = 0
        total = 0
        confidences = []
        
        with torch.no_grad():
            for batch in loader:
                if total >= n_samples:
                    break
                
                inputs = batch["x"].to(device)
                labels = batch["y"].to(device)
                
                probs = expert.predict_proba(inputs)
                preds = probs.argmax(dim=1)
                
                correct += (preds == labels).sum().item()
                total += len(labels)
                
                max_probs, _ = probs.max(dim=1)
                confidences.extend(max_probs.cpu().numpy())
        
        accuracy = correct / max(1, total)
        avg_confidence = np.mean(confidences)
        
        results[expert.name] = {
            "accuracy": accuracy,
            "confidence": avg_confidence,
            "samples": total
        }
        
        print(f"  {expert.name}: accuracy={accuracy:.3f}, confidence={avg_confidence:.3f}")
    
    return results


def debug_gating_behavior(gating, experts, loader, params, n_samples=200):
    """Debug gating network behavior."""
    print("\\nAnalyzing gating decisions...")
    
    device = next(gating.parameters()).device
    
    total = 0
    accepted = 0
    correct_accepts = 0
    expert_usage = torch.zeros(len(experts))
    
    with torch.no_grad():
        for batch in loader:
            if total >= n_samples:
                break
            
            inputs = batch["x"].to(device)
            labels = batch["y"].to(device)
            
            # Get expert predictions
            expert_probs = [expert.predict_proba(inputs) for expert in experts]
            
            # Gating forward
            features = build_gating_features(expert_probs)
            weights, _ = gating(features)
            
            # Get decisions
            mixture_probs = mixture_probabilities(weights, expert_probs)
            y_hat, reject = plugin_classifier(mixture_probs, params)
            
            # Count statistics
            batch_size = len(labels)
            total += batch_size
            
            accept_mask = (reject == 0)
            accepted += accept_mask.sum().item()
            
            correct_mask = (y_hat == labels) & accept_mask
            correct_accepts += correct_mask.sum().item()
            
            # Track expert usage
            expert_usage += weights.mean(dim=0).cpu()
    
    # Calculate metrics
    coverage = accepted / max(1, total)
    selective_risk = 1.0 - (correct_accepts / max(1, accepted))
    expert_usage = expert_usage / len(loader)
    
    print(f"  Coverage: {coverage:.3f} ({accepted}/{total})")
    print(f"  Selective Risk: {selective_risk:.3f}")
    print(f"  Expert usage: {[f'{usage:.3f}' for usage in expert_usage]}")
    
    return {
        "coverage": coverage,
        "selective_risk": selective_risk,
        "expert_usage": expert_usage.numpy().tolist(),
        "total_samples": total
    }


def debug_plugin_params(params):
    """Debug plugin parameters."""
    print("\\nPlugin Parameters:")
    print(f"  Alpha: {params.alpha.cpu().numpy()}")
    print(f"  Mu: {params.mu.cpu().numpy()}")
    print(f"  Cost: {params.cost}")
    
    # Check sanity
    alpha_sum = params.alpha.sum().item()
    print(f"  Alpha sum: {alpha_sum:.3f} (should be ~{len(params.alpha)})")


def main():
    args = parse_args()
    cfg = load_config(args.cfg)
    
    if args.seed is not None:
        cfg["seed"] = args.seed
    
    seed_everything(cfg.get("seed", 42))
    
    print("=== SIMPLE GATING DEBUG ===")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return
    
    expert_names = [name.strip() for name in args.experts.split(",")]
    
    # Build validation loader
    _, test_transform = default_transforms(cfg["dataset"])
    _, val_loader, _, group_info = build_dataloaders(
        dataset=cfg["dataset"],
        root=cfg["root"],
        imbalance_factor=cfg["imbalance_factor"],
        max_images_per_class=cfg["max_images_per_class"],
        num_classes=cfg["num_classes"],
        val_fraction=cfg["val_fraction"],
        seed=cfg["seed"],
        batch_size=32,
        test_batch_size=32,
        num_workers=2,
        transform_train=test_transform,
        transform_test=test_transform,
    )
    
    # Get class frequencies (dummy)
    class_freq = torch.ones(cfg["num_classes"])
    
    # Load experts
    experts = load_experts_simple(cfg, expert_names, class_freq)
    if not experts:
        print("ERROR: No experts loaded!")
        return
    
    # Load gating
    print("\\nLoading gating network...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        feature_dim = len(experts) * (2 * cfg["num_classes"] + 1)
        gating = GatingNetwork(feature_dim, len(experts)).to(device)
        gating.load_state_dict(checkpoint["gating"])
        gating.eval()
        
        params = PluginParameters(
            alpha=checkpoint["alpha"].to(device),
            mu=checkpoint["mu"].to(device),
            cost=cfg["abstain"]["cost"],
            class_to_group=torch.tensor(group_info.class_to_group, device=device),
        )
        
        print("  ✓ Gating loaded successfully")
        
    except Exception as e:
        print(f"ERROR loading gating: {e}")
        return
    
    # Run analysis
    try:
        expert_results = debug_expert_predictions(experts, val_loader, n_samples=200)
        gating_results = debug_gating_behavior(gating, experts, val_loader, params, n_samples=200)
        debug_plugin_params(params)
        
        # Save simple report
        report = {
            "expert_performance": expert_results,
            "gating_behavior": gating_results,
            "plugin_params": {
                "alpha": params.alpha.cpu().numpy().tolist(),
                "mu": params.mu.cpu().numpy().tolist(),
                "cost": params.cost
            }
        }
        
        with open("debug_simple_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print("\\n✓ Debug complete! Report saved to debug_simple_report.json")
        
    except Exception as e:
        print(f"ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()