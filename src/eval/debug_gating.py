"""Debug script for detailed analysis of gating network performance."""
from __future__ import annotations

import argparse
import os
import json
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt

from ..datasets.cifar_lt import build_dataloaders, default_transforms
from ..models.experts import Expert, create_model
from ..models.features import build_gating_features
from ..models.gating import GatingNetwork, PluginParameters, mixture_probabilities, plugin_classifier
from ..utils.config import add_common_args, load_config
from ..utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug gating network")
    parser.add_argument("--experts", type=str, required=True, help="Comma separated expert identifiers")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to gating checkpoint")
    parser.add_argument("--output-dir", type=str, default="./debug_output", help="Output directory for debug info")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples to analyze")
    add_common_args(parser)
    return parser.parse_args()


def load_experts_debug(cfg, expert_names, class_freq):
    """Load experts with debug info."""
    experts = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\\n=== Loading Experts ===")
    for name in expert_names:
        print(f"Loading expert: {name}")
        
        model = create_model(cfg["model"]["name"], cfg["num_classes"], cfg["model"].get("pretrained", False))
        ckpt_path = os.path.join(cfg["logging"]["output_dir"], f"expert_{name}", "best.pt")
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint for expert '{name}' not found at {ckpt_path}")
        
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        model.to(device)
        model.eval()
        
        # Create loss function
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
        elif loss_type == "ldam_drw":
            from ..losses.ldam_drw import LDAMDRWLoss
            loss_fn = LDAMDRWLoss(class_freq)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        
        experts.append(Expert(name=name, model=model, loss_fn=loss_fn))
        print(f"  ✓ Loaded {name} expert with {loss_type} loss")
    
    return experts


def analyze_expert_performance(experts, loader, class_freq, group_info, n_samples=1000):
    """Analyze individual expert performance."""
    print("\\n=== Expert Performance Analysis ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    
    for expert in experts:
        print(f"\\nAnalyzing expert: {expert.name}")
        
        expert.model.eval()
        correct_total = 0
        total = 0
        group_correct = defaultdict(int)
        group_total = defaultdict(int)
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        confidences = []
        entropies = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if total >= n_samples:
                    break
                
                inputs = batch["x"].to(device)
                labels = batch["y"].to(device)
                groups = batch["group"]
                
                # Get predictions
                probs = expert.predict_proba(inputs)
                preds = probs.argmax(dim=1)
                
                # Calculate metrics
                correct = (preds == labels)
                correct_total += correct.sum().item()
                total += len(labels)
                
                # Per-group accuracy
                for i, (pred, label, group) in enumerate(zip(preds, labels, groups)):
                    group_correct[group] += (pred == label).item()
                    group_total[group] += 1
                    
                    class_correct[label.item()] += (pred == label).item()
                    class_total[label.item()] += 1
                
                # Confidence and entropy
                max_probs, _ = probs.max(dim=1)
                confidences.extend(max_probs.cpu().numpy())
                
                entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=1)
                entropies.extend(entropy.cpu().numpy())
        
        # Compute overall accuracy
        overall_acc = correct_total / total
        
        # Compute group accuracies
        group_accs = {}
        for group in group_total:
            group_accs[group] = group_correct[group] / group_total[group]
        
        # Compute class accuracies by frequency
        head_classes = list(range(int(len(class_freq) * 0.3)))  # Top 30% classes
        tail_classes = list(range(int(len(class_freq) * 0.3), len(class_freq)))
        
        head_acc = sum(class_correct[c] for c in head_classes) / max(1, sum(class_total[c] for c in head_classes))
        tail_acc = sum(class_correct[c] for c in tail_classes) / max(1, sum(class_total[c] for c in tail_classes))
        
        results[expert.name] = {
            "overall_accuracy": overall_acc,
            "group_accuracies": group_accs,
            "head_accuracy": head_acc,
            "tail_accuracy": tail_acc,
            "avg_confidence": np.mean(confidences),
            "avg_entropy": np.mean(entropies),
            "samples_analyzed": total
        }
        
        print(f"  Overall Accuracy: {overall_acc:.4f}")
        print(f"  Group Accuracies: {group_accs}")
        print(f"  Head Accuracy: {head_acc:.4f}")
        print(f"  Tail Accuracy: {tail_acc:.4f}")
        print(f"  Avg Confidence: {np.mean(confidences):.4f}")
        print(f"  Avg Entropy: {np.mean(entropies):.4f}")
    
    return results


def analyze_gating_decisions(gating, experts, loader, params, class_freq, group_info, n_samples=1000):
    """Analyze gating network decisions in detail."""
    print("\\n=== Gating Network Analysis ===")
    
    device = next(gating.parameters()).device
    
    # Statistics
    total_samples = 0
    accept_count = 0
    reject_count = 0
    correct_accept = 0
    incorrect_accept = 0
    
    group_stats = defaultdict(lambda: {
        "total": 0, "accepted": 0, "rejected": 0,
        "correct_accept": 0, "incorrect_accept": 0
    })
    
    expert_usage = torch.zeros(len(experts), device=device)
    abstain_scores = []
    mixture_entropies = []
    expert_agreements = []
    
    gating.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if total_samples >= n_samples:
                break
            
            inputs = batch["x"].to(device)
            labels = batch["y"].to(device)
            groups = batch["group"]
            
            # Get expert predictions
            expert_probs = [expert.predict_proba(inputs) for expert in experts]
            
            # Build features
            features = build_gating_features(expert_probs)
            
            # Gating forward
            expert_weights, abstain_weight = gating(features)
            
            # Get mixture probabilities
            mixture_probs = mixture_probabilities(expert_weights, expert_probs)
            
            # Plugin classifier decision
            y_hat, reject = plugin_classifier(mixture_probs, params)
            
            # Analyze each sample
            for i in range(len(labels)):
                label = labels[i].item()
                group = groups[i]
                pred = y_hat[i].item()
                is_reject = reject[i].item()
                
                total_samples += 1
                
                if is_reject:
                    reject_count += 1
                    group_stats[group]["rejected"] += 1
                else:
                    accept_count += 1
                    group_stats[group]["accepted"] += 1
                    
                    if pred == label:
                        correct_accept += 1
                        group_stats[group]["correct_accept"] += 1
                    else:
                        incorrect_accept += 1
                        group_stats[group]["incorrect_accept"] += 1
                
                group_stats[group]["total"] += 1
            
            # Track expert usage
            expert_usage += expert_weights.mean(dim=0)
            
            # Track abstention scores
            abstain_scores.extend(abstain_weight.cpu().numpy())
            
            # Track mixture entropy
            mixture_entropy = -(mixture_probs * torch.log(mixture_probs.clamp(min=1e-8))).sum(dim=1)
            mixture_entropies.extend(mixture_entropy.cpu().numpy())
            
            # Track expert agreement
            for i in range(len(labels)):
                expert_preds = [probs[i].argmax().item() for probs in expert_probs]
                agreement = len(set(expert_preds)) == 1  # All experts agree
                expert_agreements.append(agreement)
    
    # Normalize expert usage
    expert_usage = expert_usage / len(loader)
    
    # Calculate statistics
    coverage = accept_count / total_samples
    selective_risk = incorrect_accept / max(1, accept_count)
    
    print(f"Total samples analyzed: {total_samples}")
    print(f"Coverage: {coverage:.4f} ({accept_count}/{total_samples})")
    print(f"Selective risk: {selective_risk:.4f}")
    print(f"\\nExpert usage distribution:")
    for i, usage in enumerate(expert_usage):
        print(f"  Expert {i} ({experts[i].name}): {usage:.4f}")
    
    print(f"\\nAbstention statistics:")
    print(f"  Mean abstain score: {np.mean(abstain_scores):.4f}")
    print(f"  Std abstain score: {np.std(abstain_scores):.4f}")
    
    print(f"\\nMixture statistics:")
    print(f"  Mean mixture entropy: {np.mean(mixture_entropies):.4f}")
    print(f"  Expert agreement rate: {np.mean(expert_agreements):.4f}")
    
    print(f"\\nGroup-wise statistics:")
    for group in sorted(group_stats.keys()):
        stats = group_stats[group]
        group_coverage = stats["accepted"] / max(1, stats["total"])
        group_risk = stats["incorrect_accept"] / max(1, stats["accepted"])
        
        print(f"  Group {group}:")
        print(f"    Coverage: {group_coverage:.4f} ({stats['accepted']}/{stats['total']})")
        print(f"    Selective risk: {group_risk:.4f}")
    
    return {
        "coverage": coverage,
        "selective_risk": selective_risk,
        "expert_usage": expert_usage.cpu().numpy(),
        "abstain_scores": abstain_scores,
        "mixture_entropies": mixture_entropies,
        "expert_agreements": expert_agreements,
        "group_stats": dict(group_stats)
    }


def analyze_plugin_parameters(params, class_freq, group_info):
    """Analyze plugin parameters α, μ."""
    print("\\n=== Plugin Parameters Analysis ===")
    
    print(f"Alpha (acceptance rates): {params.alpha.cpu().numpy()}")
    print(f"Mu (Lagrange multipliers): {params.mu.cpu().numpy()}")
    print(f"Cost: {params.cost}")
    
    # Check if parameters make sense
    alpha_sum = params.alpha.sum().item()
    print(f"Sum of alpha: {alpha_sum:.4f} (should be close to K={len(params.alpha)})")
    
    # Analyze per-class parameters
    print(f"\\nPer-class analysis:")
    for class_id in range(min(10, len(params.class_to_group))):  # Show first 10 classes
        group_id = params.class_to_group[class_id].item()
        alpha_val = params.alpha[group_id].item()
        mu_val = params.mu[group_id].item()
        freq = class_freq[class_id].item()
        
        print(f"  Class {class_id}: group={group_id}, α={alpha_val:.4f}, μ={mu_val:.4f}, freq={freq:.0f}")


def create_debug_plots(debug_results, output_dir):
    """Create visualization plots for debug analysis."""
    print(f"\\n=== Creating Debug Plots ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # Expert usage plot
    plt.figure(figsize=(10, 6))
    expert_names = list(debug_results["expert_performance"].keys())
    expert_usage = debug_results["gating_analysis"]["expert_usage"]
    
    plt.subplot(2, 2, 1)
    plt.bar(expert_names, expert_usage)
    plt.title("Expert Usage Distribution")
    plt.ylabel("Usage Weight")
    plt.xticks(rotation=45)
    
    # Abstention scores histogram
    plt.subplot(2, 2, 2)
    abstain_scores = debug_results["gating_analysis"]["abstain_scores"]
    plt.hist(abstain_scores, bins=50, alpha=0.7)
    plt.title("Abstention Scores Distribution")
    plt.xlabel("Abstention Score")
    plt.ylabel("Frequency")
    
    # Expert accuracy comparison
    plt.subplot(2, 2, 3)
    expert_accs = [debug_results["expert_performance"][name]["overall_accuracy"] 
                   for name in expert_names]
    plt.bar(expert_names, expert_accs)
    plt.title("Expert Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    
    # Group performance
    plt.subplot(2, 2, 4)
    group_stats = debug_results["gating_analysis"]["group_stats"]
    groups = sorted(group_stats.keys())
    coverages = [group_stats[g]["accepted"] / max(1, group_stats[g]["total"]) for g in groups]
    risks = [group_stats[g]["incorrect_accept"] / max(1, group_stats[g]["accepted"]) for g in groups]
    
    x = np.arange(len(groups))
    width = 0.35
    plt.bar(x - width/2, coverages, width, label="Coverage", alpha=0.7)
    plt.bar(x + width/2, risks, width, label="Selective Risk", alpha=0.7)
    plt.title("Group-wise Coverage vs Risk")
    plt.xlabel("Group")
    plt.ylabel("Rate")
    plt.xticks(x, [f"Group {g}" for g in groups])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "debug_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ Saved debug plots to {output_dir}/debug_analysis.png")


def save_debug_report(debug_results, output_dir, cfg):
    """Save detailed debug report to JSON."""
    report_path = os.path.join(output_dir, "debug_report.json")
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    report = {
        "config": cfg,
        "debug_results": convert_numpy(debug_results),
        "summary": {
            "overall_coverage": debug_results["gating_analysis"]["coverage"],
            "overall_selective_risk": debug_results["gating_analysis"]["selective_risk"],
            "expert_usage_entropy": -sum(p * np.log(p + 1e-8) for p in debug_results["gating_analysis"]["expert_usage"] if p > 0),
            "expert_agreement_rate": np.mean(debug_results["gating_analysis"]["expert_agreements"]),
        }
    }
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"  ✓ Saved debug report to {report_path}")


def main():
    args = parse_args()
    cfg = load_config(args.cfg)
    
    if args.seed is not None:
        cfg["seed"] = args.seed
    
    seed_everything(cfg.get("seed", 42))
    
    print("=== GATING NETWORK DEBUG ANALYSIS ===")
    print(f"Config: {args.cfg}")
    print(f"Experts: {args.experts}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    
    # Parse expert names
    expert_names = [name.strip() for name in args.experts.split(",") if name.strip()]
    if not expert_names:
        raise ValueError("No experts specified")
    
    # Build datasets
    train_transform, test_transform = default_transforms(cfg["dataset"])
    train_loader, val_loader, test_loader, group_info = build_dataloaders(
        dataset=cfg["dataset"],
        root=cfg["root"],
        imbalance_factor=cfg["imbalance_factor"],
        max_images_per_class=cfg["max_images_per_class"],
        num_classes=cfg["num_classes"],
        val_fraction=cfg["val_fraction"],
        seed=cfg["seed"],
        batch_size=32,  # Smaller batch for debug
        test_batch_size=32,
        num_workers=2,
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
    experts = load_experts_debug(cfg, expert_names, class_freq)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load gating network
    print(f"\\n=== Loading Gating Network ===")
    feature_dim = len(experts) * (2 * cfg["num_classes"] + 1)
    gating = GatingNetwork(feature_dim, len(experts)).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    gating.load_state_dict(checkpoint["gating"])
    gating.eval()
    
    # Load plugin parameters
    params = PluginParameters(
        alpha=checkpoint["alpha"].to(device),
        mu=checkpoint["mu"].to(device),
        cost=cfg["abstain"]["cost"],
        class_to_group=torch.tensor(group_info.class_to_group, device=device),
    )
    
    print(f"  ✓ Loaded gating network from {args.checkpoint}")
    
    # Run debug analysis
    debug_results = {}
    
    # 1. Analyze expert performance
    debug_results["expert_performance"] = analyze_expert_performance(
        experts, val_loader, class_freq, group_info, args.n_samples
    )
    
    # 2. Analyze gating decisions
    debug_results["gating_analysis"] = analyze_gating_decisions(
        gating, experts, val_loader, params, class_freq, group_info, args.n_samples
    )
    
    # 3. Analyze plugin parameters
    analyze_plugin_parameters(params, class_freq, group_info)
    
    # 4. Create debug plots
    create_debug_plots(debug_results, args.output_dir)
    
    # 5. Save debug report
    save_debug_report(debug_results, args.output_dir, cfg)
    
    print("\\n=== DEBUG ANALYSIS COMPLETE ===")
    print(f"Check {args.output_dir} for detailed results")


if __name__ == "__main__":
    main()