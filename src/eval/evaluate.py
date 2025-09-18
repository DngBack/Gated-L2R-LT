"""Command-line evaluation for gating ensemble with detailed reporting."""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import List

import torch

from ..datasets.cifar_lt import build_dataloaders, default_transforms
from ..eval.metrics import compute_metrics
from ..eval.rc_curve import risk_coverage_curve, trapezoidal_area
from ..models.experts import Expert, create_model
from ..models.gating import GatingNetwork, PluginParameters
from ..utils.config import add_common_args, load_config
from ..utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate gating ensemble")
    parser.add_argument("--experts", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to gating checkpoint")
    parser.add_argument(
        "--costs", type=str, default="0.01,0.05,0.1", help="Comma-separated abstention costs for RC curve"
    )
    parser.add_argument("--results-json", type=str, default=None, help="Optional path to dump metrics as JSON")
    parser.add_argument("--rc-csv", type=str, default=None, help="Optional CSV file for risk-coverage curves")
    add_common_args(parser)
    return parser.parse_args()


def load_experts(cfg, expert_names: List[str]) -> List[Expert]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experts: List[Expert] = []
    for name in expert_names:
        model = create_model(cfg["model"]["name"], cfg["num_classes"], cfg["model"].get("pretrained", False))
        ckpt_path = os.path.join(cfg["logging"]["output_dir"], f"expert_{name}", "best.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing expert checkpoint {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        model.to(device)
        model.eval()
        experts.append(Expert(name=name, model=model, loss_fn=torch.nn.CrossEntropyLoss()))
    return experts


def _print_metrics(split: str, metrics: dict, group_names: List[str]) -> None:
    print(f"\n[{split}] selective metrics")
    print(f"  Coverage: {metrics['coverage']:.4f} (reject rate {1.0 - metrics['coverage']:.4f})")
    print(f"  Selective risk: {metrics['selective_risk']:.4f}")
    print(f"  Balanced error: {metrics['balanced_error']:.4f}")
    print(f"  Worst-group error: {metrics['worst_group_error']:.4f}")
    print(f"  Min group coverage: {metrics['min_group_coverage']:.4f}")
    print(f"  Accepted samples: {metrics['num_accepted']}/{metrics['num_samples']}")
    per_group_cov = metrics.get("per_group_coverage", [])
    per_group_err = metrics.get("per_group_error", [])
    per_group_counts = metrics.get("per_group_counts", [])
    per_group_acc = metrics.get("per_group_accepted", [])
    for gid, name in enumerate(group_names):
        cov = per_group_cov[gid] if gid < len(per_group_cov) else 0.0
        err = per_group_err[gid] if gid < len(per_group_err) else 0.0
        total = per_group_counts[gid] if gid < len(per_group_counts) else 0
        accepted = per_group_acc[gid] if gid < len(per_group_acc) else 0
        print(
            f"    - Group {gid} ({name}): coverage={cov:.4f}, selective_error={err:.4f}, accepted={accepted}/{total}"
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg)
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed_everything(cfg.get("seed", 42))

    expert_names = [name.strip() for name in args.experts.split(",") if name.strip()]
    experts = load_experts(cfg, expert_names)

    train_transform, test_transform = default_transforms(cfg["dataset"])
    _, val_loader, test_loader, group_info = build_dataloaders(
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

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    feature_dim = len(experts) * (2 * cfg["num_classes"] + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gating = GatingNetwork(feature_dim, len(experts)).to(device)
    gating.load_state_dict(checkpoint["gating"])
    gating.eval()

    params = PluginParameters(
        alpha=checkpoint["alpha"].to(device),
        mu=checkpoint.get("mu", torch.zeros(group_info.num_groups(), device=device)),
        cost=cfg["abstain"]["cost"],
        class_to_group=torch.tensor(group_info.class_to_group, device=device),
    )

    val_metrics = compute_metrics(gating, experts, val_loader, params)
    test_metrics = compute_metrics(gating, experts, test_loader, params)

    _print_metrics("Validation", val_metrics, group_info.group_names)
    _print_metrics("Test", test_metrics, group_info.group_names)

    costs = [float(x) for x in args.costs.split(",") if x]
    cov_bal, risk_bal, rc_balanced = risk_coverage_curve(gating, experts, val_loader, params, costs, metric="balanced_error")
    cov_wg, risk_wg, rc_worst = risk_coverage_curve(gating, experts, val_loader, params, costs, metric="worst_group_error")

    aurc_bal = trapezoidal_area(cov_bal, risk_bal)
    aurc_wg = trapezoidal_area(cov_wg, risk_wg)

    print("\n[Validation] Risk-Coverage (balanced error)")
    for cost, cov, risk in zip(costs, cov_bal, risk_bal):
        print(f"  cost={cost:.3f} -> coverage={cov:.4f}, balanced_error={risk:.4f}")
    print(f"  AURC (balanced error) = {aurc_bal:.4f}")

    print("\n[Validation] Risk-Coverage (worst-group error)")
    for cost, cov, risk in zip(costs, cov_wg, risk_wg):
        print(f"  cost={cost:.3f} -> coverage={cov:.4f}, worst_group_error={risk:.4f}")
    print(f"  AURC (worst-group error) = {aurc_wg:.4f}")

    results_payload = {
        "config": args.cfg,
        "experts": expert_names,
        "checkpoint": args.checkpoint,
        "cost": cfg["abstain"]["cost"],
        "evaluation_costs": costs,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "rc_curve": {
            "balanced": {"coverage": cov_bal, "risk": risk_bal, "aurc": aurc_bal, "metrics": rc_balanced},
            "worst_group": {"coverage": cov_wg, "risk": risk_wg, "aurc": aurc_wg, "metrics": rc_worst},
        },
    }

    if args.results_json:
        json_dir = os.path.dirname(args.results_json)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        with open(args.results_json, "w", encoding="utf-8") as f:
            json.dump(results_payload, f, indent=2)
        print(f"\nSaved detailed metrics to {args.results_json}")

    if args.rc_csv:
        csv_dir = os.path.dirname(args.rc_csv)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        with open(args.rc_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "cost",
                    "coverage_balanced",
                    "balanced_error",
                    "coverage_worst_group",
                    "worst_group_error",
                ]
            )
            for idx, cost in enumerate(costs):
                cov_b = cov_bal[idx] if idx < len(cov_bal) else ""
                risk_b = risk_bal[idx] if idx < len(risk_bal) else ""
                cov_w = cov_wg[idx] if idx < len(cov_wg) else ""
                risk_w = risk_wg[idx] if idx < len(risk_wg) else ""
                writer.writerow([cost, cov_b, risk_b, cov_w, risk_w])
        print(f"Saved risk-coverage table to {args.rc_csv}")


if __name__ == "__main__":
    main()
