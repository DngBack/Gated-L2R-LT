# Gated-L2R-LT

Implementation scaffolding for the paper idea **"Gating Mechanisms in Ensemble Models for Robust Rejection Learning on Long-Tail Data"**.

The repository provides:

- Long-tailed dataset builders for CIFAR-10/100 with group metadata (head/tail).
- Expert training pipeline with configurable losses (cross-entropy, Balanced Softmax, Logit Adjustment, LDAM-DRW).
- Gating network training with plug-in parameters following the "Learning to Reject Meets Long-Tail Learning" formulation.
- Worst-group optimisation via exponentiated gradient.
- Evaluation utilities for balanced / worst-group selective metrics and risk-coverage curves.

## Repository structure

```
configs/                # YAML experiment configurations
src/
  datasets/            # Long-tail dataset helpers
  losses/              # Loss implementations (Balanced Softmax, LDAM-DRW, ...)
  models/              # Expert networks, gating network, feature builder
  train/               # Training scripts for experts, gating, worst-group EG
  eval/                # Metric computation, RC-curve tooling, CLI evaluator
  utils/               # Config loader, seeding, logging utilities
scripts/               # Placeholder for launch scripts
```

## Quick start

1. **Train experts**
   ```bash
   python -m src.train.train_expert --cfg configs/cifar100lt.yaml --expert head
   python -m src.train.train_expert --cfg configs/cifar100lt.yaml --expert tail
   python -m src.train.train_expert --cfg configs/cifar100lt.yaml --expert balanced
   ```

2. **Train gating network (balanced risk)**
   ```bash
   python -m src.train.train_gating_bal --cfg configs/cifar100lt.yaml --experts head,tail,balanced --epochs 1
   ```

3. **Worst-group refinement (optional)**
   ```bash
   python -m src.train.train_wg_eg --cfg configs/cifar100lt.yaml --experts head,tail,balanced --epochs 1
   ```

4. **Evaluate**
   ```bash
   python -m src.eval.evaluate --cfg configs/cifar100lt.yaml \
      --experts head,tail,balanced \
      --checkpoint outputs/cifar100lt/gating/gating.pt \
      --costs 0.01,0.05,0.1 \
      --results-json reports/cifar100lt_metrics.json \
      --rc-csv reports/cifar100lt_rc.csv
   ```

   The evaluator prints coverage, selective risk, balanced / worst-group errors and per-group breakdowns for validation and test sets.
   The optional `--results-json` argument stores a structured summary (including risk-coverage curves) while `--rc-csv` writes
   paper-ready tables of coverage vs. risk at different abstention costs.

The scripts assume checkpoints are written to `outputs/<dataset>/expert_<name>/best.pt` and `outputs/<dataset>/gating/gating.pt`.

## Notes

- The code emphasises clarity and extensibility for research prototyping rather than aggressive optimisation.
- Calibration (e.g. temperature scaling) can be integrated via `src/losses/calibration.py` before gating training.
- The current `mu_grid` search is implemented for two-group settings (head/tail); extending to more groups requires extending `grid_search_mu`.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torchvision 0.15+
- PyYAML

Install dependencies via:
```bash
pip install torch torchvision pyyaml einops numpy matplotlib scikit-learn
```
