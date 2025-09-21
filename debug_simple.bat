@echo off
echo === Simple Gating Debug ===
echo.

echo Running debug analysis...
python -m src.eval.debug_simple --cfg configs/cifar100lt.yaml --experts head,tail,balanced --checkpoint src/outputs/cifar100lt/gating/gating.pt

echo.
echo === Debug for Worst Group model ===
python -m src.eval.debug_simple --cfg configs/cifar100lt.yaml --experts head,tail,balanced --checkpoint src/outputs/cifar100lt/worst_group/worst_group.pt

echo.
echo Debug complete! Check debug_simple_report.json
pause