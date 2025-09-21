@echo off
REM Debug script for CIFAR-100 LT gating network
echo Starting debug analysis...

REM Debug gating network
python -m src.eval.debug_gating ^
    --cfg configs/cifar100lt.yaml ^
    --experts head,tail,balanced ^
    --checkpoint src/outputs/cifar100lt/gating/gating.pt ^
    --output-dir debug_results ^
    --n-samples 500

echo.
echo Debug analysis complete!
echo Check debug_results/ folder for detailed results
pause