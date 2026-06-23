# Optimizer Trajectory Demo

This directory compares optimizer trajectories on the same 3D loss surface.

The animation is process-oriented: each frame performs one optimizer step and records the intermediate point. If PyTorch is available, the demo uses `torch.optim` implementations step by step. If PyTorch cannot be imported, it falls back to NumPy implementations of the same update rules.

Included optimizers:

- GD
- Momentum
- Nesterov
- AdaGrad
- RMSProp
- Adam
- AdamW

Run:

```bash
python3 optimizers/main.py
```

The surface and analytic fallback gradient are defined in `surface.py`. The key rule is that no optimizer is allowed to jump directly to the final result; every trajectory is produced one step at a time.
