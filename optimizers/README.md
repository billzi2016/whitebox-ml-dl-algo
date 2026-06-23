# Optimizer Trajectory Demo

This directory compares optimizer trajectories on the same 3D loss surface.

![Optimizer trajectories demo](op-demo.png)

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

Choose a surface:

```bash
python3 optimizers/main.py --surface saddle
python3 optimizers/main.py --surface rosenbrock
python3 optimizers/main.py --surface himmelblau
python3 optimizers/main.py --surface rastrigin
python3 optimizers/main.py --surface rugged
```

Surface presets:

- `saddle`: a bounded saddle surface with quartic walls and ripples. This is the default because saddle geometry makes optimizer behavior easy to compare without letting trajectories fly out of the visible region.
- `rosenbrock`: the classic Rosenbrock banana valley, useful for showing narrow curved valleys.
- `himmelblau`: the Himmelblau function, a classic multi-minima benchmark.
- `rastrigin`: the Rastrigin function, a highly multimodal benchmark with many local basins.
- `rugged`: the original sinusoidal bowl used in this project, with smooth ripples and a mild nonlinear valley.

The surface and fallback gradients are defined in `surface.py`. The key rule is that no optimizer is allowed to jump directly to the final result; every trajectory is produced one step at a time.
