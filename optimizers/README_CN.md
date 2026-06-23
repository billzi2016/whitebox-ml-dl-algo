# 优化器轨迹演示

这个目录在同一个 3D loss surface 上对比多个优化器的小球轨迹。

![优化器轨迹演示](op-demo.png)

这个演示是过程型的：每一帧执行一次 optimizer step，并记录中间位置。如果 PyTorch 可用，就逐帧调用 `torch.optim`；如果 PyTorch 无法导入，就 fallback 到 NumPy 更新规则。

包含优化器：

- GD
- Momentum
- Nesterov
- AdaGrad
- RMSProp
- Adam
- AdamW

运行：

```bash
python3 optimizers/main.py
```

选择曲面：

```bash
python3 optimizers/main.py --surface saddle
python3 optimizers/main.py --surface rosenbrock
python3 optimizers/main.py --surface himmelblau
python3 optimizers/main.py --surface rastrigin
python3 optimizers/main.py --surface rugged
```

曲面预设：

- `saddle`：带四次边界墙和波纹的有界马鞍面；默认使用它，因为能看出不同优化器行为，同时不会让轨迹直接飞出画面。
- `rosenbrock`：经典 Rosenbrock 香蕉谷，用来看狭长弯曲 valley。
- `himmelblau`：经典 Himmelblau 多极小值函数。
- `rastrigin`：经典 Rastrigin 多峰函数，局部 basin 很多。
- `rugged`：项目最开始的波纹碗面，比较平滑但有轻微非线性 valley。

核心规则：不能直接跳到最终结果，每条轨迹都必须一步一步产生。
