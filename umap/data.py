import numpy as np


def make_geometric_manifold(seed=41):
    """生成几何感强的高维点云，用于 UMAP 优化过程演示。"""
    rng = np.random.default_rng(seed)
    points = []
    labels = []

    # 双环结构：提供清晰的局部邻域和全局分离。
    ring_count = 120
    theta = np.linspace(0, 2 * np.pi, ring_count, endpoint=False)
    outer = np.column_stack(
        [
            (3.2 + 0.25 * np.sin(8 * theta)) * np.cos(theta),
            (3.2 + 0.25 * np.sin(8 * theta)) * np.sin(theta),
        ]
    )
    inner = np.column_stack(
        [
            (1.45 + 0.18 * np.cos(5 * theta)) * np.cos(theta + 0.24),
            (1.45 + 0.18 * np.cos(5 * theta)) * np.sin(theta + 0.24),
        ]
    )
    points.extend([outer + rng.normal(0, 0.045, outer.shape), inner + rng.normal(0, 0.04, inner.shape)])
    labels.extend([0] * ring_count)
    labels.extend([1] * ring_count)

    # 对角波浪桥：让图不是简单同心圆，UMAP 会努力保留局部连接关系。
    bridge_count = 95
    t = np.linspace(-2.8, 2.8, bridge_count)
    bridge = np.column_stack([t, 0.55 * np.sin(2.4 * t)])
    bridge = bridge @ np.array([[0.78, -0.62], [0.62, 0.78]])
    bridge += rng.normal(0, 0.055, bridge.shape)
    points.append(bridge)
    labels.extend([2] * bridge_count)

    x2 = np.vstack(points)
    labels = np.array(labels)

    # 提升到六维，让 UMAP 处理的是高维流形而不是直接二维图。
    x, y = x2[:, 0], x2[:, 1]
    x_high = np.column_stack(
        [
            x,
            y,
            np.sin(0.9 * x),
            np.cos(0.9 * y),
            0.25 * x * y,
            np.sin(np.sqrt(x * x + y * y)),
        ]
    )
    x_high += rng.normal(0, 0.025, x_high.shape)
    return x_high, labels
