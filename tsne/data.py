import numpy as np


def make_geometric_point_cloud(seed=19):
    """生成一个好看的几何图案点云，用于展示 t-SNE 拉扯过程。"""
    rng = np.random.default_rng(seed)
    points = []
    labels = []

    # 外层玫瑰曲线：形成漂亮的花瓣结构。
    outer_count = 120
    theta = np.linspace(0, 2 * np.pi, outer_count, endpoint=False)
    radius = 3.0 + 0.65 * np.sin(6 * theta)
    outer = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    outer += rng.normal(0.0, 0.055, outer.shape)
    points.append(outer)
    labels.extend([0] * outer_count)

    # 中层星形：和外层有几何关系，但局部密度不同。
    star_count = 90
    theta = np.linspace(0, 2 * np.pi, star_count, endpoint=False)
    radius = 1.45 + 0.38 * np.sign(np.sin(5 * theta))
    star = np.column_stack([radius * np.cos(theta + 0.18), radius * np.sin(theta + 0.18)])
    star += rng.normal(0.0, 0.045, star.shape)
    points.append(star)
    labels.extend([1] * star_count)

    # 内层螺旋：提供非线性邻域关系，t-SNE 会把相邻点慢慢拉近。
    spiral_count = 95
    theta = np.linspace(0.3, 4.8 * np.pi, spiral_count)
    radius = np.linspace(0.15, 1.25, spiral_count)
    spiral = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    spiral += rng.normal(0.0, 0.035, spiral.shape)
    points.append(spiral)
    labels.extend([2] * spiral_count)

    x2 = np.vstack(points)
    labels = np.array(labels)

    # 把二维几何图案投影到五维空间，模拟“高维数据里有低维结构”。
    # t-SNE 输入是这个五维数据，动画展示的是它被优化回二维嵌入的过程。
    x1, x2_col = x2[:, 0], x2[:, 1]
    x_high = np.column_stack(
        [
            x1,
            x2_col,
            0.45 * x1 * x2_col,
            np.sin(1.4 * x1),
            np.cos(1.2 * x2_col),
        ]
    )
    x_high += rng.normal(0.0, 0.03, x_high.shape)
    return x_high, labels
