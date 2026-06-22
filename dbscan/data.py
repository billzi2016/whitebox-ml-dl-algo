import numpy as np


def make_density_geometry(seed=73):
    """生成密度不均、形状复杂、含噪声的二维点云。"""
    rng = np.random.default_rng(seed)
    points = []

    # 半月形结构：DBSCAN 能识别非凸簇，这是它相对 K-Means 的核心展示点。
    t = np.linspace(0.12, np.pi - 0.12, 120)
    moon_top = np.column_stack([2.1 * np.cos(t) - 3.0, 1.05 * np.sin(t) + 1.05])
    moon_top += rng.normal(0.0, 0.08, moon_top.shape)
    points.append(moon_top)

    t = np.linspace(0.15, np.pi - 0.15, 115)
    moon_bottom = np.column_stack([2.0 * np.cos(t) + 0.6, -1.0 * np.sin(t) - 1.2])
    moon_bottom += rng.normal(0.0, 0.085, moon_bottom.shape)
    points.append(moon_bottom)

    # 螺旋簇：进一步强调“按密度连通”而不是按中心距离聚类。
    t = np.linspace(0.4, 3.8 * np.pi, 135)
    r = np.linspace(0.18, 1.45, len(t))
    spiral = np.column_stack([r * np.cos(t) + 4.7, r * np.sin(t) + 1.25])
    spiral += rng.normal(0.0, 0.055, spiral.shape)
    points.append(spiral)

    # 紧密小岛：展示 DBSCAN 可以发现大小不同的簇。
    island = rng.multivariate_normal(
        mean=np.array([-4.8, -2.35]),
        cov=np.array([[0.12, 0.03], [0.03, 0.16]]),
        size=70,
    )
    points.append(island)

    # 背景噪声：DBSCAN 会把密度不够的孤立点标记为 noise。
    noise = rng.uniform(low=[-6.0, -3.35], high=[6.35, 3.35], size=(32, 2))
    points.append(noise)

    x = np.vstack(points)
    rng.shuffle(x)
    return x
