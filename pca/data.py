import numpy as np


def make_correlated_3d_data(seed=29):
    """生成带明显相关结构的三维点云，适合展示 PCA 主方向。"""
    rng = np.random.default_rng(seed)

    # 先在潜在二维空间里做一个弯曲但整体有主方向的数据。
    n = 260
    t = rng.normal(0.0, 1.2, n)
    u = rng.normal(0.0, 0.38, n)
    latent = np.column_stack([t, 0.55 * t + u, 0.25 * np.sin(2.3 * t) + 0.25 * u])

    # 旋转到三维空间，让坐标轴本身不等于主成分方向。
    rotation = np.array(
        [
            [0.72, -0.48, 0.50],
            [0.56, 0.82, -0.12],
            [-0.41, 0.33, 0.85],
        ]
    )
    x = latent @ rotation.T
    x += rng.normal(0.0, 0.08, x.shape)

    # 加少量离群点，让均值中心化和主方向更容易看出作用。
    outliers = rng.multivariate_normal(
        mean=np.array([2.0, -1.9, 1.2]),
        cov=np.diag([0.08, 0.05, 0.06]),
        size=14,
    )
    return np.vstack([x, outliers])
