import numpy as np


def make_regression_data(seed=501):
    """生成带噪声、离群点和非均匀采样的一维回归数据。"""
    rng = np.random.default_rng(seed)
    x_main = rng.uniform(-3.2, 3.4, 120)
    x_dense = rng.normal(0.8, 0.55, 55)
    x = np.concatenate([x_main, x_dense])

    y = 1.35 * x - 0.75 + rng.normal(0.0, 0.55, len(x))

    # 少量离群点用于展示 MSE 对异常值比较敏感。
    outlier_x = np.array([-2.8, -1.9, 2.4, 3.1])
    outlier_y = np.array([2.7, -4.1, 4.8, 1.0])
    x = np.concatenate([x, outlier_x])
    y = np.concatenate([y, outlier_y])
    return x[:, None], y
