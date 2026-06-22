import numpy as np


def make_boosting_classification_data(seed=211):
    """生成适合展示梯度提升逐步修正残差的二维二分类数据。"""
    rng = np.random.default_rng(seed)
    points = []
    labels = []

    theta = rng.uniform(0, 2 * np.pi, 190)
    radius = rng.normal(1.65, 0.22, len(theta))
    inner = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    points.append(inner)
    labels.extend([1] * len(inner))

    theta = rng.uniform(0, 2 * np.pi, 210)
    radius = rng.normal(3.05, 0.32, len(theta))
    outer = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    points.append(outer)
    labels.extend([0] * len(outer))

    # 加两块偏置区域，避免数据只是简单同心圆。
    blob_a = rng.multivariate_normal([-2.4, 2.1], [[0.25, 0.1], [0.1, 0.45]], 70)
    blob_b = rng.multivariate_normal([2.55, -2.0], [[0.35, -0.12], [-0.12, 0.28]], 65)
    points.extend([blob_a, blob_b])
    labels.extend([1] * len(blob_a))
    labels.extend([1] * len(blob_b))

    noise = rng.uniform(low=[-4.1, -3.8], high=[4.1, 3.8], size=(45, 2))
    noise_label = ((noise[:, 0] ** 2 + noise[:, 1] ** 2) < 5.6).astype(int)
    points.append(noise)
    labels.extend(noise_label.tolist())

    x = np.vstack(points)
    y = np.array(labels)
    order = rng.permutation(len(x))
    return x[order], y[order]
