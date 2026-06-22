import numpy as np


def make_forest_classification_data(seed=101):
    """生成适合展示随机森林非线性边界的二维分类数据。"""
    rng = np.random.default_rng(seed)
    points = []
    labels = []

    # 类 0：两个斜向椭圆岛。
    for mean, cov, count in [
        ([-2.6, 1.4], [[0.42, 0.25], [0.25, 0.55]], 90),
        ([2.2, -1.45], [[0.5, -0.28], [-0.28, 0.45]], 85),
    ]:
        points.append(rng.multivariate_normal(mean, cov, count))
        labels.extend([0] * count)

    # 类 1：环形和波浪带，故意让单棵浅树很难一次切好。
    theta = rng.uniform(0, 2 * np.pi, 150)
    radius = rng.normal(1.85, 0.18, len(theta))
    ring = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    ring += rng.normal(0, 0.05, ring.shape)
    points.append(ring)
    labels.extend([1] * len(ring))

    t = rng.uniform(-3.2, 3.2, 105)
    wave = np.column_stack([t, 0.72 * np.sin(1.7 * t) + 2.35])
    wave += rng.normal(0, 0.16, wave.shape)
    points.append(wave)
    labels.extend([1] * len(wave))

    # 少量噪声点，让投票边界更有随机森林特征。
    noise = rng.uniform(low=[-4.2, -3.3], high=[4.2, 3.4], size=(45, 2))
    noise_label = ((noise[:, 0] * noise[:, 1] + 0.35 * noise[:, 0]) > 0).astype(int)
    points.append(noise)
    labels.extend(noise_label.tolist())

    x = np.vstack(points)
    y = np.array(labels)
    order = rng.permutation(len(x))
    return x[order], y[order]
