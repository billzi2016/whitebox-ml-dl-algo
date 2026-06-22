import numpy as np


def make_complex_gaussian_data(seed=23):
    """生成一个有重叠、有不同形状、有噪声点的复杂二维高斯混合数据集。"""
    rng = np.random.default_rng(seed)

    # 每个簇故意设置不同的协方差矩阵：
    # 有的拉长、有的倾斜、有的比较紧，避免数据看起来像教科书里的完美圆团。
    components = [
        {
            "mean": np.array([-4.0, 1.8]),
            "cov": np.array([[0.65, 0.45], [0.45, 0.85]]),
            "count": 90,
        },
        {
            "mean": np.array([-1.0, -1.7]),
            "cov": np.array([[1.15, -0.65], [-0.65, 0.75]]),
            "count": 110,
        },
        {
            "mean": np.array([2.1, 1.5]),
            "cov": np.array([[0.9, 0.1], [0.1, 0.35]]),
            "count": 85,
        },
        {
            "mean": np.array([4.3, -1.3]),
            "cov": np.array([[0.45, 0.28], [0.28, 1.0]]),
            "count": 75,
        },
    ]

    clouds = [
        rng.multivariate_normal(item["mean"], item["cov"], item["count"])
        for item in components
    ]

    # 加一点均匀噪声点，让 K-Means 的簇边界和中心移动更有观察价值。
    noise = rng.uniform(low=[-5.8, -3.8], high=[5.8, 3.8], size=(35, 2))
    x = np.vstack([*clouds, noise])
    rng.shuffle(x)
    return x
