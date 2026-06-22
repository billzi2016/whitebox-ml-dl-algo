import numpy as np
from sklearn import datasets


def load_iris_2d():
    """加载 Iris，并只取前两个特征，保证分类区域能画在二维平面上。"""
    iris = datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target
    names = iris.target_names
    return x, y, names, "RBF SVM optimization on Iris"


def make_windmill_xor():
    """生成 16 个红蓝交替的风车 XOR 点，专门用于展示非线性边界。"""
    point_count = 16
    angles = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)

    # 半径随角度起伏，让点像旋转风车叶片，而不是简单圆环。
    radius = 2.0 + 0.45 * np.sin(4.0 * angles)
    rotation = np.pi / 16.0

    x = np.column_stack(
        [
            radius * np.cos(angles + rotation),
            radius * np.sin(angles + rotation),
        ]
    )

    # 红蓝交替：0, 1, 0, 1...。这种结构单条直线分不开。
    y = np.arange(point_count) % 2
    names = np.array(["red arm", "blue arm"])
    return x, y, names, "RBF SVM optimization on Windmill XOR"


def load_dataset(name):
    """按命令行入口加载数据。"""
    if name == "iris":
        return load_iris_2d()
    if name == "xor":
        return make_windmill_xor()
    raise ValueError(f"Unknown dataset: {name}")
