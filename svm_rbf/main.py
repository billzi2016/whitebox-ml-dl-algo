import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_iris_data():
    """加载 Iris 数据，并取前两个特征用于二维 RBF SVM 可视化。"""
    # 这个入口的意图：和 svm_linear 里的 Iris 线性 SVM 做对照。
    # 同样只取前两个特征，区别是这里换成 RBF 核，让用户看到非线性核
    # 会怎样弯曲决策边界，而不是只能画直线或分段线性边界。
    iris = datasets.load_iris()

    # 只取 sepal length / sepal width，是为了保证结果能直接画成二维图。
    # 如果取四个特征，模型当然可以训练，但决策区域无法用一张普通平面图完整表示。
    x = iris.data[:, :2]
    y = iris.target
    names = iris.target_names
    return x, y, names, "RBF SVM on Iris"


def make_windmill_xor_data():
    """生成 16 个红蓝交替的旋转风车点，用来展示 RBF 的非线性分类能力。"""
    # 这个入口的意图：专门构造一个线性 SVM 很难处理、但 RBF SVM 很适合处理的数据。
    # 16 个点按角度围成一圈，并且红蓝标签交替出现，视觉上像一个大风车。
    # 这样的点没有办法用一条直线分开，因为任意相邻方向上的类别都在交替变化。
    point_count = 16

    # endpoint=False 表示最后一个角度不重复 2π，避免第一个点和最后一个点重合。
    angles = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)

    # 故意让半径随角度轻微起伏，点不会落在完美圆上，视觉上更像旋转风车。
    # 标签按角度交替：红、蓝、红、蓝……这类数据无法被单条直线分开。
    # sin(4θ) 会制造 4 个方向上的半径起伏，让点位更像旋转叶片，而不是机械圆环。
    radius = 2.0 + 0.45 * np.sin(4.0 * angles)

    # 整体旋转一点角度，避免点刚好压在 x/y 坐标轴上，让图更容易看。
    rotation = np.pi / 16.0

    # 把极坐标 (radius, angle) 转成二维平面坐标 (x, y)。
    x = np.column_stack(
        [
            radius * np.cos(angles + rotation),
            radius * np.sin(angles + rotation),
        ]
    )

    # 0,1,0,1... 交替标注类别，对应红蓝交替的 XOR / 风车结构。
    # 这里叫 XOR 是因为它表达的是“空间上交错、不能线性分割”的核心现象，
    # 不是只在四个象限放 4 个点的最小 XOR 玩具例子。
    y = np.arange(point_count) % 2
    names = np.array(["red arm", "blue arm"])
    return x, y, names, "RBF SVM on Windmill XOR"


def make_grid(x, padding=0.8, grid_size=500):
    """围绕样本点生成二维网格，用来给整张背景区域染色。"""
    # 绘制分类区域的基本做法：
    # 1. 在样本点外面扩出一圈 padding。
    # 2. 在这个矩形范围内生成很多密集网格点。
    # 3. 让模型预测每个网格点的类别。
    # 4. 用 contourf 把预测结果染成背景颜色。
    x_min, x_max = x[:, 0].min() - padding, x[:, 0].max() + padding
    y_min, y_max = x[:, 1].min() - padding, x[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )
    return xx, yy


def fit_rbf_svm(x, y, gamma, c):
    """训练 RBF 核 SVM；标准化放进 Pipeline，避免特征尺度影响核距离。"""
    # RBF 核函数可以写成：
    # K(x, z) = exp(-gamma * ||x - z||^2)
    #
    # 它的核心意图是：两个点越近，相似度越高；距离越远，相似度指数级衰减。
    # 所以 RBF SVM 可以围绕局部样本弯曲边界，适合处理风车 XOR 这种非线性结构。
    #
    # gamma 控制“局部影响范围”：
    # - gamma 小：每个点影响范围大，边界更平滑。
    # - gamma 大：每个点影响范围小，边界更容易弯曲，也更容易过拟合。
    #
    # C 控制“分错惩罚”：
    # - C 小：允许更多训练误差，边界更保守。
    # - C 大：更努力分对训练点，边界可能更复杂。
    #
    # 这里使用 sklearn 的 SVC，因为 RBF SVM 的教学重点是核映射后的非线性边界；
    # 如果要逐帧展示对偶优化，需要实现 SMO/二次规划细节，复杂度明显高于线性版本。
    model = make_pipeline(
        # RBF 依赖欧氏距离 ||x-z||，特征尺度会直接影响“距离”。
        # 因此必须标准化，否则某个量纲大的特征会支配核函数。
        StandardScaler(),
        SVC(kernel="rbf", gamma=gamma, C=c, decision_function_shape="ovr"),
    )
    model.fit(x, y)
    return model


def plot_decision_regions(x, y, names, title, gamma, c):
    """绘制 RBF SVM 的预测区域、边界和样本点。"""
    # 先训练 RBF SVM，再用训练好的模型预测二维网格。
    # 背景颜色展示“模型认为平面上每个位置属于哪一类”。
    model = fit_rbf_svm(x, y, gamma=gamma, c=c)
    xx, yy = make_grid(x)
    grid = np.c_[xx.ravel(), yy.ravel()]
    prediction = model.predict(grid).reshape(xx.shape)

    class_count = len(names)
    fig, ax = plt.subplots(figsize=(11, 8))

    # 二分类 XOR 和三分类 Iris 的颜色、marker、contour levels 不一样，
    # 所以这里按类别数量分开设置，避免绘图参数混在一起变得难读。
    if class_count == 2:
        # 二分类时背景只有红/蓝两块，边界只需要画 0 和 1 之间的一条等高线。
        fill_colors = ["#f7b6a6", "#a9c7f5"]
        line_colors = ["#7a2e1d"]
        levels = [-0.5, 0.5, 1.5]
        contour_levels = [0.5]
        markers = ["o", "^"]
        point_colors = ["#c43d1f", "#2859b8"]
    else:
        # Iris 是三分类，因此需要三种背景色和两条类别分界等高线。
        fill_colors = ["#f6c7b6", "#bfe3c0", "#bed4f7"]
        line_colors = ["#7a2e1d", "#1f5e36"]
        levels = [-0.5, 0.5, 1.5, 2.5]
        contour_levels = [0.5, 1.5]
        markers = ["o", "s", "^"]
        point_colors = ["#c43d1f", "#287a3e", "#2859b8"]

    # contourf 负责填充分类区域；颜色来自模型对每个网格点的预测类别。
    ax.contourf(xx, yy, prediction, levels=levels, colors=fill_colors, alpha=0.75)

    # contour 只画类别边界，让用户能清楚看到 RBF SVM 弯出来的决策曲线。
    ax.contour(
        xx,
        yy,
        prediction,
        levels=contour_levels,
        colors=line_colors,
        linewidths=1.7,
        alpha=0.9,
    )

    # 再把真实样本点盖在背景上。
    # 黑色边框是为了让点在浅色背景上更清楚。
    for class_id, name in enumerate(names):
        mask = y == class_id
        ax.scatter(
            x[mask, 0],
            x[mask, 1],
            c=point_colors[class_id],
            marker=markers[class_id],
            s=95,
            edgecolor="black",
            linewidth=0.8,
            label=name,
        )

    # 标题里保留 gamma 和 C，是为了方便用户调整参数后直接比较边界变化。
    ax.set_title(f"{title} | gamma={gamma} | C={c}", fontsize=15)
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.legend(loc="upper right", framealpha=0.92)
    ax.grid(alpha=0.18)

    # XOR 风车点的横纵轴同量纲，强制 equal 可以避免图形被拉伸变形。
    # Iris 两个特征本来就是不同范围，使用 auto 更接近常规散点图展示。
    ax.set_aspect("equal" if class_count == 2 else "auto")
    plt.tight_layout()
    plt.show()


def parse_args():
    # main.py 提供两个入口：
    # python main.py iris
    # python main.py xor
    #
    # gamma 和 C 也暴露出来，便于演示 RBF SVM 参数对边界复杂度的影响。
    parser = argparse.ArgumentParser(
        description="RBF SVM visualization with Iris and windmill XOR entrances."
    )
    parser.add_argument(
        "dataset",
        choices=["iris", "xor"],
        help="Choose iris for 3-class Iris, or xor for 16 alternating windmill points.",
    )
    parser.add_argument("--gamma", default="scale", help="RBF gamma, for example scale, auto, 0.5, 2.0.")
    parser.add_argument("--c", type=float, default=10.0, help="SVM C regularization strength.")
    return parser.parse_args()


def main():
    args = parse_args()

    # argparse 收到的 gamma 默认是字符串。
    # sklearn 允许 "scale"/"auto"，也允许浮点数，所以这里做一次兼容转换。
    gamma = float(args.gamma) if args.gamma not in {"scale", "auto"} else args.gamma

    # 根据命令行入口选择数据集：
    # iris 展示三分类 RBF 边界；xor 展示非线性二分类风车边界。
    if args.dataset == "iris":
        x, y, names, title = make_iris_data()
    else:
        x, y, names, title = make_windmill_xor_data()

    plot_decision_regions(x, y, names, title, gamma=gamma, c=args.c)


if __name__ == "__main__":
    main()
