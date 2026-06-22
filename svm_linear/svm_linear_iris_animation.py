import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from sklearn import datasets
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class IrisLinearSVMAnimator:
    """用 Matplotlib 逐帧演示 Iris 三分类线性 SVM 的训练效果。"""

    def __init__(self):
        # Iris 是一个经典三分类数据集：
        # 0=setosa, 1=versicolor, 2=virginica。
        iris = datasets.load_iris()

        # 为了能在二维 plt 坐标轴中直观看到分类区域，这里只取前两个特征：
        # sepal length 和 sepal width。真实 SVM 可以使用全部四个特征，
        # 但四维空间无法直接画成一个普通二维动态图。
        self.x = iris.data[:, :2]
        self.y = iris.target
        self.target_names = iris.target_names

        # sklearn 自带的 Iris 数据默认按类别排序。
        # 如果直接从前往后取训练样本，前几帧会只有一个类别，SVM 无法训练多分类模型。
        # 因此这里先构造一个“类别均衡”的样本顺序：每轮依次取三类各一个样本。
        self.order = self._balanced_order(self.y)

        # 每一帧代表“当前已经拿多少样本训练 SVM”。
        # 从 9 个样本开始，保证三类都至少有若干样本；之后每帧增加 3 个样本。
        self.min_train_size = 9
        self.step = 3
        self.frame_sizes = list(range(self.min_train_size, len(self.x) + 1, self.step))
        if self.frame_sizes[-1] != len(self.x):
            self.frame_sizes.append(len(self.x))

        # frame_index 控制当前显示第几帧；playing 控制是否自动播放。
        # last_tick 用于计算自动播放时距离上一帧过去了多久。
        self.frame_index = 0
        self.playing = False
        self.last_tick = time.monotonic()

        # 网格点用于绘制背景染色区域。
        # SVM 会对网格中每个点预测类别，然后 contourf 把不同类别区域染成不同颜色。
        self.xx, self.yy = self._make_grid()

        # 主图显示样本点、决策区域和分类边界。
        # 下方留出空间放“上一帧 / 自动播放 / 下一帧 / 速度”控件。
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        plt.subplots_adjust(bottom=0.24)

        # add_axes 的四个数字分别表示 [left, bottom, width, height]，
        # 使用 figure 坐标系，范围是 0 到 1。
        self.prev_ax = self.fig.add_axes([0.17, 0.08, 0.12, 0.06])
        self.play_ax = self.fig.add_axes([0.32, 0.08, 0.16, 0.06])
        self.next_ax = self.fig.add_axes([0.51, 0.08, 0.12, 0.06])
        self.speed_ax = self.fig.add_axes([0.68, 0.09, 0.22, 0.035])

        # Matplotlib 原生 widgets。这样运行脚本后就是一个 plt 交互窗口，
        # 不需要额外做网页或 GUI 框架。
        self.prev_button = Button(self.prev_ax, "Previous")
        self.play_button = Button(self.play_ax, "Auto Play")
        self.next_button = Button(self.next_ax, "Next")
        self.speed_slider = Slider(
            self.speed_ax,
            "Speed",
            # 速度含义是每秒播放多少帧：0.2 很慢，3.0 较快。
            valmin=0.2,
            valmax=3.0,
            valinit=1.0,
            valstep=0.1,
        )

        # 把按钮点击事件绑定到对应的帧控制函数。
        self.prev_button.on_clicked(self.previous_frame)
        self.play_button.on_clicked(self.toggle_play)
        self.next_button.on_clicked(self.next_frame)

        # 定时器每 60ms 检查一次是否需要切到下一帧。
        # 实际换帧速度由 speed_slider 决定，而不是固定 60ms 一帧。
        self.timer = self.fig.canvas.new_timer(interval=60)
        self.timer.add_callback(self.on_timer)
        self.timer.start()

    def _balanced_order(self, labels):
        """把按类别排列的数据改成交替出现的类别顺序。"""
        groups = [np.where(labels == class_id)[0] for class_id in np.unique(labels)]
        ordered = []
        for row in zip(*groups):
            ordered.extend(row)
        return np.array(ordered)

    def _make_grid(self):
        """生成二维平面网格，用于绘制 SVM 对整个平面的分类结果。"""
        x_min, x_max = self.x[:, 0].min() - 0.8, self.x[:, 0].max() + 0.8
        y_min, y_max = self.x[:, 1].min() - 0.8, self.x[:, 1].max() + 0.8
        return np.meshgrid(
            np.linspace(x_min, x_max, 400),
            np.linspace(y_min, y_max, 400),
        )

    def fit_model(self, train_count):
        """使用当前帧指定数量的样本训练一个线性 SVM。"""
        selected = self.order[:train_count]
        model = make_pipeline(
            # SVM 对特征尺度敏感，因此先标准化，再训练线性核 SVM。
            StandardScaler(),
            # kernel="linear" 表示线性 SVM。
            # decision_function_shape="ovr" 表示三分类时使用 one-vs-rest 策略。
            SVC(kernel="linear", decision_function_shape="ovr", C=1.0),
        )
        model.fit(self.x[selected], self.y[selected])
        return model, selected

    def draw_frame(self):
        """重绘当前帧：训练模型、预测网格类别、画背景区域和样本点。"""
        train_count = self.frame_sizes[self.frame_index]
        model, selected = self.fit_model(train_count)

        # 对二维网格中的每个点做预测，得到这个点属于哪一类。
        # reshape 回网格形状后，就可以用 contourf 给整张图染色。
        prediction = model.predict(np.c_[self.xx.ravel(), self.yy.ravel()])
        prediction = prediction.reshape(self.xx.shape)

        self.ax.clear()

        # 背景三种颜色分别代表 SVM 当前预测的三个类别区域。
        # levels 把类别 0、1、2 切成三个连续区间。
        self.ax.contourf(
            self.xx,
            self.yy,
            prediction,
            levels=[-0.5, 0.5, 1.5, 2.5],
            colors=["#f6c7b6", "#bfe3c0", "#bed4f7"],
            alpha=0.75,
        )

        # contour 画出类别区域之间的边界。
        # 在线性 SVM 中，这些边界会表现为直线或分段线性边界。
        self.ax.contour(
            self.xx,
            self.yy,
            prediction,
            levels=[0.5, 1.5],
            colors=["#7a2e1d", "#1f5e36"],
            linewidths=1.5,
            alpha=0.85,
        )

        # 每个类别使用不同 marker 和颜色。
        # 未参与当前帧训练的点用透明样式，已参与训练的点用黑边高亮。
        markers = ["o", "s", "^"]
        colors = ["#c43d1f", "#287a3e", "#2859b8"]
        for class_id, name in enumerate(self.target_names):
            class_mask = self.y == class_id

            # train_mask 标记哪些样本已经进入当前帧的训练集。
            train_mask = np.zeros_like(self.y, dtype=bool)
            train_mask[selected] = True

            # 透明点：还没有被模型看到的样本，用来对比泛化效果。
            self.ax.scatter(
                self.x[class_mask & ~train_mask, 0],
                self.x[class_mask & ~train_mask, 1],
                c=colors[class_id],
                marker=markers[class_id],
                s=42,
                alpha=0.25,
                edgecolor="none",
                label=f"{name} not trained",
            )

            # 实心黑边点：当前帧已经用于训练 SVM 的样本。
            self.ax.scatter(
                self.x[class_mask & train_mask, 0],
                self.x[class_mask & train_mask, 1],
                c=colors[class_id],
                marker=markers[class_id],
                s=72,
                alpha=0.95,
                edgecolor="black",
                linewidth=0.7,
                label=f"{name} trained",
            )

        # 标题展示当前帧编号和训练样本数量，方便逐帧观察决策区域变化。
        self.ax.set_title(
            "Linear SVM on Iris: frame "
            f"{self.frame_index + 1}/{len(self.frame_sizes)} | "
            f"trained samples: {train_count}/{len(self.x)}",
            fontsize=14,
        )
        self.ax.set_xlabel("sepal length (cm)")
        self.ax.set_ylabel("sepal width (cm)")
        self.ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
        self.ax.grid(alpha=0.18)

        # draw_idle 表示“有空时重绘”，适合交互场景，避免频繁阻塞 GUI。
        self.fig.canvas.draw_idle()

    def previous_frame(self, _event=None):
        """上一帧按钮：向前回退一帧，但不会小于第 0 帧。"""
        self.frame_index = max(0, self.frame_index - 1)
        self.draw_frame()

    def next_frame(self, _event=None):
        """下一帧按钮：向后前进一帧，但不会超过最后一帧。"""
        self.frame_index = min(len(self.frame_sizes) - 1, self.frame_index + 1)
        self.draw_frame()

    def toggle_play(self, _event=None):
        """自动播放按钮：在播放和暂停之间切换。"""
        self.playing = not self.playing
        self.play_button.label.set_text("Pause" if self.playing else "Auto Play")
        self.last_tick = time.monotonic()
        self.fig.canvas.draw_idle()

    def on_timer(self):
        """定时器回调：播放状态下按速度滑块决定是否进入下一帧。"""
        if not self.playing:
            return True

        # speed_slider.val 是每秒帧数，因此每帧间隔为 1 / speed。
        seconds_per_frame = 1.0 / self.speed_slider.val
        now = time.monotonic()
        if now - self.last_tick < seconds_per_frame:
            return True

        self.last_tick = now

        # 播放到最后一帧后自动回到第一帧，方便循环观察整个过程。
        if self.frame_index >= len(self.frame_sizes) - 1:
            self.frame_index = 0
        else:
            self.frame_index += 1
        self.draw_frame()
        return True

    def show(self):
        """绘制初始帧并打开 Matplotlib 窗口。"""
        self.draw_frame()
        plt.show()


if __name__ == "__main__":
    # 直接运行本文件即可启动交互式演示。
    IrisLinearSVMAnimator().show()
