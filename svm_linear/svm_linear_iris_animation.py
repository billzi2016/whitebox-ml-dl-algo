import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from sklearn import datasets


class IrisLinearSVMAnimator:
    """用 Matplotlib 逐帧演示 Iris 三分类线性 SVM 的参数优化过程。"""

    def __init__(self):
        # Iris 是一个经典三分类数据集：
        # 0=setosa, 1=versicolor, 2=virginica。
        iris = datasets.load_iris()

        # 为了能在二维 plt 坐标轴中直观看到分类区域，这里只取前两个特征：
        # sepal length 和 sepal width。真实 SVM 可以使用全部四个特征，
        # 但四维空间无法直接画成一个普通二维动态图。
        self.x_raw = iris.data[:, :2]
        self.y = iris.target
        self.target_names = iris.target_names

        # SVM 对特征尺度敏感。这里手写标准化，而不是用 sklearn 的 Pipeline，
        # 这样后面的优化过程完全由本文件控制，便于逐帧展示。
        self.x_mean = self.x_raw.mean(axis=0)
        self.x_std = self.x_raw.std(axis=0)
        self.x = (self.x_raw - self.x_mean) / self.x_std

        # 三分类线性 SVM 用 one-vs-rest 思路实现：
        # 每个类别训练一个二分类器，类别 k 的标签为 +1，其余类别为 -1。
        self.class_count = len(self.target_names)
        self.sample_count, self.feature_count = self.x.shape

        # w 和 b 是真正被优化的参数。
        # w.shape = (3, 2)，表示三个 one-vs-rest 分类器，每个分类器有两个特征权重。
        # b.shape = (3,)，表示三个分类器各自的偏置。
        rng = np.random.default_rng(7)
        self.w = rng.normal(loc=0.0, scale=0.02, size=(self.class_count, self.feature_count))
        self.b = np.zeros(self.class_count)

        # C 控制 hinge loss 惩罚强度；learning_rate 控制每次梯度下降步长。
        # updates_per_frame 表示每一帧内部做多少次完整 batch 梯度更新。
        self.c = 1.0
        self.learning_rate = 0.03
        self.updates_per_frame = 4
        self.max_frames = 160

        # 为了支持“上一帧”，每帧训练完后保存参数快照。
        # 这样回退时不是重新随机训练，而是回到当时真实的优化状态。
        self.history = []
        self._save_snapshot()

        # frame_index 控制当前显示第几帧；playing 控制是否自动播放。
        # last_tick 用于计算自动播放时距离上一帧过去了多久。
        self.frame_index = 0
        self.playing = False
        self.last_tick = time.monotonic()

        # 网格点用于绘制背景染色区域。
        # SVM 会对网格中每个点预测类别，然后 contourf 把不同类别区域染成不同颜色。
        self.xx, self.yy = self._make_grid()

        # 主图显示样本点、决策区域和分类边界。
        # 下方留出空间放进度条，以及 Previous / Next / Auto Play / Speed 控件。
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.08, right=0.96, top=0.9, bottom=0.27)

        # add_axes 的四个数字分别表示 [left, bottom, width, height]，
        # 使用 figure 坐标系，范围是 0 到 1。
        # 进度条整体下移，并给左侧 Frame 标签、右侧数字留出空间。
        # 这样“标签 + 滑条 + 数字”的整体视觉范围不会超过主图左右边界。
        self.progress_ax = self.fig.add_axes([0.13, 0.15, 0.79, 0.032])

        # Previous 和 Next 靠近，表达它们是一组逐帧按钮。
        # Auto Play 和 Speed 靠近，表达它们是一组播放控制。
        # Previous 左边和主图左边对齐；Speed 右边和主图右边对齐。
        self.prev_ax = self.fig.add_axes([0.08, 0.06, 0.13, 0.065])
        self.next_ax = self.fig.add_axes([0.215, 0.06, 0.13, 0.065])
        self.play_ax = self.fig.add_axes([0.54, 0.065, 0.15, 0.065])
        self.speed_ax = self.fig.add_axes([0.755, 0.083, 0.19, 0.035])

        # Matplotlib 原生 widgets。这样运行脚本后就是一个 plt 交互窗口，
        # 不需要额外做网页或 GUI 框架。
        self.prev_button = Button(self.prev_ax, "Previous")
        self.play_button = Button(self.play_ax, "Auto Play")
        self.next_button = Button(self.next_ax, "Next")
        self.progress_slider = Slider(
            self.progress_ax,
            "Frame",
            valmin=0,
            valmax=self.max_frames,
            valinit=0,
            valstep=1,
        )
        self.speed_slider = Slider(
            self.speed_ax,
            "Speed",
            # 速度含义是每秒播放多少帧：0.2 很慢，3.0 较快。
            valmin=0.2,
            valmax=8.0,
            valinit=1.0,
            valstep=0.1,
        )

        # 把按钮点击事件绑定到对应的帧控制函数。
        self.prev_button.on_clicked(self.previous_frame)
        self.play_button.on_clicked(self.toggle_play)
        self.next_button.on_clicked(self.next_frame)
        self.progress_slider.on_changed(self.go_to_frame)

        # 定时器每 60ms 检查一次是否需要切到下一帧。
        # 实际换帧速度由 speed_slider 决定，而不是固定 60ms 一帧。
        self.timer = self.fig.canvas.new_timer(interval=60)
        self.timer.add_callback(self.on_timer)
        self.timer.start()

    def _make_grid(self):
        """生成二维平面网格，用于绘制 SVM 对整个平面的分类结果。"""
        x_min, x_max = self.x_raw[:, 0].min() - 0.8, self.x_raw[:, 0].max() + 0.8
        y_min, y_max = self.x_raw[:, 1].min() - 0.8, self.x_raw[:, 1].max() + 0.8
        return np.meshgrid(
            np.linspace(x_min, x_max, 400),
            np.linspace(y_min, y_max, 400),
        )

    def _standardize(self, x_raw):
        """把原始二维特征变换到和训练数据相同的标准化空间。"""
        return (x_raw - self.x_mean) / self.x_std

    def _scores(self, x_raw):
        """计算三个 one-vs-rest 分类器对输入点的线性打分。"""
        x_scaled = self._standardize(x_raw)
        return x_scaled @ self.w.T + self.b

    def _loss_and_gradients(self):
        """计算当前参数下的 SVM 目标函数，以及 w、b 的 batch 梯度。"""
        grad_w = np.zeros_like(self.w)
        grad_b = np.zeros_like(self.b)
        total_hinge = 0.0
        total_violations = 0

        for class_id in range(self.class_count):
            # 当前 one-vs-rest 分类器的二分类标签。
            # 属于当前类别记为 +1，不属于当前类别记为 -1。
            y_binary = np.where(self.y == class_id, 1.0, -1.0)
            scores = self.x @ self.w[class_id] + self.b[class_id]
            margins = y_binary * scores

            # hinge loss = max(0, 1 - y * score)。
            # margins < 1 表示样本在间隔内或被分错，需要对梯度产生贡献。
            active = margins < 1.0
            hinge = np.maximum(0.0, 1.0 - margins)
            total_hinge += hinge.sum()
            total_violations += int(active.sum())

            # 目标函数是 1/2 ||w||^2 + C * sum(hinge)。
            # 对 w 的梯度：w - C * sum(y_i * x_i)，只累加 active 样本。
            # 对 b 的梯度：-C * sum(y_i)，同样只累加 active 样本。
            grad_w[class_id] = self.w[class_id] - self.c * (
                y_binary[active, None] * self.x[active]
            ).sum(axis=0)
            grad_b[class_id] = -self.c * y_binary[active].sum()

        regularization = 0.5 * np.sum(self.w * self.w)
        total_loss = regularization + self.c * total_hinge
        return total_loss, total_violations, grad_w, grad_b

    def _step_optimizer(self):
        """执行一次完整 batch 梯度下降，模拟线性 SVM 的一步优化。"""
        _loss, _violations, grad_w, grad_b = self._loss_and_gradients()

        # 除以样本数可以让梯度尺度更稳定，学习率更容易设置。
        self.w -= self.learning_rate * grad_w / self.sample_count
        self.b -= self.learning_rate * grad_b / self.sample_count

    def _save_snapshot(self):
        """保存当前参数，供上一帧/下一帧回放使用。"""
        loss, violations, _grad_w, _grad_b = self._loss_and_gradients()
        self.history.append(
            {
                "w": self.w.copy(),
                "b": self.b.copy(),
                "loss": loss,
                "violations": violations,
            }
        )

    def _restore_snapshot(self):
        """把模型参数恢复到当前 frame_index 对应的历史状态。"""
        snapshot = self.history[self.frame_index]
        self.w = snapshot["w"].copy()
        self.b = snapshot["b"].copy()

    def _ensure_history_until(self, target_index):
        """如果用户跳到未来帧，就先把中间缺失的优化快照补算出来。"""
        while len(self.history) <= target_index:
            for _ in range(self.updates_per_frame):
                self._step_optimizer()
            self._save_snapshot()

    def draw_frame(self):
        """重绘当前帧：使用当前参数预测网格类别、画背景区域和样本点。"""
        self._restore_snapshot()
        snapshot = self.history[self.frame_index]

        # 对二维网格中的每个点做预测，得到这个点属于哪一类。
        # reshape 回网格形状后，就可以用 contourf 给整张图染色。
        grid_raw = np.c_[self.xx.ravel(), self.yy.ravel()]
        prediction = np.argmax(self._scores(grid_raw), axis=1)
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

        # 每个类别使用不同 marker 和颜色。这里所有样本从第一帧开始都参与优化；
        # 帧的变化来自 w、b 参数一步步更新，而不是来自样本数量变化。
        markers = ["o", "s", "^"]
        colors = ["#c43d1f", "#287a3e", "#2859b8"]
        for class_id, name in enumerate(self.target_names):
            class_mask = self.y == class_id

            self.ax.scatter(
                self.x_raw[class_mask, 0],
                self.x_raw[class_mask, 1],
                c=colors[class_id],
                marker=markers[class_id],
                s=72,
                alpha=0.95,
                edgecolor="black",
                linewidth=0.7,
                label=name,
            )

        # 标题展示当前帧、优化步数、目标函数值和违反间隔约束的 one-vs-rest 样本数。
        self.ax.set_title(
            "Linear SVM optimization on Iris: "
            f"frame {self.frame_index + 1}/{self.max_frames + 1} | "
            f"updates {self.frame_index * self.updates_per_frame} | "
            f"loss {snapshot['loss']:.2f} | "
            f"margin violations {snapshot['violations']}",
            fontsize=14,
        )
        self.ax.set_xlabel("sepal length (cm)")
        self.ax.set_ylabel("sepal width (cm)")
        self.ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
        self.ax.grid(alpha=0.18)

        # draw_idle 表示“有空时重绘”，适合交互场景，避免频繁阻塞 GUI。
        self.fig.canvas.draw_idle()

    def _sync_progress_slider(self):
        """按钮或自动播放改变帧时，同步更新进度条位置。"""
        if int(self.progress_slider.val) != self.frame_index:
            self.progress_slider.eventson = False
            self.progress_slider.set_val(self.frame_index)
            self.progress_slider.eventson = True

    def previous_frame(self, _event=None):
        """上一帧按钮：向前回退一帧，但不会小于第 0 帧。"""
        self.frame_index = max(0, self.frame_index - 1)
        self._sync_progress_slider()
        self.draw_frame()

    def next_frame(self, _event=None):
        """下一帧按钮：向后前进一帧，但不会超过最后一帧。"""
        if self.frame_index >= self.max_frames:
            return

        next_index = self.frame_index + 1
        self._ensure_history_until(next_index)

        self.frame_index = next_index
        self._sync_progress_slider()
        self.draw_frame()

    def go_to_frame(self, value):
        """进度条：拖动到指定帧，并恢复或补算对应的优化状态。"""
        target_index = int(value)
        self._ensure_history_until(target_index)
        self.frame_index = target_index
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
        if self.frame_index >= self.max_frames:
            self.frame_index = 0
            self._sync_progress_slider()
            self.draw_frame()
        else:
            self.next_frame()
        return True

    def show(self):
        """绘制初始帧并打开 Matplotlib 窗口。"""
        self.draw_frame()
        plt.show()


if __name__ == "__main__":
    # 直接运行本文件即可启动交互式演示。
    IrisLinearSVMAnimator().show()
