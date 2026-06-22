import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


class RBFSVMAnimator:
    """逐帧展示 RBF 特征空间中 SVM 参数优化过程。"""

    def __init__(
        self,
        optimizer,
        names,
        title,
        updates_per_frame=4,
        max_frames=160,
    ):
        self.optimizer = optimizer
        self.names = names
        self.title = title
        self.updates_per_frame = updates_per_frame
        self.max_frames = max_frames

        self.frame_index = 0
        self.playing = False
        self.last_tick = time.monotonic()
        self.history = [self.optimizer.snapshot()]

        self.xx, self.yy = self._make_grid()

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.08, right=0.96, top=0.9, bottom=0.27)

        # 布局和 linear 版本保持一致，方便两个 demo 对照。
        self.progress_ax = self.fig.add_axes([0.13, 0.15, 0.79, 0.032])
        self.prev_ax = self.fig.add_axes([0.08, 0.06, 0.13, 0.065])
        self.next_ax = self.fig.add_axes([0.215, 0.06, 0.13, 0.065])
        self.play_ax = self.fig.add_axes([0.54, 0.065, 0.15, 0.065])
        self.speed_ax = self.fig.add_axes([0.755, 0.083, 0.19, 0.035])

        self.progress_slider = Slider(
            self.progress_ax,
            "Frame",
            valmin=0,
            valmax=self.max_frames,
            valinit=0,
            valstep=1,
        )
        self.prev_button = Button(self.prev_ax, "Previous")
        self.next_button = Button(self.next_ax, "Next")
        self.play_button = Button(self.play_ax, "Auto Play")
        self.speed_slider = Slider(
            self.speed_ax,
            "Speed",
            valmin=0.2,
            valmax=8.0,
            valinit=1.0,
            valstep=0.1,
        )

        self.progress_slider.on_changed(self.go_to_frame)
        self.prev_button.on_clicked(self.previous_frame)
        self.next_button.on_clicked(self.next_frame)
        self.play_button.on_clicked(self.toggle_play)

        self.timer = self.fig.canvas.new_timer(interval=60)
        self.timer.add_callback(self.on_timer)
        self.timer.start()

    def _make_grid(self):
        """生成二维网格，用来画整张背景分类区域。"""
        x = self.optimizer.x_raw
        padding = 0.8
        x_min, x_max = x[:, 0].min() - padding, x[:, 0].max() + padding
        y_min, y_max = x[:, 1].min() - padding, x[:, 1].max() + padding
        return np.meshgrid(
            np.linspace(x_min, x_max, 450),
            np.linspace(y_min, y_max, 450),
        )

    def _ensure_history_until(self, target_index):
        """补算未来帧，保证进度条可以直接跳转。"""
        while len(self.history) <= target_index:
            for _ in range(self.updates_per_frame):
                self.optimizer.step()
            self.history.append(self.optimizer.snapshot())

    def _restore_current_frame(self):
        """恢复当前帧对应的模型参数。"""
        self.optimizer.restore(self.history[self.frame_index])

    def _sync_progress_slider(self):
        """按钮或自动播放改变帧时，同步进度条。"""
        if int(self.progress_slider.val) != self.frame_index:
            self.progress_slider.eventson = False
            self.progress_slider.set_val(self.frame_index)
            self.progress_slider.eventson = True

    def draw_frame(self):
        """根据当前 RBF SVM 参数重画分类区域。"""
        self._restore_current_frame()
        snapshot = self.history[self.frame_index]

        grid = np.c_[self.xx.ravel(), self.yy.ravel()]
        prediction = self.optimizer.predict(grid).reshape(self.xx.shape)

        self.ax.clear()
        class_count = len(self.names)

        if class_count == 2:
            fill_colors = ["#f7b6a6", "#a9c7f5"]
            line_colors = ["#7a2e1d"]
            levels = [-0.5, 0.5, 1.5]
            contour_levels = [0.5]
            markers = ["o", "^"]
            point_colors = ["#c43d1f", "#2859b8"]
        else:
            fill_colors = ["#f6c7b6", "#bfe3c0", "#bed4f7"]
            line_colors = ["#7a2e1d", "#1f5e36"]
            levels = [-0.5, 0.5, 1.5, 2.5]
            contour_levels = [0.5, 1.5]
            markers = ["o", "s", "^"]
            point_colors = ["#c43d1f", "#287a3e", "#2859b8"]

        self.ax.contourf(
            self.xx,
            self.yy,
            prediction,
            levels=levels,
            colors=fill_colors,
            alpha=0.75,
        )
        self.ax.contour(
            self.xx,
            self.yy,
            prediction,
            levels=contour_levels,
            colors=line_colors,
            linewidths=1.7,
            alpha=0.9,
        )

        for class_id, name in enumerate(self.names):
            mask = self.optimizer.y == class_id
            self.ax.scatter(
                self.optimizer.x_raw[mask, 0],
                self.optimizer.x_raw[mask, 1],
                c=point_colors[class_id],
                marker=markers[class_id],
                s=90,
                edgecolor="black",
                linewidth=0.8,
                label=name,
            )

        self.ax.set_title(
            f"{self.title}: frame {self.frame_index + 1}/{self.max_frames + 1} | "
            f"updates {self.frame_index * self.updates_per_frame} | "
            f"loss {snapshot['loss']:.2f} | "
            f"margin violations {snapshot['violations']}",
            fontsize=14,
        )
        self.ax.set_xlabel("feature 1")
        self.ax.set_ylabel("feature 2")
        self.ax.legend(loc="upper right", framealpha=0.92)
        self.ax.grid(alpha=0.18)
        self.ax.set_aspect("equal" if class_count == 2 else "auto")
        self.fig.canvas.draw_idle()

    def previous_frame(self, _event=None):
        self.frame_index = max(0, self.frame_index - 1)
        self._sync_progress_slider()
        self.draw_frame()

    def next_frame(self, _event=None):
        if self.frame_index >= self.max_frames:
            return
        self.frame_index += 1
        self._ensure_history_until(self.frame_index)
        self._sync_progress_slider()
        self.draw_frame()

    def go_to_frame(self, value):
        self.frame_index = int(value)
        self._ensure_history_until(self.frame_index)
        self.draw_frame()

    def toggle_play(self, _event=None):
        self.playing = not self.playing
        self.play_button.label.set_text("Pause" if self.playing else "Auto Play")
        self.last_tick = time.monotonic()
        self.fig.canvas.draw_idle()

    def on_timer(self):
        if not self.playing:
            return True

        seconds_per_frame = 1.0 / self.speed_slider.val
        now = time.monotonic()
        if now - self.last_tick < seconds_per_frame:
            return True

        self.last_tick = now
        if self.frame_index >= self.max_frames:
            self.frame_index = 0
            self._sync_progress_slider()
            self.draw_frame()
        else:
            self.next_frame()
        return True

    def show(self):
        self.draw_frame()
        plt.show()
