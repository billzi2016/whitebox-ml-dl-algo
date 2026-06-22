import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


class KMeansAnimator:
    """用 Matplotlib 逐帧展示 K-Means 的 assignment/update 过程。"""

    def __init__(self, optimizer, max_frames=60):
        self.optimizer = optimizer
        self.max_frames = max_frames
        self.frame_index = 0
        self.playing = False
        self.last_tick = time.monotonic()
        self.history = [self.optimizer.snapshot()]

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.08, right=0.96, top=0.9, bottom=0.27)

        # 控件布局和 SVM demo 保持一致，方便横向比较不同算法。
        self.progress_ax = self.fig.add_axes([0.13, 0.15, 0.79, 0.032])
        self.prev_ax = self.fig.add_axes([0.08, 0.06, 0.13, 0.065])
        self.next_ax = self.fig.add_axes([0.215, 0.06, 0.13, 0.065])
        self.play_ax = self.fig.add_axes([0.54, 0.065, 0.15, 0.065])
        self.speed_ax = self.fig.add_axes([0.755, 0.083, 0.19, 0.035])

        self.progress_slider = Slider(
            self.progress_ax,
            "Frame",
            valmin=0,
            valmax=max_frames,
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

    def _ensure_history_until(self, target_index):
        """补算历史，让进度条可以直接跳到未来帧。"""
        while len(self.history) <= target_index:
            self.optimizer.step()
            self.history.append(self.optimizer.snapshot())

    def _restore_current_frame(self):
        """恢复当前帧对应的 K-Means 状态。"""
        self.optimizer.restore(self.history[self.frame_index])

    def _sync_progress_slider(self):
        """按钮或自动播放改变帧时，同步进度条。"""
        if int(self.progress_slider.val) != self.frame_index:
            self.progress_slider.eventson = False
            self.progress_slider.set_val(self.frame_index)
            self.progress_slider.eventson = True

    def draw_frame(self):
        """画出当前簇分配、中心位置和中心移动轨迹。"""
        self._restore_current_frame()
        snapshot = self.history[self.frame_index]

        self.ax.clear()
        colors = ["#c43d1f", "#287a3e", "#2859b8", "#c58a16", "#7c3fb0", "#008c8c"]

        for cluster_id in range(self.optimizer.k):
            mask = snapshot["labels"] == cluster_id
            self.ax.scatter(
                self.optimizer.x[mask, 0],
                self.optimizer.x[mask, 1],
                c=colors[cluster_id % len(colors)],
                s=38,
                alpha=0.72,
                edgecolor="none",
                label=f"cluster {cluster_id}",
            )

        centers = snapshot["centers"]
        self.ax.scatter(
            centers[:, 0],
            centers[:, 1],
            c="black",
            marker="X",
            s=220,
            edgecolor="white",
            linewidth=1.4,
            label="centers",
        )

        # 画中心历史轨迹，帮助观察 update step 如何移动中心。
        for cluster_id in range(self.optimizer.k):
            path = np.array(
                [
                    item["centers"][cluster_id]
                    for item in self.history[: self.frame_index + 1]
                ]
            )
            self.ax.plot(
                path[:, 0],
                path[:, 1],
                color="black",
                alpha=0.35,
                linewidth=1.2,
            )

        self.ax.set_title(
            "K-Means optimization: "
            f"frame {self.frame_index + 1}/{self.max_frames + 1} | "
            f"iteration {snapshot['iteration']} | "
            f"phase {snapshot['phase']} | "
            f"SSE {snapshot['sse']:.2f}",
            fontsize=14,
        )
        self.ax.set_xlabel("feature 1")
        self.ax.set_ylabel("feature 2")
        self.ax.legend(loc="upper right", framealpha=0.92)
        self.ax.grid(alpha=0.18)
        self.ax.set_aspect("equal")
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
