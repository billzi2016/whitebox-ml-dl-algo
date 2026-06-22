import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


class PCAAnimator:
    """展示 PCA 主成分方向通过幂迭代逐步稳定的过程。"""

    def __init__(self, optimizer, steps_per_frame=1, max_frames=40):
        self.optimizer = optimizer
        self.steps_per_frame = steps_per_frame
        self.max_frames = max_frames
        self.frame_index = 0
        self.playing = False
        self.last_tick = time.monotonic()
        self.history = [self.optimizer.snapshot()]

        self.fig = plt.figure(figsize=(12, 8))
        self.ax3d = self.fig.add_axes([0.07, 0.28, 0.45, 0.6], projection="3d")
        self.ax2d = self.fig.add_axes([0.58, 0.28, 0.35, 0.6])

        self.progress_ax = self.fig.add_axes([0.13, 0.15, 0.79, 0.032])
        self.prev_ax = self.fig.add_axes([0.08, 0.06, 0.13, 0.065])
        self.next_ax = self.fig.add_axes([0.215, 0.06, 0.13, 0.065])
        self.play_ax = self.fig.add_axes([0.54, 0.065, 0.15, 0.065])
        self.speed_ax = self.fig.add_axes([0.755, 0.083, 0.19, 0.035])

        self.progress_slider = Slider(self.progress_ax, "Frame", 0, max_frames, valinit=0, valstep=1)
        self.prev_button = Button(self.prev_ax, "Previous")
        self.next_button = Button(self.next_ax, "Next")
        self.play_button = Button(self.play_ax, "Auto Play")
        self.speed_slider = Slider(self.speed_ax, "Speed", 0.2, 8.0, valinit=1.0, valstep=0.1)

        self.progress_slider.on_changed(self.go_to_frame)
        self.prev_button.on_clicked(self.previous_frame)
        self.next_button.on_clicked(self.next_frame)
        self.play_button.on_clicked(self.toggle_play)

        self.timer = self.fig.canvas.new_timer(interval=60)
        self.timer.add_callback(self.on_timer)
        self.timer.start()

    def _ensure_history_until(self, target_index):
        """补算未来帧。"""
        while len(self.history) <= target_index:
            for _ in range(self.steps_per_frame):
                self.optimizer.step()
            self.history.append(self.optimizer.snapshot())

    def _restore_current_frame(self):
        """恢复当前帧。"""
        self.optimizer.restore(self.history[self.frame_index])

    def _sync_progress_slider(self):
        """同步进度条。"""
        if int(self.progress_slider.val) != self.frame_index:
            self.progress_slider.eventson = False
            self.progress_slider.set_val(self.frame_index)
            self.progress_slider.eventson = True

    def _set_3d_bounds(self):
        """固定 3D 坐标范围，避免视角和范围跳动。"""
        x = self.optimizer.x
        mins = x.min(axis=0)
        maxs = x.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = float((maxs - mins).max() * 0.58)
        self.ax3d.set_xlim(center[0] - radius, center[0] + radius)
        self.ax3d.set_ylim(center[1] - radius, center[1] + radius)
        self.ax3d.set_zlim(center[2] - radius, center[2] + radius)

    def draw_frame(self):
        """绘制原始三维点云、当前主成分方向和二维投影。"""
        self._restore_current_frame()
        snapshot = self.history[self.frame_index]

        self.ax3d.clear()
        self.ax2d.clear()

        x = self.optimizer.x
        reconstructed = snapshot["reconstructed"]
        self.ax3d.scatter(x[:, 0], x[:, 1], x[:, 2], s=22, alpha=0.45, c="#275db3")
        self.ax3d.scatter(
            reconstructed[:, 0],
            reconstructed[:, 1],
            reconstructed[:, 2],
            s=12,
            alpha=0.35,
            c="#d84a2b",
        )

        mean = self.optimizer.mean
        scale = 2.8
        for vector, color, name in [
            (snapshot["v1"], "#111111", "PC1"),
            (snapshot["v2"], "#d84a2b", "PC2"),
        ]:
            start = mean - scale * vector
            end = mean + scale * vector
            self.ax3d.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=color,
                linewidth=3,
                label=name,
            )

        projected = snapshot["projected"]
        self.ax2d.scatter(projected[:, 0], projected[:, 1], s=25, alpha=0.68, c="#2c7f4f")
        self.ax2d.axhline(0, color="#999999", linewidth=0.8)
        self.ax2d.axvline(0, color="#999999", linewidth=0.8)

        self.fig.suptitle(
            "PCA power iteration: "
            f"frame {self.frame_index + 1}/{self.max_frames + 1} | "
            f"iteration {snapshot['iteration']} | "
            f"var1 {snapshot['var1']:.3f} | var2 {snapshot['var2']:.3f}",
            fontsize=14,
        )
        self.ax3d.set_title("3D data and current principal directions")
        self.ax2d.set_title("Projection onto current PC1 / PC2")
        self.ax3d.legend(loc="upper right")
        self.ax2d.grid(alpha=0.15)
        self.ax2d.set_xlabel("PC1 coordinate")
        self.ax2d.set_ylabel("PC2 coordinate")
        self.ax2d.set_aspect("auto")
        self._set_3d_bounds()
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
