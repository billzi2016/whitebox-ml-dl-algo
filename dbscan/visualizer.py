import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.widgets import Button, Slider


class DBSCANAnimator:
    """逐帧展示 DBSCAN 如何按密度连通扩展簇。"""

    def __init__(self, optimizer, steps_per_frame=5, max_frames=180):
        self.optimizer = optimizer
        self.steps_per_frame = steps_per_frame
        self.max_frames = max_frames
        self.frame_index = 0
        self.playing = False
        self.last_tick = time.monotonic()
        self.history = [self.optimizer.snapshot()]

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.08, right=0.96, top=0.9, bottom=0.27)
        self.plot_box = [0.08, 0.27, 0.88, 0.63]

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
        """恢复当前帧 DBSCAN 状态。"""
        self.optimizer.restore(self.history[self.frame_index])

    def _sync_progress_slider(self):
        """同步进度条。"""
        if int(self.progress_slider.val) != self.frame_index:
            self.progress_slider.eventson = False
            self.progress_slider.set_val(self.frame_index)
            self.progress_slider.eventson = True

    def _set_bounds(self):
        """固定 axes 位置，只按数据范围设置坐标。"""
        x = self.optimizer.x
        x_min, x_max = float(x[:, 0].min()), float(x[:, 0].max())
        y_min, y_max = float(x[:, 1].min()), float(x[:, 1].max())
        self.ax.set_xlim(x_min - 0.6, x_max + 0.6)
        self.ax.set_ylim(y_min - 0.6, y_max + 0.6)

    def draw_frame(self):
        """绘制当前簇扩展状态。"""
        self._restore_current_frame()
        snapshot = self.history[self.frame_index]
        labels = snapshot["labels"]

        self.ax.clear()
        colors = ["#d84a2b", "#2c7f4f", "#275db3", "#c58a16", "#7c3fb0", "#008c8c"]

        unvisited = labels == self.optimizer.UNVISITED
        noise = labels == self.optimizer.NOISE
        self.ax.scatter(
            self.optimizer.x[unvisited, 0],
            self.optimizer.x[unvisited, 1],
            c="#c7c7c7",
            s=30,
            alpha=0.45,
            label="unvisited",
        )
        self.ax.scatter(
            self.optimizer.x[noise, 0],
            self.optimizer.x[noise, 1],
            c="#222222",
            marker="x",
            s=48,
            alpha=0.85,
            label="noise",
        )

        for cluster_id in sorted(set(labels[labels >= 0])):
            mask = labels == cluster_id
            self.ax.scatter(
                self.optimizer.x[mask, 0],
                self.optimizer.x[mask, 1],
                c=colors[cluster_id % len(colors)],
                s=42,
                alpha=0.78,
                edgecolor="white",
                linewidth=0.35,
                label=f"cluster {cluster_id}",
            )

        core = self.optimizer.core_points & (labels != self.optimizer.UNVISITED)
        self.ax.scatter(
            self.optimizer.x[core, 0],
            self.optimizer.x[core, 1],
            facecolors="none",
            edgecolors="black",
            s=92,
            linewidth=0.75,
            label="core points",
        )

        current = snapshot["current_point"]
        if current is not None:
            x, y = self.optimizer.x[current]
            self.ax.scatter([x], [y], c="#ffeb3b", edgecolor="black", s=170, zorder=6, label="current")
            self.ax.add_patch(
                Circle((x, y), self.optimizer.eps, fill=False, color="#111111", linewidth=1.4, alpha=0.8)
            )

        self.ax.set_title(
            "DBSCAN expansion: "
            f"frame {self.frame_index + 1}/{self.max_frames + 1} | "
            f"phase {snapshot['phase']} | "
            f"clusters {snapshot['cluster_id']} | "
            f"assigned {snapshot['assigned']} | noise {snapshot['noise']}",
            fontsize=14,
        )
        self.ax.set_xlabel("feature 1")
        self.ax.set_ylabel("feature 2")
        self.ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
        self.ax.grid(alpha=0.14)
        self.ax.set_position(self.plot_box)
        self.ax.set_aspect("auto")
        self._set_bounds()
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
