import time

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider


class TSNEAnimator:
    """逐帧展示 t-SNE 点云被吸引和排斥拉扯的过程。"""

    def __init__(self, optimizer, labels, updates_per_frame=5, max_frames=220):
        self.optimizer = optimizer
        self.labels = labels
        self.updates_per_frame = updates_per_frame
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
        """补算未来帧。"""
        while len(self.history) <= target_index:
            for _ in range(self.updates_per_frame):
                self.optimizer.step()
            self.history.append(self.optimizer.snapshot())

    def _set_current_bounds(self, y):
        """按当前点云最大边界设置坐标范围，显示比例保持 1:1。"""
        x_min, x_max = float(y[:, 0].min()), float(y[:, 0].max())
        y_min, y_max = float(y[:, 1].min()), float(y[:, 1].max())

        x_span = max(x_max - x_min, 0.7)
        y_span = max(y_max - y_min, 0.7)
        x_pad = x_span * 0.12
        y_pad = y_span * 0.12

        self.ax.set_xlim(x_min - x_pad, x_max + x_pad)
        self.ax.set_ylim(y_min - y_pad, y_max + y_pad)

    def _restore_current_frame(self):
        """恢复当前帧坐标。"""
        self.optimizer.restore(self.history[self.frame_index])

    def _sync_progress_slider(self):
        """同步进度条。"""
        if int(self.progress_slider.val) != self.frame_index:
            self.progress_slider.eventson = False
            self.progress_slider.set_val(self.frame_index)
            self.progress_slider.eventson = True

    def draw_frame(self):
        """绘制当前 t-SNE embedding。"""
        self._restore_current_frame()
        snapshot = self.history[self.frame_index]
        y = snapshot["y"]

        self.ax.clear()
        colors = ["#d84a2b", "#2c7f4f", "#275db3"]
        markers = ["o", "s", "^"]

        for label in sorted(set(self.labels)):
            mask = self.labels == label
            self.ax.scatter(
                y[mask, 0],
                y[mask, 1],
                c=colors[label % len(colors)],
                marker=markers[label % len(markers)],
                s=36,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.35,
                label=f"geometry {label}",
            )

        self.ax.set_title(
            "t-SNE optimization: "
            f"frame {self.frame_index + 1}/{self.max_frames + 1} | "
            f"iteration {snapshot['iteration']} | "
            f"KL {snapshot['loss']:.4f}",
            fontsize=14,
        )
        self.ax.set_xlabel("embedding 1")
        self.ax.set_ylabel("embedding 2")
        self.ax.legend(loc="upper right", framealpha=0.92)
        self.ax.grid(alpha=0.12)
        self.ax.set_position(self.plot_box)
        self.ax.set_aspect("auto")
        self._set_current_bounds(y)
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
