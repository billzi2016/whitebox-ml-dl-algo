import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


class LinearRegressionAnimator:
    """逐帧展示线性回归用梯度下降拟合直线。"""

    def __init__(self, optimizer, updates_per_frame=3, max_frames=120):
        self.optimizer = optimizer
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
        while len(self.history) <= target_index:
            for _ in range(self.updates_per_frame):
                self.optimizer.step()
            self.history.append(self.optimizer.snapshot())

    def _restore_current_frame(self):
        self.optimizer.restore(self.history[self.frame_index])

    def _sync_progress_slider(self):
        if int(self.progress_slider.val) != self.frame_index:
            self.progress_slider.eventson = False
            self.progress_slider.set_val(self.frame_index)
            self.progress_slider.eventson = True

    def draw_frame(self):
        self._restore_current_frame()
        snapshot = self.history[self.frame_index]
        x = self.optimizer.x_raw[:, 0]
        y = self.optimizer.y
        line_x = np.linspace(x.min() - 0.4, x.max() + 0.4, 200)[:, None]
        line_y = self.optimizer.predict(line_x)

        self.ax.clear()
        self.ax.scatter(x, y, c="#2859b8", s=34, alpha=0.72, edgecolor="white", linewidth=0.35, label="data")
        self.ax.plot(line_x[:, 0], line_y, color="#c43d1f", linewidth=3, label="current fit")

        predictions = self.optimizer.predict(self.optimizer.x_raw)
        for xi, yi, pi in zip(x[::6], y[::6], predictions[::6]):
            self.ax.plot([xi, xi], [yi, pi], color="#777777", alpha=0.25, linewidth=0.8)

        self.ax.set_title(
            "Linear Regression gradient descent: "
            f"frame {self.frame_index + 1}/{self.max_frames + 1} | "
            f"iteration {snapshot['iteration']} | MSE {snapshot['loss']:.4f} | "
            f"w {snapshot['w'][0]:.3f} | b {snapshot['b']:.3f}",
            fontsize=14,
        )
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.legend(loc="upper left", framealpha=0.9)
        self.ax.grid(alpha=0.16)
        self.ax.set_position(self.plot_box)
        self.ax.set_aspect("auto")
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
