import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from optimizers.surface import SURFACE_PRESETS, loss_numpy, make_surface_grid


class OptimizerAnimator:
    """3D surface animation comparing optimizer trajectories step by step."""

    def __init__(self, comparison, max_frames=180):
        self.comparison = comparison
        self.max_frames = max_frames
        self.frame_index = 0
        self.playing = False
        self.last_tick = time.monotonic()
        self.history = [self.comparison.snapshot()]
        self.xx, self.yy, self.zz = make_surface_grid(self.comparison.surface)

        self.fig = plt.figure(figsize=(15, 8))
        self.ax = self.fig.add_axes([0.06, 0.25, 0.72, 0.68], projection="3d")
        self.info_ax = self.fig.add_axes([0.80, 0.28, 0.17, 0.60])
        self.info_ax.axis("off")

        self.progress_ax = self.fig.add_axes([0.13, 0.15, 0.76, 0.032])
        self.prev_ax = self.fig.add_axes([0.08, 0.06, 0.12, 0.065])
        self.next_ax = self.fig.add_axes([0.205, 0.06, 0.12, 0.065])
        self.play_ax = self.fig.add_axes([0.50, 0.065, 0.15, 0.065])
        self.speed_ax = self.fig.add_axes([0.735, 0.083, 0.18, 0.035])

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
            self.comparison.step()
            self.history.append(self.comparison.snapshot())

    def _sync_progress_slider(self):
        if int(self.progress_slider.val) != self.frame_index:
            self.progress_slider.eventson = False
            self.progress_slider.set_val(self.frame_index)
            self.progress_slider.eventson = True

    def draw_frame(self):
        snapshot = self.history[self.frame_index]
        self.ax.clear()
        self.info_ax.clear()
        self.info_ax.axis("off")

        self.ax.plot_surface(self.xx, self.yy, self.zz, cmap="viridis", alpha=0.55, linewidth=0, antialiased=True)

        colors = {
            "GD": "#e53935",
            "Momentum": "#fb8c00",
            "Nesterov": "#fdd835",
            "AdaGrad": "#43a047",
            "RMSProp": "#00acc1",
            "Adam": "#1e88e5",
            "AdamW": "#8e24aa",
        }

        info_lines = [f"backend: {snapshot['backend']}", ""]
        for name, point in snapshot["points"].items():
            path = np.array([item["points"][name] for item in self.history[: self.frame_index + 1]])
            z_path = loss_numpy(path, self.comparison.surface)
            self.ax.plot(path[:, 0], path[:, 1], z_path, color=colors[name], linewidth=2.0)
            self.ax.scatter([point[0]], [point[1]], [snapshot["losses"][name]], color=colors[name], s=55, label=name)
            info_lines.append(f"{name}: {snapshot['losses'][name]: .4f}")

        surface_title = SURFACE_PRESETS[snapshot["surface"]]["title"]
        self.ax.set_title(f"{surface_title} | step {snapshot['step']}")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("loss")
        self.ax.view_init(elev=35, azim=-55)
        self.ax.legend(loc="upper left", fontsize=8)
        self.info_ax.text(0.0, 1.0, "\n".join(info_lines), va="top", family="monospace", fontsize=10)
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
