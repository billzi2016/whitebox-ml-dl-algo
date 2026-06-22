import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


class RandomForestAnimator:
    """展示随机森林逐棵树、逐节点分裂和投票边界变化。"""

    def __init__(self, forest, steps_per_frame=3, max_frames=160):
        self.forest = forest
        self.steps_per_frame = steps_per_frame
        self.max_frames = max_frames
        self.frame_index = 0
        self.playing = False
        self.last_tick = time.monotonic()
        self.history = [self.forest.snapshot()]
        self.xx, self.yy = self._make_grid()

        self.fig = plt.figure(figsize=(22, 8))
        self.ax_data = self.fig.add_axes([0.045, 0.27, 0.39, 0.63])
        self.ax_tree = self.fig.add_axes([0.47, 0.27, 0.50, 0.63])

        self.progress_ax = self.fig.add_axes([0.11, 0.15, 0.78, 0.032])
        self.prev_ax = self.fig.add_axes([0.05, 0.06, 0.09, 0.065])
        self.next_ax = self.fig.add_axes([0.145, 0.06, 0.09, 0.065])
        self.play_ax = self.fig.add_axes([0.55, 0.065, 0.12, 0.065])
        self.speed_ax = self.fig.add_axes([0.755, 0.083, 0.16, 0.035])

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

    def _make_grid(self):
        x = self.forest.x
        x_min, x_max = x[:, 0].min() - 0.8, x[:, 0].max() + 0.8
        y_min, y_max = x[:, 1].min() - 0.8, x[:, 1].max() + 0.8
        return np.meshgrid(np.linspace(x_min, x_max, 350), np.linspace(y_min, y_max, 350))

    def _ensure_history_until(self, target_index):
        while len(self.history) <= target_index:
            for _ in range(self.steps_per_frame):
                self.forest.step()
            self.history.append(self.forest.snapshot())

    def _restore_current_frame(self):
        self.forest.restore(self.history[self.frame_index])

    def _sync_progress_slider(self):
        if int(self.progress_slider.val) != self.frame_index:
            self.progress_slider.eventson = False
            self.progress_slider.set_val(self.frame_index)
            self.progress_slider.eventson = True

    def _tree_node_positions(self, tree):
        """给当前树的节点计算一个稳定的二维布局。"""
        positions = {}
        leaf_cursor = 0

        def assign(node_id, depth):
            nonlocal leaf_cursor
            node = tree.nodes[node_id]
            if node["is_leaf"]:
                positions[node_id] = [leaf_cursor, -depth]
                leaf_cursor += 1
                return positions[node_id][0]

            left_x = assign(node["left"], depth + 1)
            right_x = assign(node["right"], depth + 1)
            positions[node_id] = [(left_x + right_x) / 2.0, -depth]
            return positions[node_id][0]

        assign(0, 0)
        return positions

    def _node_label(self, node, node_id):
        """生成右侧树节点文本。"""
        sample_count = len(node["indices"])
        if node["is_leaf"]:
            return f"#{node_id}\nleaf={node['prediction']}\nn={sample_count}\ngini={node['gini']:.2f}"
        feature = "x" if node["feature"] == 0 else "y"
        return (
            f"#{node_id}\n{feature} <= {node['threshold']:.2f}\n"
            f"n={sample_count}\ngini={node['gini']:.2f}"
        )

    def _draw_current_tree(self, snapshot):
        """右侧绘制当前正在构造的树。"""
        self.ax_tree.clear()
        if not snapshot["trees"]:
            self.ax_tree.set_title("Current tree structure")
            self.ax_tree.axis("off")
            return

        tree_index = max(0, snapshot["current_tree_index"])
        tree_index = min(tree_index, len(self.forest.trees) - 1)
        tree = self.forest.trees[tree_index]
        positions = self._tree_node_positions(tree)
        latest_node = None
        if tree.last_split is not None:
            latest_node = tree.last_split["node_id"]

        for node_id, node in enumerate(tree.nodes):
            if not node["is_leaf"]:
                x0, y0 = positions[node_id]
                for child_id in [node["left"], node["right"]]:
                    x1, y1 = positions[child_id]
                    self.ax_tree.plot([x0, x1], [y0, y1], color="#777777", linewidth=1.4)

        for node_id, node in enumerate(tree.nodes):
            x, y = positions[node_id]
            if node_id == latest_node:
                face = "#fff2a8"
                edge = "#111111"
                line_width = 2.2
            elif node["is_leaf"]:
                face = "#c9daf8" if node["prediction"] == 1 else "#f6c7b6"
                edge = "#333333"
                line_width = 1.2
            else:
                face = "#f1f1f1"
                edge = "#333333"
                line_width = 1.2

            self.ax_tree.text(
                x,
                y,
                self._node_label(node, node_id),
                ha="center",
                va="center",
                fontsize=7,
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": face,
                    "edgecolor": edge,
                    "linewidth": line_width,
                },
            )

        xs = [pos[0] for pos in positions.values()]
        ys = [pos[1] for pos in positions.values()]
        self.ax_tree.set_xlim(min(xs) - 1.25, max(xs) + 1.25)
        self.ax_tree.set_ylim(min(ys) - 1.05, 0.95)
        self.ax_tree.set_title(f"Current tree structure: tree {tree_index + 1}/{self.forest.n_trees}")
        self.ax_tree.axis("off")

    def draw_frame(self):
        self._restore_current_frame()
        snapshot = self.history[self.frame_index]
        grid = np.c_[self.xx.ravel(), self.yy.ravel()]
        votes = self.forest.predict_votes(grid).reshape(self.xx.shape)

        self.ax_data.clear()
        self.ax_data.contourf(
            self.xx,
            self.yy,
            votes,
            levels=[0.0, 0.35, 0.5, 0.65, 1.0],
            colors=["#f5b8a8", "#f8d8cd", "#c9daf8", "#8fb2ed"],
            alpha=0.82,
        )
        self.ax_data.contour(self.xx, self.yy, votes, levels=[0.5], colors=["#1f1f1f"], linewidths=1.6)

        x, y = self.forest.x, self.forest.y
        self.ax_data.scatter(x[y == 0, 0], x[y == 0, 1], c="#c43d1f", s=38, edgecolor="white", linewidth=0.35, label="class 0")
        self.ax_data.scatter(x[y == 1, 0], x[y == 1, 1], c="#2859b8", marker="^", s=42, edgecolor="white", linewidth=0.35, label="class 1")

        current_tree = None
        if 0 <= snapshot["current_tree_index"] < len(self.forest.trees):
            current_tree = self.forest.trees[snapshot["current_tree_index"]]
            split = current_tree.last_split
            if split is not None:
                if split["feature"] == 0:
                    self.ax_data.axvline(split["threshold"], color="#111111", linestyle="--", linewidth=2.0, label="latest split")
                else:
                    self.ax_data.axhline(split["threshold"], color="#111111", linestyle="--", linewidth=2.0, label="latest split")

        built_trees = len(snapshot["trees"])
        node_count = sum(len(tree["nodes"]) for tree in snapshot["trees"])
        self.fig.suptitle(
            "Random Forest building: "
            f"frame {self.frame_index + 1}/{self.max_frames + 1} | "
            f"phase {snapshot['phase']} | trees {built_trees}/{self.forest.n_trees} | nodes {node_count}",
            fontsize=14,
        )
        self.ax_data.set_title("Forest vote boundary and latest split")
        self.ax_data.set_xlabel("feature 1")
        self.ax_data.set_ylabel("feature 2")
        self.ax_data.legend(loc="upper right", framealpha=0.9)
        self.ax_data.grid(alpha=0.14)
        self.ax_data.set_aspect("auto")
        self._draw_current_tree(snapshot)
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
