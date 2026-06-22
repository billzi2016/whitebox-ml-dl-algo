from random_forest.visualizer import RandomForestAnimator


class XGBoostAnimator(RandomForestAnimator):
    """复用随机森林宽屏布局，展示 XGBoost 逐棵提升树构造。"""

    def __init__(self, booster, steps_per_frame=3, max_frames=160):
        self.forest = booster
        super().__init__(booster, steps_per_frame=steps_per_frame, max_frames=max_frames)

    def draw_frame(self):
        self._restore_current_frame()
        snapshot = self.history[self.frame_index]
        grid = self._grid_points()
        probabilities = self.forest.predict_proba(grid).reshape(self.xx.shape)

        self.ax_data.clear()
        self.ax_data.contourf(
            self.xx,
            self.yy,
            probabilities,
            levels=[0.0, 0.35, 0.5, 0.65, 1.0],
            colors=["#f5b8a8", "#f8d8cd", "#c9daf8", "#8fb2ed"],
            alpha=0.82,
        )
        self.ax_data.contour(self.xx, self.yy, probabilities, levels=[0.5], colors=["#1f1f1f"], linewidths=1.6)

        x, y = self.forest.x, self.forest.y
        self.ax_data.scatter(x[y == 0, 0], x[y == 0, 1], c="#c43d1f", s=38, edgecolor="white", linewidth=0.35, label="class 0")
        self.ax_data.scatter(x[y == 1, 0], x[y == 1, 1], c="#2859b8", marker="^", s=42, edgecolor="white", linewidth=0.35, label="class 1")

        if 0 <= snapshot["current_tree_index"] < len(self.forest.trees):
            tree = self.forest.trees[snapshot["current_tree_index"]]
            split = tree.last_split
            if split is not None:
                if split["feature"] == 0:
                    self.ax_data.axvline(split["threshold"], color="#111111", linestyle="--", linewidth=2.0, label="latest split")
                else:
                    self.ax_data.axhline(split["threshold"], color="#111111", linestyle="--", linewidth=2.0, label="latest split")

        built_trees = len(snapshot["trees"])
        node_count = sum(len(tree["nodes"]) for tree in snapshot["trees"])
        self.fig.suptitle(
            "XGBoost-style boosting: "
            f"frame {self.frame_index + 1}/{self.max_frames + 1} | "
            f"phase {snapshot['phase']} | trees {built_trees}/{self.forest.n_estimators} | "
            f"nodes {node_count} | logloss {snapshot['logloss']:.4f}",
            fontsize=14,
        )
        self.ax_data.set_title("Boosted probability boundary and latest split")
        self.ax_data.set_xlabel("feature 1")
        self.ax_data.set_ylabel("feature 2")
        self.ax_data.legend(loc="upper right", framealpha=0.9)
        self.ax_data.grid(alpha=0.14)
        self.ax_data.set_aspect("auto")
        self._draw_current_tree(snapshot)
        self.fig.canvas.draw_idle()

    def _grid_points(self):
        import numpy as np

        return np.c_[self.xx.ravel(), self.yy.ravel()]
