import numpy as np

from random_forest.tree import DecisionTreeBuilder


class RandomForestBuilder:
    """逐步构建随机森林：bootstrap + 随机特征 + 多树投票。"""

    def __init__(
        self,
        x,
        y,
        n_trees=9,
        max_depth=5,
        min_samples_split=8,
        max_features=1,
        seed=17,
    ):
        self.x = x
        self.y = y
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = np.random.default_rng(seed)

        self.trees = []
        self.tree_bootstrap_indices = []
        self.current_tree_index = -1
        self.phase = "start"
        self.finished = False
        self._start_next_tree()

    def _start_next_tree(self):
        """用 bootstrap 样本启动下一棵树。"""
        if len(self.trees) >= self.n_trees:
            self.finished = True
            self.phase = "finished"
            return

        bootstrap = self.rng.integers(0, len(self.x), size=len(self.x))
        tree = DecisionTreeBuilder(
            self.x[bootstrap],
            self.y[bootstrap],
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            seed=int(self.rng.integers(0, 1_000_000)),
        )
        self.trees.append(tree)
        self.tree_bootstrap_indices.append(bootstrap)
        self.current_tree_index = len(self.trees) - 1
        self.phase = "bootstrap tree"

    def step(self):
        """推进随机森林构建一步。"""
        if self.finished:
            return

        tree = self.trees[self.current_tree_index]
        if tree.finished:
            self._start_next_tree()
            return

        tree.step()
        self.phase = "split node"

    def predict_votes(self, x):
        """返回当前已存在树的平均投票概率。"""
        active_trees = [tree for tree in self.trees if len(tree.nodes) > 0]
        votes = np.zeros(len(x))
        for tree in active_trees:
            votes += tree.predict(x)
        return votes / max(len(active_trees), 1)

    def predict(self, x):
        """多数投票分类。"""
        return (self.predict_votes(x) >= 0.5).astype(int)

    def snapshot(self):
        """保存森林状态。"""
        return {
            "trees": [tree.snapshot() for tree in self.trees],
            "tree_bootstrap_indices": [idx.copy() for idx in self.tree_bootstrap_indices],
            "current_tree_index": self.current_tree_index,
            "phase": self.phase,
            "finished": self.finished,
        }

    def restore(self, snapshot):
        """恢复森林状态。"""
        self.tree_bootstrap_indices = [idx.copy() for idx in snapshot["tree_bootstrap_indices"]]
        self.current_tree_index = snapshot["current_tree_index"]
        self.phase = snapshot["phase"]
        self.finished = snapshot["finished"]

        while len(self.trees) < len(snapshot["trees"]):
            bootstrap = self.tree_bootstrap_indices[len(self.trees)]
            self.trees.append(
                DecisionTreeBuilder(
                    self.x[bootstrap],
                    self.y[bootstrap],
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    max_features=self.max_features,
                )
            )

        self.trees = self.trees[: len(snapshot["trees"])]
        for tree, tree_snapshot in zip(self.trees, snapshot["trees"]):
            tree.restore(tree_snapshot)
