import numpy as np

from xgboost.tree import BoostingTreeBuilder


class XGBoostBuilder:
    """手写二分类 XGBoost 风格的梯度提升树过程。"""

    def __init__(
        self,
        x,
        y,
        n_estimators=12,
        max_depth=3,
        learning_rate=0.35,
        reg_lambda=1.0,
    ):
        self.x = x
        self.y = y
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

        positive_rate = np.clip(y.mean(), 1e-4, 1 - 1e-4)
        self.base_score = float(np.log(positive_rate / (1.0 - positive_rate)))
        self.logits = np.full(len(x), self.base_score)
        self.trees = []
        self.current_tree_index = -1
        self.phase = "start"
        self.finished = False
        self._start_next_tree()

    def _sigmoid(self, values):
        """把 logit 转成概率。"""
        return 1.0 / (1.0 + np.exp(-values))

    def _gradients_hessians(self):
        """logloss 的一阶和二阶导数。"""
        p = self._sigmoid(self.logits)
        gradients = p - self.y
        hessians = p * (1.0 - p)
        return gradients, hessians

    def _start_next_tree(self):
        """基于当前预测残差启动下一棵提升树。"""
        if len(self.trees) >= self.n_estimators:
            self.finished = True
            self.phase = "finished"
            return

        gradients, hessians = self._gradients_hessians()
        tree = BoostingTreeBuilder(
            self.x,
            gradients,
            hessians,
            max_depth=self.max_depth,
            reg_lambda=self.reg_lambda,
        )
        self.trees.append(tree)
        self.current_tree_index = len(self.trees) - 1
        self.phase = "start tree"

    def _commit_current_tree(self):
        """当前树构造完成后，把它的输出加到全局 logits。"""
        tree = self.trees[self.current_tree_index]
        self.logits += self.learning_rate * tree.predict(self.x)

    def step(self):
        """推进 XGBoost 构建一步。"""
        if self.finished:
            return

        tree = self.trees[self.current_tree_index]
        if tree.finished:
            self._commit_current_tree()
            self._start_next_tree()
            return

        tree.step()
        self.phase = "split node"

    def predict_proba(self, x):
        """用已提交树和当前树预测概率。"""
        logits = np.full(len(x), self.base_score)
        for tree_index, tree in enumerate(self.trees):
            if tree_index < self.current_tree_index or tree.finished:
                logits += self.learning_rate * tree.predict(x)
        return self._sigmoid(logits)

    def logloss(self):
        """当前训练集 logloss。"""
        p = np.clip(self._sigmoid(self.logits), 1e-6, 1 - 1e-6)
        return float(-np.mean(self.y * np.log(p) + (1 - self.y) * np.log(1 - p)))

    def snapshot(self):
        """保存当前提升状态。"""
        return {
            "trees": [tree.snapshot() for tree in self.trees],
            "current_tree_index": self.current_tree_index,
            "phase": self.phase,
            "finished": self.finished,
            "logits": self.logits.copy(),
            "logloss": self.logloss(),
        }

    def restore(self, snapshot):
        """恢复提升状态。"""
        self.current_tree_index = snapshot["current_tree_index"]
        self.phase = snapshot["phase"]
        self.finished = snapshot["finished"]
        self.logits = snapshot["logits"].copy()

        while len(self.trees) < len(snapshot["trees"]):
            gradients, hessians = self._gradients_hessians()
            self.trees.append(
                BoostingTreeBuilder(
                    self.x,
                    gradients,
                    hessians,
                    max_depth=self.max_depth,
                    reg_lambda=self.reg_lambda,
                )
            )

        self.trees = self.trees[: len(snapshot["trees"])]
        for tree, tree_snapshot in zip(self.trees, snapshot["trees"]):
            tree.restore(tree_snapshot)
