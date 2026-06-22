import numpy as np


class BoostingTreeBuilder:
    """手写 XGBoost 风格的二阶 CART 回归树，用来拟合 logloss 梯度。"""

    def __init__(self, x, gradients, hessians, max_depth=3, reg_lambda=1.0, gamma=0.0):
        self.x = x
        self.gradients = gradients
        self.hessians = hessians
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.nodes = [
            {
                "indices": np.arange(len(x)),
                "depth": 0,
                "is_leaf": True,
                "feature": None,
                "threshold": None,
                "left": None,
                "right": None,
                "weight": self._leaf_weight(np.arange(len(x))),
                "gain": 0.0,
            }
        ]
        self.queue = [0]
        self.last_split = None
        self.finished = False

    def _leaf_weight(self, indices):
        """XGBoost 叶子分数：w* = -G / (H + lambda)。"""
        g = self.gradients[indices].sum()
        h = self.hessians[indices].sum()
        return float(-g / (h + self.reg_lambda))

    def _score(self, indices):
        """节点结构分数：1/2 * G^2 / (H + lambda)。"""
        g = self.gradients[indices].sum()
        h = self.hessians[indices].sum()
        return 0.5 * float(g * g / (h + self.reg_lambda))

    def _best_split(self, indices):
        """寻找二阶增益最大的 split。"""
        parent_score = self._score(indices)
        best = None

        for feature in range(self.x.shape[1]):
            values = self.x[indices, feature]
            thresholds = np.unique(values)
            if len(thresholds) > 30:
                thresholds = np.quantile(values, np.linspace(0.05, 0.95, 30))

            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask
                if left_mask.sum() < 8 or right_mask.sum() < 8:
                    continue

                left_indices = indices[left_mask]
                right_indices = indices[right_mask]
                gain = self._score(left_indices) + self._score(right_indices) - parent_score - self.gamma
                if best is None or gain > best["gain"]:
                    best = {
                        "feature": int(feature),
                        "threshold": float(threshold),
                        "gain": float(gain),
                        "left_indices": left_indices,
                        "right_indices": right_indices,
                    }

        return best

    def step(self):
        """分裂一个叶子节点。"""
        if not self.queue:
            self.finished = True
            self.last_split = None
            return False

        node_id = self.queue.pop(0)
        node = self.nodes[node_id]
        indices = node["indices"]

        if node["depth"] >= self.max_depth:
            self.last_split = None
            return True

        split = self._best_split(indices)
        if split is None or split["gain"] <= 1e-10:
            self.last_split = None
            return True

        left_id = len(self.nodes)
        right_id = left_id + 1
        for child_indices in [split["left_indices"], split["right_indices"]]:
            self.nodes.append(
                {
                    "indices": child_indices,
                    "depth": node["depth"] + 1,
                    "is_leaf": True,
                    "feature": None,
                    "threshold": None,
                    "left": None,
                    "right": None,
                    "weight": self._leaf_weight(child_indices),
                    "gain": 0.0,
                }
            )

        node["is_leaf"] = False
        node["feature"] = split["feature"]
        node["threshold"] = split["threshold"]
        node["left"] = left_id
        node["right"] = right_id
        node["gain"] = split["gain"]
        self.queue.extend([left_id, right_id])
        self.last_split = {"node_id": node_id, **split}
        return True

    def predict_one(self, row):
        """返回当前树对一个样本的 logit 增量。"""
        node = self.nodes[0]
        while not node["is_leaf"]:
            if row[node["feature"]] <= node["threshold"]:
                node = self.nodes[node["left"]]
            else:
                node = self.nodes[node["right"]]
        return node["weight"]

    def predict(self, x):
        """批量预测 logit 增量。"""
        return np.array([self.predict_one(row) for row in x])

    def snapshot(self):
        """保存树状态。"""
        return {
            "nodes": [dict(node) for node in self.nodes],
            "queue": list(self.queue),
            "last_split": None if self.last_split is None else dict(self.last_split),
            "finished": self.finished,
        }

    def restore(self, snapshot):
        """恢复树状态。"""
        self.nodes = [dict(node) for node in snapshot["nodes"]]
        self.queue = list(snapshot["queue"])
        self.last_split = None if snapshot["last_split"] is None else dict(snapshot["last_split"])
        self.finished = snapshot["finished"]
