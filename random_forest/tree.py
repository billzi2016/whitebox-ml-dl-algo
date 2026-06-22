import numpy as np


class DecisionTreeBuilder:
    """手写 CART 分类树构建器，每次 step 只分裂一个叶子节点。"""

    def __init__(self, x, y, max_depth=5, min_samples_split=8, max_features=1, seed=0):
        self.x = x
        self.y = y
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = np.random.default_rng(seed)

        self.nodes = [
            {
                "indices": np.arange(len(x)),
                "depth": 0,
                "prediction": self._majority(y),
                "is_leaf": True,
                "feature": None,
                "threshold": None,
                "left": None,
                "right": None,
                "gini": self._gini(y),
            }
        ]
        self.queue = [0]
        self.last_split = None
        self.finished = False

    def _majority(self, values):
        """返回当前节点内最多的类别。"""
        counts = np.bincount(values, minlength=2)
        return int(np.argmax(counts))

    def _gini(self, values):
        """Gini impurity：1 - sum(p_k^2)。"""
        if len(values) == 0:
            return 0.0
        counts = np.bincount(values, minlength=2) / len(values)
        return float(1.0 - np.sum(counts * counts))

    def _best_split(self, indices):
        """在随机特征子集上寻找 Gini gain 最大的切分。"""
        parent_y = self.y[indices]
        parent_gini = self._gini(parent_y)
        best = None
        features = self.rng.choice(self.x.shape[1], size=self.max_features, replace=False)

        for feature in features:
            values = self.x[indices, feature]
            thresholds = np.unique(values)
            if len(thresholds) > 24:
                thresholds = np.quantile(values, np.linspace(0.08, 0.92, 24))

            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_y = parent_y[left_mask]
                right_y = parent_y[right_mask]
                weighted = (
                    len(left_y) / len(parent_y) * self._gini(left_y)
                    + len(right_y) / len(parent_y) * self._gini(right_y)
                )
                gain = parent_gini - weighted
                if best is None or gain > best["gain"]:
                    best = {
                        "feature": int(feature),
                        "threshold": float(threshold),
                        "gain": float(gain),
                        "left_indices": indices[left_mask],
                        "right_indices": indices[right_mask],
                    }

        return best

    def step(self):
        """分裂一个待处理叶子节点。"""
        if not self.queue:
            self.finished = True
            self.last_split = None
            return False

        node_id = self.queue.pop(0)
        node = self.nodes[node_id]
        indices = node["indices"]

        if (
            node["depth"] >= self.max_depth
            or len(indices) < self.min_samples_split
            or self._gini(self.y[indices]) == 0.0
        ):
            self.last_split = None
            return True

        split = self._best_split(indices)
        if split is None or split["gain"] <= 1e-12:
            self.last_split = None
            return True

        left_id = len(self.nodes)
        right_id = left_id + 1
        for child_indices in [split["left_indices"], split["right_indices"]]:
            self.nodes.append(
                {
                    "indices": child_indices,
                    "depth": node["depth"] + 1,
                    "prediction": self._majority(self.y[child_indices]),
                    "is_leaf": True,
                    "feature": None,
                    "threshold": None,
                    "left": None,
                    "right": None,
                    "gini": self._gini(self.y[child_indices]),
                }
            )

        node["is_leaf"] = False
        node["feature"] = split["feature"]
        node["threshold"] = split["threshold"]
        node["left"] = left_id
        node["right"] = right_id
        self.queue.extend([left_id, right_id])
        self.last_split = {"node_id": node_id, **split}
        return True

    def predict_one(self, row):
        """沿树从根节点走到叶子，返回叶子多数类。"""
        node = self.nodes[0]
        while not node["is_leaf"]:
            if row[node["feature"]] <= node["threshold"]:
                node = self.nodes[node["left"]]
            else:
                node = self.nodes[node["right"]]
        return node["prediction"]

    def predict(self, x):
        """批量预测。"""
        return np.array([self.predict_one(row) for row in x])

    def snapshot(self):
        """保存当前树状态。"""
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
