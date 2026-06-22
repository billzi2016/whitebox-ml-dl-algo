import numpy as np


class KMeansOptimizer:
    """手写 K-Means：初始化中心、分配样本、更新中心、记录 SSE。"""

    def __init__(self, x, k=4, seed=7):
        self.x = x
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.iteration = 0
        self.phase = "initial"

        # K-Means 对初始中心敏感。这里不用 sklearn 的 k-means++，
        # 而是手写一个简单的 farthest-first 初始化，让初始点分散一些，
        # 动画更容易看到中心逐步吸附到数据团。
        self.centers = self._initialize_centers()
        self.labels = self.assign_labels()

    def _initialize_centers(self):
        """用 farthest-first 思路手写初始化中心。"""
        first_index = self.rng.integers(0, len(self.x))
        centers = [self.x[first_index]]

        while len(centers) < self.k:
            current = np.vstack(centers)
            distances = self._squared_distances_to_centers(current)
            nearest_distances = distances.min(axis=1)
            next_index = int(np.argmax(nearest_distances))
            centers.append(self.x[next_index])

        return np.vstack(centers)

    def _squared_distances_to_centers(self, centers):
        """计算每个样本到每个中心的平方距离。"""
        diff = self.x[:, None, :] - centers[None, :, :]
        return np.sum(diff * diff, axis=2)

    def assign_labels(self):
        """Assignment step：把每个点分给最近的中心。"""
        distances = self._squared_distances_to_centers(self.centers)
        return np.argmin(distances, axis=1)

    def update_centers(self):
        """Update step：把每个中心移动到当前簇内样本的均值位置。"""
        new_centers = self.centers.copy()

        for cluster_id in range(self.k):
            mask = self.labels == cluster_id
            if np.any(mask):
                new_centers[cluster_id] = self.x[mask].mean(axis=0)
            else:
                # 空簇说明当前中心没有吸引到任何点。
                # 为了让动画继续运行，把它重新放到当前 SSE 最大的样本附近。
                distances = self._squared_distances_to_centers(self.centers)
                nearest_distances = distances.min(axis=1)
                new_centers[cluster_id] = self.x[int(np.argmax(nearest_distances))]

        return new_centers

    def sse(self):
        """K-Means 目标函数：所有点到所属中心的平方距离之和。"""
        distances = self._squared_distances_to_centers(self.centers)
        return float(distances[np.arange(len(self.x)), self.labels].sum())

    def step(self):
        """执行半步 K-Means，让动画能分别展示分配和更新。"""
        if self.phase in {"initial", "updated"}:
            self.labels = self.assign_labels()
            self.phase = "assigned"
        else:
            self.centers = self.update_centers()
            self.iteration += 1
            self.phase = "updated"

    def snapshot(self):
        """保存当前状态，支持上一帧和进度条回放。"""
        return {
            "centers": self.centers.copy(),
            "labels": self.labels.copy(),
            "sse": self.sse(),
            "iteration": self.iteration,
            "phase": self.phase,
        }

    def restore(self, snapshot):
        """恢复某一帧的聚类状态。"""
        self.centers = snapshot["centers"].copy()
        self.labels = snapshot["labels"].copy()
        self.iteration = snapshot["iteration"]
        self.phase = snapshot["phase"]
