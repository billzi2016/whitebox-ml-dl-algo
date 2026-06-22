import numpy as np


class PCAOptimizer:
    """手写 PCA：中心化、协方差矩阵、幂迭代求主成分。"""

    def __init__(self, x, seed=3):
        self.x = x
        self.mean = self.x.mean(axis=0)
        self.centered = self.x - self.mean
        self.covariance = self.centered.T @ self.centered / (len(self.centered) - 1)

        rng = np.random.default_rng(seed)
        self.v1 = self._normalize(rng.normal(size=3))
        self.v2 = self._normalize(rng.normal(size=3))
        self.iteration = 0

    def _normalize(self, vector):
        """单位化向量，避免幂迭代中长度爆炸。"""
        norm = np.linalg.norm(vector)
        return vector / max(norm, 1e-12)

    def _orthogonalize(self, vector, basis):
        """从 vector 中减掉 basis 方向，得到正交方向。"""
        return vector - np.dot(vector, basis) * basis

    def step(self):
        """执行一次幂迭代，逐步逼近第一和第二主成分。"""
        self.v1 = self._normalize(self.covariance @ self.v1)

        # 第二主成分需要和第一主成分正交。
        candidate = self.covariance @ self.v2
        candidate = self._orthogonalize(candidate, self.v1)
        self.v2 = self._normalize(candidate)
        self.iteration += 1

    def explained_variance(self, vector):
        """某个方向上的方差，也就是 v^T Cov v。"""
        return float(vector.T @ self.covariance @ vector)

    def projected_2d(self):
        """把中心化数据投影到当前两个主成分方向上。"""
        components = np.column_stack([self.v1, self.v2])
        return self.centered @ components

    def reconstruction_2d_plane(self):
        """用前两个方向重建三维数据，展示 PCA 平面。"""
        components = np.column_stack([self.v1, self.v2])
        projected = self.projected_2d()
        return projected @ components.T + self.mean

    def snapshot(self):
        """保存当前主方向和投影状态。"""
        return {
            "v1": self.v1.copy(),
            "v2": self.v2.copy(),
            "iteration": self.iteration,
            "var1": self.explained_variance(self.v1),
            "var2": self.explained_variance(self.v2),
            "projected": self.projected_2d(),
            "reconstructed": self.reconstruction_2d_plane(),
        }

    def restore(self, snapshot):
        """恢复某一帧主成分方向。"""
        self.v1 = snapshot["v1"].copy()
        self.v2 = snapshot["v2"].copy()
        self.iteration = snapshot["iteration"]
