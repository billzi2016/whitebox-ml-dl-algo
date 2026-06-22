import numpy as np


class TSNEOptimizer:
    """手写 t-SNE 的相似度计算、KL 散度和梯度下降。"""

    def __init__(
        self,
        x_high,
        perplexity=30.0,
        learning_rate=120.0,
        exaggeration=4.0,
        exaggeration_iters=120,
        seed=5,
    ):
        self.x_high = self._standardize(x_high)
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.exaggeration = exaggeration
        self.exaggeration_iters = exaggeration_iters
        self.iteration = 0

        self.p = self._joint_probabilities()

        rng = np.random.default_rng(seed)
        self.y = rng.normal(0.0, 0.0008, size=(len(x_high), 2))
        self.velocity = np.zeros_like(self.y)

    def _standardize(self, x):
        """标准化高维输入，避免某个维度支配距离。"""
        return (x - x.mean(axis=0)) / x.std(axis=0)

    def _squared_distances(self, x):
        """计算成对平方距离矩阵。"""
        diff = x[:, None, :] - x[None, :, :]
        return np.sum(diff * diff, axis=2)

    def _binary_search_row(self, distances):
        """为单个样本搜索 sigma，使条件分布熵接近 perplexity。"""
        target_entropy = np.log(self.perplexity)
        beta = 1.0
        beta_min = -np.inf
        beta_max = np.inf

        for _ in range(50):
            weights = np.exp(-distances * beta)
            weights_sum = np.sum(weights)
            probabilities = weights / max(weights_sum, 1e-12)
            entropy = np.log(max(weights_sum, 1e-12)) + beta * np.sum(distances * probabilities)
            entropy_diff = entropy - target_entropy

            if abs(entropy_diff) < 1e-5:
                break
            if entropy_diff > 0:
                beta_min = beta
                beta = beta * 2.0 if np.isinf(beta_max) else (beta + beta_max) / 2.0
            else:
                beta_max = beta
                beta = beta / 2.0 if np.isinf(beta_min) else (beta + beta_min) / 2.0

        return probabilities

    def _joint_probabilities(self):
        """计算高维空间的对称联合相似度 P。"""
        distances = self._squared_distances(self.x_high)
        n = len(self.x_high)
        conditional = np.zeros((n, n))

        for i in range(n):
            mask = np.arange(n) != i
            row = self._binary_search_row(distances[i, mask])
            conditional[i, mask] = row

        p = (conditional + conditional.T) / (2.0 * n)
        p = np.maximum(p, 1e-12)
        p /= p.sum()
        return p

    def low_dimensional_probabilities(self):
        """计算低维空间的 Student-t 相似度 Q。"""
        distances = self._squared_distances(self.y)
        inv = 1.0 / (1.0 + distances)
        np.fill_diagonal(inv, 0.0)
        q = inv / max(inv.sum(), 1e-12)
        q = np.maximum(q, 1e-12)
        return q, inv

    def loss_and_gradient(self):
        """计算 KL(P||Q) 和 t-SNE 梯度。"""
        q, inv = self.low_dimensional_probabilities()

        # early exaggeration 会在前期放大吸引力，让局部簇更快拉开。
        p_used = self.p * self.exaggeration if self.iteration < self.exaggeration_iters else self.p
        loss = float(np.sum(self.p * np.log(self.p / q)))

        forces = (p_used - q) * inv
        gradient = np.zeros_like(self.y)
        for i in range(len(self.y)):
            diff = self.y[i] - self.y
            gradient[i] = 4.0 * np.sum(forces[i, :, None] * diff, axis=0)

        return loss, gradient

    def step(self):
        """执行一次带动量的梯度下降，让点云持续被吸引和排斥拉扯。"""
        _loss, gradient = self.loss_and_gradient()
        momentum = 0.5 if self.iteration < 120 else 0.8
        self.velocity = momentum * self.velocity - self.learning_rate * gradient
        self.y += self.velocity

        # 去掉整体平移，只保留相对结构。
        self.y -= self.y.mean(axis=0)
        self.iteration += 1

    def snapshot(self):
        """保存当前嵌入坐标、速度和 loss。"""
        loss, _gradient = self.loss_and_gradient()
        return {
            "y": self.y.copy(),
            "velocity": self.velocity.copy(),
            "iteration": self.iteration,
            "loss": loss,
        }

    def restore(self, snapshot):
        """恢复某一帧对应的 t-SNE 状态。"""
        self.y = snapshot["y"].copy()
        self.velocity = snapshot["velocity"].copy()
        self.iteration = snapshot["iteration"]
