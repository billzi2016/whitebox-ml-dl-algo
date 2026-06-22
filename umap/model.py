import numpy as np


class UMAPOptimizer:
    """手写 UMAP 风格的近邻图和低维吸引/排斥优化。"""

    def __init__(
        self,
        x_high,
        n_neighbors=12,
        learning_rate=0.08,
        repulsion_strength=0.8,
        seed=13,
    ):
        self.x_high = self._standardize(x_high)
        self.n_neighbors = n_neighbors
        self.learning_rate = learning_rate
        self.repulsion_strength = repulsion_strength
        self.iteration = 0

        self.graph = self._build_fuzzy_graph()

        rng = np.random.default_rng(seed)
        self.y = rng.normal(0.0, 0.01, size=(len(x_high), 2))

    def _standardize(self, x):
        """标准化高维输入，避免某个维度主导邻域距离。"""
        return (x - x.mean(axis=0)) / x.std(axis=0)

    def _pairwise_distances(self, x):
        """计算高维成对欧氏距离。"""
        diff = x[:, None, :] - x[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2))

    def _build_fuzzy_graph(self):
        """构造简化版 UMAP fuzzy nearest-neighbor graph。"""
        distances = self._pairwise_distances(self.x_high)
        n = len(self.x_high)
        graph = np.zeros((n, n))

        for i in range(n):
            order = np.argsort(distances[i])
            neighbors = order[1 : self.n_neighbors + 1]
            neighbor_distances = distances[i, neighbors]

            # rho 是最近邻距离，表示局部连通性下限。
            rho = float(neighbor_distances[0])
            adjusted = np.maximum(0.0, neighbor_distances - rho)

            # sigma 简化为局部距离均值，保留“局部尺度自适应”的意图。
            sigma = max(float(adjusted.mean()), 1e-3)
            weights = np.exp(-adjusted / sigma)
            graph[i, neighbors] = weights

        # UMAP 的模糊集合 union：w = a + b - a*b。
        graph = graph + graph.T - graph * graph.T
        np.fill_diagonal(graph, 0.0)
        return graph

    def _low_similarity(self):
        """低维空间相似度，用 1 / (1 + d^2) 近似 UMAP 曲线。"""
        diff = self.y[:, None, :] - self.y[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        similarity = 1.0 / (1.0 + dist2)
        np.fill_diagonal(similarity, 0.0)
        return similarity, diff, dist2

    def loss_and_gradient(self):
        """计算交叉熵式目标和低维坐标梯度。"""
        q, diff, dist2 = self._low_similarity()
        p = self.graph
        eps = 1e-6

        attractive_loss = -p * np.log(q + eps)
        repulsive_loss = -(1.0 - p) * np.log(1.0 - q + eps)
        loss = float((attractive_loss + 0.002 * repulsive_loss).sum())

        gradient = np.zeros_like(self.y)

        # 正边吸引：高维近邻在低维中应该靠近。
        attractive = p * q

        # 负边排斥：所有非邻近点有弱排斥，避免塌缩到一起。
        repulsive = self.repulsion_strength * (1.0 - p) * (q * q) * 0.002

        forces = attractive - repulsive
        for i in range(len(self.y)):
            gradient[i] = 2.0 * np.sum(forces[i, :, None] * diff[i], axis=0)

        return loss, gradient

    def step(self):
        """执行一次低维坐标优化。"""
        _loss, gradient = self.loss_and_gradient()
        self.y -= self.learning_rate * gradient
        self.y -= self.y.mean(axis=0)
        self.iteration += 1

    def snapshot(self):
        """保存当前 embedding 和 loss。"""
        loss, _gradient = self.loss_and_gradient()
        return {
            "y": self.y.copy(),
            "iteration": self.iteration,
            "loss": loss,
        }

    def restore(self, snapshot):
        """恢复某一帧的 embedding。"""
        self.y = snapshot["y"].copy()
        self.iteration = snapshot["iteration"]
