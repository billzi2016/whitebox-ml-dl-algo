import numpy as np


class RBFKernelSVMOptimizer:
    """手写 RBF 特征空间中的 one-vs-rest SVM 优化器。"""

    def __init__(self, x_raw, y, gamma=1.0, c=1.0, learning_rate=0.08, seed=11):
        # 原始二维坐标保留给绘图使用；训练时会先手写标准化。
        self.x_raw = x_raw
        self.y = y
        self.gamma = gamma
        self.c = c
        self.learning_rate = learning_rate

        self.classes = np.unique(y)
        self.output_class_count = len(self.classes)
        self.is_binary = self.output_class_count == 2

        # 二分类只需要一个 SVM：score >= 0 判为正类，score < 0 判为负类。
        # 如果二分类也强行训练两个 one-vs-rest 分类器，两个打分函数可能互相抵消，
        # 背景区域会出现“训练点分对了，但大面积颜色仍然不对”的视觉问题。
        self.classifier_count = 1 if self.is_binary else self.output_class_count
        self.sample_count = len(x_raw)

        # RBF 核依赖欧氏距离，所以必须统一尺度。
        self.x_mean = self.x_raw.mean(axis=0)
        self.x_std = self.x_raw.std(axis=0)
        self.x = self._standardize(self.x_raw)

        # 教学版 RBF 显式映射：把每个训练样本当作一个 RBF 中心。
        # phi_j(x) = exp(-gamma * ||x - center_j||^2)
        # 这样可以在 RBF 特征空间里手写线性 SVM 的优化过程。
        self.centers = self.x.copy()
        self.phi = self.rbf_features(self.x_raw)
        self.feature_count = self.phi.shape[1]

        rng = np.random.default_rng(seed)
        self.w = rng.normal(
            loc=0.0,
            scale=0.01,
            size=(self.classifier_count, self.feature_count),
        )
        self.b = np.zeros(self.classifier_count)

    def _standardize(self, x_raw):
        """用训练集统计量标准化二维坐标。"""
        return (x_raw - self.x_mean) / self.x_std

    def rbf_features(self, x_raw):
        """把二维输入映射成以训练样本为中心的 RBF 特征。"""
        x = self._standardize(x_raw)

        # squared_distances[i, j] = ||x_i - center_j||^2。
        diff = x[:, None, :] - self.centers[None, :, :]
        squared_distances = np.sum(diff * diff, axis=2)
        return np.exp(-self.gamma * squared_distances)

    def scores(self, x_raw):
        """计算每个类别的 one-vs-rest 打分。"""
        phi = self.rbf_features(x_raw)
        return phi @ self.w.T + self.b

    def predict(self, x_raw):
        """取打分最高的类别作为预测类别。"""
        scores = self.scores(x_raw)
        if self.is_binary:
            return np.where(scores[:, 0] >= 0.0, self.classes[1], self.classes[0])
        return self.classes[np.argmax(scores, axis=1)]

    def loss_and_gradients(self):
        """计算 RBF 特征空间中 SVM 目标函数和 batch 梯度。"""
        grad_w = np.zeros_like(self.w)
        grad_b = np.zeros_like(self.b)
        total_hinge = 0.0
        total_violations = 0

        for classifier_id in range(self.classifier_count):
            if self.is_binary:
                # 二分类：只训练一个分类器，第二个类别是 +1，第一个类别是 -1。
                y_binary = np.where(self.y == self.classes[1], 1.0, -1.0)
            else:
                # 多分类：one-vs-rest，当前类别为 +1，其他类别为 -1。
                y_binary = np.where(self.y == self.classes[classifier_id], 1.0, -1.0)

            scores = self.phi @ self.w[classifier_id] + self.b[classifier_id]
            margins = y_binary * scores

            # margin < 1 的样本会产生 hinge loss 和梯度。
            active = margins < 1.0
            hinge = np.maximum(0.0, 1.0 - margins)
            total_hinge += hinge.sum()
            total_violations += int(active.sum())

            # L = 1/2 ||w||^2 + C * sum(max(0, 1 - y f(x)))
            grad_w[classifier_id] = self.w[classifier_id] - self.c * (
                y_binary[active, None] * self.phi[active]
            ).sum(axis=0)
            grad_b[classifier_id] = -self.c * y_binary[active].sum()

        regularization = 0.5 * np.sum(self.w * self.w)
        total_loss = regularization + self.c * total_hinge
        return total_loss, total_violations, grad_w, grad_b

    def step(self):
        """执行一次完整 batch 梯度下降。"""
        _loss, _violations, grad_w, grad_b = self.loss_and_gradients()

        # 用样本数归一化梯度，避免数据量变化导致学习率尺度失控。
        self.w -= self.learning_rate * grad_w / self.sample_count
        self.b -= self.learning_rate * grad_b / self.sample_count

    def snapshot(self):
        """保存当前参数和指标，用于上一帧/进度条回放。"""
        loss, violations, _grad_w, _grad_b = self.loss_and_gradients()
        return {
            "w": self.w.copy(),
            "b": self.b.copy(),
            "loss": loss,
            "violations": violations,
        }

    def restore(self, snapshot):
        """恢复到某一帧的参数状态。"""
        self.w = snapshot["w"].copy()
        self.b = snapshot["b"].copy()
