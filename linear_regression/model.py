import numpy as np


class LinearRegressionOptimizer:
    """手写一元线性回归的 MSE 梯度下降优化。"""

    def __init__(self, x, y, learning_rate=0.08):
        self.x_raw = x
        self.y = y
        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0)
        self.x = (x - self.x_mean) / self.x_std
        self.learning_rate = learning_rate
        self.w = np.array([0.0])
        self.b = 0.0
        self.iteration = 0

    def predict_scaled(self, x_scaled):
        """在标准化特征空间中预测。"""
        return x_scaled @ self.w + self.b

    def predict(self, x_raw):
        """对原始 x 坐标预测。"""
        x_scaled = (x_raw - self.x_mean) / self.x_std
        return self.predict_scaled(x_scaled)

    def loss_and_gradients(self):
        """计算 MSE 和参数梯度。"""
        predictions = self.predict_scaled(self.x)
        errors = predictions - self.y
        loss = float(np.mean(errors * errors))
        grad_w = 2.0 * (self.x.T @ errors) / len(self.x)
        grad_b = 2.0 * float(errors.mean())
        return loss, grad_w, grad_b

    def step(self):
        """执行一次梯度下降。"""
        _loss, grad_w, grad_b = self.loss_and_gradients()
        self.w -= self.learning_rate * grad_w
        self.b -= self.learning_rate * grad_b
        self.iteration += 1

    def snapshot(self):
        """保存当前参数状态。"""
        loss, _grad_w, _grad_b = self.loss_and_gradients()
        return {
            "w": self.w.copy(),
            "b": self.b,
            "iteration": self.iteration,
            "loss": loss,
        }

    def restore(self, snapshot):
        """恢复某一帧参数。"""
        self.w = snapshot["w"].copy()
        self.b = snapshot["b"]
        self.iteration = snapshot["iteration"]
