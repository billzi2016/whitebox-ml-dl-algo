import importlib

import numpy as np

from optimizers.surface import grad_numpy, loss_numpy, loss_torch


OPTIMIZER_NAMES = [
    "GD",
    "Momentum",
    "Nesterov",
    "AdaGrad",
    "RMSProp",
    "Adam",
    "AdamW",
]


class NumpyOptimizerState:
    """Numpy fallback that mirrors common optimizer update rules."""

    def __init__(self, name, start, lr):
        self.name = name
        self.theta = np.array(start, dtype=float)
        self.lr = lr
        self.t = 0
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        self.beta = 0.9
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.weight_decay = 0.04

    def step(self):
        self.t += 1

        if self.name == "Nesterov":
            lookahead = self.theta - self.lr * self.beta * self.m
            grad = grad_numpy(lookahead)
        else:
            grad = grad_numpy(self.theta)

        if self.name == "GD":
            self.theta -= self.lr * grad
        elif self.name == "Momentum":
            self.m = self.beta * self.m + grad
            self.theta -= self.lr * self.m
        elif self.name == "Nesterov":
            self.m = self.beta * self.m + grad
            self.theta -= self.lr * self.m
        elif self.name == "AdaGrad":
            self.v += grad * grad
            self.theta -= self.lr * grad / (np.sqrt(self.v) + self.eps)
        elif self.name == "RMSProp":
            self.v = 0.99 * self.v + 0.01 * grad * grad
            self.theta -= self.lr * grad / (np.sqrt(self.v) + self.eps)
        elif self.name == "Adam":
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1 - self.beta2) * grad * grad
            m_hat = self.m / (1 - self.beta1**self.t)
            v_hat = self.v / (1 - self.beta2**self.t)
            self.theta -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        elif self.name == "AdamW":
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1 - self.beta2) * grad * grad
            m_hat = self.m / (1 - self.beta1**self.t)
            v_hat = self.v / (1 - self.beta2**self.t)
            self.theta -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * self.theta)

    def point(self):
        return self.theta.copy()


class TorchOptimizerState:
    """Thin wrapper around torch.optim. Each animation step calls optimizer.step()."""

    def __init__(self, name, start, lr, torch):
        self.name = name
        self.torch = torch
        self.theta = torch.tensor(start, dtype=torch.float32, requires_grad=True)

        if name == "GD":
            self.optimizer = torch.optim.SGD([self.theta], lr=lr)
        elif name == "Momentum":
            self.optimizer = torch.optim.SGD([self.theta], lr=lr, momentum=0.9)
        elif name == "Nesterov":
            self.optimizer = torch.optim.SGD([self.theta], lr=lr, momentum=0.9, nesterov=True)
        elif name == "AdaGrad":
            self.optimizer = torch.optim.Adagrad([self.theta], lr=lr)
        elif name == "RMSProp":
            self.optimizer = torch.optim.RMSprop([self.theta], lr=lr, alpha=0.99)
        elif name == "Adam":
            self.optimizer = torch.optim.Adam([self.theta], lr=lr)
        elif name == "AdamW":
            self.optimizer = torch.optim.AdamW([self.theta], lr=lr, weight_decay=0.04)
        else:
            raise ValueError(name)

    def step(self):
        self.optimizer.zero_grad()
        loss = loss_torch(self.theta, self.torch)
        loss.backward()
        self.optimizer.step()

    def point(self):
        return self.theta.detach().cpu().numpy().copy()


class OptimizerComparison:
    """Runs several optimizers step-by-step and stores trajectories."""

    def __init__(self, start=(-4.2, 3.6), lr=0.055):
        self.start = np.array(start, dtype=float)
        self.lr = lr
        self.backend = "numpy"
        self.torch = None

        try:
            torch = importlib.import_module("torch")
            _ = torch.tensor([0.0])
            self.torch = torch
            self.backend = "torch"
        except Exception:
            self.torch = None
            self.backend = "numpy"

        self.states = [
            self._make_state(name)
            for name in OPTIMIZER_NAMES
        ]
        self.step_index = 0

    def _make_state(self, name):
        if self.torch is not None:
            return TorchOptimizerState(name, self.start, self.lr, self.torch)
        return NumpyOptimizerState(name, self.start, self.lr)

    def step(self):
        for state in self.states:
            state.step()
        self.step_index += 1

    def snapshot(self):
        points = {state.name: state.point() for state in self.states}
        losses = {name: float(loss_numpy(point)) for name, point in points.items()}
        return {
            "points": points,
            "losses": losses,
            "step": self.step_index,
            "backend": self.backend,
        }

    def restore(self, snapshot):
        # The animator only restores by replaying history. Optimizer internal state is
        # not reconstructed from snapshots because torch optimizer buffers are opaque.
        pass
