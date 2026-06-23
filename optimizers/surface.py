import numpy as np


def loss_numpy(theta):
    """A non-convex but smooth 2D surface for optimizer trajectory demos."""
    x, y = theta[..., 0], theta[..., 1]
    bowl = 0.12 * (x * x + 1.8 * y * y)
    waves = 0.9 * np.sin(1.7 * x) * np.cos(1.3 * y)
    valley = 0.22 * np.sin(0.7 * x * y)
    return bowl + waves + valley


def grad_numpy(theta):
    """Analytic gradient of loss_numpy."""
    x, y = theta[0], theta[1]
    dx = 0.24 * x + 1.53 * np.cos(1.7 * x) * np.cos(1.3 * y) + 0.154 * y * np.cos(0.7 * x * y)
    dy = 0.432 * y - 1.17 * np.sin(1.7 * x) * np.sin(1.3 * y) + 0.154 * x * np.cos(0.7 * x * y)
    return np.array([dx, dy])


def make_surface_grid(limit=5.0, size=180):
    """Create mesh grid for drawing the 3D surface."""
    xs = np.linspace(-limit, limit, size)
    ys = np.linspace(-limit, limit, size)
    xx, yy = np.meshgrid(xs, ys)
    zz = loss_numpy(np.stack([xx, yy], axis=-1))
    return xx, yy, zz


def loss_torch(theta, torch):
    """Torch version of the same surface, used when torch optimizers are available."""
    x, y = theta[0], theta[1]
    bowl = 0.12 * (x * x + 1.8 * y * y)
    waves = 0.9 * torch.sin(1.7 * x) * torch.cos(1.3 * y)
    valley = 0.22 * torch.sin(0.7 * x * y)
    return bowl + waves + valley
