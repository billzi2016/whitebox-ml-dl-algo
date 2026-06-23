import numpy as np


SURFACE_PRESETS = {
    "rugged": {
        "title": "Rugged sinusoidal bowl",
        "limit": 5.0,
        "start": (-4.2, 3.6),
        "lr": 0.055,
    },
    "saddle": {
        "title": "Bounded saddle with quartic walls",
        "limit": 3.4,
        "start": (-2.2, 1.9),
        "lr": 0.02,
    },
    "rosenbrock": {
        "title": "Rosenbrock banana valley",
        "limit": 2.4,
        "start": (-1.6, 1.8),
        "lr": 0.004,
    },
    "himmelblau": {
        "title": "Himmelblau multi-minima function",
        "limit": 5.2,
        "start": (-4.0, 4.0),
        "lr": 0.003,
    },
    "rastrigin": {
        "title": "Rastrigin highly multimodal surface",
        "limit": 5.12,
        "start": (-4.2, 3.7),
        "lr": 0.01,
    },
}


def loss_numpy(theta, surface="rugged"):
    """Evaluate one of several classic optimizer-demo surfaces."""
    x, y = theta[..., 0], theta[..., 1]

    if surface == "rugged":
        bowl = 0.12 * (x * x + 1.8 * y * y)
        waves = 0.9 * np.sin(1.7 * x) * np.cos(1.3 * y)
        valley = 0.22 * np.sin(0.7 * x * y)
        return bowl + waves + valley

    if surface == "saddle":
        monkey = 0.045 * (x**3 - 3 * x * y * y)
        bowl = 0.16 * (x * x + y * y)
        walls = 0.018 * (x**4 + y**4)
        ripples = 0.18 * np.sin(2.3 * x) * np.cos(1.7 * y)
        return monkey + bowl + walls + ripples

    if surface == "rosenbrock":
        return 0.18 * ((1 - x) ** 2 + 100 * (y - x * x) ** 2)

    if surface == "himmelblau":
        return 0.02 * ((x * x + y - 11) ** 2 + (x + y * y - 7) ** 2)

    if surface == "rastrigin":
        return 0.08 * (20 + x * x + y * y - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))

    raise ValueError(f"Unknown surface: {surface}")


def grad_numpy(theta, surface="rugged"):
    """Finite-difference gradient for the numpy fallback path."""
    eps = 1e-5
    grad = np.zeros(2)
    for axis in range(2):
        step = np.zeros(2)
        step[axis] = eps
        grad[axis] = (loss_numpy(theta + step, surface) - loss_numpy(theta - step, surface)) / (2 * eps)
    return grad


def make_surface_grid(surface="rugged", size=180):
    """Create mesh grid for drawing the selected 3D surface."""
    limit = SURFACE_PRESETS[surface]["limit"]
    xs = np.linspace(-limit, limit, size)
    ys = np.linspace(-limit, limit, size)
    xx, yy = np.meshgrid(xs, ys)
    zz = loss_numpy(np.stack([xx, yy], axis=-1), surface)
    return xx, yy, zz


def loss_torch(theta, torch, surface="rugged"):
    """Torch version of the selected surface, used with torch.optim."""
    x, y = theta[0], theta[1]

    if surface == "rugged":
        bowl = 0.12 * (x * x + 1.8 * y * y)
        waves = 0.9 * torch.sin(1.7 * x) * torch.cos(1.3 * y)
        valley = 0.22 * torch.sin(0.7 * x * y)
        return bowl + waves + valley

    if surface == "saddle":
        monkey = 0.045 * (x**3 - 3 * x * y * y)
        bowl = 0.16 * (x * x + y * y)
        walls = 0.018 * (x**4 + y**4)
        ripples = 0.18 * torch.sin(2.3 * x) * torch.cos(1.7 * y)
        return monkey + bowl + walls + ripples

    if surface == "rosenbrock":
        return 0.18 * ((1 - x) ** 2 + 100 * (y - x * x) ** 2)

    if surface == "himmelblau":
        return 0.02 * ((x * x + y - 11) ** 2 + (x + y * y - 7) ** 2)

    if surface == "rastrigin":
        return 0.08 * (20 + x * x + y * y - 10 * (torch.cos(2 * torch.pi * x) + torch.cos(2 * torch.pi * y)))

    raise ValueError(f"Unknown surface: {surface}")
