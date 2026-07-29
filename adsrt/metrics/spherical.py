from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Array, Metric4D


@dataclass
class StaticSphericalAdS4(Metric4D):
    """ds^2 = -f dt^2 + dr^2/f + r^2(dtheta^2 + sin^2 theta dphi^2)."""

    L: float = 1.0
    mu: float = 0.0

    active_derivative_indices = (1, 2)

    def __post_init__(self) -> None:
        if self.L <= 0.0:
            raise ValueError("L must be positive")
        if self.mu < 0.0:
            raise ValueError("mu must be nonnegative")

    def f(self, r: float) -> float:
        return 1.0 - 2.0 * self.mu / r + (r / self.L) ** 2

    def f_prime(self, r: float) -> float:
        return 2.0 * self.mu / (r * r) + 2.0 * r / (self.L * self.L)

    def metric(self, x: Array) -> Array:
        _, r, theta, _ = np.asarray(x, dtype=float)
        if r <= 0.0:
            raise ValueError("spherical chart requires r>0")
        s = np.sin(theta)
        f = self.f(float(r))
        return np.diag([-f, 1.0 / f, r * r, r * r * s * s]).astype(float)

    def inverse_metric(self, x: Array) -> Array:
        _, r, theta, _ = np.asarray(x, dtype=float)
        s = np.sin(theta)
        f = self.f(float(r))
        return np.diag([-1.0 / f, f, 1.0 / (r * r), 1.0 / (r * r * s * s)]).astype(float)

    def inverse_metric_derivatives(self, x: Array) -> Array:
        _, r, theta, _ = np.asarray(x, dtype=float)
        f = self.f(float(r))
        fp = self.f_prime(float(r))
        s = np.sin(theta)
        c = np.cos(theta)
        d = np.zeros((4, 4, 4), dtype=float)
        d[1, 0, 0] = fp / (f * f)
        d[1, 1, 1] = fp
        d[1, 2, 2] = -2.0 / (r**3)
        d[1, 3, 3] = -2.0 / (r**3 * s * s)
        d[2, 3, 3] = -2.0 * c / (r * r * s**3)
        return d

    def horizon_radius(self) -> float | None:
        if self.mu == 0.0:
            return None
        roots = self.positive_real_roots((1.0 / self.L**2, 0.0, 1.0, -2.0 * self.mu))
        return roots[-1] if roots else None


@dataclass
class PureAdS4(StaticSphericalAdS4):
    mu: float = 0.0
    regular_origin = True


@dataclass
class SchwarzschildAdS4(StaticSphericalAdS4):
    mu: float = 0.25

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mu <= 0.0:
            raise ValueError("SchwarzschildAdS4 requires mu>0")
