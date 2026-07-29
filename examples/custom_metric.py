"""Example plug-in metric: four-dimensional Reissner-Nordstrom-AdS."""

from dataclasses import dataclass

import numpy as np

from adsrt.metrics.spherical import StaticSphericalAdS4


@dataclass
class ReissnerNordstromAdS4(StaticSphericalAdS4):
    charge: float = 0.15

    def f(self, r: float) -> float:
        return 1.0 - 2.0 * self.mu / r + self.charge**2 / r**2 + (r / self.L) ** 2

    def f_prime(self, r: float) -> float:
        return 2.0 * self.mu / r**2 - 2.0 * self.charge**2 / r**3 + 2.0 * r / self.L**2

    def horizon_radius(self) -> float | None:
        roots = self.positive_real_roots(
            (1.0 / self.L**2, 0.0, 1.0, -2.0 * self.mu, self.charge**2)
        )
        return roots[-1] if roots else None
