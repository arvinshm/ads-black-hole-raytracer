from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class CoordinateInfo:
    names: tuple[str, str, str, str] = ("t", "r", "theta", "phi")
    periodic: tuple[bool, bool, bool, bool] = (False, False, False, True)


class Metric4D(ABC):
    """Base class for a 4D Lorentzian metric with signature (-,+,+,+).

    The default chart is (t, r, theta, phi).  The ray tracer only assumes that
    coordinate 1 is radial and that the outer observing/source surface is a
    constant-r timelike cylinder.  Custom metrics can override any method.
    """

    coordinate_info = CoordinateInfo()
    active_derivative_indices: tuple[int, ...] = (0, 1, 2, 3)
    derivative_step: float = 2.0e-6
    regular_origin: bool = False

    @abstractmethod
    def metric(self, x: Array) -> Array:
        """Return g_{mu nu}(x) as a symmetric 4x4 array."""

    def inverse_metric(self, x: Array) -> Array:
        return np.linalg.inv(self.metric(x))

    def inverse_metric_derivatives(self, x: Array) -> Array:
        """Return partial_mu g^{alpha beta} using centered finite differences.

        Stationary/axisymmetric metrics should set ``active_derivative_indices``
        to only the coordinates on which the metric actually depends.  Analytic
        overrides are strongly recommended for expensive custom metrics.
        """
        x = np.asarray(x, dtype=float)
        derivatives = np.zeros((4, 4, 4), dtype=float)
        for mu in self.active_derivative_indices:
            scale = max(1.0, abs(float(x[mu])))
            h = self.derivative_step * scale
            xp = x.copy()
            xm = x.copy()
            xp[mu] += h
            xm[mu] -= h
            # Avoid stepping through r=0 or exactly onto a polar coordinate pole.
            if mu == 1 and xm[1] <= 0.0:
                xm[1] = x[1]
                xp[1] = x[1] + h
                derivatives[mu] = (self.inverse_metric(xp) - self.inverse_metric(x)) / h
            elif mu == 2 and xm[2] <= 0.0:
                xm[2] = x[2]
                xp[2] = x[2] + h
                derivatives[mu] = (self.inverse_metric(xp) - self.inverse_metric(x)) / h
            elif mu == 2 and xp[2] >= np.pi:
                xp[2] = x[2]
                xm[2] = x[2] - h
                derivatives[mu] = (self.inverse_metric(x) - self.inverse_metric(xm)) / h
            else:
                derivatives[mu] = (self.inverse_metric(xp) - self.inverse_metric(xm)) / (2.0 * h)
        return derivatives

    def horizon_radius(self) -> float | None:
        """Return the outer horizon radius, or None if no horizon is present."""
        return None

    def validate_point(self, x: Array) -> None:
        g = self.metric(x)
        if g.shape != (4, 4):
            raise ValueError("metric(x) must return a 4x4 array")
        if not np.all(np.isfinite(g)):
            raise ValueError(f"metric is non-finite at x={x}")
        if not np.allclose(g, g.T, atol=1e-12, rtol=1e-12):
            raise ValueError("metric must be symmetric")
        eig = np.linalg.eigvalsh(g)
        if np.count_nonzero(eig < 0.0) != 1:
            raise ValueError(f"metric must have Lorentzian signature (-,+,+,+); eigenvalues={eig}")

    def default_angular_velocity(self, x: Array) -> float:
        """Angular velocity of a local zero-angular-momentum observer (ZAMO).

        For diagonal static metrics this is zero.  At large r in Kerr-AdS this
        tends to the nonrotating-at-infinity angular velocity -a/L^2.
        """
        g = self.metric(x)
        if abs(g[3, 3]) < 1e-15:
            return 0.0
        return float(-g[0, 3] / g[3, 3])

    def wrap_coordinates(self, x: Array) -> Array:
        y = np.asarray(x, dtype=float).copy()
        y[3] = np.mod(y[3], 2.0 * np.pi)
        return y

    @staticmethod
    def positive_real_roots(coefficients: Iterable[float], tol: float = 1e-9) -> list[float]:
        roots = np.roots(np.asarray(tuple(coefficients), dtype=float))
        values = [float(z.real) for z in roots if abs(z.imag) < tol and z.real > tol]
        return sorted(values)
