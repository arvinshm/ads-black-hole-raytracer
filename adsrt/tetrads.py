from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import metric_dot
from .metrics.base import Array, Metric4D


@dataclass(frozen=True)
class OrthonormalTetrad:
    u: Array
    e_r: Array
    e_theta: Array
    e_phi: Array

    @property
    def spatial(self) -> tuple[Array, Array, Array]:
        return self.e_r, self.e_theta, self.e_phi


def normalize_timelike(g: Array, vector: Array) -> Array:
    norm2 = metric_dot(g, vector, vector)
    if not np.isfinite(norm2) or norm2 >= 0.0:
        raise ValueError(f"observer vector is not timelike: norm^2={norm2}")
    result = np.asarray(vector, dtype=float) / np.sqrt(-norm2)
    if result[0] < 0.0:
        result = -result
    return result


def normalize_spacelike(g: Array, vector: Array) -> Array:
    norm2 = metric_dot(g, vector, vector)
    if not np.isfinite(norm2) or norm2 <= 0.0:
        raise ValueError(f"spatial vector is not spacelike: norm^2={norm2}")
    return np.asarray(vector, dtype=float) / np.sqrt(norm2)


def project_orthogonal(g: Array, vector: Array, basis: list[Array]) -> Array:
    v = np.asarray(vector, dtype=float).copy()
    for b in basis:
        sign = -1.0 if metric_dot(g, b, b) < 0.0 else 1.0
        v -= sign * metric_dot(g, v, b) * b
    return v


def observer_four_velocity(metric: Metric4D, x: Array, angular_velocity: float | None = None) -> Array:
    omega = metric.default_angular_velocity(x) if angular_velocity is None else angular_velocity
    candidate = np.array([1.0, 0.0, 0.0, omega], dtype=float)
    return normalize_timelike(metric.metric(x), candidate)


def make_tetrad(
    metric: Metric4D,
    x: Array,
    angular_velocity: float | None = None,
) -> OrthonormalTetrad:
    """Construct a right-handed local tetrad from coordinate-direction seeds."""
    g = metric.metric(x)
    u = observer_four_velocity(metric, x, angular_velocity)
    seeds = [
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    ]
    spatial: list[Array] = []
    for seed in seeds:
        v = project_orthogonal(g, seed, [u, *spatial])
        spatial.append(normalize_spacelike(g, v))
    return OrthonormalTetrad(u=u, e_r=spatial[0], e_theta=spatial[1], e_phi=spatial[2])


def boosted_observer(tetrad: OrthonormalTetrad, velocity_local: tuple[float, float, float]) -> Array:
    v = np.asarray(velocity_local, dtype=float)
    speed2 = float(v @ v)
    if speed2 >= 1.0:
        raise ValueError("local emitter velocity must have |v|<1")
    gamma = 1.0 / np.sqrt(1.0 - speed2)
    spatial = v[0] * tetrad.e_r + v[1] * tetrad.e_theta + v[2] * tetrad.e_phi
    return gamma * (tetrad.u + spatial)


def tetrad_error(metric: Metric4D, x: Array, tetrad: OrthonormalTetrad) -> float:
    g = metric.metric(x)
    vectors = [tetrad.u, tetrad.e_r, tetrad.e_theta, tetrad.e_phi]
    gram = np.array([[metric_dot(g, a, b) for b in vectors] for a in vectors])
    return float(np.max(np.abs(gram - np.diag([-1.0, 1.0, 1.0, 1.0]))))
