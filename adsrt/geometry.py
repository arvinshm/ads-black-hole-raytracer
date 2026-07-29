from __future__ import annotations

import numpy as np

from .metrics.base import Array, Metric4D


def metric_dot(g: Array, a: Array, b: Array) -> float:
    return float(np.asarray(a) @ g @ np.asarray(b))


def hamiltonian(metric: Metric4D, x: Array, p_cov: Array) -> float:
    g_inv = metric.inverse_metric(x)
    return 0.5 * float(np.real(np.asarray(p_cov) @ g_inv @ np.asarray(p_cov)))


def null_residual(metric: Metric4D, x: Array, p_cov: Array) -> float:
    """Scale-independent residual |g^{ab}p_a p_b| / sum|term_ab|."""
    p = np.asarray(p_cov, dtype=float)
    g_inv = np.asarray(metric.inverse_metric(x), dtype=float)
    contraction = float(p @ g_inv @ p)
    scale = float(np.sum(np.abs(np.einsum("a,b,ab->ab", p, p, g_inv))))
    return abs(contraction) / max(scale, np.finfo(float).tiny)


def hamilton_rhs(metric: Metric4D, _lambda: float, y: Array) -> Array:
    x = np.asarray(y[:4], dtype=float)
    p = np.asarray(y[4:], dtype=float)
    g_inv = metric.inverse_metric(x)
    d_g_inv = metric.inverse_metric_derivatives(x)
    dx = g_inv @ p
    dp = -0.5 * np.einsum("a,b,mab->m", p, p, d_g_inv, optimize=True)
    return np.concatenate((dx, dp))


def covector_to_vector(metric: Metric4D, x: Array, p_cov: Array) -> Array:
    return metric.inverse_metric(x) @ np.asarray(p_cov, dtype=float)


def vector_to_covector(metric: Metric4D, x: Array, vector: Array) -> Array:
    return metric.metric(x) @ np.asarray(vector, dtype=float)


def angular_separation(theta1: float, phi1: float, theta2: float, phi2: float) -> float:
    dphi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
    cosine = (
        np.sin(theta1) * np.sin(theta2) * np.cos(dphi)
        + np.cos(theta1) * np.cos(theta2)
    )
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def antipode(theta: float, phi: float) -> tuple[float, float]:
    return float(np.pi - theta), float(np.mod(phi + np.pi, 2.0 * np.pi))
