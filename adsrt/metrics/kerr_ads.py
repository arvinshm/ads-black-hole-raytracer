from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Array, Metric4D


@dataclass
class KerrAdS4(Metric4D):
    """Four-dimensional Kerr-AdS in Boyer-Lindquist coordinates.

    rho^2 = r^2 + a^2 cos^2(theta)
    Delta_r = (r^2+a^2)(1+r^2/L^2) - 2 mu r
    Delta_theta = 1 - a^2 cos^2(theta)/L^2
    Xi = 1 - a^2/L^2
    """

    L: float = 1.0
    mu: float = 0.5
    a: float = 0.25

    active_derivative_indices = (1, 2)
    derivative_step = 1.0e-6

    def __post_init__(self) -> None:
        if self.L <= 0.0:
            raise ValueError("L must be positive")
        if self.mu <= 0.0:
            raise ValueError("mu must be positive")
        if abs(self.a) >= self.L:
            raise ValueError("Kerr-AdS requires |a|<L so Xi>0")
        if self.horizon_radius() is None:
            raise ValueError("parameters describe no positive-radius event horizon")

    @property
    def Xi(self) -> float:
        return 1.0 - (self.a / self.L) ** 2

    def rho2(self, r, theta):
        return r * r + self.a * self.a * np.cos(theta) ** 2

    def delta_r(self, r):
        return (r * r + self.a * self.a) * (1.0 + r * r / self.L**2) - 2.0 * self.mu * r

    def delta_theta(self, theta):
        return 1.0 - (self.a * np.cos(theta) / self.L) ** 2

    def metric(self, x: Array) -> Array:
        coordinates = np.asarray(x)
        _, r, theta, _ = coordinates
        if not np.iscomplexobj(r) and r <= 0.0:
            raise ValueError("Boyer-Lindquist chart requires r>0")
        s = np.sin(theta)
        rho2 = self.rho2(r, theta)
        dr = self.delta_r(r)
        dt = self.delta_theta(theta)
        xi = self.Xi

        g = np.zeros((4, 4), dtype=np.result_type(coordinates.dtype, np.float64))
        g[0, 0] = (-dr + dt * self.a**2 * s**2) / rho2
        g[0, 3] = self.a * s**2 * (dr - dt * (r * r + self.a**2)) / (rho2 * xi)
        g[3, 0] = g[0, 3]
        g[1, 1] = rho2 / dr
        g[2, 2] = rho2 / dt
        g[3, 3] = (
            s**2
            * (dt * (r * r + self.a**2) ** 2 - dr * self.a**2 * s**2)
            / (rho2 * xi**2)
        )
        return g

    def inverse_metric(self, x: Array) -> Array:
        """Analytic inverse metric in Boyer--Lindquist coordinates.

        Avoiding a dense 4x4 inversion at every ODE stage is important for
        high-resolution renders, especially for near-critical rays that spend
        a long affine time close to the photon region.
        """
        coordinates = np.asarray(x)
        _, r, theta, _ = coordinates
        s = np.sin(theta)
        c = np.cos(theta)
        rho2 = r * r + self.a * self.a * c * c
        dr = self.delta_r(r)
        dt = self.delta_theta(theta)
        xi = self.Xi
        R = r * r + self.a * self.a

        g_inv = np.zeros((4, 4), dtype=np.result_type(coordinates.dtype, np.float64))
        g_inv[0, 0] = -(R * R / dr - self.a * self.a * s * s / dt) / rho2
        g_inv[0, 3] = -self.a * xi * (R / dr - 1.0 / dt) / rho2
        g_inv[3, 0] = g_inv[0, 3]
        g_inv[1, 1] = dr / rho2
        g_inv[2, 2] = dt / rho2
        g_inv[3, 3] = xi * xi * (1.0 / (dt * s * s) - self.a * self.a / dr) / rho2
        return g_inv

    def inverse_metric_derivatives(self, x: Array) -> Array:
        """Analytic partial derivatives ``partial_mu g^{alpha beta}``.

        Only the radial and polar derivatives are nonzero.  These expressions
        are algebraically differentiated from :meth:`inverse_metric` and are
        substantially faster than complex-step differentiation inside every
        Hamiltonian ODE evaluation.
        """
        x = np.asarray(x, dtype=float)
        _, r, theta, _ = x
        a = self.a
        L = self.L
        mu = self.mu
        xi = self.Xi
        s = np.sin(theta)
        c = np.cos(theta)
        s2 = s * s
        rho = r * r + a * a * c * c
        rho_r = 2.0 * r
        rho_t = -2.0 * a * a * c * s
        R = r * r + a * a
        R_r = 2.0 * r
        D = self.delta_r(r)
        D_r = 2.0 * r * (1.0 + r * r / L**2) + 2.0 * r * R / L**2 - 2.0 * mu
        T = self.delta_theta(theta)
        T_t = 2.0 * a * a * c * s / L**2

        A = R * R / D - a * a * s2 / T
        B = R / D - 1.0 / T
        C = 1.0 / (T * s2) - a * a / D

        A_r = (2.0 * R * R_r * D - R * R * D_r) / (D * D)
        B_r = (R_r * D - R * D_r) / (D * D)
        C_r = a * a * D_r / (D * D)

        A_t = -a * a * (2.0 * s * c * T - s2 * T_t) / (T * T)
        B_t = T_t / (T * T)
        q = T * s2
        q_t = T_t * s2 + 2.0 * T * s * c
        C_t = -q_t / (q * q)

        derivatives = np.zeros((4, 4, 4), dtype=float)
        dr = derivatives[1]
        dt = derivatives[2]

        dr[0, 0] = -A_r / rho + A * rho_r / (rho * rho)
        dr[0, 3] = -a * xi * (B_r / rho - B * rho_r / (rho * rho))
        dr[3, 0] = dr[0, 3]
        dr[1, 1] = D_r / rho - D * rho_r / (rho * rho)
        dr[2, 2] = -T * rho_r / (rho * rho)
        dr[3, 3] = xi * xi * (C_r / rho - C * rho_r / (rho * rho))

        dt[0, 0] = -A_t / rho + A * rho_t / (rho * rho)
        dt[0, 3] = -a * xi * (B_t / rho - B * rho_t / (rho * rho))
        dt[3, 0] = dt[0, 3]
        dt[1, 1] = -D * rho_t / (rho * rho)
        dt[2, 2] = T_t / rho - T * rho_t / (rho * rho)
        dt[3, 3] = xi * xi * (C_t / rho - C * rho_t / (rho * rho))
        return derivatives

    def horizon_angular_velocity(self) -> float:
        """Angular velocity relative to the nonrotating frame at infinity."""
        r_plus = self.horizon_radius()
        if r_plus is None:
            raise ValueError("metric has no event horizon")
        return float(
            self.a * (1.0 + r_plus * r_plus / self.L**2)
            / (r_plus * r_plus + self.a * self.a)
        )

    def hawking_reall_ratio(self) -> float:
        """Return ``|Omega_H| L``; the Hawking--Reall bound is <= 1."""
        return abs(self.horizon_angular_velocity()) * self.L

    def satisfies_hawking_reall_bound(self, tolerance: float = 1.0e-12) -> bool:
        return self.hawking_reall_ratio() <= 1.0 + tolerance

    def horizon_radius(self) -> float | None:
        # Delta_r = r^4/L^2 + (1+a^2/L^2)r^2 - 2mu r + a^2.
        roots = self.positive_real_roots(
            (1.0 / self.L**2, 0.0, 1.0 + self.a**2 / self.L**2, -2.0 * self.mu, self.a**2)
        )
        return roots[-1] if roots else None
