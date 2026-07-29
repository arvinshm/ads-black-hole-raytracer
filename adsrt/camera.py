from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import hamiltonian, vector_to_covector
from .metrics.base import Array, Metric4D
from .tetrads import OrthonormalTetrad, make_tetrad


@dataclass
class PinholeCamera:
    metric: Metric4D
    position: Array
    width: int = 128
    height: int = 128
    horizontal_fov_degrees: float = 100.0
    angular_velocity: float | None = None

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        if self.position.shape != (4,):
            raise ValueError("camera position must be [t,r,theta,phi]")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")
        if not 0.0 < self.horizontal_fov_degrees < 179.0:
            raise ValueError("horizontal FOV must be between 0 and 179 degrees")
        self.tetrad: OrthonormalTetrad = make_tetrad(
            self.metric, self.position, self.angular_velocity
        )

    @property
    def vertical_fov_radians(self) -> float:
        horizontal = np.deg2rad(self.horizontal_fov_degrees)
        aspect = self.height / self.width
        return float(2.0 * np.arctan(np.tan(horizontal / 2.0) * aspect))

    def screen_coordinates(self, row: int, col: int) -> tuple[float, float]:
        # Pixel centers in [-1,1], with y positive upward in the final image.
        x = 2.0 * (col + 0.5) / self.width - 1.0
        y = 1.0 - 2.0 * (row + 0.5) / self.height
        return float(x), float(y)

    def initial_state(self, row: int, col: int) -> tuple[Array, Array]:
        """Return (x,p_cov) for a past-directed ray launched into the scene."""
        sx, sy = self.screen_coordinates(row, col)
        hfov = np.deg2rad(self.horizontal_fov_degrees)
        vfov = self.vertical_fov_radians
        x_plane = sx * np.tan(hfov / 2.0)
        y_plane = sy * np.tan(vfov / 2.0)

        # Backward optical direction: inward, screen-right = +phi, screen-up = -theta.
        direction = -self.tetrad.e_r + x_plane * self.tetrad.e_phi - y_plane * self.tetrad.e_theta
        g = self.metric.metric(self.position)
        norm = np.sqrt(float(direction @ g @ direction))
        direction /= norm

        # If k_future = u - direction is the future photon arriving at the camera,
        # then k_past = -k_future = -u + direction traces the same ray backward.
        k_past = -self.tetrad.u + direction
        p_cov = vector_to_covector(self.metric, self.position, k_past)
        null_error = abs(hamiltonian(self.metric, self.position, p_cov))
        if null_error > 1e-10:
            raise RuntimeError(f"camera ray initialization is not null: H={null_error}")
        return self.position.copy(), p_cov
