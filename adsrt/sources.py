from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import angular_separation, covector_to_vector, metric_dot
from .metrics.base import Array, Metric4D
from .tetrads import boosted_observer, make_tetrad


@dataclass(frozen=True)
class SourceSample:
    emitted_intensity: float
    frequency_shift: float
    observed_wavelength_nm: float
    emission_cosine: float
    angular_distance: float
    emission_time: float


def wavelength_to_rgb(wavelength_nm: float) -> Array:
    """Approximate visible-spectrum sRGB for diagnostic rendering.

    Values outside 380--780 nm are shown as black.  This is a visualization
    helper, not a calibrated color-management or detector-response model.
    """
    w = float(wavelength_nm)
    if not 380.0 <= w <= 780.0:
        return np.zeros(3, dtype=float)
    if w < 440.0:
        rgb = np.array([-(w - 440.0) / 60.0, 0.0, 1.0])
    elif w < 490.0:
        rgb = np.array([0.0, (w - 440.0) / 50.0, 1.0])
    elif w < 510.0:
        rgb = np.array([0.0, 1.0, -(w - 510.0) / 20.0])
    elif w < 580.0:
        rgb = np.array([(w - 510.0) / 70.0, 1.0, 0.0])
    elif w < 645.0:
        rgb = np.array([1.0, -(w - 645.0) / 65.0, 0.0])
    else:
        rgb = np.array([1.0, 0.0, 0.0])
    if w < 420.0:
        factor = 0.3 + 0.7 * (w - 380.0) / 40.0
    elif w <= 700.0:
        factor = 1.0
    else:
        factor = 0.3 + 0.7 * (780.0 - w) / 80.0
    return np.clip(rgb * factor, 0.0, 1.0)


@dataclass
class BoundaryPointSource:
    """A finite Gaussian patch approximating a point source on the AdS boundary."""

    theta: float = np.pi / 2.0
    phi: float = 0.0
    angular_sigma: float = 0.035
    wavelength_nm: float = 550.0
    peak_intensity: float = 1.0
    emitter_velocity_local: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: float | None = None
    pulse_center_time: float | None = None
    pulse_sigma_time: float = 0.1
    reference_time: float = 0.0
    lambertian: bool = False

    def __post_init__(self) -> None:
        if self.angular_sigma <= 0.0:
            raise ValueError("angular_sigma must be positive")
        if self.wavelength_nm <= 0.0:
            raise ValueError("wavelength_nm must be positive")
        if self.pulse_sigma_time <= 0.0:
            raise ValueError("pulse_sigma_time must be positive")

    def sample(
        self,
        metric: Metric4D,
        x_hit: Array,
        p_past_cov: Array,
        observed_frequency: float = 1.0,
    ) -> SourceSample:
        # ``phi`` is the source-center coordinate at ``reference_time``.  In
        # Kerr-AdS a nonrotating boundary observer has dphi/dt -> -a/L^2, so
        # keeping the center fixed in coordinate phi would describe the wrong
        # worldline.  At fixed r,theta in a stationary axisymmetric metric the
        # requested local velocity gives a constant coordinate angular velocity.
        center_probe = np.array(
            [float(x_hit[0]), float(x_hit[1]), self.theta, self.phi], dtype=float
        )
        center_tetrad = make_tetrad(metric, center_probe, self.angular_velocity)
        center_u = boosted_observer(center_tetrad, self.emitter_velocity_local)
        center_omega = float(center_u[3] / center_u[0])
        center_phi = float(
            np.mod(self.phi + center_omega * (float(x_hit[0]) - self.reference_time), 2.0 * np.pi)
        )
        distance = angular_separation(
            float(x_hit[2]), float(x_hit[3]), self.theta, center_phi
        )
        spatial_profile = np.exp(-0.5 * (distance / self.angular_sigma) ** 2)

        source_tetrad = make_tetrad(metric, x_hit, self.angular_velocity)
        u_emit = boosted_observer(source_tetrad, self.emitter_velocity_local)
        emitted_frequency = abs(float(np.asarray(p_past_cov) @ u_emit))
        if emitted_frequency <= 0.0 or not np.isfinite(emitted_frequency):
            return SourceSample(0.0, np.nan, np.nan, 0.0, distance, float(x_hit[0]))
        g_factor = observed_frequency / emitted_frequency

        # Reverse the past-directed ray to recover the physical future photon.
        k_future = -covector_to_vector(metric, x_hit, p_past_cov)
        n_emit = k_future / emitted_frequency - u_emit
        inward = -source_tetrad.e_r
        emission_cosine = metric_dot(metric.metric(x_hit), n_emit, inward)
        hemisphere = max(0.0, emission_cosine)
        angular_emission = hemisphere if self.lambertian else (1.0 if hemisphere > -1e-8 else 0.0)

        temporal = 1.0
        if self.pulse_center_time is not None:
            dt = (float(x_hit[0]) - self.pulse_center_time) / self.pulse_sigma_time
            temporal = np.exp(-0.5 * dt * dt)

        emitted_intensity = float(self.peak_intensity * spatial_profile * angular_emission * temporal)
        observed_wavelength = self.wavelength_nm / g_factor
        return SourceSample(
            emitted_intensity=emitted_intensity,
            frequency_shift=float(g_factor),
            observed_wavelength_nm=float(observed_wavelength),
            emission_cosine=float(emission_cosine),
            angular_distance=float(distance),
            emission_time=float(x_hit[0]),
        )
