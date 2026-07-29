from __future__ import annotations

import math

import numpy as np

from .camera import PinholeCamera
from .geometry import angular_separation, null_residual, vector_to_covector
from .metrics.kerr_ads import KerrAdS4
from .metrics.spherical import PureAdS4, SchwarzschildAdS4
from .sources import BoundaryPointSource
from .tetrads import make_tetrad, tetrad_error
from .tracer import RayStatus, RayTracer, TraceConfig


def _launch_at_local_angle(metric, x, alpha: float):
    tetrad = make_tetrad(metric, x)
    direction = -np.cos(alpha) * tetrad.e_r + np.sin(alpha) * tetrad.e_phi
    k_past = -tetrad.u + direction
    return vector_to_covector(metric, x, k_past)


def _metric_algebra_checks() -> dict[str, float]:
    metrics_and_points = [
        (PureAdS4(L=1.0), np.array([0.1, 2.3, 1.1, 0.7])),
        (SchwarzschildAdS4(L=1.0, mu=0.25), np.array([0.1, 2.3, 1.1, 0.7])),
        (KerrAdS4(L=1.0, mu=0.5, a=0.25), np.array([0.1, 2.3, 1.1, 0.7])),
    ]
    inverse_errors = []
    frame_errors = []
    for metric, x in metrics_and_points:
        metric.validate_point(x)
        inverse_errors.append(float(np.max(np.abs(metric.metric(x) @ metric.inverse_metric(x) - np.eye(4)))))
        frame_errors.append(tetrad_error(metric, x, make_tetrad(metric, x)))
    return {
        "max_metric_inverse_error": max(inverse_errors),
        "max_tetrad_orthonormality_error": max(frame_errors),
        "passed": bool(max(inverse_errors) < 1e-11 and max(frame_errors) < 1e-11),
    }


def _pure_ads_checks() -> dict[str, object]:
    metric = PureAdS4(L=1.0)
    cutoffs = [10.0, 20.0, 40.0]
    misses = []
    time_errors = []
    null_errors = []
    for radius in cutoffs:
        camera = PinholeCamera(
            metric,
            np.array([0.0, radius, np.pi / 2.0, np.pi]),
            width=3,
            height=1,
            horizontal_fov_degrees=90.0,
        )
        tracer = RayTracer(
            metric,
            TraceConfig(radius, max_affine=20.0, max_step=0.05, store_path=True),
        )
        result = tracer.trace(*camera.initial_state(0, 2))
        misses.append(angular_separation(result.x_final[2], result.x_final[3], np.pi / 2.0, 0.0))
        time_errors.append(abs((-result.x_final[0]) - np.pi))
        null_errors.append(result.max_null_error)
        if result.status not in (RayStatus.BOUNDARY, RayStatus.ORIGIN_BRIDGED):
            raise AssertionError("pure AdS ray did not return to the cutoff boundary")

    radius = 20.0
    camera = PinholeCamera(
        metric,
        np.array([0.0, radius, np.pi / 2.0, np.pi]),
        width=3,
        height=3,
        horizontal_fov_degrees=90.0,
    )
    result = RayTracer(
        metric,
        TraceConfig(radius, max_affine=20.0, max_step=0.03, store_path=True),
    ).trace(*camera.initial_state(1, 1))
    expected_radial_time = 2.0 * np.arctan(radius)
    radial_time_error = abs((-result.x_final[0]) - expected_radial_time)
    radial_endpoint_miss = angular_separation(
        result.x_final[2], result.x_final[3], np.pi / 2.0, 0.0
    )

    decreasing = all(misses[i + 1] < misses[i] for i in range(len(misses) - 1))
    approximately_inverse_cutoff = 1.6 < misses[0] / misses[1] < 2.4 and 1.6 < misses[1] / misses[2] < 2.4
    passed = (
        decreasing
        and approximately_inverse_cutoff
        and misses[-1] < 0.04
        and time_errors[-1] < 0.07
        and radial_time_error < 2e-4
        and radial_endpoint_miss < 2e-5
        and max(null_errors + [result.max_null_error]) < 2e-5
    )
    return {
        "cutoff_radii": cutoffs,
        "antipodal_angular_miss_radians": misses,
        "boundary_time_error_vs_pi": time_errors,
        "radial_exact_time_error": radial_time_error,
        "radial_endpoint_miss": radial_endpoint_miss,
        "max_relative_null_residual": max(null_errors + [result.max_null_error]),
        "passed": bool(passed),
    }


def _schwarzschild_ads_checks() -> dict[str, object]:
    metric = SchwarzschildAdS4(L=1.0, mu=0.25)
    radius = 20.0
    x = np.array([0.0, radius, np.pi / 2.0, np.pi])
    tracer = RayTracer(metric, TraceConfig(radius, max_affine=100.0, max_step=0.05, store_path=True))

    photon_sphere = 3.0 * metric.mu
    b_critical = photon_sphere / np.sqrt(metric.f(photon_sphere))
    alpha_critical = np.arcsin(b_critical * np.sqrt(metric.f(radius)) / radius)
    captured = tracer.trace(x, _launch_at_local_angle(metric, x, 0.95 * alpha_critical))
    escaped = tracer.trace(x, _launch_at_local_angle(metric, x, 1.05 * alpha_critical))
    horizon = metric.horizon_radius()
    horizon_residual = abs(metric.f(horizon)) if horizon is not None else np.inf

    passed = (
        captured.status is RayStatus.HORIZON
        and escaped.status in (RayStatus.BOUNDARY, RayStatus.ORIGIN_BRIDGED)
        and horizon_residual < 1e-10
        and max(captured.max_null_error, escaped.max_null_error) < 2e-5
    )
    return {
        "horizon_radius": horizon,
        "horizon_equation_residual": horizon_residual,
        "photon_sphere_radius": photon_sphere,
        "critical_local_angle_degrees": np.rad2deg(alpha_critical),
        "below_critical_status": captured.status.value,
        "above_critical_status": escaped.status.value,
        "max_relative_null_residual": max(captured.max_null_error, escaped.max_null_error),
        "passed": bool(passed),
    }


def _kerr_ads_checks() -> dict[str, object]:
    mu = 0.5
    spherical = SchwarzschildAdS4(L=1.0, mu=mu)
    kerr_zero = KerrAdS4(L=1.0, mu=mu, a=0.0)
    points = [
        np.array([0.0, 1.4, 0.8, 0.2]),
        np.array([0.0, 3.0, 1.2, 4.1]),
    ]
    reduction_error = max(
        float(np.max(np.abs(spherical.metric(x) - kerr_zero.metric(x)))) for x in points
    )

    metric = KerrAdS4(L=1.0, mu=mu, a=0.25)
    algebra_point = np.array([0.0, 2.3, 1.1, 0.4])
    analytic_inverse_error = float(
        np.max(np.abs(metric.inverse_metric(algebra_point) - np.linalg.inv(metric.metric(algebra_point))))
    )
    analytic_derivative_error = 0.0
    analytic_derivatives = metric.inverse_metric_derivatives(algebra_point)
    for coordinate in (1, 2):
        h = 1.0e-20 * max(1.0, abs(float(algebra_point[coordinate])))
        xc = algebra_point.astype(complex)
        xc[coordinate] += 1j * h
        reference = np.imag(np.linalg.inv(metric.metric(xc))) / h
        analytic_derivative_error = max(
            analytic_derivative_error,
            float(np.max(np.abs(analytic_derivatives[coordinate] - reference))),
        )

    near_bound = KerrAdS4(L=1.0, mu=mu, a=0.3182612449684538)
    near_bound_ratio = near_bound.hawking_reall_ratio()
    near_bound_margin = 1.0 - near_bound_ratio

    radius = 20.0
    camera = PinholeCamera(
        metric,
        np.array([0.0, radius, np.pi / 2.0, np.pi]),
        width=3,
        height=1,
        horizontal_fov_degrees=140.0,
    )
    x0, p0 = camera.initial_state(0, 0)
    result = RayTracer(
        metric,
        TraceConfig(radius, max_affine=100.0, max_step=0.05, store_path=True),
    ).trace(x0, p0)
    pt_drift = abs(result.p_final[0] - p0[0])
    pphi_drift = abs(result.p_final[3] - p0[3])
    horizon = metric.horizon_radius()
    horizon_residual = abs(metric.delta_r(horizon)) if horizon is not None else np.inf
    omega_at_large_r = metric.default_angular_velocity(np.array([0.0, 100.0, np.pi / 2.0, 0.0]))
    omega_infinity_error = abs(omega_at_large_r + metric.a / metric.L**2)

    passed = (
        reduction_error < 1e-11
        and result.status in (RayStatus.BOUNDARY, RayStatus.HORIZON)
        and pt_drift < 1e-11
        and pphi_drift < 1e-11
        and result.max_null_error < 2e-5
        and horizon_residual < 1e-9
        and omega_infinity_error < 2e-4
        and analytic_inverse_error < 1e-11
        and analytic_derivative_error < 1e-10
        and abs(near_bound_ratio - 0.98) < 2e-12
        and near_bound.satisfies_hawking_reall_bound()
    )
    return {
        "a_to_zero_metric_reduction_error": reduction_error,
        "outer_horizon_radius": horizon,
        "horizon_equation_residual": horizon_residual,
        "test_ray_status": result.status.value,
        "p_t_drift": pt_drift,
        "p_phi_drift": pphi_drift,
        "max_relative_null_residual": result.max_null_error,
        "zamo_omega_large_r": omega_at_large_r,
        "zamo_omega_error_vs_minus_a_over_L2": omega_infinity_error,
        "analytic_inverse_error": analytic_inverse_error,
        "analytic_inverse_derivative_error": analytic_derivative_error,
        "near_bound_Omega_H_times_L": near_bound_ratio,
        "near_bound_margin": near_bound_margin,
        "near_bound_satisfies_hawking_reall": near_bound.satisfies_hawking_reall_bound(),
        "passed": bool(passed),
    }


def _radiometry_checks() -> dict[str, object]:
    metric = PureAdS4(L=1.0)
    x = np.array([0.0, 20.0, np.pi / 2.0, 0.0])
    tetrad = make_tetrad(metric, x)
    k_future = tetrad.u - tetrad.e_r
    p_past = vector_to_covector(metric, x, -k_future)
    speed = 0.3
    source = BoundaryPointSource(
        theta=np.pi / 2.0,
        phi=0.0,
        angular_sigma=0.05,
        emitter_velocity_local=(-speed, 0.0, 0.0),
    )
    sample = source.sample(metric, x, p_past, observed_frequency=1.0)
    expected = math.sqrt((1.0 + speed) / (1.0 - speed))
    error = abs(sample.frequency_shift - expected)
    return {
        "computed_doppler_factor": sample.frequency_shift,
        "special_relativistic_expected_factor": expected,
        "absolute_error": error,
        "passed": bool(error < 1e-11 and sample.emitted_intensity > 0.0),
    }


def run_validation_suite() -> dict[str, dict[str, object]]:
    report = {
        "metric_algebra": _metric_algebra_checks(),
        "pure_ads": _pure_ads_checks(),
        "schwarzschild_ads": _schwarzschild_ads_checks(),
        "kerr_ads": _kerr_ads_checks(),
        "radiometry": _radiometry_checks(),
    }
    report["summary"] = {
        "passed": bool(all(section.get("passed", False) for name, section in report.items() if name != "summary")),
        "sections": len(report),
    }
    return report
