from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.integrate import solve_ivp

from .geometry import hamilton_rhs, null_residual
from .metrics.base import Array, Metric4D


class RayStatus(str, Enum):
    BOUNDARY = "boundary"
    HORIZON = "horizon"
    ORIGIN_BRIDGED = "origin_bridged"
    MAX_AFFINE = "max_affine"
    FAILED = "failed"


@dataclass
class TraceConfig:
    outer_radius: float
    max_affine: float = 250.0
    rtol: float = 2e-9
    atol: float = 2e-11
    max_step: float = 0.2
    horizon_buffer: float = 2e-4
    origin_radius: float = 2e-5
    launch_offset_fraction: float = 2e-7
    store_path: bool = False
    max_rhs_evaluations: int | None = None
    ray_wall_time_limit_seconds: float | None = None


@dataclass
class RayResult:
    status: RayStatus
    x_final: Array
    p_final: Array
    affine_final: float
    max_null_error: float
    path: Array | None = None
    message: str = ""


class _IntegrationBudgetExceeded(RuntimeError):
    def __init__(self, affine: float, state: Array, evaluations: int):
        super().__init__(f"RHS evaluation budget exceeded after {evaluations} calls")
        self.affine = float(affine)
        self.state = np.asarray(state, dtype=float).copy()
        self.evaluations = int(evaluations)


class RayTracer:
    def __init__(self, metric: Metric4D, config: TraceConfig):
        self.metric = metric
        self.config = config
        if config.outer_radius <= 0.0:
            raise ValueError("outer_radius must be positive")
        horizon = metric.horizon_radius()
        if horizon is not None and config.outer_radius <= horizon + config.horizon_buffer:
            raise ValueError("outer_radius must lie outside the event horizon")

    def _outer_event(self, _lam: float, y: Array) -> float:
        return float(y[1] - self.config.outer_radius)

    def _horizon_event(self, _lam: float, y: Array) -> float:
        horizon = self.metric.horizon_radius()
        if horizon is None:
            return 1.0
        return float(y[1] - horizon - self.config.horizon_buffer)

    def _origin_event(self, _lam: float, y: Array) -> float:
        if not self.metric.regular_origin:
            return 1.0
        return float(y[1] - self.config.origin_radius)

    def _integrate_segment(self, y0: Array, lambda0: float, lambda1: float):
        outer = lambda lam, y: self._outer_event(lam, y)
        horizon = lambda lam, y: self._horizon_event(lam, y)
        origin = lambda lam, y: self._origin_event(lam, y)
        outer.terminal = True
        outer.direction = 1.0
        horizon.terminal = True
        horizon.direction = -1.0
        origin.terminal = True
        origin.direction = -1.0
        evaluations = 0
        budget = self.config.max_rhs_evaluations

        def rhs(lam: float, y: Array) -> Array:
            nonlocal evaluations
            evaluations += 1
            if budget is not None and evaluations > budget:
                raise _IntegrationBudgetExceeded(lam, y, evaluations)
            return hamilton_rhs(self.metric, lam, y)

        return solve_ivp(
            rhs,
            (lambda0, lambda1),
            y0,
            method="DOP853",
            rtol=self.config.rtol,
            atol=self.config.atol,
            max_step=self.config.max_step,
            events=(outer, horizon, origin),
            dense_output=False,
        )

    @staticmethod
    def _bridge_origin(y: Array, origin_radius: float) -> Array:
        """Continue an exactly radial ray through the regular r=0 chart singularity."""
        result = np.asarray(y, dtype=float).copy()
        result[1] = origin_radius * (1.0 + 1e-5)
        result[2] = np.pi - result[2]
        result[3] = np.mod(result[3] + np.pi, 2.0 * np.pi)
        result[5] *= -1.0  # p_r changes sign under the antipodal spherical re-charting.
        return result

    def _project_radial_momentum_to_null(self, x: Array, p: Array) -> Array:
        """Adjust only p_r so the shifted launch state remains exactly null."""
        result = np.asarray(p, dtype=float).copy()
        g_inv = np.asarray(self.metric.inverse_metric(x), dtype=float)
        r_index = 1
        others = [0, 2, 3]
        a = float(g_inv[r_index, r_index])
        b = float(sum(g_inv[r_index, j] * result[j] for j in others))
        c = float(sum(g_inv[i, j] * result[i] * result[j] for i in others for j in others))
        discriminant = b * b - a * c
        if discriminant < 0.0 and discriminant > -1e-12 * max(1.0, b * b, abs(a * c)):
            discriminant = 0.0
        if discriminant < 0.0 or abs(a) < np.finfo(float).tiny:
            raise RuntimeError("cannot project launch momentum onto the local null cone")
        roots = [(-b + np.sqrt(discriminant)) / a, (-b - np.sqrt(discriminant)) / a]
        result[r_index] = min(roots, key=lambda value: abs(value - result[r_index]))
        return result

    def trace(self, x0: Array, p0: Array) -> RayResult:
        x0 = np.asarray(x0, dtype=float).copy()
        p0 = np.asarray(p0, dtype=float).copy()
        # Start just inside the cutoff surface, preventing a spurious lambda=0 hit.
        if abs(x0[1] - self.config.outer_radius) <= 10.0 * np.finfo(float).eps * self.config.outer_radius:
            x0[1] = self.config.outer_radius * (1.0 - self.config.launch_offset_fraction)
            p0 = self._project_radial_momentum_to_null(x0, p0)
        y = np.concatenate((x0, p0))
        lambda0 = 0.0
        paths: list[Array] = []
        origin_bridged = False

        for _segment in range(2):
            try:
                sol = self._integrate_segment(y, lambda0, self.config.max_affine)
            except _IntegrationBudgetExceeded as exc:
                return self._result(
                    RayStatus.MAX_AFFINE,
                    exc.state,
                    exc.affine,
                    paths,
                    str(exc),
                )
            if self.config.store_path:
                paths.append(sol.y.T)
            if not sol.success:
                return self._result(RayStatus.FAILED, sol.y[:, -1], sol.t[-1], paths, sol.message)

            event_times = [events[0] if len(events) else np.inf for events in sol.t_events]
            event_index = int(np.argmin(event_times))
            if not np.isfinite(event_times[event_index]):
                return self._result(RayStatus.MAX_AFFINE, sol.y[:, -1], sol.t[-1], paths)

            y_event = sol.y_events[event_index][0]
            lambda_event = float(event_times[event_index])
            if event_index == 0:
                status = RayStatus.ORIGIN_BRIDGED if origin_bridged else RayStatus.BOUNDARY
                # A bridged ray still physically ended at the boundary.
                result = self._result(status, y_event, lambda_event, paths)
                if status is RayStatus.ORIGIN_BRIDGED:
                    result.message = "boundary hit after regular-origin bridge"
                return result
            if event_index == 1:
                return self._result(RayStatus.HORIZON, y_event, lambda_event, paths)
            if event_index == 2 and self.metric.regular_origin and not origin_bridged:
                y = self._bridge_origin(y_event, self.config.origin_radius)
                lambda0 = lambda_event
                origin_bridged = True
                continue
            return self._result(RayStatus.FAILED, y_event, lambda_event, paths, "unexpected event")

        return self._result(RayStatus.FAILED, y, lambda0, paths, "too many origin bridges")

    def _result(
        self,
        status: RayStatus,
        y_final: Array,
        affine_final: float,
        paths: list[Array],
        message: str = "",
    ) -> RayResult:
        y_final = np.asarray(y_final, dtype=float)
        combined = np.concatenate(paths, axis=0) if paths else None
        samples = combined if combined is not None else y_final[None, :]
        errors = [null_residual(self.metric, row[:4], row[4:]) for row in samples[:: max(1, len(samples)//100)]]
        return RayResult(
            status=status,
            x_final=self.metric.wrap_coordinates(y_final[:4]),
            p_final=y_final[4:].copy(),
            affine_final=float(affine_final),
            max_null_error=float(max(errors, default=np.nan)),
            path=combined,
            message=message,
        )
