from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import signal

import matplotlib.pyplot as plt
import numpy as np

from .camera import PinholeCamera
from .sources import BoundaryPointSource, wavelength_to_rgb
from .tracer import RayStatus, RayTracer


class _RayWallTimeExceeded(TimeoutError):
    pass


def _ray_timeout_handler(_signum, _frame):
    raise _RayWallTimeExceeded("per-ray wall-time limit exceeded")


def _render_one_row(payload):
    """Process-pool worker; one task per row avoids per-pixel pickle overhead."""
    camera, tracer, source, row = payload
    w = camera.width
    rgb = np.zeros((w, 3), dtype=float)
    intensity = np.zeros(w, dtype=float)
    frequency_shift = np.full(w, np.nan, dtype=float)
    travel_time = np.full(w, np.nan, dtype=float)
    angular_miss = np.full(w, np.nan, dtype=float)
    status = np.empty(w, dtype="U20")
    null_error = np.full(w, np.nan, dtype=float)

    observed_frequency = 1.0
    for col in range(w):
        x0, p0 = camera.initial_state(row, col)
        wall_limit = tracer.config.ray_wall_time_limit_seconds
        use_alarm = wall_limit is not None and hasattr(signal, "setitimer")
        try:
            if use_alarm:
                signal.signal(signal.SIGALRM, _ray_timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, float(wall_limit))
            result = tracer.trace(x0, p0)
        except _RayWallTimeExceeded:
            status[col] = RayStatus.MAX_AFFINE.value
            null_error[col] = np.nan
            continue
        finally:
            if use_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
        status[col] = result.status.value
        null_error[col] = result.max_null_error
        if result.status not in (RayStatus.BOUNDARY, RayStatus.ORIGIN_BRIDGED):
            continue
        sample = source.sample(
            camera.metric,
            result.x_final,
            result.p_final,
            observed_frequency=observed_frequency,
        )
        frequency_shift[col] = sample.frequency_shift
        travel_time[col] = camera.position[0] - sample.emission_time
        angular_miss[col] = sample.angular_distance
        observed_intensity = sample.emitted_intensity * sample.frequency_shift**4
        intensity[col] = observed_intensity
        rgb[col] = wavelength_to_rgb(sample.observed_wavelength_nm) * observed_intensity

    return row, rgb, intensity, frequency_shift, travel_time, angular_miss, status, null_error


@dataclass
class RenderResult:
    rgb: np.ndarray
    intensity: np.ndarray
    frequency_shift: np.ndarray
    travel_time: np.ndarray
    angular_miss: np.ndarray
    status: np.ndarray
    null_error: np.ndarray

    def save(self, directory: str | Path, metadata: dict[str, object] | None = None) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        plt.imsave(path / "image.png", np.clip(self.rgb, 0.0, 1.0))
        np.savez_compressed(
            path / "render_data.npz",
            rgb=self.rgb,
            intensity=self.intensity,
            frequency_shift=self.frequency_shift,
            travel_time=self.travel_time,
            angular_miss=self.angular_miss,
            status=self.status,
            null_error=self.null_error,
        )
        self._save_scalar(path / "intensity.png", self.intensity, "Observed bolometric intensity")
        self._save_scalar(path / "frequency_shift.png", self.frequency_shift, "g = nu_obs / nu_emit")
        self._save_scalar(path / "travel_time.png", self.travel_time, "Coordinate travel time")
        self._save_scalar(path / "angular_miss.png", self.angular_miss, "Boundary angular distance")
        if metadata is not None:
            import json

            (path / "metadata.json").write_text(json.dumps(metadata, indent=2, default=float))

    @staticmethod
    def _save_scalar(path: Path, data: np.ndarray, title: str) -> None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        masked = np.ma.masked_invalid(data)
        image = ax.imshow(masked, origin="upper")
        ax.set_title(title)
        ax.set_xlabel("pixel x")
        ax.set_ylabel("pixel y")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)


class Renderer:
    def __init__(self, camera: PinholeCamera, tracer: RayTracer, source: BoundaryPointSource):
        self.camera = camera
        self.tracer = tracer
        self.source = source
        if abs(camera.position[1] - tracer.config.outer_radius) > 1e-9 * tracer.config.outer_radius:
            raise ValueError("camera radius must equal TraceConfig.outer_radius")

    def render(self, progress: bool = True, workers: int = 1) -> RenderResult:
        h, w = self.camera.height, self.camera.width
        rgb = np.zeros((h, w, 3), dtype=float)
        intensity = np.zeros((h, w), dtype=float)
        frequency_shift = np.full((h, w), np.nan, dtype=float)
        travel_time = np.full((h, w), np.nan, dtype=float)
        angular_miss = np.full((h, w), np.nan, dtype=float)
        status = np.empty((h, w), dtype="U20")
        null_error = np.full((h, w), np.nan, dtype=float)

        def assign(row_result):
            row, row_rgb, row_i, row_g, row_t, row_miss, row_status, row_null = row_result
            rgb[row] = row_rgb
            intensity[row] = row_i
            frequency_shift[row] = row_g
            travel_time[row] = row_t
            angular_miss[row] = row_miss
            status[row] = row_status
            null_error[row] = row_null

        workers = max(1, int(workers))
        if workers == 1:
            for row in range(h):
                if progress and (row == 0 or (row + 1) % max(1, h // 10) == 0):
                    print(f"render row {row + 1}/{h}", flush=True)
                assign(_render_one_row((self.camera, self.tracer, self.source, row)))
        else:
            completed = 0
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(_render_one_row, (self.camera, self.tracer, self.source, row))
                    for row in range(h)
                ]
                for future in as_completed(futures):
                    assign(future.result())
                    completed += 1
                    if progress and (completed == 1 or completed % max(1, h // 10) == 0):
                        print(f"render rows completed {completed}/{h}", flush=True)

        # Exposure normalization changes only display brightness, not saved intensity data.
        positive = intensity[intensity > 0.0]
        if positive.size:
            exposure = np.quantile(positive, 0.995)
            if exposure > 0.0:
                rgb = np.clip(rgb / exposure, 0.0, 1.0)
        return RenderResult(
            rgb=rgb,
            intensity=intensity,
            frequency_shift=frequency_shift,
            travel_time=travel_time,
            angular_miss=angular_miss,
            status=status,
            null_error=null_error,
        )
