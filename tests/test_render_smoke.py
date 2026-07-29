import numpy as np

from adsrt.camera import PinholeCamera
from adsrt.metrics import PureAdS4, SchwarzschildAdS4
from adsrt.renderer import Renderer
from adsrt.sources import BoundaryPointSource
from adsrt.tracer import RayTracer, TraceConfig


def test_pure_ads_small_render_is_finite_and_refocused():
    metric = PureAdS4(L=1.0)
    radius = 15.0
    camera = PinholeCamera(metric, np.array([0.0, radius, np.pi / 2.0, np.pi]), width=6, height=4, horizontal_fov_degrees=100.0)
    tracer = RayTracer(metric, TraceConfig(radius, max_affine=20.0, max_step=0.1))
    source = BoundaryPointSource(theta=np.pi / 2.0, phi=0.0, angular_sigma=0.15)
    result = Renderer(camera, tracer, source).render(progress=False)
    assert result.rgb.shape == (4, 6, 3)
    assert np.all(np.isfinite(result.rgb))
    assert np.count_nonzero(result.intensity) == 24
    assert np.nanmax(np.abs(result.frequency_shift - 1.0)) < 1e-10


def test_schwarzschild_center_ray_is_black_hole_capture():
    metric = SchwarzschildAdS4(L=1.0, mu=0.25)
    radius = 15.0
    camera = PinholeCamera(metric, np.array([0.0, radius, np.pi / 2.0, np.pi]), width=3, height=3, horizontal_fov_degrees=100.0)
    tracer = RayTracer(metric, TraceConfig(radius, max_affine=50.0, max_step=0.08))
    x, p = camera.initial_state(1, 1)
    assert tracer.trace(x, p).status.value == "horizon"
