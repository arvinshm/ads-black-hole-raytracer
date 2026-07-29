import numpy as np

from adsrt.geometry import null_residual
from adsrt.metrics import KerrAdS4, SchwarzschildAdS4
from adsrt.tetrads import make_tetrad


def test_kerr_ads_reduces_to_schwarzschild_ads_at_zero_rotation():
    x = np.array([0.0, 2.0, 1.0, 0.3])
    a0 = KerrAdS4(L=1.0, mu=0.5, a=0.0)
    spherical = SchwarzschildAdS4(L=1.0, mu=0.5)
    assert np.allclose(a0.metric(x), spherical.metric(x), atol=2e-14, rtol=2e-14)


def test_local_null_vector_is_null():
    metric = KerrAdS4(L=1.0, mu=0.5, a=0.25)
    x = np.array([0.0, 10.0, np.pi / 2.0, 0.0])
    tetrad = make_tetrad(metric, x)
    k = tetrad.u + 0.6 * tetrad.e_r + 0.8 * tetrad.e_phi
    p = metric.metric(x) @ k
    assert null_residual(metric, x, p) < 1e-13


def test_kerr_analytic_inverse_and_derivatives():
    metric = KerrAdS4(L=1.0, mu=0.5, a=0.3182612449684538)
    x = np.array([0.0, 2.3, 1.1, 0.4])
    assert np.allclose(metric.inverse_metric(x), np.linalg.inv(metric.metric(x)), atol=2e-13, rtol=2e-13)
    derivatives = metric.inverse_metric_derivatives(x)
    for coordinate in (1, 2):
        h = 1e-20 * max(1.0, abs(float(x[coordinate])))
        xc = x.astype(complex)
        xc[coordinate] += 1j * h
        reference = np.imag(np.linalg.inv(metric.metric(xc))) / h
        assert np.allclose(derivatives[coordinate], reference, atol=2e-12, rtol=2e-12)


def test_hawking_reall_ratio_helper():
    metric = KerrAdS4(L=1.0, mu=0.5, a=0.3182612449684538)
    assert abs(metric.hawking_reall_ratio() - 0.98) < 2e-12
    assert metric.satisfies_hawking_reall_bound()
