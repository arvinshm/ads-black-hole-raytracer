# AdS black-hole ray tracer

This package backward-traces null geodesics from a pinhole camera through 3+1-dimensional asymptotically AdS metrics.

Implemented metrics:

- pure AdS;
- Schwarzschild–AdS;
- Kerr–AdS in Boyer–Lindquist coordinates;
- custom metrics through the `Metric4D` interface.

The camera and source are placed on a large finite cutoff surface `r = outer_radius`. This regulates the conformal-boundary problem; increasing `outer_radius/L` approaches the boundary limit.

Each camera pixel fixes a past-directed null initial condition. The tracer integrates Hamilton's equations

```text
H = 1/2 g^{mu nu} p_mu p_nu = 0,
dx^mu/dlambda = g^{mu nu} p_nu,
dp_mu/dlambda = -1/2 partial_mu g^{alpha beta} p_alpha p_beta.
```

A ray terminates at the outer boundary, the regulated horizon, the regular pure-AdS origin bridge, or its numerical budget. Saved data include RGB intensity, bolometric intensity, frequency shift, travel time, source angular miss, status, and null-constraint residual.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
```

## Validation

```bash
pytest -q
adsrt validate --json
```

The tests check metric inversion, tetrads, pure-AdS refocusing, Schwarzschild capture, Kerr reduction at `a=0`, Kerr conserved momenta, analytic inverse-metric derivatives, and Doppler shift.

## Rendering

```bash
adsrt render examples/configs/pure_ads.json
adsrt render examples/configs/schwarzschild_ads.json
adsrt render examples/configs/kerr_ads_static.json
adsrt render examples/configs/kerr_ads.json
```

## Continuity experiment

```bash
python examples/run_continuity_experiment.py \
  --preset misaligned \
  --targets 0,0.01,0.03,0.1,0.3 \
  --width 32 --height 32 --workers 8
```

The runner holds the camera, source width, mass parameter, numerical tolerances, and exposure convention fixed while changing the Kerr horizon angular velocity. It includes a Schwarzschild case and an `a=0` Kerr case as a strict implementation check.

Targets above the Hawking–Reall value `Omega_H L = 1` can also be requested when a regular horizon exists for the chosen `(L, mu)` family. The code treats the stationary metric as a fixed background and does not simulate superradiant backreaction.

## Custom metrics

See `examples/custom_metric.py` and `adsrt/metrics/base.py`. A custom metric should inherit from `Metric4D` and provide at least `metric(x)`. Overriding `inverse_metric(x)` and `inverse_metric_derivatives(x)` is recommended for speed.
