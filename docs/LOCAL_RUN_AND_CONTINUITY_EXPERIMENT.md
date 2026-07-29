# Local run and continuity experiment

## Apple Silicon setup

```bash
uname -m
python3 -c "import platform; print(platform.machine())"
```

Both commands should report `arm64` on a native Apple Silicon installation.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
pytest -q
```

If an offline environment cannot obtain build dependencies, try:

```bash
python -m pip install -e . --no-build-isolation
```

## Fast smoke test

```bash
python examples/run_continuity_experiment.py \
  --preset exact_antipodal \
  --targets 0,0.1 \
  --width 4 --height 4 --workers 2
```

## Controlled continuity run

Exact antipodal source:

```bash
python examples/run_continuity_experiment.py \
  --preset exact_antipodal \
  --targets 0,0.01,0.03,0.1,0.3 \
  --width 96 --height 96 --workers 10
```

Slightly misaligned source:

```bash
python examples/run_continuity_experiment.py \
  --preset misaligned \
  --targets 0,0.01,0.03,0.1,0.3 \
  --width 96 --height 96 --workers 10
```

The misaligned preset moves the source by `0.10` radians in azimuth. It removes the exact Schwarzschild Einstein-ring degeneracy and is therefore a cleaner visual test of ordinary image continuity as `a -> 0`.

For strong rotation:

```bash
python examples/run_continuity_experiment.py \
  --preset misaligned \
  --targets 0,0.3,0.5,0.8,0.98,1.2 \
  --source-sigma 0.025 \
  --width 32 --height 32 --workers 10
```

The target solver locates the regular-horizon endpoint for fixed `(L, mu)` and reports a clear error when a requested `Omega_H L` is unattainable.

## Runtime

The current renderer uses CPU processes and SciPy's adaptive DOP853 integrator. It does not use a GPU. Every pixel launches an independent ODE solve, and rays near critical photon trajectories may be much more expensive than ordinary rays.

Recommended progression:

```text
4x4 or 8x8     installation smoke test
32x32          geometry and parameter check
64x64          intermediate render
96x96 or more  final render
```

On an M4 Pro, start near `--workers 10`; on an M4 Max, try `--workers 12`. Benchmark nearby values because the optimum depends on thermal conditions and process overhead.

Outputs are written beneath `outputs/continuity_experiment/<preset>/` unless `--output-root` is supplied. Each case contains images, raw NumPy arrays, status maps, metadata, and a configuration snapshot. The runner also creates a common-exposure comparison image.
