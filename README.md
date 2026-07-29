# Kerr–AdS continuity experiment

Controlled ray tracing from Schwarzschild–AdS to Kerr–AdS with matched mass, camera, source, numerical settings, and a common exposure scale.

<a href="https://raw.githubusercontent.com/arvinshm/ads-black-hole-raytracer/main/figures/continuity_experiment.png">
  <img src="https://raw.githubusercontent.com/arvinshm/ads-black-hole-raytracer/main/figures/continuity_experiment.png" alt="Kerr–AdS continuity experiment" width="100%">
</a>

[Open the full-resolution figure](https://raw.githubusercontent.com/arvinshm/ads-black-hole-raytracer/main/figures/continuity_experiment.png)

The displayed run uses

```text
Omega_H L = 0, 0.01, 0.03, 0.10, 0.30
```

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]

python examples/run_continuity_experiment.py \
  --preset misaligned \
  --width 96 --height 96 \
  --workers 10
```

Custom rotation targets can be supplied directly:

```bash
python examples/run_continuity_experiment.py \
  --targets 0,0.01,0.03,0.1,0.3 \
  --preset misaligned
```

More details are in [`docs/LOCAL_RUN_AND_CONTINUITY_EXPERIMENT.md`](docs/LOCAL_RUN_AND_CONTINUITY_EXPERIMENT.md).
