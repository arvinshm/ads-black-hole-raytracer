# Kerr–AdS continuity experiment

Controlled ray tracing from Schwarzschild–AdS to Kerr–AdS with matched mass, camera, source, numerical settings, and a common exposure scale.

![Kerr–AdS continuity results](figures/continuity_experiment/latest_targets.jpg)

The displayed `96 x 96` misaligned-source run contains:

```text
Schwarzschild–AdS
Kerr–AdS: Omega_H L = 0, 0.30, 0.70, 0.98, 1.28, 1.55
```

The last two Kerr–AdS cases are above the Hawking–Reall bound, `Omega_H L = 1`, and are fixed-background ray traces.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]

python examples/run_continuity_experiment.py \
  --preset misaligned \
  --targets 0,0.30,0.70,0.98,1.28,1.55 \
  --width 96 --height 96 \
  --workers 10
```

More details are in [`docs/LOCAL_RUN_AND_CONTINUITY_EXPERIMENT.md`](docs/LOCAL_RUN_AND_CONTINUITY_EXPERIMENT.md).
