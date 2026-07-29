# Verification

The published source was tested before upload.

```text
pytest -q
7 passed
```

A fresh-package end-to-end smoke test also rendered the controlled continuity sequence at `4 x 4` resolution, including Schwarzschild–AdS, Kerr–AdS at `a=0`, and four nonzero rotation targets. It produced the expected output folders, raw arrays, metadata, status maps, and common-exposure comparison image.

The continuity target solver was additionally checked at `Omega_H L = 0`, `0.3`, `0.98`, `1.2`, and `1.5` for `L=1`, `mu=0.5`.
