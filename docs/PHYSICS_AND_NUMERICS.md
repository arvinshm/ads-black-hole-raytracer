# Physics and numerics

The code uses signature `(-,+,+,+)` and Hamiltonian null geodesics,

```text
H = 1/2 g^{mu nu} p_mu p_nu = 0.
```

Hamilton's equations are integrated backward from the camera. A complete two-dimensional camera direction determines one unique geodesic initial-value problem. Multiple images arise because several distinct camera directions can map to the same source point.

The AdS conformal boundary is regulated by placing source and camera on a large finite-radius timelike surface. The cutoff can be increased to study convergence toward the conformal-boundary limit.

The redshift factor is computed invariantly from observer and emitter four-velocities. For observed frequency normalized to one,

```text
g = nu_obs / nu_emit,
nu_emit = |p_mu u_emit^mu|.
```

Bolometric specific intensity is displayed with the standard `g^4` transformation. The wavelength-to-RGB conversion is only a diagnostic visualization, not a calibrated detector model.

Kerr–AdS uses Boyer–Lindquist coordinates with

```text
rho^2 = r^2 + a^2 cos^2(theta)
Delta_r = (r^2+a^2)(1+r^2/L^2) - 2 mu r
Delta_theta = 1 - a^2 cos^2(theta)/L^2
Xi = 1 - a^2/L^2.
```

The horizon angular velocity relative to the nonrotating frame at infinity is

```text
Omega_H = a(1+r_+^2/L^2)/(r_+^2+a^2).
```

`Omega_H L <= 1` is the Hawking–Reall stability bound. The code can still trace fixed-background null geodesics for regular Kerr–AdS metrics above that value, but it does not model superradiant growth or backreaction.

Adaptive integration is accurate but expensive near unstable photon trajectories. Runtime is therefore not uniform across pixels. Per-ray affine, RHS-evaluation, and wall-time budgets prevent a small number of critical rays from stalling a full render. Such rays are labeled rather than interpolated silently.

The Kerr implementation uses analytic inverse-metric components and derivatives. Validation checks include the `a -> 0` reduction to Schwarzschild–AdS, horizon equations, conserved `p_t` and `p_phi`, null-constraint drift, pure-AdS refocusing, and Schwarzschild capture across the analytic photon-sphere threshold.
