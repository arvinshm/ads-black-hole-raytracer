# Component pseudocode

## Metric

```text
metric(x):
    return covariant metric g_mu_nu at x

inverse_metric(x):
    return inverse matrix g^mu_nu

inverse_metric_derivatives(x):
    return partial_mu g^alpha_beta

horizon_radius():
    solve the metric's horizon equation
```

## Camera

```text
construct local orthonormal tetrad at camera
for each pixel:
    convert pixel center to two screen-plane coordinates
    form inward spatial direction in the tetrad
    normalize direction
    combine with observer four-velocity to obtain past-directed null vector
    lower its index to get canonical momentum p_mu
```

## Hamiltonian geodesic tracer

```text
state y = (x^mu, p_mu)
while no terminal event:
    dx^mu/dlambda = g^mu_nu p_nu
    dp_mu/dlambda = -1/2 partial_mu g^alpha_beta p_alpha p_beta
    integrate adaptively with DOP853
    detect outer boundary, horizon, or regular origin
    if pure-AdS origin is reached:
        re-chart through the origin and continue
return final state, status, affine time, and null residual
```

## Boundary source

```text
find the source center at the ray's boundary-arrival time
measure angular distance from hit point to source center
apply Gaussian source profile
construct emitter four-velocity
compute emitted frequency = |p_mu u_emit^mu|
compute g-factor = frequency_observed / frequency_emitted
apply optional pulse and emission-angle factors
return emitted intensity, observed wavelength, g-factor, and miss angle
```

## Renderer

```text
parallelize camera rows across CPU worker processes
for each pixel:
    initialize one backward ray
    trace it
    if it reaches the boundary:
        sample the source
        I_observed = g^4 I_emitted
        convert shifted wavelength to diagnostic RGB
    otherwise mark horizon or numerical status
normalize display exposure without modifying raw intensity data
save images, arrays, metadata, and status maps
```

## Continuity experiment

```text
read requested Omega_H L targets
for each target:
    locate extremal endpoint of fixed-(L, mu) Kerr-AdS family
    reject target if no regular horizon can realize it
    bisect to find rotation parameter a
render Schwarzschild-AdS and every Kerr-AdS case with matched settings
assemble one comparison image with common exposure
```
