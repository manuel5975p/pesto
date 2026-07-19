# NSCBC/LODI characteristic boundaries (x extremes)

State as of 2026-07-18. Module `src/dns/gpu/comp_nscbc.*`, test
`tests/test_gpu_comp_nscbc.cpp`, flag `--nscbc` in `comp_cylinder_tunnel_gpu`.
The streamwise sponge (`DeviceCompFar`) remains the default; NSCBC is opt-in
until 3D/mesh validation and anchor regeneration.

## Design

Poinsot–Lele NSCBC: decompose the x-dynamics at each open boundary into
characteristic waves (acoustic u±c, entropy/shear u). Outgoing amplitudes are
computed from one-sided interior differences; incoming ones are modeled. The
boundary rows then evolve **in time** through these amplitudes.

Key structural choice: LODI yields d/dt of the boundary rows, so the seam is
`DeviceComp::setRhsHook(s, k)` — called after `kCompRhs` with the stage
primitives live, it overwrites rows i = 0 and i = nx−1 of the rhs `k`. The
rows then ride through SSP-RK3 via the normal `axpy`; no dt plumbing, no state
overwrite (that instantaneous overwrite was `comp_bc`'s acoustic-churn failure
mode). `kCompRhs` stays pure-periodic: the wrap-contaminated x-fluxes at the
two boundary rows are dead once overwritten, and no interior cell reads the
wrap face.

Amplitudes (λ1 = u−c, λ2..4 = u, λ5 = u+c):

    L1 = λ1 (∂p/∂x − ρc ∂u/∂x)     L5 = λ5 (∂p/∂x + ρc ∂u/∂x)
    L2 = u (c²∂ρ/∂x − ∂p/∂x)       L3 = u ∂v/∂x    L4 = u ∂w/∂x

LODI x-contribution: ∂ρ/∂t = −(L2 + (L1+L5)/2)/c², ∂u/∂t = −(L5−L1)/(2ρc),
∂v/∂t = −L3, ∂w/∂t = −L4, ∂p/∂t = −(L1+L5)/2; converted to conservative rhs.
Transverse y/z flux divergence is retained exactly (shared `faceFlux`,
z cancels at nz == 1); x-viscous terms on the boundary rows are dropped
(Poinsot–Lele viscous conditions). Lateral far field = free-slip mirror walls
(`DeviceCompFar` with sigMax = 0, which keeps `boost()` for the quiet start).

### Outflow (i = nx−1, subsonic)

L2..L5 leave (one-sided differences); incoming

    L1 = σ(1−M²) c/Lx (p − p∞)  −  β (Tp − ρc Tu)

σ = 0.25 (Rudy–Strikwerda). The second term is the **Yoo–Im transverse
correction**, β = `betaTrans` = 0.5 (`--nsbeta`), with
Tp = v∂p/∂y + γp∂v/∂y (+ z-terms), Tu = v∂u/∂y (+ z). It is NOT optional at
shedding Re: the incoming invariant obeys dR⁻/dt = −L1 − (Tp − ρc Tu), and
leaving the transverse forcing uncanceled makes every wake core crossing the
outflow radiate a spurious incoming wave. Measured scan (cyl Re800 M0.2 512²):

| β        | outcome                                                  |
|----------|----------------------------------------------------------|
| 0        | growing global mode, cl amplitude 2 → 13                 |
| M = 0.2  | bounded but still drifting upward                         |
| **0.5**  | **clean limit cycle, matches sponge cd σ and St exactly**|
| 1.0      | blowup at t = 35 (full cancellation ill-posed at low M)  |

### Inflow (i = 0, subsonic)

L1 leaves; incoming L5/L2/L3/L4 relax (u, ρ, v, w) toward freestream at rate
∝ etaIn·c/Lx. **etaIn must be stiff (default 100):** with etaIn = 2 the
body's blockage pressure gradient sagged the inflow row to 0.79 u∞ →
cd 30% low (equilibrium sag ∝ 1/etaIn). When characteristic-BC drag looks
low, check upstream freestream sag first. The inflow target tracks the
quiet-start ramp via `setFreestream`.

## Measured (512², quiet start + kick, window t ∈ [62.8, 94.3])

| case  | far field | cd            | clAmp | St     |
|-------|-----------|---------------|-------|--------|
| Re60  | NSCBC     | 1.813         | 0.05  | ~0.145 |
| Re60  | sponge-era anchor | 1.830 |       |        |
| Re800 | NSCBC     | 1.776 ± 0.180 | 1.33  | 0.2499 |
| Re800 | sponge (same-era control) | 1.894 ± 0.176 | 1.85 | 0.2503 |

Identical σ and St at Re800; cd ~6% below the walled sponge = the wake exits
carrying its real momentum deficit (less streamwise confinement — same
direction as the incompressible `--open` fringe result). Unit level: plane
acoustic pulse reflection 0.6%, vortex-exit residual 2.5%, uniform freestream
machine-exact, stiff-inflow pull exact.

## Caveats / open items

- Quiet-start-era sponge no longer reproduces the impulsive-era Re800 anchor
  (cd 1.894 vs 1.71, St 0.250 vs 0.2445): the Re800 anchors are
  start-condition-sensitive and need regeneration regardless of NSCBC.
- 2026-07-18: the comp cylinder app's grid convention was fixed to scale
  resolution with ny (domain fixed 8D; bit-identical at ny = 512). Numbers in
  this file measured at 512² hold; pre-fix n = 256 runs were geometry-broken.
- Not yet validated: 3D, `comp_mesh_tunnel_gpu` wiring, saturated
  long-window Re60 shedding.
- Reflection is small, not zero (plane-wave-normal exact only); a vortex
  crossing still leaves a few-% residual. At Re2000 this residual PUMPS:
  pure NSCBC blew up at 2048² (t = 63.6; final frames show vorticity sheets
  on the free-slip walls and shed from the stiff inflow row — box modes
  between stiff inflow, walls and soft outflow), and at 1024² it survives
  contaminated (cd σ 0.58, clAmp 4.8). Fix = `--nsstrip N`: a thin sponge
  strip (sigMax 0.5, N ≈ ny/64 ≈ 1–1.5% of nx) inside the NSCBC rows, the
  standard production hybrid. 1024² Re2000 triplet, window [50, 96]:
  pure 1.846 ± 0.582 (clAmp 4.8), strip-12 1.902 ± 0.341 (2.4, calmest of
  all three), sponge 2.087 ± 0.370 (2.8). Use --nscbc alone up to ~Re800;
  add --nsstrip beyond.

## Six-face NSCBC (2026-07-18, later)

`DeviceCompNscbc` generalized from x-rows to all open faces; one kernel
(`kNscbcCells`), disjoint slab enumeration (x faces own full slabs, y faces
interior-x strips, z faces the interior core), `openY/openZ` in Config
(openZ ignored at nz == 1; open axes need n >= 4).

- Lateral faces are characteristic far field: incoming acoustic = soft
  pressure anchor `sigLat (p - pInf) c/L` (default 0.25) with the Yoo-Im
  transverse correction (sign of the rho-c-Tu term flips with the face side:
  incoming invariant is p -+ rho c u_n at hi/lo). The boundary-normal
  velocity is NEVER relaxed — the body's displacement/dipole field vents
  through the face; a lateral velocity clamp is a wall in disguise (measured
  with the sponge far field).
- Entropy/shear waves (lambda = u_n) upwind on the sign of the outward
  normal velocity: outgoing = one-sided interior differences, incoming =
  gentle freestream relax `etaLat c/L` (default 0.25 — deliberately soft,
  unlike the stiff x-inflow).
- Edges/corners: per-axis LODI contributions summed (Lodato-style closure);
  non-boundary axes keep the exact faceFlux divergence; transverse
  derivatives go one-sided where a central stencil would wrap across another
  open face.
- `boostX(g, s, du)` is now a free function in `gpu/comp.hpp`
  (`DeviceCompFar::boost` delegates), so sponge-free apps keep the quiet
  start.

Wiring: `comp_mesh_tunnel_gpu` dropped sponge + walls entirely — always
all-face NSCBC (`--open` removed, `--nsbeta` added). `comp_cylinder_tunnel_gpu
--nscbc --open` = laterally open (walls remain its default; `--nsstrip` still
adds lateral rho/p strips through the open sponge bands).

Measured:
- Unit (test gpu_comp_nscbc): 2D/3D uniform freestream is a fixed point to
  FP32 rounding; radial acoustic pulse exits ALL faces with 2.4% residual
  (soft-inflow rig); vortex skimming the open y-hi face exits with 2.9%.
- Sphere Re100 M0.2 dr6 (comp_mesh_tunnel_gpu, quiet start): 64³ cd 1.10,
  96³ cd 1.146, cl/cs ~ 1e-3 — vs the walled-sponge-era 96³ anchor 1.226 and
  literature unbounded ~1.09: the free-slip-image blockage excess is gone.
- Cylinder Re800 512 `--nscbc --open` smoke (short window): cd 1.71 ± 0.17,
  clean shedding, stable through ramp + onset; below walled NSCBC 1.776 =
  lateral confinement removed.
- Not yet validated laterally open: long-window shedding statistics,
  Re2000-class (the stiff-inflow box-mode caution above still applies).
