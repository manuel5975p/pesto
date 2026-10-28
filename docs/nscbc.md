# NSCBC/LODI characteristic boundaries

Module `src/dns/gpu/comp_nscbc.*`, test `tests/test_gpu_comp_nscbc.cpp`,
selected with `--far nscbc` on `dns flow --physics compressible` -- the
compressible DEFAULT. The streamwise sponge (`DeviceCompFar`, `--far sponge`)
is an explicit escape hatch only.

## Design

Poinsot–Lele NSCBC: decompose the x-dynamics at each open boundary into
characteristic waves (acoustic u±c, entropy/shear u). Outgoing amplitudes are
computed from one-sided interior differences; incoming ones are modeled. The
boundary rows then evolve **in time** through these amplitudes.

Key structural choice: LODI yields d/dt of the boundary rows, so the seam is
`DeviceComp::setRhsHook` — called after the fused stage kernel with the stage
primitives live, it rewrites rows i = 0 and i = nx−1 of the stage output
through the same `StageCombine` (out = ca·a + cb·b + crAt·rhs, its own
characteristic rhs in place of the flux divergence; `kNscbcCells`). The rows
ride through SSP-RK3 like every interior cell; no dt plumbing, no state
overwrite (an instantaneous state overwrite churns acoustically). The stage
kernel stays pure-periodic: the wrap-contaminated x-fluxes at the two boundary
rows are dead once rewritten, and no interior cell reads the wrap face. The
hook re-reads `cmb.a`, which is why a stage output must never alias it
(CLAUDE.md).

Amplitudes (λ1 = u−c, λ2..4 = u, λ5 = u+c):

    L1 = λ1 (∂p/∂x − ρc ∂u/∂x)     L5 = λ5 (∂p/∂x + ρc ∂u/∂x)
    L2 = u (c²∂ρ/∂x − ∂p/∂x)       L3 = u ∂v/∂x    L4 = u ∂w/∂x

LODI x-contribution: ∂ρ/∂t = −(L2 + (L1+L5)/2)/c², ∂u/∂t = −(L5−L1)/(2ρc),
∂v/∂t = −L3, ∂w/∂t = −L4, ∂p/∂t = −(L1+L5)/2; converted to conservative rhs.
Transverse y/z flux divergence is retained exactly (shared `faceFlux`,
z cancels at nz == 1); x-viscous terms on the boundary rows are dropped
(Poinsot–Lele viscous conditions). Lateral far field = free-slip mirror walls
(`DeviceCompFar` with sigMax = 0, which keeps `boost()` for the quiet start),
or characteristic laterals with `--open`.

### Outflow (i = nx−1, subsonic)

L2..L5 leave (one-sided differences); incoming

    L1 = σ(1−M²) c/Lx (p − p∞)  −  β (Tp − ρc Tu)

σ = 0.25 (Rudy–Strikwerda). The second term is the **Yoo–Im transverse
correction**, β = `betaTrans` = 0.5 (`--nscbc-beta`), with
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

Backflow through x-hi needs no special treatment: the λ = u amplitudes are
proportional to un and shut off smoothly through a stagnation, and the brief
inflow advects back out (vortex-on-face residual 0.0280 at a swirl 2× the
freestream, 0.0513 at 4×; pinned by `backflowVortexResidual`).

### Inflow (i = 0, subsonic)

L1 leaves; incoming L5/L2/L3/L4 relax (u, ρ, v, w) toward freestream at rate
∝ etaIn·c/Lx (`--nscbc-etain`, default 100). **etaIn must be stiff when there
is a freestream to sag:** with etaIn = 2 the body's blockage pressure gradient
sagged the inflow row to 0.79 u∞ → cd 30% low (equilibrium sag ∝ 1/etaIn).
When characteristic-BC drag looks low, check upstream freestream sag first.
The inflow target tracks the quiet-start ramp via `setFreestream`. Without a
freestream (a static rotor) the stiff inflow is a pure acoustic mirror and a
soft etaIn is the right setting (below).

## Measured (512², quiet start + kick, window t ∈ [62.8, 94.3])

| case  | far field | cd            | clAmp | St     |
|-------|-----------|---------------|-------|--------|
| Re60  | NSCBC     | 1.813         | 0.05  | ~0.145 |
| Re800 | NSCBC     | 1.776 ± 0.180 | 1.33  | 0.2499 |
| Re800 | sponge (same-era control) | 1.894 ± 0.176 | 1.85 | 0.2503 |

Identical σ and St at Re800; cd ~6% below the walled sponge = the wake exits
carrying its real momentum deficit (less streamwise confinement). Unit level:
plane acoustic pulse reflection 0.6%, vortex-exit residual 2.5%, uniform
freestream machine-exact, stiff-inflow pull exact.

## Interior acoustics (test `gpu_acoustics`)

Before any boundary: what the scheme does to a sound wave. A linear wave
(p'/p∞ = 1e-3) in a PERIODIC box returns to its own initial state after every
traversal, so amplitude ratio and phase shift after k of them ARE the numerical
dissipation and dispersion. Per wavelength travelled, 128 cells, cfl 0.4:

| ppw | amp ratio | phase err (rad) | c_num/c |
|-----|-----------|-----------------|---------|
| 128 | 0.999937  | -2.55e-03 | 0.99959 |
| 64  | 0.999976  | -1.00e-02 | 0.99840 |
| 32  | 0.999902  | -4.03e-02 | 0.99358 |
| 16  | 0.999111  | -1.60e-01 | 0.97451 |
| 8   | 0.994831  | -6.25e-01 | 0.90050 |
| 6.1 | 0.992451  | -1.05e+00 | 0.83240 |
| 4   | 0.991869  | -2.28e+00 | 0.63701 |
| 3   | 0.996581  | -3.73e+00 | 0.40651 |
| 2   | **1.000000** | -6.28e+00 | **0.00000** |

Dispersion is clean second order (measured exponents 1.96 / 1.99 / 2.01 / 1.98
over 8→128 ppw). The wave rides a mean flow exactly — phase error identical at
M 0, 0.2 and 0.5 — and the time error is invisible next to the spatial one
(unchanged from cfl 0.4 down to 0.1), so the table is the stencil alone.
Linear in amplitude to p'/p∞ = 1e-2; at 1e-1 the wave steepens.

**The 2Δ mode is exactly stationary and exactly undamped.** A central
skew-symmetric flux has no upwind dissipation, so acoustic content at the mesh
scale does not decay — it stops travelling. Sound put into 3-4 cells crawls at
0.4-0.6 c and stays. The only sinks in the interior scheme are molecular ν and
the explicit SGS model: the cure for mesh-scale content is to stop generating
it (resolve the source), not to filter it.

## Small scales and the sink (test `gpu_comp_sink`)

What the interior facts mean for a BOUNDARY. A wavepacket at wavenumber kh
(amp 1e-3, envelope σ = 0.32, quiescent 256-cell box) released mid-domain
against the characteristic outflow; the fraction of its acoustic energy still
inside after 1.3 crossing times is the fraction the face never received:

| kh    | ppw  | E_left/E_0 | why |
|-------|------|-----------|-----|
| π/4   | 8    | 0.0215    | resolved: leaves, face absorbs |
| π/2   | 4    | 0.2319    | group velocity c·cos(kh) ≈ 0 |
| 3π/4  | 2.67 | 0.6869    | group velocity negative |
| π     | 2    | 0.9783    | stationary and undamped |

Same for the vortical family (compact solenoidal patch, \|u'\| = 5% c, σ =
16h, advected at M 0.2 through the outflow, non-wrapping scan, sampled at one
and two exit windows): the 8Δ patch drains through a slow face tail (residual
0.55 → 0.17, peak at the face rows), the 4Δ patch PARKS (0.55 → 0.40 with its
peak still at the release point after two full windows: the advective group
speed u·cos(kh) is ~0 there).

**None of this is the face's to absorb.** No face-local closure can act on
content that does not arrive, and the interior flux will never damp it. The
cure is an EXPLICIT sink confined to a boundary band (`DeviceCompSink`,
`--nscbc-sink N`, `--nscbc-sink-amp` ∈ (0, 2], the SSP-RK3 bound): D4/sin⁴
selective damping in conservative flux form, additive rhs UNDER the NSCBC
overwrite, exactly zero on constant and linear fields (no freestream
reference). Scale-selective, so resolved content above the 6Δ knee reaches the
characteristic rows unchanged and every reflection coefficient on this page
stays valid. Contract between the two layers: the face owns the extreme rows'
RHS; the sink is an additive term in the band inside; the face's one-sided
`Lout` reads row nx−2, so the sink's field IS the face's upstream sample — the
combined system is measured together, never inferred from the parts.

Measured. Operator: a 2Δ ripple in the band decays on the analytic σ·sin⁴
profile to 3% at α = 1 and at the α = 2 cap; a 16Δ packet crosses losing < 2‰;
a 4Δ packet in the band drops 0.769 → 0.023; all five fields conserve to
roundoff (flux form). System (`gpu_nscbc_absorb`): the resolved outflow
reflection is unchanged-to-improved (R 0.0116 → 0.0063 at M 0 — the sink eats
the small-scale part of the residual), and the inflow mirror is NOT fixed
(0.810 → 0.802): that reflection is resolved-scale and belongs to etaIn or the
strip. Near field: a resolved vortex (core 16h) riding M 0.2 reaches mid-band
with 0.9999 of its peak vorticity under sink 16, and 0.6813 under strip 16 —
the freestream relaxation eats a third of the wake before it leaves the box.

High Re: Re 2000 cylinder, 1024², `--far nscbc --nscbc-sink 16` and NO strip:
clean to t = 64.8 (198k steps), cd[50, 64.8] = 1.919 ± 0.300, cl range ±2.3 —
the strip-12 hybrid's calm statistics (1.902 ± 0.341, clAmp 2.4, window [50,
96]) without a freestream reference anywhere, where pure NSCBC is
contaminated (1.846 ± 0.582, clAmp 4.8). Above Re 800 the validator requires
the sink or `--nscbc-strip` and suggests the sink first.

**Defaults.** The sink is ON (16-cell band, amp 1) for `--far nscbc` wherever
it is defined — uniform mesh, whole faces, band fits twice per open axis —
and measured free where it has nothing to absorb (Re60 nscbc cylinder cd
+0.003%, 1.786575 → 1.786622; open Re60 cylinder −0.003%). `--nscbc-sink 0`
is the bare boundary, bit-for-bit. On `--stretch`/`--focal`/`--ogrid`/`--body
floor` the default silently stays off (a default that does not apply is
absence), while an EXPLICIT sink there is rejected. `--nscbc-etalat` defaults
to 0.05 (below).

## Absorption (test `gpu_nscbc_absorb`)

Reflection coefficients: a plane acoustic pulse (p'/p∞ = 1%, σ = 0.08 in a
2.56 box) released mid-domain, integrated until it plus three σ of tail have
cleared the face, residual max |p − p∞| normalized by the incident amplitude.
n = 256 unless stated.

| mean Mach | R outflow | R inflow (etaIn 100) |
|-----------|-----------|----------------------|
| 0.2       | 0.0073    | 0.8435               |
| 0.05      | 0.0106    | 0.8148               |
| 0.01      | 0.0115    | 0.8126               |
| 0 (quiescent) | 0.0116 | 0.8100              |

**The outflow absorbs at every Mach including zero** and refines away: n =
128/256/512 gives 0.0258/0.0073/0.0041 at M 0.2 (observed order 1.8 then 0.8;
the σ pressure anchor is the floor). Linear over amp 1e-3…1e-2; below that a
fixed residual floor ~1e-5 p∞ dominates the ratio, and at amp 0.1 nonlinear
steepening puts it back at 3.2%.

**The stiff inflow is an acoustic mirror, and it does not refine away**
(order ≈ 0 to −0.2 over the same triplet). `etaIn` is sized against blockage
sag, not acoustics, and the trade is steep — R inflow at M 0 for etaIn
0.5/2/10/30/100 = 0.020/0.060/0.231/0.493/0.810, with the quiescent all-face
radial residual 0.0034/0.0036/0.0156/0.0336/0.0520.

**Per-family inflow stiffness** (`--nscbc-etain-shear`): splits the λ = u
family (entropy L2, shear L3/L4) off the sag guard; L5 keeps `--nscbc-etain`,
the family rides its own rate (default: follow etaIn, bit-identical).
- The acoustic mirror is the sag guard's own: softening the family 400×
  (etaInShear 100 → 0.25, etaIn held at 100) moves R_inflow by 0.1% (0.8100
  → 0.8093). A normal wave reflects off the `u = uInf` clamp itself; only
  `--nscbc-etain` or the strip can fix the mirror.
- What the family rate owns is what the face writes under vortical load: a
  resolved vortex parked three cores off the inflow makes the face rows write
  vorticity at 0.0069 of the core scale at 100, falling monotonely to 0.0050
  at 0.25. Its real customer is turbulent inflow, where L3/L4 at 100 would
  clamp every injected eddy.

## Measuring a boundary

Two rules, both enforced by the tests:

- **Never difference across the periodic wrap.** A whole-grid scan with a
  periodic central difference subtracts the outflow row from the inflow row
  and reads an O(amp) jump as O(amp/h) vorticity, growing with refinement.
  Peeling cells off the faces is not the fix either: it also removes where a
  boundary-made sheet would live. Use a NON-WRAPPING (one-sided at every
  extreme) scan, margin 0; `modeSplit` takes a margin ≥ 2 and
  `gpu_nscbc_order` carries the non-wrapping scan and pins the peak location.
- **An oblique reflection has a measurement WINDOW.** An oblique plane ridge
  is a diagonal band spanning `x = xc ± (Ly/2)·tanθ`; it has not cleared the
  box until its most-upstream point has covered `(Lbox − xc + (Ly/2)tanθ +
  3σ)/(c cosθ)`, while the reflection starts leaving the FAR face after
  `Lbox/(c cosθ)`. In a square box those cross at ~45° and no valid time
  exists past it: size the rig so `xc > (Ly/2)tanθ + 3σ`.

Any boundary number states where its peak sits and shows it is flat in time.

## The lateral closure (test `gpu_nscbc_order`)

45° ridge, 1024×128 rig, peak |ω|/wScale by the non-wrapping scan:

| laterals | sink | peak \|ω\| | at (i,j) |
|----------|------|-----------|----------|
| free-slip walls | — | 0.2198 | (846,126) — the y wall |
| free-slip walls | x-band 16 | 0.2200 | unchanged: an x band cannot reach it |
| open (etaLat 0.25) | — | **0.8017** | (1007,0) — the lateral FACE |
| open (etaLat 0.25) | 16 | 0.1164 | face rows |
| open, etaLat 0.05 | — | 0.2005 | face rows |
| open, etaLat 1.0 | — | 2.1649 | face rows |
| open, etaLat 0.05 | 16 | **0.0319** | face rows |

(1) With free-slip walls the residual (0.22 of the incident scale) sits in the
x-INTERIOR against the y wall, not on the characteristic face: the outflow row
lays down no vortex sheet, and the residual of an oblique wave is acoustic. It
is resolved-scale vorticity parked in a quiescent box; nothing local removes
it without eating the near field, a freestream advects it out (the
`vortexResidual` band). (2) **Bare characteristic laterals write vorticity in
proportion to their rate**: the incoming relax `etaLat·c/L·(v_t − tgt)` writes
tangential velocity wherever the acoustic load varies along the face (0.20 /
0.80 / 2.16 at etaLat 0.05 / 0.25 / 1.0) — a stiff lateral clamp is a wall in
disguise. The validator warns above etaLat 0.5. (3) The working open-box
configuration is **soft etaLat + the sink: 0.0319**, seven times below the
walls baseline; each piece is necessary (the sink eats what the face writes at
small scale, the soft rate stops writing it at large scale). It is what plain
`--open` resolves to (`--nscbc-etalat` default 0.05, sink default 16). The
vortex-on-face transparency rig (`gpu_comp_nscbc`) is etaLat-insensitive
(0.026–0.027), and the open Re60 cylinder cd moved −0.003%.

## Angle of incidence (test `gpu_nscbc_order`)

Rig sized so the window exists at every angle: 1024×128, σ = 0.03, band parked
at 0.9·Lbox, sampled at 1.3·t_clear. Fraction of incident acoustic energy
reflected off the OUTFLOW, quiescent:

| θ | β=0 | β=0.2 | β=0.3 | β=0.4 | β=0.5 | β=0.6 | adaptive | R₁² (EM-1) | β_opt |
|---|-----|-------|-------|-------|-------|-------|----------|------------|-------|
| 0°  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.500 |
| 15° | 0.0115 | 0.0100 | 0.0096 | 0.0093 | 0.0092 | 0.0092 | 0.0096 | 0.0003 | 0.491 |
| 30° | 0.0489 | 0.0399 | 0.0376 | 0.0364 | 0.0361 | 0.0366 | 0.0367 | 0.0052 | 0.464 |
| 45° | 0.1071 | 0.0683 | 0.0614 | 0.0594 | 0.0608 | 0.0647 | 0.0627 | 0.0294 | 0.414 |
| 60° | 0.2238 | 0.1116 | 0.1001 | 0.1017 | 0.1104 | 0.1232 | **0.0968** | 0.1111 | 0.333 |
| 75° | 0.5682 | 0.2744 | 0.2889 | 0.3202 | 0.3552 | 0.3900 | 0.3677 | 0.3467 | 0.206 |

β = 0 lands on the Engquist–Majda first-order coefficient
R₁ = (1−cosθ)/(1+cosθ) within a factor of 2 at 60–75°, which is the check that
the rig measures a reflection and not a leftover. The closure owns the oblique
error, not the dn() stencil.

**The measured optimum tracks β_opt(θ) = cosθ/(1+cosθ), not the constant 0.5.**
0.5 is only its normal-incidence limit; the best fixed β falls from ~0.5 at 30°
to 0.2 at 75°, exactly as β_opt does (0.464 → 0.206).

### `--nscbc-abc adaptive|modal`: the exact coefficient, read off the face

The face can recover its own incidence angle with no extra state, because the
two quantities it already computes give it away. For an outgoing acoustic wave,
`Tp/Lout = (1−cosθ)/cosθ` (Lout = L5 at a hi face, L1 at a lo one) — pointwise
and independent of the pulse shape, since both are the same `f′(s)` times an
angle factor. Writing X = Tp/Lout and substituting β = betaT·2/(2+X) into
`Lin = −β·Tp`:

    Lin = −Tp·Lout/(2·Lout + Tp) = −i ω p cosθ(1 − cosθ) = L1_exact

**exactly, at every angle** — not through O(θ²). It is the Engquist–Majda
symbol summed rather than truncated, obtained without the auxiliary time
history an exact ABC normally costs. X is clamped to [0, 18]; X = 0 returns
betaT bit-for-bit, so a normal-incidence face is unchanged.

Two ways to form that ratio, and the difference is the whole point:

- **`adaptive`** — per face cell, inside the NSCBC kernel. Needs no plane and
  no transform, so it works on **every** open face and on any mesh, including
  the all-open box a static rotor runs in.
- **`modal`** — per transverse DCT-II mode of the whole outflow plane
  (`gpu/nscbc_modal.*`). With free-slip mirror laterals the cosine basis is
  the *exact* transverse eigenbasis, so each mode carries one k_t and the
  ratio is formed inside it. x-hi plane only, uniform mesh, walled laterals —
  rejected, never ignored, with `--open`, `--stretch`, `--focal`, `--ogrid` or
  `--body floor`.

| θ | best fixed β | adaptive | modal |
|---|---|---|---|
| 0°  | 0.0000 | 0.0000 (+0.0%) | 0.0000 (+0.0%) |
| 15° | 0.0092 | 0.0096 (+3.8%) | 0.0094 (+1.9%) |
| 30° | 0.0361 | 0.0367 (+1.6%) | 0.0367 (+1.6%) |
| 45° | 0.0594 | 0.0627 (+5.6%) | 0.0632 (+6.4%) |
| 60° | 0.1001 | **0.0968 (−3.2%)** | 0.1031 (+3.0%) |
| 75° | 0.2744 | 0.3677 (+34.0%) | **0.2771 (+1.0%)** |

Both reach the per-angle optimum without being told the angle. **`adaptive`
collapses at grazing incidence and `modal` does not**: at 75° the field a face
cell sees is a superposition — incident wave, its wall images, and a
reflection that is no longer small — so a pointwise ratio reads an *average*
angle. The DCT separates it instead. Pinned in `gpu_nscbc_order`.

Cost: `modal` is 3 Real planes + 1 complex plane on the face (12 and 20 B per
face cell), 2 transforms per stage, and 0.20 → 0.31 ms/step on a 192² 2D
cylinder — launch-latency bound, not bandwidth. `adaptive` is free.

A real face sees every angle at once. Radial pulse, all four faces
characteristic, quiescent, n = 256, fraction of pulse energy left:

| etaIn | β=0.2 | β=0.3 | β=0.4 | β=0.5 | β=0.6 | adaptive |
|-------|-------|-------|-------|-------|-------|------|
| 100 (production) | 0.1156 | 0.1133 | 0.1121 | 0.1116 | 0.1117 | 0.1129 |
| 0.5 (soft)       | 0.0007 | 0.0006 | 0.0005 | 0.0004 | 0.0004 | **0.0004** |

In the geometry a static rotor runs in — compact source, all faces open, no
freestream — the transverse coefficient is worth ~3% and the inflow stiffness
a factor of 280 (`modal` is not available there: an all-characteristic lateral
face is not separable). `--nscbc-etain` is the boundary condition on that
case; β is a rounding error on it. With a soft inflow `--far nscbc` absorbs a
radial pulse to **0.04% of its energy**.

### On Mur / Engquist–Majda second order

Mur's second-order ABC is the FDTD discretization of the Engquist–Majda B2
one-way operator, `∂ₓ∂ₜp + (1/c)∂ₜₜp − (c/2)∂_yy p = 0`, whose reflection
coefficient is the square of B1's: R₂ = [(1−cosθ)/(1+cosθ)]².

In a characteristic framework that operator is a statement about the incoming
amplitude. For an outgoing oblique wave at the x-hi face, with `p = A
exp(i(kx x + ky y − ωt))`, `u = kx p/(ρω)`, `v = ky p/(ρω)`:

    L1_exact = (u−c)(∂ₓp − ρc ∂ₓu) = −i ω p cosθ (1 − cosθ)
    Tp       = γ p ∂_y v  =  i ω p sin²θ        (the code's transverse term)

so `L1 = −½ Tp` reproduces L1_exact through O(θ²) — and `Lin = … − betaT·(Tp −
s ρc Tun)` with **betaT = 0.5 is exactly that**. The default already carries
the Mur-2 content; β = 1 is the full cancellation that is ill-posed at low M,
β = 0 is bare B1. The useful move beyond it is not a higher fixed order but the
exact coefficient, which `--nscbc-abc` provides without truncation.
