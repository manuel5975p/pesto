# The incompressible FV path — OpenFOAM simpleFoam on the GPU

`plans/masterplan_meshsolver.md` P5b (user decision U7: "duplicating OpenFOAM
but with GPU solvers is the goal"). `dns flow --polymesh CASE` runs OpenFOAM
v2606's **simpleFoam** algorithm — SIMPLE / SIMPLEC, laminar or RAS
(kOmegaSST, SpalartAllmaras; section RAS), optional MRF —
on the IDENTICAL polyMesh of an OpenFOAM case on the GPU — in FP32 by default
(fields, matrices, Krylov, AMG in float, every reduction in FP64, each solve
stopped at max(tolerance, a measured FP32 floor); section Precision) or in
FP64 with `--fv-precision fp64`, the parity mode every parity number below
was measured in — with the linear solves on `gpu/la` (`docs/la.md`). The case IS the OpenFOAM case:
`0/U`, `0/p`, `system/{controlDict,fvSchemes,fvSolution}`,
`constant/{transportProperties,MRFProperties,turbulenceProperties}` are read
as OpenFOAM reads them, and every entry the solver does not reproduce exactly
is rejected by name (exit 3): among them `constant/fvOptions` /
`system/fvOptions` (fv::options sources simpleFoam applies; rejected on
presence) and a forces object without `CofR` (OpenFOAM's
`coordSystem::cartesian(dict)` then reads `origin` + `e1`/`e3` and writes
force/moment in that rotated frame, or stops when they are missing:
`origin` alone or nothing is an OpenFOAM error; with `CofR` OpenFOAM reads
neither `origin` nor `e1`/`e3`, and CofR is the moment centre here too).
An empty `fvOptions` (no sources) is rejected too: on presence, by design. The mesh is the FaceMesh (`docs/mesh.md`) in
SOURCE order, i.e. OpenFOAM's own cell and face numbering, so every per-cell
sum runs in OpenFOAM's order.

Measured (gfx1100, FP64 = `--fv-precision fp64`; OpenFOAM v2606 serial or np 12):

| case | cells | vs simpleFoam | result |
|---|---|---|---|
| IF1a lid-driven cavity Re 100 (SIMPLE, p 0.3 / U 0.7, closed: setReference) | 576 | identical blockMesh, linear solves tight on both sides | U, p, phi at it 100 / 400: 3.5e-14 / 6.6e-14 of scale; initial residuals of 50 iterations 2.0e-12 relative |
| IF1b skewed cavity (26.6°, SIMPLEC, 2 non-orth correctors, p relax 1) | 576 | same | 4.3e-14 / 1.1e-12; residuals 4.1e-10 |
| IF1c graded channel (inletOutlet, fixedValue p, slip, linearUpwind, cellLimited) | 800 | same | 3.6e-13 / 2.0e-13; residuals 3.0e-11 |
| IF1d 3D duct (non-orthogonal, cyclic z, freestreamVelocity/Pressure, symmetryPlane) | 512 | same | 1.4e-13 / 3.0e-14; residuals 2.9e-9 |
| IF2 snappy sphere Re 100, 5 prism layers, B_G1s schemes and solvers (relTol 0.01) | 65 859 | np 12, identical mesh | cd (Fx) at 1500 it: 0.436612919585 vs 0.436612919585 (8.8e-13); transient (it 1–300) within 1e-3–2e-6 (the relTol-0.01 iterates differ) |
| DTMB 4119 F1 + MRF, tight solves, 3 iterations | 708 044 | np 12 | U 2.1e-12, p 2.0e-12, phi 7.6e-13 at it 3; residual history equal to 6 digits |
| DTMB 4119 F1 + MRF (B_F1full: relTol 0.01) | 708 044 | np 12, 600 it | thrust Fx 4.3e-6, torque Mx 3.7e-8 relative at it 600 |
| DTMB 4119 G1 + MRF (B_G1: relTol 0.01) | 988 762 | np 12, 700 it | thrust 5.3e-6, torque 2.0e-9 relative at it 700 |
| IF1e cavity_lim: IF1b with `limited corrected 0.33` laplacians + snGrad (4% off the unlimited fields) | 576 | serial, tight | 3.5e-14 / 2.2e-13; residuals 1.7e-10 |
| IF1f mrf2: two ADJACENT MRF zones + an inactive one, nonRotatingPatches literal + regex, rotating wall, MRF-included open / slip / symmetryPlane faces | 1 152 | serial, tight | 1.6e-13 / 1.6e-13 (phi 3.9e-13); residuals 9.3e-12 |
| rotor2 L1 / L2 + MRF, laminar Re 220, J 0.63 (`runs/parity` meshes; `tools/fv/make_rotor_cases.sh`), corrected | 114 834 / 144 692 | np 6, 1500 it | CT 8.6e-12 / 8.7e-12, CQ 1.2e-13 / 1.2e-13 |
| the same, `limited corrected 0.33` + cellLimited grad(U) | 114 834 / 144 692 | np 6, 1500 it | CT 8.4e-12 / 1.3e-11, CQ 1.2e-13 / 1.2e-13 |
| DTMB 4119 F1 + MRF converged (1200 it, residual 1e-6 at it 944 / 945) | 708 044 | np 12 | CT 2.1e-9, CQ 1.2e-9 |

Wall time (IF3, `simjob --bench`; foam np 12 `--bind-to core` on the
i5-14600KF; dns one RX 7900 XTX):

| case | dns ms/it | foam ms/it | dns wall (all) | foam wall | to matched residual |
|---|---|---|---|---|---|
| sphere 66k, 1500 it | 6.38 | 12.4 | 9.8 s | 20.9 s | 1.43–1.54× (it 110–322: same iteration counts) |
| F1 708k, 600 it | 33.5 | 291 | 21.5 s | 181.9 s | 8.2–8.7× (1e-2…1e-4: it 28/205/391/549 vs 28/206/391/550) |
| G1 989k, 700 it | 51.1 | 418 (50-it bench) | 37.7 s | 296.7 s (older run, not `--bench`) | 7.9× (wall) |
| rotor2 L1 115k + MRF, 1500 it (foam np 6; np 12: 41 s) | 8.6 | 21.3 | 13.2 s | 32 s (ClockTime) | 2.1× to CT/CQ within 1e-3 of final (it 357 both: 4.5 vs 9.5 s loop) |
| rotor2 L2 145k + MRF, 1500 it (foam np 6; np 12: 53 s) | 10.3 | 29.4 | 15.8 s | 44 s | 2.3× (it 437 both: 6.7 vs 15.2 s) |
| F1 708k + MRF, 600 it (MRF phase re-bench) | 34.7 | 287 | 22.3 s | 174 s (ClockTime 172.5) | 8.2× (it 514 both: 18.0 vs 148.3 s) |

The small mesh is launch / host-sync bound; the prop meshes are mostly
bandwidth bound (the fine-level SpMV / smoother), with the coarse levels and
the GS colours latency bound. CORRECTED count (rocprof, rotor2 L1, per SIMPLE
iteration, pre-performance binary): ~1470 kernel dispatches (incl. 96 memset
fills) and ~71 synchronous `hipMemcpy`, not "~40 launches + ~15 syncs";
after the performance round (below) 905 dispatches and 21 syncs.

**Final prop parity + wall time** (phase FINAL, `plans/verify/impl_meshsolver_final.md`;
fresh case copies, cold runs, both `simjob --bench`; simpleFoam np 12
`--bind-to core`; same endTime and relTol 0.01 inner stops on both sides;
CT = −Fx, CQ = −Mx, n = D = ρ = 1; settle = CT and CQ within 1e-3 of each
side's own final value):

| case | cells | it | CT rel | CQ rel | settle it (foam / dns) | dns wall | foam ClockTime | to settle (dns loop / foam Exec) |
|---|---|---|---|---|---|---|---|---|
| rotor2 L1 laminar MRF | 114 834 | 1500 | −1.1e-12 | −1.2e-13 | 357 / 357 | 13.2 s | 41 s | 4.54 / 11.83 s (2.6×) |
| rotor2 L2 laminar MRF | 144 692 | 1500 | +1.1e-11 | +2.4e-13 | 437 / 437 | 15.8 s | 55 s | 6.61 / 18.91 s (2.9×) |
| rotor2 L1 layered, SST + MRF (U: bounded Gauss linearUpwind grad(U), grad(U) cellLimited 1, laplacian/snGrad limited corrected 0.33; k/omega upwind — NOT the laminar rows' central schemes) | 123 511 | 800 | −2.0e-6 (serial foam: −2.3e-8) | −8.6e-8 (serial: −1.3e-9) | 209 / 209 | 12.1 s | 42 s | 3.46 / 11.85 s (3.4×) |
| DTMB 4119 F1 laminar MRF | 708 044 | 1200 | −2.7e-9 | −1.2e-9 | 506 / 507 | 40.6 s | 345 s | 17.71 / 149.17 s (8.4× vs np 12; ≈ 6.9× vs np 6, below) |

np 12 is NOT simpleFoam's fastest setup on any of these meshes. np 6
(verifier FINAL, `--bench`): rotor2 L1 21.4 ms/it vs 26.9 at np 12 (to
settle 9.50 s: dns 2.1×, whole run 2.4×); F1 240.3 ms/it (200-it bench) vs
291.5 at np 12 over the same 200 it (284.3 over 1200), so F1 to settle is
≈ 122 s np 6 (extrapolated) vs 17.6 s dns = **≈ 6.9×**, the whole run ≈
7.2×. Quote speed against np 6. simpleFoam np 12 is not deterministic run
to run (rotor2 L2 CT moved 9e-12 relative between two runs; F1 2.2e-10);
dns is byte-identical.

The SST np 12 offset is simpleFoam's own decomposition: foam np 12 vs foam
serial differs by exactly the dns-vs-np 12 field numbers (nut 4.3e-3, k
5.6e-4 of scale), while dns vs serial foam is ≤ 2.3e-8 (U) / 8.8e-9 (nut).
Quote tight parity against serial simpleFoam on RAS cases.

**Performance round** (`plans/masterplan_performance.md`, report
`plans/verify/impl_performance.md`; full-length runs on fresh case copies,
`simjob --bench`, gfx1100). Every change is BIT-IDENTICAL: stats (all columns
but `ms`), `force.dat` / `moment.dat` and the log (all but its timing lines)
equal `build/final/out` byte for byte on all four cases, the fields equal the
pre-change binary's — so the parity table above holds unchanged. Foam
references: simpleFoam np 12 from the FINAL bench above; np 6 = foam's
fastest measured setup (rotor2 L1 21.4 ms/it, 32 s, settle 9.50 s; L2
29.4 ms/it, 44 s, settle 15.2 s; SST 39 s = 48.75 ms/it, settle 12.7 s; F1
240.3 ms/it from a 200-it bench, whole run ≈ 288 s and settle ≈ 122 s
extrapolated).

| case | dns ms/it before → after | active (pIt > 0) / tail (pIt 0) ms/it after | dns wall | to settle (dns loop) | vs foam np 6: ms/it / whole run / to settle | vs np 12: ms/it / whole / settle |
|---|---|---|---|---|---|---|
| rotor2 L1 115k, 1500 it | 8.69 → 3.96 (2.19×) | 5.55 (774 it) / 2.28 (726 it) | 13.3 → 6.23 s | 4.57 → 2.17 s | 5.4× / 5.1× / 4.4× | 6.8× / 6.6× / 5.5× |
| rotor2 L2 145k, 1500 it | 10.11 → 4.99 (2.03×) | 6.53 (915) / 2.57 (585) | 15.5 → 7.84 s | 6.32 → 3.17 s | 5.9× / 5.6× / 4.8× | 7.3× / 7.0× / 6.0× |
| rotor2 L1 SST 124k, 800 it | 14.58 → 7.18 (2.03×) | 7.54 (694) / 4.78 (106) | 12.0 → 6.14 s | 3.46 → 1.74 s | 6.8× / 6.4× / 7.3× | 7.2× / 6.8× / 6.8× |
| DTMB F1 708k, 1200 it | 32.02 → 22.07 (1.45×) | 22.07 (all active) | 39.9 → 27.9 s | 17.39 → 11.98 s | 10.9× / ≈ 10.3× / ≈ 10.2× | 13.0× / 12.4× / 12.5× |

("before" = the pre-change binary in the same session, `build/perf_base/dns`;
the FINAL report's 8.58 / 10.27 / 14.62 / 32.67 ms/it are the same code.)
What moved, rotor2 L1 per SIMPLE iteration: prefetched ELL row loops; fused
Jacobi sweeps + zero-start (no memset) in the AMG cycle; the scaled-V line
search folded into the prolongation; one sync per PCG iteration and two per
BiCGStab iteration (α on the device), the normFactor terms riding the first
batch, no duplicate A x0; deferred factor flags; the register-resident
coarsest inverse (0.80 → 0.51 ms); gradient reuse (`forces()` / RAS / the
iteration start) and the second non-orthogonal corrector reusing matrix and
`sumA`; MC-GS on a private colour-major copy with the idempotent turn pass
skipped; the three U components as ONE `bicgstabBatch`; and in the tail no
AMG at all: r0 is tested before the V-cycle (the corrector converged at
r0 last time) and the hierarchy is refreshed at its first use. GPU busy
4.2–4.6 ms (rocprof, run to run) of the ~5.6 ms active-phase iteration (L1);
the rest is launch latency and the 21 syncs.

## Precision (`--fv-precision fp32|fp64`, report `plans/verify/impl_fv_fp32.md`)

**FP32 is the default; FP64 is the parity mode.** One source, two
instantiations: every kernel, `Mesh`, `Ctl`, the RAS state and
`SimpleImpl<R>` (gpu/internal/fv_device.hpp) are templates on the arithmetic
R; `gpu/la` was already templated (`BlockEll<R>`, `MultiColourGS<R>`,
`Amg<R>`, `Krylov<R>`, block shapes unchanged). R = double is the FP64 solver
BIT FOR BIT: `tools/fv/bit_perf.sh` (14 cases, `FV_FLAGS="--fv-precision
fp64"`) vs the pre-FP32 binary `build/fp64_base/dns`, and the four
full-length FINAL cases vs `build/final/out` (stats but `ms`, force.dat,
moment.dat, log but timing / `#` / note lines) are byte-identical.

FP32 = float for: the fields (U, p, phi, k, omega, nuTilda, nut), the
geometry (S, weights, dc, V, C, Cf: rounded once from the host FP64 setup),
the face/row coefficients, both matrices, the Krylov vectors, the AMG
hierarchy values and every assembly kernel. FP64 is kept where the answer
depends on it, all of them reductions: batched dots and norms (FP64
accumulation from float inputs, the same partial count from N only — the
determinism contract of `gpu/la`), OpenFOAM's normFactor and residual sums,
xRef = gAverage(x) (rounded once to float), adjustPhi's three flux sums,
the continuity errors, the forces / moments (per-face float, summed in FP64),
and the smoothers' / coarsest level's factorisations (FP64 per thread, as in
`gpu/la`). Constants follow the arithmetic: OpenFOAM ROOTVSMALL 1e-18 for
float (1e-150 for double); SMALL stays 1e-15 (representable). Wall-function
constants (`FvWallFnPatch`, double on the host) are rounded to R in the
kernel, so the wall functions run in float too (nvcc rejects the mixed
double/float `fmin`/`fmax` calls; measured unchanged on rotor2 SST). The host
interface (FvState, logs, force sums) is FP64 either way.

**The tolerance.** An FP32 solve's initial residual cannot fall below the
arithmetic's noise (CLAUDE.md: true-residual floor ~ε_f32‖A‖‖x‖/‖b‖), so
fvSolution's tolerances (here 1e-10 p, 1e-12 U / k / omega) are never met:
without a floor every SIMPLE iteration solves noise to relTol and the
converged tail never starts. In FP32 each solve therefore stops at
`max(fvSolution tolerance, floor)` in OpenFOAM's normalised-residual units,
relTol unchanged (`dns/fvprecision.hpp` `kFp32Floors`): **U 1e-6, p 1e-6,
turbulence scalars 1e-7**. A solve that stops on the floor alone (its
residual above both the tolerance and relTol × initial) is not an error: the
log line ends `, floor-limited`, the run prints the counts at the end, and
the startup line (and the stats header) states every effective tolerance:

```
# FP32 (FP64 reductions): absolute tolerance = max(fvSolution, FP32 floor): U 1e-06 (FP32 floor; fvSolution 1e-12, relTol 0.01), p 1e-06 (FP32 floor; fvSolution 1e-10, relTol 0.01)
```

The floors are MEASURED. With no floor, the initial residual of the
converged FP32 tail sits at (median of the last 100 iterations; rotor2 L1 /
L1 SST / F1): U 3.1e-7 / 7.9e-7 / 4.6e-7, p 4.6e-7 / 6.6e-7 / 6.8e-7, k
2.2e-8, omega 7.3e-10 (SST). Each floor is the first decade above its
field's noise. The "smallest floor that never stalls a solve to maxIter"
criterion is DEGENERATE here: no solve of any case reached maxIter at any
floor tried (0, 1e-8, 1e-7, 3e-7, 1e-6, 3e-6; max 9 p, 5 U iterations) — the
FP32 recursive estimate keeps falling below the true floor (`gpu_amg_fp32`:
512² Poisson, the estimate reaches 7.5e-11 while the true residual stalls at
1.17e-5 = 1.28 × ε_f32 Σ|A||x|/Σ|b|), so a stall is not how FP32 fails;
chasing noise is. Bracket (CT vs FP64, whole wall):

| floor (U = p = s) | rotor2 L1 | L1 SST | F1 |
|---|---|---|---|
| 0 | −1.3e-7, 9.39 s | +3.1e-6, 6.68 s | −2.3e-6, 18.7 s |
| 1e-7 | −2.6e-6, 5.02 s | +4.3e-6, 4.29 s | — |
| 3e-7 | −3.4e-7, 4.52 s | — | — |
| 1e-6 | +6.2e-7, 4.24 s | +1.5e-5, 3.52 s | −2.5e-8, 11.3 s |
| 3e-6 | −1.2e-6, 3.46 s | — | — |

The turbulence floor is separate because it sets the nut FIELD (SST, U = p =
1e-6, max-norm vs FP64): s 0 → nut 4.3e-5, 1e-8 → 7.9e-4, 3e-8 → 1.2e-3,
1e-7 → 6.8e-3, 1e-6 → 0.40 (loop 3.76 / 3.37 / 3.32 / 3.09 / — s; CT within
the 2.6e-5 jitter at every value). 1e-7 keeps the rule (first decade above
k's noise). FP32 arithmetic alone (all floors 0) is nut 4.8e-5, U 1.0e-6, k
3.6e-7 — the field offsets below are the floors', not the float's. Quote
field-level parity in `--fv-precision fp64`. (The sweeps above ran before
the wall-function constants became R; re-measured after: all floors 0 nut
4.8e-5, s 0 with U = p 1e-6 nut 5.2e-5, the default 6.8e-3 — unchanged.)

**Measured** (full length, fresh outputs, gfx1100; FP32 job 553 (SST: job
557, after the wall-function cleanup) and FP64
walls job 548 under `simjob --bench`, FP64 ms/it from the byte-identical
rerun of job 555; FP64 = this binary in `--fv-precision fp64` = the
performance round; simpleFoam
from the FINAL / performance rounds above; settle = CT and CQ within 1e-3 of
the run's final value, loop time to it):

| case | ms/it FP64 → FP32 | wall FP64 → FP32 | to settle (loop) FP64 → FP32, settle it | FP32 vs foam np 6: ms/it / wall / settle | vs np 12 |
|---|---|---|---|---|---|
| rotor2 L1 115k, 1500 it | 3.98 → **2.63** (1.51×) | 6.21 → 4.24 s | 2.17 → 1.42 s (1.53×), it 357 / 357 | 8.1× / 7.5× / 6.7× | 10.3× / 9.7× / 8.3× |
| rotor2 L2 145k, 1500 it | 4.99 → **2.30** (2.17×) | 7.87 → 3.80 s | 3.17 → 1.76 s (1.80×), it 437 / 438 | 12.8× / 11.6× / 8.6× | 15.7× / 14.5× / 10.7× |
| rotor2 L1 SST 124k, 800 it | 7.19 → **3.79** (1.90×) | 6.13 → 3.41 s | 1.75 → 1.13 s (1.55×), it 209 / 209 | 12.9× / 11.4× / 11.2× | 13.7× / 12.3× / 10.5× |
| DTMB F1 708k, 1200 it | 21.88 → **8.18** (2.67×) | 27.35 → 11.29 s | 11.88 → 5.32 s (2.23×), it 507 / 507 | 29.4× / ≈ 25.5× / ≈ 22.9× | 35.1× / 30.6× / 28.0× |

Two effects: the float kernels (F1 with no floor, every iteration active:
21.88 → 14.37 ms/it, 1.52×; the small rotor2 meshes are launch / sync bound,
first 357 L1 iterations 2.16 → 2.08 s with no floor) and the floor, which
ends the active phase early (iterations with a p solve: L1 774 → 301, L2
915 → 352, SST 694 → 180, F1 1200 → 524; F1 in FP64 never reaches its 1e-10
p tolerance). Inner iterations over the run (FP64 → FP32): L1 U 4595 →
3076, p 9254 → 3431; L2 U 4821 → 3334, p 11889 → 2962; SST U 4992 → 2827, p
6235 → 1992, omega 1163 → 144, k 1553 → 871; F1 U 3661 → 3334, p 13863 →
4266. No solve reached maxIter.

Relative differences below are (x − x_ref)/|x_ref| (`tools/fv/fp32_compare.py`):
rotor2 L1 / L2 have CT < 0, so their CT sign is opposite to dns/ref − 1.

| case | CT, CQ vs FP64 at the last iteration (last-100 means) | vs simpleFoam (SST: serial) | CT / CQ jitter, last 100 it | fields FP32 vs FP64 at the last iteration (max-norm of scale) |
|---|---|---|---|---|
| rotor2 L1 | +6.2e-7, −2.2e-9 (−2.6e-7, +8.2e-9) | +6.2e-7, −2.2e-9 (np 12) | 3.2e-6 / 1.4e-7 | U 1.4e-5, p 1.2e-6, phi 2.7e-6 |
| rotor2 L2 | +1.7e-6, +5.2e-8 (+5.6e-7, +2.6e-8) | +1.7e-6, +5.2e-8 (np 12) | 1.9e-6 / 8.6e-8 | U 3.9e-5, p 2.6e-6, phi 2.4e-6 |
| rotor2 L1 SST | +1.25e-5, +8.8e-7 (+5.6e-6, +3.5e-7) | +1.25e-5, +8.8e-7 (serial) | 3.0e-5 / 2.5e-6 | U 1.2e-4, p 3.5e-5, phi 1.0e-5, k 2.1e-4, omega 1.1e-4, nut 6.8e-3 |
| DTMB F1 | −2.5e-8, −4.8e-8 (−1.0e-6, −1.1e-7: FP64 still drifts, jitter 3.1e-6) | −2.8e-8, −4.9e-8 (np 12) | 4.6e-6 / 3.8e-7 | U 5.5e-5, p 1.0e-5, phi 3.6e-6 |

SIMPLE settles in FP32 exactly as in FP64 (settle-1e-3 iteration equal ± 1)
and the end-of-run jitter is below 1e-4 relative everywhere (CT 3.0e-5 at
most: the SST state wanders on its 1e-6 floor; FP64 1.5e-7) — the SST
last-iteration CT offset (1.25e-5) is that jitter, not a bias. FP32 reruns
are byte-identical (rotor2 L1 and L1 SST full length; `gpu_fv_simple`
channel, sst_plate, mrf2). Label `[FV path, body-fitted, <physics>, FP32]`
(`FP64` in parity mode); `--dry-run` prints the precision and the floors.

RAS (phase BL, `plans/verify/impl_meshsolver_bl.md`; OpenFOAM v2606 serial
unless stated):

| case | cells | vs simpleFoam | result |
|---|---|---|---|
| IF-T sst_plate: 2D plate, kOmegaSST, kqR / omegaWallFunction (binomial) / nutUSpalding, symmetryPlane run-up, slip `wall` lid, omega bounded EVERY iteration | 960 | identical blockMesh, tight solves | U, p, phi ≤ 1.2e-13, k 4.7e-13, omega 3.7e-12, nut 1.6e-10 of scale at it 100 / 400; residuals of 50 iterations 7e-10 |
| IF-T sst_skew: 26.6° channel, nutk + omegaWallFunction stepwise / nutLowRe + tanh, inletOutlet k / omega, cellLimited grad(k), 1 non-orth corrector, `exactDistance` | 600 | same | ≤ 3.2e-13 (nut 4.3e-12); residuals 7.9e-10 |
| IF-T sa_skew: SpalartAllmaras ft2 on, Spalding / nutLowRe, meshWave | 600 | same | ≤ 1.5e-13; residuals 6.9e-12 |
| BL-a TMR-like flat plate (the tutorial `turbulentFlatPlate` at 1/4 resolution, y+ 1 grading 9300), Re_L 5e6, kOmegaSST, its schemes (`bounded Gauss limitedLinear 1` k / omega), relTol 0.1, 8000 it | 13 056 | identical mesh | drag Fx 5.7e-6; cf(x) ≤ 2.6e-4 (x ≥ 0.05; 4e-6 at mid plate); u(y) profiles ≤ 2.2e-5 U at x = 0.5 / 1.0 / 1.5 |
| BL-b rotor2 L1 + PRISM LAYERS (snappy addLayers), kOmegaSST + MRF, nu 5e-4 (blade y+ mean 0.85, max 7.6), relTol 0.01, 800 it | 123 511 | identical mesh, serial | CT 2.3e-8, CQ 1.3e-9; fields U 2.3e-8, k 2.8e-9, nut 8.8e-9 of scale; settle / residual iteration counts equal ± 3 |
| the same vs simpleFoam np 6 | 123 511 | np 6 | CT 4.8e-7, CQ 4.5e-8 (np 6 vs serial simpleFoam itself: 5e-7 — parallel meshWave / GAMG) |

Wall time (`simjob --bench`): plate 8000 it dns 36.7 s (4.5 ms/it) vs
simpleFoam serial 49.9 s; rotor2 L1 SST + MRF 800 it dns 12.1 s (14.7 ms/it)
vs simpleFoam np 6 39 s / serial 144 s (ClockTime, verifier re-run under
`--bench`, jobs 469 / 466; the first np 6 35 s / serial 145 s were NOT
`--bench`) (3.3× / 12×; CT within 1e-3 of final: it 209 vs 210, 3.6 s vs
12.7 s np 6).

## The algorithm (simpleFoam/UEqn.H + pEqn.H, verbatim)

One `DeviceSimple::iterate()`:

1. `storePrevIter` of p (when `relaxationFactors/fields/p` exists).
2. `MRF.correctBoundaryVelocity`: U_b = Ω × (Cf − o) on every zone's
   included boundary faces (section MRF).
3. UEqn: `fvm::div(phi, U)` (`[bounded] Gauss linear | linearUpwind grad(U) |
   upwind`), `+ MRF.DDt(U)` (source −V Ω_z×U in zone z), `− fvc::div(ν
   dev2(T(grad U))) − fvm::laplacian(ν, U)` (`Gauss linear corrected |
   uncorrected | limited [corrected] ψ`), then
   `UEqn.relax(α_U)` exactly as `fvMatrix::relax` (D + Σ cmptMax|ic|, max with
   Σ|off|, /α, − Σ cmptMin ic, S += (D − D₀) U).
4. Momentum predictor `solve(UEqn == −grad p)` per component (the empty
   axis of a 2D mesh is not solved, as `validComponents`), then
   `U.correctBoundaryConditions()`. The components' systems share UEqn's
   off-diagonals: they run as ONE `bicgstabBatch` (each component bit for
   bit its own solve; the first failing component's error, as the
   component-by-component loop reported it).
5. `rAU = 1/A`, `HbyA = rAU H` (`fvMatrix::A/H/H1` with the boundary
   internal/boundary coefficients exactly as `addBoundaryDiag`,
   `addCmptAvBoundaryDiag`, `addBoundarySource`), `constrainHbyA`,
   `phiHbyA = flux(HbyA)` (`λ(P − N) + N`), `MRF.makeRelative`, `adjustPhi`
   (closed domains), SIMPLEC: `rAtU = 1/(1/rAU − H1)`,
   `phiHbyA += interpolate(rAtU − rAU) snGrad(p) |S|`, `HbyA −= (rAU − rAtU) grad p`.
6. nNonOrthogonalCorrectors + 1 times: `fvm::laplacian(rAtU, p) ==
   fvc::div(phiHbyA)` with the explicit non-orthogonal correction
   (`faceFluxCorrectionPtr`), `setReference(pRefCell, pRefValue)` when no p
   patch fixes the level, solve, `p.correctBoundaryConditions()`; on the
   last: `phi = phiHbyA − pEqn.flux()`.
7. continuity errors (as `continuityErrs.H`), `p.relax()` (internal AND
   boundary values, `operator==`), `U = HbyA − rAtU grad p`,
   `U.correctBoundaryConditions()` (the mixed patches re-read phi / U_b
   first: their `updateCoeffs`).

Geometric factors are `basicFvGeometryScheme`'s: weights
`|S·(Cf−C_o)| / (|S·(Cf−C_o)| + |S·(C_n−Cf)|)`, `nonOrthDeltaCoeffs =
1/max(n·δ, 0.05|δ|)`, correction vectors `n − δ·dcNO` (0 on non-coupled
patches), patch deltaCoeffs `1/|n (n·(Cf − C))|`. Merged cyclic faces carry
the separation in δ and an owner-frame `rO = Cf − C_o` / neighbour-frame
`rN = rO − δ` (the FaceMesh contract: a cyclic face's Cf may sit in either
frame).

**Constraint patches.** OpenFOAM gives every DERIVED "calculated" field
(H, grad U, ...) the constraint patch type on `symmetry` / `symmetryPlane`
(and `empty`, `cyclic`) polyMesh patches. So on a symmetry patch the
boundary value of H is the reflected average `0.5 (H_P + R H_P)`,
R = I − 2nn — phiHbyA there is ~0, not `S · rAU_P H_P` — and the boundary
gradient of U starts from `0.5 (G_P + R G_P Rᵀ)` before gaussGrad's normal
correction. A `slip` BC on a plain `patch` is NOT a constraint patch (it
uses the extrapolated values). Missing this was a 3.7% field error on IF1d
(measured), invisible on every other case.

Boundary conditions (`fvPatchField` value/gradient coefficients verbatim):

| U | p | notes |
|---|---|---|
| `fixedValue`, `noSlip`, `movingWallVelocity` (static mesh) | `fixedValue` | not assignable (constrainHbyA) |
| `zeroGradient` | `zeroGradient` | |
| `inletOutlet` (`phi` only) | — | mixed, vf = neg(phi); assignable; adjustable in adjustPhi |
| `freestream` (= inletOutlet on freestreamValue) | — | |
| `freestreamVelocity` | `freestreamPressure` | mixed, vf = 0.5 ∓ 0.5 (U_b·n)/\|U_b\| (`supersonic no`) |
| `slip` | `slip` | basicSymmetry, not assignable; p zeroGradient |
| `symmetry`, `symmetryPlane` | same | constraint patches (above); per-face normal (planar patches: equal to the plane normal to rounding) |
| `empty`, `cyclic` | same | no faces / merged into internal faces |

Schemes accepted: `ddtSchemes default steadyState`; `grad(U)`, `grad(p)`
(or `default`): `Gauss linear`, `cellLimited Gauss linear K` (minmod, as
`cellLimitedGrad`); `div(phi,U)`: `[bounded] Gauss linear | linearUpwind
grad(U) | upwind`; `div((nuEff*dev2(T(grad(U)))))` `Gauss linear`;
`laplacianSchemes` `Gauss linear corrected | uncorrected | limited
[corrected] ψ` (one for both); `interpolationSchemes default linear`;
`snGradSchemes default corrected | uncorrected | limited [corrected] ψ`.
`limited` is `limitedSnGrad`: the corrected scheme's correction (U: `k ·
interpolate(grad U)` with grad(U)'s own scheme; p: grad(p)'s) times ONE
limiter per face, `min(ψ |δ_c (x_N − x_P)| / ((1 − ψ) |corr| + 1e-15), 1)`
(|·| the vector magnitude for U), in both laplacians (`faceFluxCorrection`)
and in SIMPLEC's `fvc::snGrad(p)`; ψ in [0, 1], `limited uncorrected`
rejected. Controls: `SIMPLE {nNonOrthogonalCorrectors, consistent,
momentumPredictor, pRefCell, pRefValue, residualControl, transonic no}`,
`relaxationFactors {fields {p}, equations {U}}` (regex keys), `cache {}`.

## Linear solves and OpenFOAM's stop

`gpu/la` in the run's arithmetic (FP32 default, FP64 parity; section
Precision): U by BiCGStab + multicolour Gauss–Seidel, p by flexible
AMG-PCG on the NEGATED (SPD) Laplacian, AMG hierarchy built once on the
geometric weights |S| dcNO, refreshed once per SIMPLE iteration AT ITS FIRST
USE (an iteration whose p solves all start converged — the SIMPLE tail —
refreshes nothing; a later corrector reuses the matrix, so the first use
sees the same values); scaled-V
cycle (OpenFOAM GAMG's `scaleCorrection`), 2 damped Jacobi sweeps per level
(measured on F1: 34.9 ms/iteration vs 45.2 for the gpu/la K-cycle default,
54.1 for a plain V-cycle). The STOP is OpenFOAM's: residual = Σ|b − Ax| /
normFactor, normFactor = Σ(|Ax₀ − ΣA x̄| + |b − ΣA x̄|) + 1e-20 (x̄ the
initial guess's mean, `lduMatrix::solver::normFactor`), stop when below
`tolerance` (FP32: max(tolerance, the floor)) or `relTol` × the initial one,
between `minIter` and `maxIter`
(`KrylovOptions::normFactor`, `minIter`). OpenFOAM's solver NAMES (GAMG,
PCG, smoothSolver, PBiCGStab) select our solver per field; their iterates are
not reproduced (`notes` in the case description), so a relTol-0.01 transient
agrees to ~1e-3, the fixed point to solver tolerance.

## Output and CLI

```
dns flow --polymesh CASE [--max-iters N] [--fv-precision fp32|fp64] [--out PREFIX] [--emit stats|none] [--dry-run]
```

- `--fv-precision` (default fp32; section Precision) — rejected without
  `--polymesh` (exit 2).
- `--fv-coupled N` [`--fv-coupled-eta X`, `--fv-coupled-alpha A`,
  `--fv-coupled-check`] — the EXPERIMENTAL coupled prototype (section
  Coupled prototype); needs `--fv-precision fp64`; the tuning flags need
  `--fv-coupled` (exit 2).

- `--dry-run` reads the polyMesh manifest and the case dictionaries (phase
  1.5, the `--scene` precedent) and prints the resolved case.
- Phase 3 reads the mesh (exit 3 before any device), creates PREFIX/,
  `PREFIX/postProcessing/<forces>/0/{force,moment}.dat` and
  `PREFIX_stats.csv` before the first device allocation.
- The run prints OpenFOAM-style log lines (`Time = N`, `Solving for Ux,
  Initial residual = …`, `time step continuity errors`), writes
  `PREFIX/<time>/{U,p,phi}` (RAS: + `k`, `omega` | `nuTilda`, `nut`; the
  stats CSV appends the turbulence solves' initial residual and iteration
  count, in solve order) (ASCII, `writePrecision`, OpenFOAM cell / face /
  patch-face order, boundary `calculated`) at `writeInterval` and at the end,
  and the forces function object's force.dat / moment.dat (pressure +
  viscous, `rhoInf`, `CofR`, `−ρν devTwoSymm(grad U_b)` on the corrected
  boundary gradient) every iteration. `tools/fv/foamdiff.py A B` diffs two
  time directories cell by cell.
- Every other `dns flow` flag is rejected with `--polymesh` (exit 2); so is
  `--physics compressible` (the compressible FV path, P4/P7, is not built).

Label (log and stats header): `[FV path, body-fitted, laminar, FP32]` (RAS:
the model label and the wall-distance method, section RAS; `FP64` in the
parity mode; the next header line states every effective tolerance), plus
`[MRF frozen rotor: APPROXIMATION, steady rotating frame]` with an active MRF
zone; the dry-run names each zone `[MRF frozen rotor]`.

## MRF (OpenFOAM `MRFZoneList` / `MRFZone`, v2606 source)

`constant/MRFProperties`: one sub-dictionary per zone, applied ZONE BY ZONE in
file order (MRFZoneList). Per zone: `active` (default yes; an inactive zone
reads nothing else, as `MRFZone::read`), `cellZone` (a polyMesh cellZone),
`origin`, `axis` (normalised), `omega` (rad/s, `omega X;` or `omega constant
X;` — other Function1 types are time-dependent: rejected), and
`nonRotatingPatches` (wordRes: a word is a literal name, a quoted string a
regex; a selector matching no patch is an error here, OpenFOAM ignores it).
`setMRFFaces` verbatim:

- internal faces with the owner OR neighbour in the zone: `makeRelative`
  subtracts `(Ω × (Cf − o)) · S` — so a face between two zones' cells is
  subtracted TWICE, once per zone (`phi − frame_lo − frame_hi`; dropping the
  second makes IF1f 13% wrong, measured);
- boundary faces of zone cells on a non-coupled, non-empty patch:
  INCLUDED — `phi = 0`, `U_b == Ω × (Cf − o)` (`correctBoundaryVelocity`,
  start of every UEqn) — whatever the patch is (walls, inlets, slip,
  symmetryPlane); on a `nonRotatingPatches` patch: EXCLUDED — `phi −=
  frame` and U_b keeps its BC;
- `MRF.DDt(U)`: −V (Ω_z × U) in the momentum source of zone-z cells.

`createPhi` is not made relative (simpleFoam calls `makeRelative` on
phiHbyA only); `makeAbsolute` is never called by simpleFoam and is not
built. The forces function object takes the moment about `CofR` (required;
`origin` is not read) on the face centres, pressure and viscous parts separately.
Rejected by name: two active zones on one cellZone (overlap; disjoint
cellZones are a polyMesh reader invariant), a zone cell on a merged cyclic
face (OpenFOAM subtracts each half's own frame), a zero axis, more than 8
active zones (`FvSimpleSetup::kMaxMrfZones`).

Kernel variants are separate template instantiations (`kURow<Multi>`,
`kPhiH<Two>`, `kUFace/kPFace/kConsistentF<Lim>`): a single-zone corrected
case runs the P5b kernels and is byte-identical to the pre-MRF-phase binary
(stats, force.dat, fields: `build/mrfbit/bit.sh`).

## RAS (phase BL: kOmegaSST, SpalartAllmaras as simpleFoam runs them)

`constant/turbulenceProperties` `simulationType RAS; RAS { RASModel
kOmegaSST | SpalartAllmaras; turbulence on; <model>Coeffs {...} }` (a
`momentumTransport` beside it must agree). Fields `0/nut` + `0/k`, `0/omega`
(SST) or `0/nuTilda` (SA). `gpu/fv_ras.hip`, called as simpleFoam does:
`turbulence->validate()` (correctNut on the initial fields) after createPhi,
`turbulence->correct()` after the momentum corrector of every iteration; the
momentum equation's `divDevSigma` takes nuEff = nu + nut (cell values,
LINEAR face interpolation for `fvm::laplacian(nuEff, U)`, cell products for
`fvc::div(nuEff dev2(T(grad U)))`, nu + nut_b on the boundary), and the
forces function object `devRhoReff` = −ρ nuEff_b dev(twoSymm(grad U_b)).

**kOmegaSST** = `kOmegaSSTBase::correct()` verbatim: S2 = 2|symm(grad U)|²,
GbyNu0 = grad U && devTwoSymm(grad U), G = nut GbyNu0; omega's
`updateCoeffs` (the wall function, below) BEFORE CDkOmega = 2 αω2 (∇k·∇ω)/ω;
F1 = tanh(min(min(max(√k/(β* ω y), 500ν/(y²ω)), 4αω2 k/(max(CDkOmega,
1e-10) y²)), 10)⁴), F23 = F2 (× F3 when `F3 on`), GbyNu limited by (c1/a1)
β* ω max(a1 ω, b1 F23 √S2); the omega equation (γ GbyNu, −SuSp(2/3 γ divU),
−Sp(β ω), −SuSp((F1−1) CDkOmega/ω)), `relax`, `boundaryManipulate`
(setValues on the wall-function cells), solve, `bound(omega, omegaMin)`;
then the k equation with the NEW omega (Pk = min(G, c1 β* k ω),
−SuSp(2/3 divU), −Sp(β* ω)), relax, solve, `bound(k, kMin)`; correctNut nut
= a1 k/max(a1 ω, b1 F23 √S2) with F23 of the new fields. Default
coefficients of `kOmegaSSTBase` (overrides by name; `decayControl yes`
rejected). Label `RANS kOmegaSST (OpenFOAM v2606), steady (SIMPLE
pseudo-time: not time-accurate)`.

**SpalartAllmaras** = `SpalartAllmarasBase::correct()` verbatim (OpenFOAM's
SA: Stilda = max(Ω + fv2 ν̃/(κd)², Cs Ω), r ≤ 10, the Cb2/σ|∇ν̃|² source,
`bound(nuTilda, 0)`); `ft2` is OpenFOAM's switch (default OFF — CLAUDE.md
parity note: set `ft2 true`; Prt is irrelevant here, no temperature).
Label `RANS SpalartAllmaras (OpenFOAM v2606, ft2 on|off)`.

**Boundary values of derived fields** follow OpenFOAM's field algebra:
`calculated` nut patches take the model expression of the BOUNDARY values
(k_b, ω_b, the boundary gradient's S2, F23 with y_b); DkEff / DomegaEff
take F1 of the boundary values; y_b = SMALL on wall patches (wallDist's
fixedValue), y_P elsewhere (zeroGradient). fixedValue-derived patches
(every nut wall function, fixedValue) ignore the assignment: Spalding's
first guess is the PREVIOUS wall nut. `bound()` max-assigns zeroGradient-type
patches (kqR included: the stale boundary value), re-evaluates
slip/symmetry, leaves fixedValue / mixed alone.

**Wall functions** (`wallFunctionCoefficients` Cmu 0.09, κ 0.41, E 9.8,
yPlusLam = 10 fixed-point iterations): `omegaWallFunction` — blending
`stepwise | max | binomial (default, n 2) | exponential | tanh`, cornerWeights
1/(the cell's omega-wall faces), ω and G of the wall cells set, the cells
FIXED in the omega matrix; `nutLowReWallFunction` (0),
`nutUSpaldingWallFunction` (Newton on Spalding's law, maxIter 10, tolerance
0.01 — another tolerance takes calcNut's second pass: rejected),
`nutkWallFunction` (its blenders); `kqRWallFunction` = zeroGradient. y of
the wall functions = `nearWallDist`: the cell centre's distance to the wall
face and its point neighbours on the combined wall patch.

**Wall distance (masterplan U9 knob, stated in the log label and dry-run)**:
`wallDist { method meshWave; }` = FaceCellWave of `wallPoint` (nearest wall
FACE CENTRE, propagation tolerance 0.01) in OpenFOAM's SERIAL order —
`primitiveMesh::cells()` lists a cell's OWNED faces then its NEIGHBOUR
faces, and with the 1% tolerance that order changes the result (an
ascending face order was 14% off at 10k cells of the rotor2 mesh) — then
`correctBoundaryCells` (wall-face cells and wall-point cells: the exact
distance to the faces around them). Equal to simpleFoam's serial
`wallDistance` to 4e-14 (unlayered rotor2), 5e-12 (layered, its 12-digit
output), 2.6e-15 (plate) — `fvwalldist`, `fvwalldist_rotor`. A decomposed
simpleFoam runs the wave per processor (np 6 vs serial: CT 5e-7).
`exactDistance` = nearest point on the wall faces (the faces' own fan
triangulation; simpleFoam needs `libs ("libdistributed.so")` for it, and
its exact method fails in parallel). meshWave with cyclic patches is
rejected (the cyclic transfer of the wave is not reproduced).

**Scalar transport**: `fvm::div(phi, X)` `[bounded] Gauss linear | upwind |
linearUpwind grad(X) | limitedLinear k` (`bounded` = −Sp(div phi)), the
gradient of X with its own `grad(X)` scheme (Gauss linear | cellLimited);
`fvm::laplacian(Gamma, X)` with Gamma interpolated linearly, the laplacian
scheme of the case (one for all fields); `fvMatrix::relax` (scalar: D +
Σ|ic|, max with Σ|off|, /α, − Σ ic); `setValues` (fixed cells: off entries
zeroed, the neighbours' coefficients moved to their sources); OpenFOAM's
solver stop (`smoothSolver` / `PBiCGStab` / `PBiCG` / `GAMG` names → gpu/la
BiCGStab + MC-GS, the run's precision; FP32 floor 1e-7). BCs: `fixedValue`, `zeroGradient`,
`kqRWallFunction`, `inletOutlet`, `freestream`, `omegaWallFunction`, `slip`,
`symmetry`, `symmetryPlane`, `empty`, `cyclic`; nut `calculated`,
`fixedValue`, `zeroGradient`, the three wall functions, constraint types.
`relaxationFactors/equations` for k / omega / nuTilda, `residualControl`
entries for them.

**`limitedLinear` is not a tight-parity scheme.** On a field uniform to the
bit (the initial freestream) gradf = 0 and r = 2000 sign(d·∇X) − 1 takes
the sign of the gradient's ROUNDING NOISE: the limiter is 0 or 1 by the
summation order (it differs between simpleFoam decompositions too). The
tight IF-T cases therefore use linearUpwind; limitedLinear's parity is the
converged plate (BL-a).

**Prism layers (snappy addLayers, `tools/fv/make_rotor_bl.py`).** The
rotor2 L1 mesh re-meshed with 8 requested layers (relative first layer
0.03, expansion 1.3), `nBufferCellsNoExtrude 0`, `relaxed {
minTetQuality -1e30; }`: snappy extrudes 79% of the 5440 blade faces but
truncates the stacks around every unextruded spot (+1 layer per face ring),
so the blade averages 1.52 layers and 41% of the target thickness (snappy
log); 123 511 cells (110 988 hex, 731 prisms, 11 789 polyhedra), checkMesh
non-orth max 65.2°, 2 faces with skewness 17.9. FaceMesh reads it
unchanged. `nBufferCellsNoExtrude -1` (no truncation) removes ALL extrusion
on this mesh; `nOuterIter`, medial-axis and feature-angle variants did not
raise the coverage (report). At nu 2e-5 (y+ ~ 10–20) neither code reaches a
steady state on this mesh (CT oscillates ±3% from it ~100 in simpleFoam;
central U diverges there), so the gate runs at nu 5e-4: blade y+ mean 0.85.

## Coupled prototype (EXPERIMENTAL, `--fv-coupled N`; plan_perf_5 P0a + P0b)

`plans/plan_perf_5_coupled.md`. Opt-in, FP64 only (`--fv-precision fp64`,
else exit 2), laminar only (RAS: exit 2). Absent = SIMPLE byte for byte.
After N SIMPLE iterations every iteration is one coupled outer iteration
(`gpu/fv_coupled.hip`):

1. R* from SIMPLE's own stage methods without a solve: R_U = rhs3 − A_rel U
   (U_old = U, so the UNRELAXED residual), R_c = rhs − AP p (= −div of
   simpleFoam's φ at (U, p)), R_c* = R_c + div(interp(rAU R_U / V)·S) (a
   row transform with the same zero set). The zero set is simpleFoam's
   fixed point: the Rhie–Chow coefficient is the case's α_U-relaxed rAU.
2. Probe: simpleFoam's normalised initial residuals at this state (U on the
   relaxed system, p on AP); the run stops when they reach the fvSolution
   tolerances (`coupled solution converged`).
3. K dx = R*, K = −dR*/dx with φ, rAU, rAtU, value fractions and the U-side
   deferred corrections frozen (Picard), composed: AU off + D0 + τ Dm + ic,
   V Ω×, the slip/symmetry bc coupling, V grad, div(interp(·)·S) with HbyA's
   boundary rule, the linearised Rhie–Chow and SIMPLEC terms, setReference.
   FGMRES(30), relTol η (`--fv-coupled-eta`, 0.1), right-preconditioned by
   one linearised SIMPLEC step (MC-GS on the case's relaxed U system, one
   AMG V-cycle on AP, u = u* − rAtU grad δp). τ = (1/α − 1) min(1, r/r0)
   (`--fv-coupled-alpha`, 1 = none).
4. x += dx; φ = F(U, p) with this iteration's matrices (kRAU … kPhi).

P0a split `SimpleImpl<R>::iterate()` into `assembleU` (`boundaryU`,
`rowsU`), `systemU`, `evalU`, `prepP`, `assembleP`, `evalP`, `fluxes`,
`correctU`: `tools/fv/bit_perf.sh` 14 cases byte-identical in FP32 and FP64
against the pre-refactor binary, reruns identical (jobs 584, 587).

Measured (FP64, gfx1100, job 586; settle = CT and CQ within 1e-3 of the
converged value; tolerance = probe ≤ U 1e-12, p 1e-10; C_K = Krylov ms per
iteration, C_fixed = residual + setup + update per outer):

| run | outer to settle | outer to tolerance | K̄ to settle | C_K / C_fixed | loop to settle | loop to tolerance |
|---|---|---|---|---|---|---|
| rotor2 L1 from IC, η 0.1 | 18 | 66 | 42.8 | 1.02 / 2.6 ms | 0.834 s | 2.07 s |
| rotor2 L1, 30 SIMPLE + coupled, η 0.1 | 15 | 62 | 38.4 | 1.02 / 2.1 ms | 0.810 s | 2.03 s |
| rotor2 L1 from IC, η 0.3 | 20 | 76 | 20.1 | 0.99 / 2.5 ms | 0.449 s | 1.32 s |
| F1, 30 SIMPLE + coupled, η 0.1 | 18 | 64 | 76.9 | 6.20 / 10.3 ms | 9.47 s | 18.4 s |
| SIMPLE FP64 (reference) | L1 357 / F1 507 it | never (1500 / 1200 it) | — | — | 2.17 / 11.88 s | — |

Converged CT / CQ vs SIMPLE FP64 1500 it and simpleFoam: L1 +2.1e-11 /
−7.1e-13 (IC), +6.8e-11 / −2.1e-12 (hybrid), +3.8e-11 / −1.3e-12 (η 0.3).
F1 +2.8e-6 / +2.4e-7: both references are still drifting at 1200 it
(issues.md 63c). G-JAC (|K v − FD| / |K v|, frozen coefficients): cavity
8.2e-16 / 1.7e-15, mrf2 1.5e-14 / 2.9e-14, channel 4.5e-14 / 3.7e-14,
rotor2 L1 2.7e-14 / 1.4e-14 (U / continuity rows); duct 1.2e-4 / 3.8e-4
(freestream value fractions not frozen in the FD). G-FP0 (1500 SIMPLE
iterations, then coupled): the first probe is SIMPLE's tail level (U
1.4e-11, p 1.04e-10), CT moves ≤ 6e-13 per outer, the probe reaches 2.5e-13 /
6.7e-13 in two outer iterations. Coupled reruns byte-identical (stats but
timing, fields, force / moment).

Kill gate (plan §6: ≥ 1.5× to settle on L1 AND F1 vs SIMPLE FP32 1.42 /
5.32 s; FP32 projection = FP64 time / the SIMPLE float-kernel ratio, L1
1.04×, F1 1.52× (2.0× optimistic)): L1 1.77× (IC) / 1.97× (hybrid) / 3.3×
(η 0.3); **F1 0.87–1.13× at η 0.1: NOT PASSED**. F1 at η 0.3 MODELLED from
L1's Krylov reduction (0.52×): 1.57–2.0×, unmeasured. P1 not started.

## Files

| file | content |
|---|---|
| `gpu/fv_coupled.hip` | EXPERIMENTAL coupled p-U prototype (`--fv-coupled`): R*, probe, composed Jacobian, SIMPLEC-step preconditioner, FGMRES |
| `dns/foamdict.{hpp,cpp}` | OpenFOAM dictionary tokenizer/parser (balanced-paren words, regex keys, directives kept by name) |
| `dns/fvcase.{hpp,cpp}` | `FoamCase`: BCs, schemes, controls, MRF, forces; `parseFoamCase` (pure), `readFoamCaseTexts` |
| `dns/fvsimple.{hpp,cpp}` | `loadFvCase`, `buildSimpleSetup`: geometric factors, BC codes per face, CSR + ELL slots, MRF frame flux, pressure reference |
| `dns/foamio.{hpp,cpp}` | OpenFOAM ASCII field writer / internalField reader |
| `dns/fvprecision.hpp` | `FvPrecision`, `FvFloors`, the measured `kFp32Floors`, `fvNumerics` |
| `gpu/fv_simple.{hpp,hip}` | `DeviceSimple`: the kernels and one SIMPLE iteration, forces (templates on R; both instantiated) |
| `gpu/internal/fv_device.hpp` | the device mesh view, shared helpers / gradient kernel, `DeviceSimple::Impl` (interface), `SimpleImpl<R>`, `RasState<R>` |
| `tools/fv/fp32_compare.py`, `tools/fv/bit_perf.sh` (`FV_FLAGS`) | CT / CQ vs a reference, settle, jitter, the log's per-field solve statistics (maxIter, floor-limited, tail residual); the FP64 byte gate |
| `gpu/fv_ras.hip` | RAS: kOmegaSST, SpalartAllmaras, wall functions, scalar transport, bound, correctNut |
| `dns/fvwalldist.{hpp,cpp}` | meshWave (FaceCellWave, serial order) + correctBoundaryCells, exactDistance (AABB tree), nearWallDist |
| `tools/fv/make_ras_cases.py`, `make_ras_test_data.sh`, `make_rotor_bl.py`, `plate_compare.py`, `fieldloc.py` | the RAS parity cases + references, the layered rotor2 case, the plate cf / profile gate, a field-difference locator |
| `tests/data/fv/{sst_plate,sst_skew,sa_skew}` (1.7 MB), `tests/test_fvwalldist.cpp` (`fvwalldist`, `fvwalldist_rotor`) | the IF-T cases with simpleFoam's fields, residuals and `wallDistance` |
| `gpu/runner_fv.{hpp,hip}` | the run loop, log, residualControl, outputs |
| `tools/fv/make_cases.py`, `run_foam.sh`, `make_test_data.sh`, `foamdiff.py` | the parity cases, their OpenFOAM runs, the checked-in references, the field diff |
| `tools/fv/make_rotor_cases.sh`, `mrf_compare.py` | the rotor2 L1/L2 MRF cases from the parity suite's meshes; CT/CQ, settle and residual iteration counts, times of a dns run vs its simpleFoam run |
| `tests/data/fv/{cavity,cavity_skew,channel,duct,cavity_lim,mrf2}` (2.6 MB) | the IF1 cases with OpenFOAM's fields at it 100 / last and the initial-residual history |
| `tests/test_fvcase.cpp` (`fvcase`), `tests/test_gpu_fv_simple.cpp` (`gpu_fv_simple`) | the case layer (M1: MRFZoneList parsing / setup / rejections); IF1 a–f, determinism, 2D w ≡ 0 |

## Invariants

- Deterministic: one thread per face (face coefficients) or per row (CSR
  gather in ascending face order, ELL entries zeroed then accumulated by
  their own row), fixed-order FP64 reductions (also of float inputs), no
  atomics: two runs are byte-identical in either precision (`gpu_fv_simple`).
- `--fv-precision fp64` is the FP64 solver bit for bit (R = double of the same
  templates); never fork a kernel per precision.
- The MRF frame flux is OpenFOAM's POINTWISE `(Ω × (Cf − o))·S` on the
  zone's internal faces (MRFZone::makeRelativeRhoFlux), not the telescoping
  edge-potential form of the masterplan's compressible FV MRF (§3, P9): this
  path reproduces simpleFoam. On planar faces both are exact.
- 2D: the empty axis is never solved; U on it stays exactly 0.
- Coupled prototype: the residual is simpleFoam's fixed point, built from
  SIMPLE's own stage methods; the Rhie–Chow coefficient carries the case's
  α_U (`kRAU`), never the coupled pseudo-time or FGMRES tolerance — those
  live only in the Jacobian / preconditioner (path, not answer).
- Feature off is every existing path unchanged: the FV path is a separate
  family behind `--polymesh` (BIT: 19 dry-runs and 5 GPU stats files
  byte-identical to the pre-P5b binary).

## Not built

- PISO / PIMPLE (pimpleFoam: `ddtCorr`, Euler ddt), transient controls.
- Other RAS models (kEpsilon, realizableKE, kOmegaSSTLM, ...), the
  compressible / variable-density turbulence forms, `kLowReWallFunction`,
  rough wall functions, `decayControl`, meshWave with cyclic patches,
  `limitedLinearV`; heat transfer, compressible FV (P4/P7), species.
- `fixedFluxPressure`, `totalPressure`, `pressureInletOutletVelocity`,
  `rotatingWallVelocity`, `limited uncorrected`, other snGrad schemes,
  `leastSquares` gradients, `pRefPoint`, `startFrom` a non-zero time,
  overlapping MRF zones, MRF zones touching cyclic faces, a time-dependent
  `omega`, `makeAbsolute` (unused by simpleFoam), field relaxation of U /
  equation relaxation of p.
- `phi` output on meshes with cyclic patches (OpenFOAM's face order needs
  the cyclic patch faces the FaceMesh merges); U and p are written.
- Rounding-level replication of OpenFOAM's per-term accumulation order (the
  fixed points agree to ~1e-13 regardless).
