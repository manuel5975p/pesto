# gpu/la — the GPU sparse linear-algebra library

The solver library of `plans/masterplan_meshsolver.md` §2.4 (phases P3, P5):
block matrices on a `CellGraph`, block Jacobi, multicolour block Gauss–Seidel,
batched block-tridiagonal line solves, FGMRES / PCG (classic or flexible) /
BiCGStab with batched FP64 inner products, and aggregation AMG (B = 1 and every
block shape; V-, scaled-V and K-cycle). Own kernels, no vendor sparse library (§7), one source for
HIP and the CUDA shim. Mesh-agnostic: a matrix sees rows, ascending columns and
per-face slots, never faces — the structured NK (P6), the incompressible FV
SIMPLE/PISO solver (P5b: pressure B = 1, momentum B = 3 or 3 × B = 1), the FV
implicit solver (P7, B = 5/6/7) and chemistry (B = 5 + nT + nY) all plug in.

At P3 and P5 it was test-only. Its first runner customer is the
incompressible FV path (P5b, `dns flow --polymesh`, `docs/fv.md`): U by
BiCGStab + multicolour GS, p by flexible AMG-PCG (scaled-V, 2 Jacobi sweeps),
FP64 (`--fv-precision fp64`) or FP32 (its default, "Precision" below), stopped on OpenFOAM's normalised residual (`KrylovOptions::
normFactor`, below). NK and every structured path are untouched.

## Files

| file | content |
|---|---|
| `dns/la/ell.{hpp,cpp}` | `BlockShape`, `supportedBlockSizes`, `EllPattern` (sliced ELL), host FP64 `BlockMatrix` |
| `dns/la/lines.{hpp,cpp}` | `LineSet`, `lineSet` (validated chains), `strongLines` (algebraic) |
| `dns/la/structured.{hpp,cpp}` | `structuredGraph` (7/5-point, periodic per axis), `gridLines` |
| `gpu/la/blockell.{hpp,hip}` | `BlockEll<T>`: upload, `spmv`, `residual`, `spmvMulti` (≤ 3 B = 1 systems sharing the off-diagonals), raw `EllView` for assembly |
| `gpu/la/smooth.{hpp,hip}` | `BlockJacobi<T>`, `MultiColourGS<T>` (private colour-major ELL; `factorMulti` / `applyMulti`), `LineSmoother<T>`; `factorDeferred` on all three |
| `gpu/la/krylov.{hpp,hip}` | `Krylov<T>`: `fgmres`, `pcg`, `bicgstab`, `bicgstabBatch`, batched `dots`, `KrylovStart`, `preFlag` |
| `dns/la/amg.{hpp,cpp}` | AMG host setup: `AmgSetup`, `amgHierarchy` (pairwise aggregation, coarse graphs, gather maps), `amgStrength`, `amgGalerkin` (host restatement of the refresh) |
| `gpu/la/amg.{hpp,hip}` | `Amg<T>`: device hierarchy, `refresh` (coarse values), `apply` (one cycle), dense coarsest inverse |
| `gpu/la/internal/ellops.hpp` | shape dispatch, logical block of an entry, FP64 Gauss–Jordan, the (prefetched) ELL row walk `ellRow` / `spmvRow` / `ellRowMulti` |
| `gpu/la/internal/dots.hpp` | the batched fixed-order FP64 dot kernels (Krylov and the AMG inner steps) |

`T ∈ {float, double}` for matrix values AND vectors (one type per object).
Dots are FP64 always.

## Block shapes

`BlockShape{B, Bd, nS}`: the diagonal block is dense B × B; an off-diagonal
block is a dense Bd × Bd top-left part plus, on rows Bd..B−1, a diagonal of nS
scalars (nS = 1 shared by all such rows, or one per row). Compiled:

| B | Bd | customer |
|---|---|---|
| 1 | 1 | pressure (P5b), scalar transport |
| 3 | 3 | coupled momentum (P5b) |
| 5 | 5 | mean flow (P6/P7) |
| 6, 7 | 6/7 or 5 | + SA / SST scalars (shared or per-row scalar off-diagonals) |
| 9, 11 | 9/11 or 5 | + 4 / 6 species (masterplan "5×5 + shared scalar") |

Any other (B, Bd) is an error by name (`blockShape`); chemistry with
5 + nT + nY outside the list must add one `case` to `internal::withShape` and
to `blockShape`'s list.

## Storage: sliced ELL, component-major

Row c sits in slice c/64 at lane c%64; slice s stores `sliceWidth[s]` entries
per row (its widest row), entry k of row c at `E = sliceBase[s] + 64k + lane`,
columns ascending, padding `col = −1` after the real entries:

    diag[(r B + q) n + c]     off[(r Bd + q) nE + E]     sc[e nE + E]     col[E]

Vectors are flat, component-major `v[q n + c]`. A slice's 64 rows read
coalesced; a mesh of mixed degree pays padding only up to its slice's widest
row. `EllPattern::slotPos[slot]` maps a `CellGraph` slot to its E, so a device
assembly kernel scatters face blocks to `ownSlot`/`neiSlot` in face order (no
atomics) straight into `BlockEll::off()`.

## Smoothers

- **BlockJacobi**: per-row dense inverse by FP64 Gauss–Jordan with partial
  pivoting, stored in T. `factor()` errors on a zero/non-finite pivot or a
  non-finite inverse in T (a user-causable condition: `std::expected`).
  `smooth` is one FUSED launch per step (`kJacobiSweep`: kSpmv's residual row
  and kDiagApply's update, ping-pong between x and `work`; an odd count from x
  runs its first step as the old in-place two launches); `smoothFromZero` =
  a zeroed x then `smooth`, bit for bit, without the memset (`kJacobiZero`:
  `b − A·0 == b` exactly, the update keeps the literal `T(0) + ω s`).
- **MultiColourGS**: distance-1 colours (`la::greedyColour`), validated at
  `make` (a coupled same-colour pair is an error naming both rows). One kernel
  per colour, `x_c = (1−ω)x_c + ω D_c⁻¹(b_c − Σ_{j≠c} A_cj x_j)`. `apply` = one
  symmetric sweep from 0 (a symmetric M for symmetric A: PCG-safe). The
  smoother keeps a PRIVATE colour-major copy of A's off-diagonal ELL (a
  colour's rows contiguous, ascending cell order inside it, each row's entries
  in A's column order) and its permuted D⁻¹; vectors stay in cell order, A is
  untouched, so any cell order runs coalesced and the bits are the cell-order
  walk's. `factor(offChanged)` re-gathers the off values only when they
  changed (the FV U components share them). At ω = 1 a colour pass that
  directly repeats the previous one (the turn of a symmetric sweep, the seam
  between two) is skipped: its inputs are unchanged, so it would rewrite the
  same bits (`apply`: 2K−1 launches, not 2K). `factorMulti` / `applyMulti`:
  up to 3 B = 1 systems sharing the off-diagonals with their own diagonals,
  each bit for bit `factor` / `apply` of it, one launch per colour for all.
- **LineSmoother**: lines from a `LineSet` (every row exactly once, singletons
  allowed), sorted by length (descending, stable) in groups of 64 padded to
  their longest; one thread per line. Factor: block Thomas in FP64,
  `T_p = D_p − L_p C_{p−1}`, stored `T_p⁻¹`, `T_p⁻¹L_p`, `C_p = T_p⁻¹U_p`
  (3 B² per position, coalesced across lanes). `apply` = M⁻¹ r for M = the
  line tridiagonal; `smooth` = line-Jacobi, off-line couplings lagged.
- **Line sets**: `gridLines` (structured axis lines), `lineSet` (any chains,
  validated by name: out of range, a row twice, empty, non-adjacent), and
  `strongLines` (algebraic: per-slot coupling weights, a row is anisotropic if
  max ≥ ratio × min, links strong both ways; seeds in descending anisotropy,
  grown both ways along the strongest link). On a 12 × 9 grid with y-weights
  100× x it reproduces `gridLines(axis y)` exactly; isotropic → all singletons;
  `cyl2d_snappy` with |S|/|d| at ratio 4: 32 multi-row lines, max 3 (its 3
  prism layers). Seeding from wall faces (masterplan) is P7's, when wall
  patches reach the solver.

## Krylov

- **FGMRES(m)** (m ≤ 64), right preconditioned and flexible (Z stored), CGS2:
  pass 1 batches (V_i, w) for i ≤ j (one sync), pass 2 batches (V_i, w) AND
  (w, w) (one sync) and takes `h_{j+1,j}² = |w|² − Σ h2_i²`, re-measured
  directly when that cancels below 1% of |w|². Givens on the host in FP64;
  `x += Z y` in one FP64-accumulating launch; the true residual at restart.
- **PCG**: ONE sync per iteration: (p, Ap) is reduced into device memory,
  α = (r, z)/(p, Ap) is formed by a one-thread kernel (`kAlpha`, the host's
  division) and read by the x / r updates (`kLinDev` = `kLin` with b loaded,
  same expression); (p, Ap) itself rides the next (r, z), (r, r) batch, where
  the host takes the old branch: non-positive (p, Ap) sets `breakdown` (A or
  M not SPD) — the device updates were skipped (a `bad` flag), so x and r
  are where the two-sync loop left them. With `minIter` 0 and
  `KrylovOptions::r0First` (default) r0 is read BEFORE the first M apply: an
  already-converged solve costs no preconditioner (the FV SIMPLE tail) for one
  extra sync otherwise; `r0First = false` keeps M first. Same results either
  way (`gpu_la_ref`).
  `KrylovOptions::flexible` = flexible CG (Polak–Ribière
  β = (r', z' − z)/(r, z), the third dot rides the same batch, z alternates
  between two buffers) for a VARIABLE preconditioner — the AMG K-cycle.
  `flexible = false` is the classic loop, byte for byte.
- **BiCGStab**, right preconditioned: two syncs per iteration (α = ρ/(r̂, v)
  on the device, (r̂, v) read with the (t, s) batch, as PCG's); half-step
  convergence on |s|. **Shadow-residual restart** when |(r̂, r)| ≤ 2⁻⁵² |r̂||r|
  (below the FP64 dot's own rounding). Measured: on the Pe 1e3 cell with
  lines, ρ fell to −7.8e-16 then exactly 0 at iteration 254 (a Lanczos
  breakdown); with the restart it converges (one restart, 890 iterations). A
  1e-12 threshold restarted a converging B3 run 27 times and stalled it.
- **Stop**: |r| ≤ max(relTol |r0|, absTol) on the 2-norm — OpenFOAM's
  `relTol` / `tolerance` split. `KrylovOptions::normFactor > 0` (PCG,
  BiCGStab) replaces the 2-norm by OpenFOAM's own measure Σ|r_i| /
  normFactor (the caller computes `lduMatrix::solver::normFactor` from the
  initial guess; the L1 sum rides the same dot batch: a pair with b = null
  is Σ|a_i|), so r0, residual and the stop are OpenFOAM's; BiCGStab keeps
  its 2-norms for the shadow-restart test. `minIter` = OpenFOAM's. Both
  default off: the 2-norm loop byte for byte. `KrylovResult` reports iterations, r0, the
  recursive |r| and |r|/|r0|, converged, breakdown, BiCGStab restarts.
- **Device-side start** (`KrylovStart`): `Ax` = A x0 computed by the caller
  (the solver skips its own, bitwise the same product) and `normTerms`
  (OpenFOAM's per-row normFactor terms): normFactor = Σ normTerms + 1e-20 is
  reduced in the solver's FIRST batch. With it the FV solve's setup costs no
  sync of its own (it had two).
- **`preFlag(j)`**: a device int a caller's kernel may set (a smoother's
  `factorDeferred` flag) before the next solve; the solve reads it with its
  first batch, stops with `KrylovResult::flagged` (x untouched) and clears it.
- **`bicgstabBatch(nb ≤ 3, mask, …)`**: nb independent systems (system j at
  offset jN) solved EXACTLY as nb `bicgstab` calls, in lockstep: A and M once
  per step for all, every batch carries each active system's pairs (a pair's
  result does not depend on the batch: partials fixed by N), per-system
  scalars, branches, restarts and stop; a stopped system is masked out of
  every later update. Per system its result or its error text. The FV U
  components: 3 solves → 1, syncs 3 × (1 + 2 it) → 1 + 2 it.
- **Batched dots**: `kDotPartial` (grid P × K, each block a fixed strided loop
  and a fixed 256-wide shared-memory tree — no shuffles, wave32 = wave64) then
  `kDotFinal` (one block per dot, fixed tree), one host copy for the batch. P
  depends on N only → byte-identical reruns on any device. Products are exact
  in FP64 for float inputs.

### Precision: an FP32 iterate has a floor

With T = float the TRUE residual of the solved operator stops near
`ε_f32 ‖A‖‖x‖/‖b‖` while the recursive estimate keeps falling (measured,
relTol 1e-5): Poisson 128² PCG true 1.3–1.6e-5 vs estimate 9e-6; BiCGStab on
the Pe 1e3 cell true 8e-5–2.4e-4 vs 6e-6. T = double tracks to 1e-15. So a
solve that must match OpenFOAM (FP64) to solver tolerance uses `T = double`
for the operator and the iterate (the FV path's parity mode,
`--fv-precision fp64`), or FP64 defect correction around an FP32 inner
solve; FP32 is right for a preconditioner-grade inner solve (P7's Newton:
forcing 0.1) and for a solve stopped at a floor ABOVE the arithmetic's noise.

The FV path's default is that last case (`--fv-precision fp32`, docs/fv.md
"Precision"): `T = float` for its matrices, Krylov vectors and AMG, OpenFOAM's
normFactor measure, each solve's absolute tolerance max(fvSolution, floor),
floors measured on the SIMPLE tail's own noise (U, p 1e-6; turbulence 1e-7).
Library-level evidence (`gpu_amg_fp32`, the FV p configuration — scaled V,
2 damped Jacobi sweeps, PCG — on 512² Dirichlet Poisson, OpenFOAM measure at
x0 = 0): at absTol 1e-6 float and double both take 47 iterations, the float
solution is 2.1e-6 of scale off the double one, two float solves are
byte-identical; asking float for 1e-7 / 1e-8 / 1e-10 it still converges (55
/ 64 / 81 iterations, estimate 7.5e-11) while the TRUE normalised residual
stays 1.16–1.17e-5 = 1.28 × ε_f32 Σ|A||x|/Σ|b|. An FP32 solve does not stall
below its floor — it certifies noise; the FV path re-measures r0 from the
state every SIMPLE iteration, which is what sees the floor.

## AMG (P5)

**Setup (host, once per mesh; `dns/la/amg`).** Input: the fine `CellGraph`,
one strength weight ≥ 0 per slot, B. `amgStrength(m)` gives
`max(−a_ij, 0)` for B = 1 (Notay: negative couplings) and the Frobenius norm
of the logical off block for B > 1; the masterplan's physical weights
(`|S|/d`, `|ṁ| + ρν|S|/d + …`) plug in the same way. A level is `passes` (2)
pairwise passes (aggregates ≤ 4 rows):

- `S_i = {j : w_ij ≥ β max_k w_ik > 0}`, β = 0.25.
- Notay's ORDER: the next row is an unassigned one with the fewest unassigned
  rows j having it in `S_j` (a bucket queue; most recently filed first); it
  takes its strongest unassigned strong neighbour (ties: the lower column) or
  stays a singleton. Natural row order instead stalled on the Morton
  `cyl2d_snappy` (1084 → 336 → 128 → 64 → 47, ratio 1.36); with the priority
  order every test graph coarsens ~4× per level, Morton or source order.
- Pass-2 weights are the pass-1 weights summed between pairs.
- Stop at `rows·B ≤ coarseUnknowns` (128), a stalled ratio (< 1.2), or
  `maxLevels`; a coarsest level above `kAmgMaxCoarseUnknowns` (1024) is an
  error by name ("coarsening stalled … rows without strong couplings?").
- The transfer of a level: `agg`, members (CSR, ascending), the fine slots
  INSIDE each aggregate (they land in the coarse diagonal block), and per coarse
  slot the fine slots it sums. Every list ascending. Per level: greedy colours
  (empty when a row exceeds 254 entries — MC-GS is then refused by name), and
  `strongLines` when `lineRatio > 1`.

**Device (`gpu/la/amg`).** P is piecewise constant, so `P^T A P` is a fixed
sum: `refresh()` gathers each coarse diagonal block (member diagonal blocks
ascending, then the intra-aggregate off blocks) and each coarse off entry (its
fine entries ascending) in FP64 and rounds once to T — no SpGEMM, no atomics.
The device matches `amgGalerkin` (the host restatement, same order) BIT FOR BIT
on every level, every compiled shape, float and double (`gpu_amg_refresh`).
`refresh` then refactors every level's smoother and inverts the coarsest matrix
in FP64 (Gauss–Jordan, partial pivoting, one 1024-thread block; a pivot below
1e-14 max|diag| is an error by name). Every factor flag and the coarsest's
land in one device array read with ONE download (was one per level); the
lowest failing level's error first, as before. At ≤ 128 unknowns
(`coarseUnknowns`' default) the inverse runs `kDenseInvertReg`: [M | X]
resident in registers (thread t: column t/4, rows t%4 + 4i), pivot column and
scaled pivot row through LDS, every element the same operations in the same
order (the pivot rule is a total order, so the reduction shape cannot change
it) → the same Xt bits; 0.80 → 0.51 ms per refresh at m = 114 (rotor2 L1).
It is FP64-throughput bound on one workgroup (≈ 1.5 m³ FMAs); a 4-barrier
variant was SLOWER (0.65–0.68 ms) and is not used. `apply(r, z)` = one cycle from z = 0:

- pre-smooth, residual, restrict (members summed in FP64), coarse correction,
  prolong-add, post-smooth; pre = post (GS: symmetric sweeps), so the V-cycle is
  a symmetric preconditioner. A Jacobi level starts with `smoothFromZero` (no
  memset); with 2 sweeps a level visit is 7 launches (zero-start, 3 fused
  sweeps, residual, restrict, prolongation), was 12 (memset, 4 + 4 sweep
  launches, residual, restrict, prolongation); the
  scaled-V line search's `x = coef c` rides the caller's prolongation
  (`kProlongAddScaled`: the product rounded to T, then the add — two
  roundings as before, contraction off / `__dmul_rn`).
- coarse correction on level k: `inner` = 0 plain recursion (V); 1 one line
  search `x = ((c, b)/(c, Ac)) c` (scaled V = OpenFOAM GAMG's
  `scaleCorrection`); 2 the K-cycle (Notay 2008: two recursions combined by a
  2-step flexible CG, ρ₂ ≤ 0 falls back to the line search). Its dots and
  coefficients stay ON the device (batched dots into device memory, one-thread
  coefficient kernels, vector kernels that read them): no host sync inside a
  cycle. The outer PCG must then be flexible.
- `inner` applies on coarse levels 1 and 2 always and deeper only at
  ≥ `innerMinRows` (10 000) rows (below it the visit is launch cost, not work;
  the V-cycled tail is a bounded depth, so h-independence survives).
- smoothers: `smoother` on level 0 (MC-GS default), `coarseSmoother` on levels
  ≥ 1 (Jacobi default, ω 0.67), lines on any level from the hierarchy's line
  sets (line-Jacobi, damped by ω).
- `constantNullSpace` (pure Neumann / periodic pressure): `apply` projects the
  per-component mean out of r and z (one batched dot each), and the coarsest
  inverse is of `A + α 1 1ᵀ` (α = mean diagonal / n). Without the projection
  FP64 PCG broke down at relTol 1e-12 on the periodic cut-cell operator (est
  12.9 at iteration 59, measured).

**Why these defaults (measured, `test_gpu_amg tune`, double, relTol 1e-8):**
coarse levels are LAUNCH-bound. A K-cycle on every level doubles the visits of
every level below it (2⁶ coarsest visits at 1024²), and an MC-GS sweep is two
launches per colour on aggregated graphs with many colours: the full K-cycle
with MC-GS everywhere took 22 ms per iteration at 1024² (13 it, 287 ms) vs
1.7 ms for a V-cycle (110 it). Jacobi on coarse levels plus the 10 000-row K
threshold: 1024² 15 it, 45.6 ms; 96³ 14 it, 46 ms; rotor2_L1 26 it, 27 ms.
`coarseUnknowns` 1024 made the single-block dense inverse 218 ms per refresh at
96³ (256: 9.6 ms, 128: 1.4 ms), hence 128.

## Performance kernels (`plans/masterplan_performance.md`, report `plans/verify/impl_performance.md`)

Contract: every rewritten kernel keeps each output element's FP expression,
operand order and literal form (`T(0) + ω s`, `s = T(0); s += …`) verbatim,
so hipcc's contraction (`-ffp-contract=fast-honor-pragmas`) makes the same
choices; only kernel boundaries, load scheduling and sync placement move.
Acceptance = BYTE identity: the pre-change kernels live on verbatim in
`tests/test_gpu_la_ref.hip` (`gpu_la_ref`, memcmp, float + double, every
compiled shape, structured + polyhedral graphs: spmv / residual, Jacobi
smooth / smoothFromZero 1–3 steps with ±0 right-hand sides, MC-GS sweep /
smooth / apply / factor(false), the AMG cycle V / scaled-V / K with coarsest
sizes 1–200 incl. pivot ties and the null space, pcg / bicgstab with and
without a start, r0First, maxIter, breakdown, a set pre-flag,
`bicgstabBatch` vs one `bicgstab` per system), and the FV byte gate
(`tools/fv/bit_perf.sh`: 14 cases vs the pre-change binary).

- **Prefetched ELL row walk** (`internal::ellRow`, B = 1): 8 column loads,
  then the predicated value and x loads, then the FMA chain in ascending entry
  order; the first padding entry still ends the row; every load stays inside
  the slice. B ≥ 3 keeps the one-entry loop. Average per call on rotor2 L1
  (rocprof, all levels): kSpmv 8.0 → 4.8 µs, the MC-GS colour pass 7.6 → 5.0
  µs (with the private colour-major storage).
- **Deferred failure flags**: `factorDeferred(flag)` (all three smoothers)
  launches without the host read; the flag is read with the next batch the
  result must pass anyway (AMG refresh: one download; FV U / RAS scalars:
  `Krylov::preFlag`). The text and exit code are unchanged; a failed factor's
  numbers never reach a field.
- **Launches / syncs** (rocprof, rotor2 L1, iterations 11–40 per SIMPLE
  iteration): 1470 dispatches (incl. 96 memset fills, 71 copy kernels) and 71
  `hipMemcpy` (synchronous) → 905 dispatches (3.5 fills, 25 copies) and 21
  `hipMemcpy`. D2D copies and memsets in the hot paths are stream-ordered
  `hipMemcpyAsync` / `hipMemsetAsync` on stream 0.

## Gates (P3; gfx1100, HIP, through simjob)

**LA1** (`gpu_la1`): device vs the host FP64 restatement of the SAME
(T-rounded) operator (`tests/la_test_util.hpp`), 7 graphs × 12 shapes (all of
the table above + B 11 per-row scalars) × {float, double}: structured 2D
5-point 37 × 23, 3D 7-point 13 × 9 × 7, 3D cyclic 8 × 6 × 5, 2D cyclic 2 × 40
(neighbour-unique wrap), `cyl2d_snappy` (Morton and ColourMajor), `box_refined`;
lines = grid lines on the structured graphs, deterministic 12-row walks on the
polyhedral ones. Worst over all cases, relative to the row scale Σ|A||x|
(Jacobi/GS: |D⁻¹|(|b| + Σ|A||x|); lines: the line's max |z|):

| | spmv | residual | Jacobi | MC-GS sweep | lines |
|---|---|---|---|---|---|
| float (gate 2e-6; lines 1e-6) | 6.10e-7 | 5.76e-7 | 3.50e-7 | 3.77e-7 | 5.22e-7 |
| double (gate 1e-12) | 4.93e-16 | 4.73e-16 | 1.01e-15 | 7.35e-16 | 1.16e-15 |

Lines with B ≤ 7 against `dns::solveBlockBand` (K = 1), B 9/11 against a dense
solve of the line (host `test_la`: band == dense to 0 at B 1/3/5/7). Batched
dots, 66 × 100003: 0 relative to Σ|ab| (float and double), byte-identical
rerun.

**LA2** (`gpu_la2`): |true − estimate| ≤ 1e-5 (units of |r0|) gated on every
double run, convergence where the method is expected to converge; every run
twice → byte-identical x and residual. Iterations to relTol 1e-8 (double):

| problem | method / preconditioner: iterations (gap) |
|---|---|
| conv-diff 8² Pe 1e3, full GMRES(64) | none: 64 (finite termination, 1.2e-15) |
| conv-diff 128² Pe 1e3 uniform flow, B1 | FGMRES(30): Jacobi 449, MC-SGS 276, lines 256; BiCGStab: 202 / 107 / 98 |
| conv-diff 128² Pe 1e3 recirculating cell, B1 | BiCGStab: Jacobi 1325, MC-SGS 651, lines 890 (1 restart); FGMRES(30) STAGNATES (3000 it: 2.3e-3 / 8.8e-4 / 6.6e-4; restart 60 the same) — gap gated only |
| Poisson 128² B1, PCG | none 385, Jacobi 385, MC-SGS 192, lines 298 |
| Poisson 32³ B1, PCG | MC-SGS 59 |
| uniform flow B3, skew (Coriolis-like) coupling | FGMRES MC-SGS 269, BiCGStab MC-SGS 110 |
| cell flow B3 skew coupling | BiCGStab MC-SGS 828 |
| anisotropic diffusion 64 × 256 a_y 1e3, B5 | FGMRES lines 40, BiCGStab lines 30; FGMRES MC-SGS 3000 it at 2.2e-5 (gap only) |

All double gaps ≤ 3e-15. A random NON-skew block coupling on a weakly dominant
operator is indefinite (the first attempt: FGMRES stalled at 0.2, BiCGStab
broke down) — test operators must be well posed.

**LA-perf** (`gpu_la_perf`, `simjob --bench`):

| kernel | measured | gate |
|---|---|---|
| SpMV B5 float, 1M cells (100³ periodic) | **1.163 ns/cell** | ≤ 1.3 (A3: 1.12) |
| batched lines 8192 × 128 B5 float, apply | **0.637 ms** (0.61 ns/cell) | ≤ 0.8 (A3: 0.65) |
| lines factor (FP64 Thomas) | 4.10 ms | — |
| SpMV B1 / B3 float | 0.062 / 0.434 ns/cell | — (B1 fits the 96 MB Infinity Cache) |
| SpMV B1 / B3 double | 0.081 / 0.771 ns/cell | — |
| SpMV B7 / B11 (Bd 5, shared scalar) float | 1.367 / 1.821 ns/cell | — |
| MC-GS forward sweep B5, natural order (2 colours) | 1.86 SpMV | — |
| same, colour-major numbering | **1.09 SpMV** | masterplan "~1 SpMV" |
| batched dots, 31 distinct × 5.24M float | 1.18 ms incl. host copy (551 GB/s) | — |

**Shim**: `DNS_WITH_CUDA=ON` (CUDA 13.3, sm_86, host g++-15, no NVIDIA
device): the three `gpu/la` TUs, `test_gpu_la` and `test_la` compile and link;
no new shim name (the library uses only `hipMalloc/Free/Memcpy/Memset/
GetLastError/GetErrorString/DeviceSynchronize`). `test_la` (CUDA build) passes;
`test_gpu_la` skips (77). Linking needed one pre-existing fix outside gpu/la:
`sstGamma1/2` and `sstCdFloor` (`dns/rans.hpp`) are plain `constexpr` and nvcc
refuses them in a kernel (`gpu/rans.hip:127`); they are now `DNS_HD` (HIP
codegen unchanged: stripped `dns` byte-identical).

## Gates (P5; gfx1100, HIP, through simjob)

**Refresh** (`gpu_amg_refresh`): device coarse values vs `amgGalerkin` of the
level above, 5 graphs (2D 37 × 23, 3D 13 × 9 × 7, 3D cyclic 8 × 6 × 5,
`cyl2d_snappy`, `box_refined`) × 12 shapes × {float, double}: worst
|device − host| = **0** (bit for bit), 4–6 levels each. Dense coarsest vs
`A z = r`: double 9.4e-16 (B1) / 1.7e-15 (B3) / 2.7e-15 (B5), float 6–8e-8,
null space 3.6e-15 / 3.2e-7; a zero row is rejected by name; reruns and two
independent setups + cycles byte-identical.

**LA3** (`gpu_amg_la3`, double, relTol 1e-8, random b, FCG + default AMG):

| problem | iterations |
|---|---|
| Poisson 64² / 256² / 1024² (Dirichlet) | **12 / 16 / 15** (growth ×1.33, ×0.94; gate ≤ 1.5) |
| same, V-cycle / scaled V | 29 / 71 / 148 and 23 / 51 / 82 (not h-independent) |
| same, K-cycle with Jacobi on level 0 too | 14 / 19 / 18 |
| 3D 64³ isotropic | 15 |
| 64³ z aspect 100 uniform: default / lines (ω 0.67) / lines ω 1 | 19 / **6** / 4 |
| 64³ z aspect 100 graded (h/100 at k = 0 → h): default / lines / lines ω 1 | 19 / **21** / 35 (gate ≤ 2 × 15) |
| 64³ iso with lines (all singletons = damped Jacobi) / ω 1 | 17 / 117 |

Aggregation semi-coarsens along the strong axis by itself (aspect 100: default
19 vs iso 15); lines help the uniform case only. FP32 hierarchy and iterate at
1024²: 10 it, estimate 3.6e-6, true 2.8e-5 (the FP32 floor).

Block B = 3 (skew Coriolis-like coupling, 128², Pe 1e3, relTol 1e-8):
uniform flow FGMRES/K 48 (GS coarse: 20), BiCGStab/V 20 (GS coarse: 12);
recirculating cell FGMRES/K 194 (GS coarse: 90), BiCGStab/V 85 (GS coarse: 38).
For convective operators use `coarseSmoother = MultiColourGS` (Jacobi is the
pressure default).

**LA4** (`gpu_amg_la4`): the cut-cell pressure operator (`gpu/cutpoisson`'s
`kApplyCut`, assembled on the 63 560 fluid cells of a periodic 256² box with a
r = 0.1 cylinder, sanitized; constant null space). FFT-PCG (FP32, tol 1e-6 of
|b|): **14** iterations; AMG-PCG at relTol 1e-6: **15** (double), 15 (float),
61 (V-cycle). p(FFT-PCG) vs p(AMG-PCG, relTol 1e-12): **rel L2 9.4e-8**, max
2.7e-7 of max|p| (gate 1e-4). The FFT result's own FP64 residual is 1.2e-4 (the
FP32 floor of an h⁻² operator), yet its p agrees to 1e-7.

**Setup** (`amg_setup`, host): 182³ = 6.03M-row 7-point graph, unit weights:
**1.83 s** (lines on: 2.01 s), 9 levels 6 028 568 → 1 507 145 → 376 841 →
94 314 → 23 592 → 5 901 → 1 476 → 370 → 93, grid complexity 1.333, operator
1.559; two setups identical (`AmgHierarchy ==`, every vector).

**Polyhedral meshes** (`gpu_amg_meshes` + local paths; OpenFOAM's
`Gauss linear corrected` Laplacian, γ = 1: internal `|S|/max(n·d, 0.05|d|)`,
outlet fixedValue `|S|/(n·d_b)`, b = V; `tests/la_test_util.hpp`
`foamLaplacian`), relTol 1e-8, default AMG, host setup / refresh / solve:

| mesh | cells | levels / op cx | it (1e-8) | it (1e-2) | host / refresh / solve (1e-8) |
|---|---|---|---|---|---|
| cyl2d_snappy | 1 084 | 3 / 1.38 | 16 | 6 | 0.2 ms / 0.4 / 9.7 |
| box_refined (pure Neumann) | 276 | 2 / 1.40 | 9 | 3 | — |
| rotor2_L1 | 114 834 | 6 / 1.54 | 18 | 7 | 44 ms / 2.3 / 30 |
| rotor2_L3 | 320 692 | 7 / 1.58 | 20 | 8 | 142 ms / 1.5 / 39 |
| B_F1s (prop) | 708 044 | 8 / 1.57 | 18 | 7 | 332 ms / 1.2 / 72 |
| mB | 831 578 | 8 / 1.55 | 18 | 7 | 380 ms / 1.8 / 81 |
| B_G1s (prop) | 988 762 | 8 / 1.59 | 18 | 7 | 473 ms / 2.3 / 90 |

**Against OpenFOAM v2606 GAMG on the SAME matrix** (`laplacianFoam`, steady,
DT 1, unit `scalarSemiImplicitSource`, outlet fixedValue 0, the rest
zeroGradient, `nNonOrthogonalCorrectors 0`, serial, x0 = 0; OpenFOAM's residual
is `Σ|b − Ax| / Σ|b|` at x0 = 0, so AMG-PCG's L1 history is read at the same
norm). The two tight solutions (OpenFOAM PCG+GAMG to 1e-14, AMG-PCG to 1e-13)
agree to **1.0e-14 / 1.1e-14 of max|T|** (rotor2_L1 / B_G1s): the matrix is
identical.

| mesh | OpenFOAM solver | its | final res | time (solve incl. agglomeration) | AMG-PCG its to the same L1 | AMG solve |
|---|---|---|---|---|---|---|
| rotor2_L1 | GAMG GaussSeidel relTol 0.01 (the prop setting) | 6 | 8.0e-3 | 55 ms | 7 | 15 ms (+2.3 refresh) |
| rotor2_L1 | GAMG relTol 1e-6 | 21 | 9.3e-7 | 130 ms | 14 | ≈ 24 ms |
| rotor2_L1 | PCG + GAMG (2 V-cycles each) relTol 1e-6 | 8 | 6.1e-7 | 116 ms | 15 | ≈ 25 ms |
| B_G1s | GAMG relTol 0.01 | 8 | 8.0e-3 | 701 ms | 6 | ≈ 36 ms (+2 refresh) |
| B_G1s | GAMG relTol 1e-6 | 26 | 8.4e-7 | 1689 ms | 14 | ≈ 70 ms |
| B_G1s | PCG + GAMG relTol 1e-6 | 13 | 4.8e-7 | 1713 ms | 14 | ≈ 70 ms |

The AMG host setup (0.04 / 0.47 s) is once per mesh; OpenFOAM's agglomeration
is inside its first solve (cached after). AMG times interpolate the measured
per-iteration cost (5.0 ms/it on B_G1s at 1e-8: 90 ms / 18).

## Not built (P5)

- The AMG in a runner (P5b's pressure is its first customer); FP32 AMG as the
  preconditioner of an FP64 Krylov (a type-mixing composition); hipGraph
  capture of a cycle (the cycle is launch-bound below ~1e4 rows: graph replay
  would remove ~3 µs per launch); a multi-block dense coarsest factorization
  (one block: 9.6 ms at 216 unknowns, 218 ms at ~1000).
- Device assembly kernels (P5b/P6/P7 write into `EllView` slots),
  coloured line-GS (lines are line-Jacobi), ILU (never, §7).
- FP64 defect correction around an FP32 solve (a composition the caller owns).
- `strongLines` seeded from wall faces (graph weights only).
- hipGraph capture; multi-GPU.
