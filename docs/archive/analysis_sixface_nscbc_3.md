# Six-face NSCBC/LODI hardening review (agent 3)

Scope: `DeviceCompNscbc` all-face kernel, its wiring in the two comp apps, the unit test, and interaction with the ghost wall / far module / quiet start. Files: `src/dns/gpu/comp_nscbc.{hpp,hip}`, `comp.{hpp,hip}`, `comp_flux.hpp`, `comp_far.hip`, `apps/comp_mesh_tunnel_gpu_main.hip`, `apps/comp_cylinder_tunnel_gpu_main.hip`, `tests/test_gpu_comp_nscbc.cpp`.

Sign conventions, disjoint enumeration (both-lateral-open case), LODI amplitude mapping, and nz==1 degeneration all check out (see "Verified correct"). Ranked issues:

## BUGS (correctness)

**B1 — coverage gap when `openZ && !openY` (latent) — comp_nscbc.hip:38-62.** Slab enumeration: `nAh=ny*nz; nBh=openY?(nx-2)*nz:0; nCh=openZ?(nx-2)*(ny-2):0`. The z-slab omits `j∈{0,ny-1}` on the assumption the y-slab owns those `k∈{0,nz-1}` strips (y-slab runs all k, line 54). That holds only when `openY` is true. With `openZ && !openY`, the cells `i∈[1,nx-2], j∈{0,ny-1}, k∈{0,nz-1}` (the four z-face edge lines along x) are enumerated by no thread — x-slab skips interior i, y-slab is empty, z-slab requires `j∈[1,ny-2]`. They keep the raw periodic `kCompRhs` (z-wrap-contaminated, no characteristic treatment). `make()` (comp_nscbc.hip:176-188) does not forbid `openZ && !openY`. Not hit today (mesh app forces `openY=true`; cyl ties `openZ` to `open` which also sets `openY`) — latent, but a silent wrong-answer trap with no guard. Fix (cheap): `if (c.openZ) c.openY = true;` in `make()`, or reject the combo. Proper fix: when `!openY`, span z-slab full `j∈[0,ny-1]` (`nCh=(nx-2)*ny`, drop the `1+` on j).

**B2 — lateral-wall / NSCBC seam wraps at the x-face edge columns — comp_nscbc.hip:80-87, 94-101.** In cyl `--nscbc` (no `--open`) the stage hook is `far->apply` with `sigMax=0`: sponge no-ops but `kFarWallY` still overwrites `j∈{0,ny-1}` with free-slip mirrors (comp_far.hip:122-126). NSCBC then treats x rows; at an x-face cell `j=0` the y axis is not onFace, so (a) the retained transverse y-flux `faceFlux(...,axis=1)` lo face at `j=0` reads wrap neighbor `j=ny-1` (line 96) — the opposite wall, not the mirror of `j=1`; (b) Yoo-Im `ct(...,d=1)` (line 132) takes the unguarded central branch (line 86) and also reads `j=ny-1`. So the x-boundary edge columns mix opposite-wall values into their transverse terms. Localized (2 rows per x face), small at freestream, grows in the wake edge. Fully-open mode is clean (every boundary axis is LODI; non-boundary axes provably interior; `ct` open-guard at 84-85 fires). Fix: wall hook writes a true ghost the NSCBC transverse stencil can read, or special-case the mirror in the x-edge transverse flux.

## RISKS (physics / numerics)

**R1 — FP32 one-sided pressure derivatives degrade with resolution & falling Mach — comp_nscbc.hip:108-113, 65-67.** All of `kNscbcCells` is `Real=float`. At M0.2 `pInf=ρu²/(γM²)≈17.86`; the physical boundary `dp/dx` per cell shrinks as the grid refines while the FP32 floor of `(p[c]-p[cn])` (both ≈17.86) is ~1e-6 abs → ~8e-5 in `dp/dx` at cyl h. Outgoing `L1/L5` SNR is only ~10:1 at 512² and degrades linearly in nx and in 1/M². The anchor `sig·(p-pI)` (line 138) subtracts two ~17.86 FP32 numbers, floor ~1e-6. Concrete mechanism feeding the documented Re2000 2048² blow-up (docs/nscbc.md:90-95); bites any M≤0.1 or finer run. Experiment: FP64 gauge-pressure `(p-pInf)` and normal derivatives in the boundary kernel only; re-run 2048² Re2000 strip and radial-pulse residual at M0.05.

**R2 — entropy/shear upwind switch chatters at u_n≈0 on lateral faces — comp_nscbc.hip:140-148.** `if (a==0 || s*un>0)` hard-switches L2/L3/L4 between one-sided interior differences and the `etaLat` relax with no dead-band. On a lateral face `u_n` oscillates through 0 (wake cores skimming); the branch flips per-cell and per-RK-stage, and the shear limits are discontinuous (`L3=un·∂v_t/∂n→0` vs finite `etaLat·c/L·(v_t−target)`). Small near freestream, not for a near-stationary vortex parked on the face. Only tested by a vortex that advects through (`lateralVortexResidual`, test:294), never parked. Fix: blend over a velocity band, `w=smoothstep(s·un/(εc))`, `L=w·Lout+(1-w)·Lin`, ε~1e-2.

**R3 — transverse faceFlux at boundary rows reads the wrap normal-gradient (viscous) — comp_nscbc.hip:94-101 + comp_flux.hpp:59-72.** Even in fully-open mode, a pure x-face cell's retained y/z `faceFlux` computes its viscous cross-derivative `du[c][x]` from `i±1` (comp_flux.hpp:67-71); at `i=0`, `i-1` wraps to `nx-1`. Poinsot-Lele drops the normal viscous term but it leaks back through the transverse face's cross-derivative — a systematic wrong-gradient on every boundary cell, not just edges. Small at high Re. Consider zeroing the wrap-direction `du[c][normal]` for boundary rows or one-siding it.

**R4 — inflow stiffness (etaIn=100) not in the CFL; RK-unstable on small domains — comp_nscbc.hip:118-121, comp.hip:356-384.** Relax rate `etaIn·c/Lx`; `stableDt` accounts only for acoustic CFL + viscosity. On the unit rigs `Lx=nx·0.01` is tiny → rate O(1e3), survives only because dt is tiny (rate·dt≈0.7 at nx=48). Nothing checks `etaIn·c·dt/Lx≲1`; a small production domain with etaIn=100 would be unstable with no diagnostic. Fold the relax rate into `stableDt` or clamp it in `make()`.

**R5 — mesh app has no absorbing-strip fallback; six-face open + stiff inflow unprotected against box modes — apps/comp_mesh_tunnel_gpu_main.hip:222-234.** The cyl app exposes `--nsstrip` (documented production hybrid for Re>~800, docs/nscbc.md:90-100); the mesh app wires pure six-face NSCBC with no strip and no walls, and it is the only 3D path. The documented failure (stiff inflow + reflecting faces pump box modes; pure NSCBC blew up Re2000 2048²) applies directly — any turbulent/LES/high-Re 3D mesh run has no mitigation knob. Add an optional sponge strip (reuse `DeviceCompFar` with small `ns`, `sigMax` as the cyl app does) before running past sphere-Re100.

**R6 — Lodato per-axis summation at edges/corners unquantified — comp_nscbc.hip:92-163.** Corner cells sum independent 1D LODI sharing one `c2,r` (line 67); a corner gets 2-3× the acoustic relaxation of a face cell. Documented as "Lodato-style" but never error-quantified in 3D; `lateralVortexResidual` (<20%) is the only edge probe and is a single 2D corner.

**R7 — quiet-start boost applied to boundary rows outside the characteristic treatment; one-step target lag — apps:497-501 / 309-314.** `boostX` adds `du` to all cells incl. NSCBC rows, after the step, while the in-step inflow relaxed toward the previous `uT` (`setFreestream` called after boost). Benign (boost is exact/silent) but the inflow target lags the frame by one step during ramp. Low priority.

## VALIDATION GAPS

- **V1 — long-window open-lateral shedding stats.** docs/nscbc.md:143-144 admits only a short-window cyl Re800 `--nscbc --open` smoke (cd 1.71±0.17). No saturated-window cd/cl/St with error bars, no Re60 long onset. This is the mesh app's production config and is unvalidated.
- **V2 — Re2000-class six-face open robustness.** Only walled/x-only NSCBC pushed to Re2000. Six-face open + no strip (mesh app) untested at high Re; R5 predicts box-mode pumping.
- **V3 — 3D mesh anchors.** Only sphere Re100 at 64³/96³ (cd 1.10/1.146). No open-BC grid convergence, no lift/St, no non-sphere body, no mass/momentum flux budget across open faces. `data/validation` anchors remain sponge/walled era.
- **V4 — no `openZ && !openY` unit test** → B1 invisible to CI. Add an OpenRig variant `openY=false, openZ=true` (3D): freestream fixed point + z-normal pulse exit.
- **V5 — lateral `betaTrans` unvalidated.** The β scan was x-outflow only; lateral faces reuse `betaT` with the `s`-flipped sign (line 139), never scanned. A wake core exiting a lateral face may pump the growing mode β was introduced to kill.
- **V6 — no low-Mach test.** All tests at M0.2; R1's FP32 SNR collapse needs an M0.05 pulse/vortex-residual check.
- **V7 — no parked-vortex chattering test (R2)** and no conservation/flux-budget test (the drift test checks only the uniform fixed point).
- **V8 — corner/edge closure (R6)** has no dedicated 3D unit test.

## Verified correct (no action)

- LODI amplitude→primitive map (comp_nscbc.hip:158-162) matches header/doc; conservative assembly (165-171) is the standard chain.
- Lo/hi outgoing-vs-incoming split is sign-correct: x-lo computes only L1 (u−c leaves), models L2..L5; x-hi/lateral compute the +axis acoustic that leaves, model the other; `dn` is `(larger−smaller)/h` on both sides (108-110) so +axis orientation is consistent.
- Pressure anchor restores p→pInf at both faces (dp/dt uses symmetric `L1+L5`).
- Disjoint enumeration with both lateral axes open: every ≥1-open-face cell enumerated exactly once, and the per-axis `onFace` loop sums all open-axis LODI contributions regardless of owning slab — overlap-free, verified case-by-case.
- `ct` open-guard (84-85) correctly one-sides the Yoo-Im transverse derivative at open-face edges in fully-open mode; only unguarded path is the wall axis (B2).
- nz==1: `openZ` forced false (179); a=2 falls to transverse branch where axis-2 `faceFlux`/`ct` cancel by wrap, w stays 0, kmz=0.
- `boostX` (comp.hip:203-221) is an exact Galilean change, no acoustic injection.

## Suggested priority order
1. B1 guard + V4 test (cheap; prevents silent wrong answers).
2. R5 strip knob for the mesh app + V1/V2 (blocks trusting any turbulent 3D open run).
3. R1 FP64 boundary pressure differencing + V6 (fixes the real high-res/low-M failure).
4. R2 dead-band blend + V7.
5. B2 / R3 seam cleanups.
6. R4 CFL guard, R6/V8 corner validation, V5 lateral-β scan.
