# Six-face NSCBC/LODI hardening review — analysis 1

Files reviewed: `src/dns/gpu/comp_nscbc.{hpp,hip}`, `comp.{hpp,hip}` (`rhs`/`step`/`setRhsHook`/`boostX`), `comp_flux.hpp`, both apps, `tests/test_gpu_comp_nscbc.cpp`, `docs/nscbc.md`.

**Verdict:** The LODI algebra (amplitudes, lo/hi orientation, primitive→conservative conversion, edge/corner summation) is correct. The slab enumeration is disjoint/complete **only under an unenforced assumption openZ ⇒ openY** — that is the one hard bug. Everything else is physics risk (entropy/shear chattering at grazing lateral flow, transverse-viscous wrap contamination, FP32 low-Mach anchor SNR, no mesh-app strip fallback) plus a large unexercised validation surface.

## LODI correctness (checked, PASS)
- `dn` (comp_nscbc.hip:108-110) yields the +axis-oriented derivative on both sides. Correct.
- Incoming/outgoing split per face correct: x-lo inflow only L1 leaves (:116); hi faces L5 computed / L1=Lin (:149-151); lo lateral L1 computed / L5=Lin (:152-154).
- Yoo–Im sign `s=hi?1:-1` on the ρc·Tun term (:135,139) matches incoming invariant p∓ρc·u_n. Correct.
- LODI→primitive (:158-162), primitive→conservative (:165-171), and edge/corner per-axis summation with transverse `faceFlux` kept only on non-open axes all match Poinsot–Lele. Correct.
- Dimensional check of modeled inflow/lateral relax amplitudes (:117-121,:145-147) consistent.

## Findings, ranked

**BUG 1 — enumeration drops z-face j-edges when openY=false, openZ=true.** `comp_nscbc.hip:37-39`: `nBh = openY?(nx-2)*nz:0`, `nCh = openZ?(nx-2)*(ny-2):0`. Slab C (z-faces) excludes j∈{0,ny-1} (:60-61), relying on slab B (y-faces, all k) to cover those j-edge strips. If `openY=false && openZ=true`, slab B is empty, so cells `(i∈[1,nx-2], j∈{0,ny-1}, k∈{0,nz-1})` — the four j-edge seams of the z-faces, 4·(nx−2) cells — are enumerated by **no slab**; their rhs keeps the pure-periodic wrap-contaminated `kCompRhs` value → a reflecting seam. `make()` (:176-188) permits this config (Config default openY=false). Not hit by the two apps (both set openY=true whenever openZ), but a latent bug and a violated contract. **Fix:** in `make()` add `if (c.openZ) c.openY = true;` (matches design assumption), or generalize slab C to span full y when openY is false. Add a unit test for `{openY=false,openZ=true}`.

**BUG 2 (minor/localized) — transverse viscous flux at x-faces reads the wrapped x-neighbor.** The retained transverse `faceFlux` (comp_nscbc.hip:94-102) computes tangential viscous derivatives `du[c][d]` reaching `i±1` (comp_flux.hpp:67-72). For an x-face cell (i=0/nx−1) the y/z transverse flux samples `i∓1`, wrapping to the opposite x boundary (inflow reads outflow-row gradients). Contradicts the "no cell reads the wrap face" claim for the boundary rows themselves. Effect is O(μ) and x-faces are freestream-ish, so small; lateral y/z faces are unaffected (their transverse axes are interior/periodic). **Fix:** one-side or drop the boundary-normal tangential-viscous contribution in the 1-cell boundary strip (consistent with already dropping normal viscous), or document + accept (vanishes with μ/grid).

**RISK 1 — entropy/shear upwind switch chatters at grazing lateral flow.** `comp_nscbc.hip:140-148`: lateral entropy/shear switch on `s*un>0`. Outgoing branch L2=un·(…) → 0 as un→0; incoming branch → nonzero relax `etaLat·c·c²/L·(ρ−ρ∞)`. So the amplitude is **discontinuous in un at un=0**, jump ~etaLat·c·(ρ−ρ∞)·c²/L. A lateral far-field face in a near-symmetric wake sits at u_n≈0 and dithers about 0 across stages/steps → L2/L3/L4 chatter, injecting boundary noise. Freestream is still a fixed point (both branches →0 at ρ=ρ∞), so `openFreestreamDrift` passes and hides it — it only bites with a live perturbed wake (the untested regime). Plausible contributor to the Re2000 lateral-open box-mode blowups. **Fix/experiment:** smooth blend w=½(1+tanh(un/(εc))) between the two forms, or for a true far field (u_n≈0) always use the gentle relax on lateral faces and reserve the computed-outgoing form for a==0 outflow. Add a grazing-flow chatter unit test.

**RISK 2 — FP32 low-Mach anchor SNR.** Fields are FP32. Anchor `sig·(p−pInf)` and outgoing `dp=(p[c]−p[cn])/h` subtract two ~pInf values; at M=0.2 pInf≈17.9, FP32 ulp ≈1e-6 → ~0.1% noise on p−pInf, worse as M drops. Same subtraction in `c2=gam·p/r` and Yoo–Im `Tp`. Caps how non-reflecting the boundary can be at low M (consistent with measured 2.4–2.9% floors). **Fix/experiment:** carry a boundary-local FP32 perturbation p′=p−pInf before differencing, or document the M-dependent floor.

**RISK 3 — mesh app has no strip/sponge/wall fallback; Re>800 3D will box-mode.** `comp_mesh_tunnel_gpu` is hard-wired to pure six-face NSCBC (grep: no `--nsstrip`, no sponge). Docs (nscbc.md:80-100, CLAUDE.md) establish pure NSCBC is good only ≤Re800 and needs the thin sponge strip beyond. The cylinder app exposes `--nsstrip`; the mesh app does not, so transitional/turbulent 3D mesh runs have zero mitigation. **Fix:** port `--nsstrip` (thin `DeviceCompFar` sigMax≈0.5 band on all open faces) into the mesh app, default on (~ny/64) for 3D; until then cap documented validity at ~Re800.

**RISK 4 (awareness) — quiet-start boost/target one-step lag.** `boostX` (comp.hip:214-221) boosts all cells incl. boundary rows, then `setFreestream` retargets (mesh app :497-501, cyl :311-313); NSCBC uses previous `uT`/`omM2` during the step. `omM2=1−M²` and auto-`betaT` recomputed from current `u_` each `apply` (:199,206), so early-ramp transverse cancellation is weak. Benign for smooth ramp; the interaction with the t=0 kick vortex already in the field is untested.

## Validation gaps
Unit test covers: freestream fixed point (2D/3D), 1D pulse reflection, 1D vortex outflow, inflow pull, radial pulse all-faces (nz=1), lateral vortex skimming one y-face (nz=1). Missing:
1. Edge/corner exit (two-/three-face) — the Lodato summation path is never exercised by a moving structure.
2. 3D lateral characteristic — `radialPulseResidual`/`lateralVortexResidual` are both nz=1; the openZ z-face acoustic/entropy path is only touched by a trivially-passing 3D freestream fixed point.
3. Grazing-flow chatter (RISK 1) — nothing probes u_n≈0.
4. openY=false/openZ=true (BUG 1) — permitted, untested.
5. Body+boundary coupling — no unit test has a ghost wall; wake→outflow (the whole point of Yoo–Im) is only in app smoke runs.
6. Long-window lateral-open shedding stats — docs (nscbc.md:143) admit unvalidated; need Re60/Re800 `--open` cd/St/clAmp saturated windows vs walled-NSCBC and sponge controls, grid-converged.
7. 3D mesh anchors — only sphere Re100 96³ short (cd 1.146, 2230 steps); no St, no spectrum, no long window, no second geometry, no six-face LES case; `data/validation/anchors.csv` has no six-face entry.
8. Re2000-class laterally open — untested and (RISK 3) unmitigable in the mesh app.

Proposed experiments: unit (3D corner-exit vortex; z-face pulse nz>1; grazing-flow chatter probe; openY=false/openZ=true after BUG 1); regression (cyl Re60/Re800 `--nscbc --open` long window @512²/1024² vs walled+sponge, as its own anchor truth class); mesh (sphere Re300 against `val_mesh_sphere`, second body long-window, add `--nsstrip` and repeat Re>800).

## Quick reference (file:line)
- Slab enumeration / BUG 1: `comp_nscbc.hip:34-62` (nBh/nCh :37-39).
- LODI amplitudes & lo/hi split: `comp_nscbc.hip:104-163`.
- Entropy/shear upwind switch / RISK 1: `comp_nscbc.hip:140-148`.
- Yoo–Im transverse + s-sign: `comp_nscbc.hip:129-139`.
- one-sided transverse `ct` edge guard: `comp_nscbc.hip:80-87`.
- retained transverse `faceFlux` wrap / BUG 2: `comp_nscbc.hip:94-102` + `comp_flux.hpp:67-72`.
- `make()` (no openZ⇒openY): `comp_nscbc.hip:176-188`.
- rhs-hook seam in RK3: `comp.hip:301-354` (hook :323, step :326).
- boost + retarget: `comp.hip:214-221`; apps `comp_mesh_tunnel_gpu_main.hip:497-501`, `comp_cylinder_tunnel_gpu_main.hip:311-313`.
- mesh app pure NSCBC, no strip / RISK 3: `comp_mesh_tunnel_gpu_main.hip:220-239`.
