# Masterplan — Six-face NSCBC/LODI hardening

Reviewed: `src/dns/gpu/comp_nscbc.{hpp,hip}`, `comp.{hpp,hip}`, `comp_flux.hpp`, `comp_far.hip`, both comp apps, `tests/test_gpu_comp_nscbc.cpp`. Every finding below was checked against the actual code; line numbers are verbatim from the current tree.

## Verified findings (ranked)

1. **CONFIRMED BUG — enumeration drops z-face j-edges when `openZ && !openY`.** `comp_nscbc.hip:38-39` (`nBh=openY?…:0`, `nCh=openZ?(nx-2)*(ny-2):0`) + z-slab `j=1+tt/(nx-2)` at `:60-61`. With `openZ=true, openY=false`, slab B is empty and slab C omits `j∈{0,ny-1}`, so cells `i∈[1,nx-2], j∈{0,ny-1}, k∈{0,nz-1}` (4·(nx−2) cells) are enumerated by no thread and keep the wrap-contaminated periodic `kCompRhs`. `make()` (`:176-188`) does not forbid the combo. All three agents agree. Not hit today (cyl ties `openZ` to `open`⇒`openY`; mesh forces `openY=true`), so **zero effect on any validated run** — but a silent wrong-answer trap with no guard.

2. **CONFIRMED RISK — lateral entropy/shear switch is discontinuous at `u_n=0`.** `comp_nscbc.hip:140-148`: `if (a==0 || s*un>0)` outgoing branch scales L2/L3/L4 by `un`(→0), incoming branch is finite `etaLat·c/L·(…)`; the RHS jumps as `u_n` crosses 0. On a lateral far-field face in a wake, `u_n` oscillates through 0 → per-stage/per-step chatter feeding box modes. All three agree. Freestream is still an exact fixed point (both branches →0 at ρ=ρ∞, tangential u=uInf=uT), so `openFreestreamDrift` masks it. **Affects only the `--open`/mesh six-face path (unvalidated); walled `--nscbc` uses `openY=false` so `a>0` never executes — Re60/Re800 anchors untouched.**

3. **CONFIRMED GAP — mesh app has no strip/wall/sponge fallback.** `comp_mesh_tunnel_gpu_main.hip:220-239` hard-wires pure six-face NSCBC (`openY=true, openZ=nz>1`), stage hook = ghost only, no `--nsstrip`. The cyl app has the knob (`:82`, `:110-113`: `sigMax=0.5`, `fc.ns=nsStrip`). Docs/CLAUDE.md establish pure NSCBC is good only ≤Re800 and needs the thin sponge strip beyond. The mesh app is the *only* 3D path and has zero mitigation. All three agree.

4. **CONFIRMED RISK (moderate) — FP32 low-Mach anchor SNR.** `comp_nscbc.hip:65-67,108-113,138`: `Real=float`; `dp=(p[c]−p[cn])` and `sig·(p−pI)` subtract two ≈pInf values (17.86 at M0.2), FP32 ulp ≈1.9e-6. **Agent 3 overstates this as ~10:1 SNR and the direct cause of the Re2000 2048² blow-up; downgrade to agent 1's read** — the cancellation floor is on the cell-to-cell *difference* (~0.05 for a resolved wave), giving SNR ~2.6e4 at M0.2 (consistent with the 5e-6 freestream-fixed-point test passing). It degrades ∝nx and ∝1/M² and becomes real at M≤0.1 and/or 2048², but the documented Re2000 blow-up is attributed to residual outflow reflection pumping box modes (a BC/physics issue), not FP. Real but not urgent; fix needs its own M0.05 gate. **Defer.**

5. **CONFIRMED LATENT, currently MASKED — transverse `ct`/`faceFlux` wrap at x-face edge columns when a lateral axis is walled.** `comp_nscbc.hip:80-87,94-101`: at a walled-y x-face cell (i=0/nx−1, j=0/ny−1), `onFace(1)=false`, so the retained y-`faceFlux` reads `j−1→ny−1` (opposite wall) and `ct(q.p,1)` takes the unguarded central branch. Verified harmless: `kFarWallY` (`comp_far.hip:62-77,122-126`) overwrites the entire `j∈{0,ny-1}` rows every stage via the state hook, which runs *before* `rhs` each stage (`comp.hip:345-353`), so those corner NSCBC rhs values are discarded before any flux reads them. Bites only if walls are removed / hook order changes / y made periodic. **Defer (defensive only).**

6. **REJECTED for now — agent 2's proposal to drop `a==0 ||` so x-outflow entropy/shear upwinds on backflow** (`comp_nscbc.hip:140`). This is exactly the validated x-only path; the `a==0` compute-branch is conventional NSCBC subsonic outflow, and the wake-core-crossing growing mode is *already* controlled by the tuned Yoo-Im `betaTrans=0.5` (`:129-139`, Config doc `:38-43`). Removing it would change the Re60 cd 1.813 / Re800 cd 1.776 anchors and re-introduce the `u_n=0` discontinuity (agent 2's own R2). Agents 1 and 3 do not propose it. **Do not touch x-outflow; defer to a separate β/backflow validation campaign if ever pursued.**

7. **CONFIRMED MINOR — LES eddy viscosity dropped from boundary transverse flux.** `comp_nscbc.hip:95-96` calls `faceFlux` with default `nut=nullptr`; interior `kCompRhs` threads `nut`/`kapScale` (`comp.hip:310-321`). For `--les Cs>0` mesh runs the boundary rows lose SGS transverse diffusion. Six-face LES is unvalidated and the boundary is near-freestream. **Defer.**

8. **CONFIRMED MINOR — inflow relax rate not in `stableDt`.** `comp_nscbc.hip:118-121` rate `etaIn·c/Lx`; `make()` (`:182`) only checks `etaIn>0`. Safe in production (rate·dt ≈0.06 on the 512² cyl) but unguarded on tiny domains. **Defer (document ceiling).**

9. **REJECTED as action — quiet-start boost/target one-step lag** (`comp.hip:214-221` + apps). Boost is an exact silent Galilean change; the one-step inflow-target lag during ramp is benign. No change.

## Implementation plan (NOW, dependency order)

Coding style: terse, `std::expected`, no exceptions, keep `nz==1` cancellation exact, narrow visibility. **Do not** modify the x-outflow branch, the `betaTrans` logic, `comp_far.*`, or the sponge default.

### Fix A — close the `openZ && !openY` enumeration hole (finding 1)
Generalize slab C to span full `j∈[0,ny-1]` when `openY` is false (slab B is empty there, so no overlap; when `openY` is true the formula is bit-identical to today, so both-open and all validated paths are unchanged).

- `comp_nscbc.hip:39` — replace
  `const std::size_t nCh = openZ ? std::size_t(g.nx - 2) * (g.ny - 2) : 0;`
  with
  `const std::size_t nCh = openZ ? std::size_t(g.nx - 2) * (openY ? g.ny - 2 : g.ny) : 0;`
- `comp_nscbc.hip:60-61` — z-slab index:
  `i = 1 + int(tt % (g.nx - 2));`
  `j = (openY ? 1 : 0) + int(tt / (g.nx - 2));`
- `comp_nscbc.hip:202-205` (`apply`, host-side `nb`) — mirror the same formula:
  `(c_.openZ ? std::size_t(g_.nx - 2) * (c_.openY ? g_.ny - 2 : g_.ny) : 0)`.

`kNscbcCells` already receives `openY`, so no signature change. Verify: when `openY==true`, `nCh` and `j` are unchanged ⇒ `openFreestreamDrift(48,40)` (3D both-open) and every current path bit-identical.

*(Alternative considered and rejected: `if (c.openZ) c.openY=true;` in `make()` — silently opens Y when only Z requested, wrong physics for a z-open/y-walled body. The slab generalization keeps the capability and is equally cheap.)*

### Fix B — continuous lateral entropy/shear blend (finding 2)
Only the lateral branch (`a>0`); leave `a==0` exactly as-is. Blend the outgoing (computed one-sided) and incoming (relax) forms of L2/L3/L4 with a smooth weight in `s*un`, so the branch no longer jumps at `u_n=0`. Both forms already vanish at freestream, so the fixed point (and `freestreamDrift`/`openFreestreamDrift`) is preserved exactly for any weight.

In `comp_nscbc.hip`, replace the `if (a==0 || s*un>0){…}else{…}` block (`:140-148`) with:
- keep the pure compute branch for `a==0` (unchanged);
- for `a>0`: compute both `Lout{2,3,4}` (the `un·…` forms) and `Lin{2,3,4}` (the `etaLat` relax forms), then
  `const Real w = Real(0.5)*(Real(1)+tanh(s*un/(kLatBand*cs)));`
  `L2 = w*Lout2 + (Real(1)-w)*Lin2;` (same for L3, L4),
  with a file-scope `constexpr Real kLatBand = Real(0.01);` (dead-band width as a fraction of `cs`; comment that it can be promoted to `Config` if a scan wants it).

Rationale for the form: `w→1` when strongly outgoing (`s·un≫0`), `w→0` when strongly incoming; near `u_n=0` the transition is C∞. Keep the existing `Lin` target expressions verbatim (tangential-x relaxes to `uT`, others to 0).

Verify: walled `--nscbc` (openY=false) never enters `a>0`, so cyl Re60/Re800 anchors are untouched. The two open-path unit tests (`radialPulseResidual<0.2`, `lateralVortexResidual<0.2`) may shift slightly; implementer must confirm they still pass (thresholds are loose).

### Fix C — port `--nsstrip` into the mesh app (finding 3)
Give the only 3D path the same production hybrid the cyl app has, **default off** so the validated sphere Re100 run is byte-for-byte unchanged.

In `comp_mesh_tunnel_gpu_main.hip`:
- parse `const int nsStrip` from `--nsstrip` (default 0) near the other flags.
- build a `gpu::DeviceCompFar` exactly as the cyl app does when `nsStrip>0`: `fc.openY=true; fc.openZ=nz>1; fc.ns=nsStrip; fc.sigMax=0.5;`. On `nsStrip==0`, do **not** construct/apply it (preserves current pure six-face behavior).
- when the strip is active, extend the stage hook to apply it and track the ramp target: `applyBc = [&](st){ if(far) far->apply(st); ghost->apply(st); };` and in the ramp block call `far->setFreestream(uNew)` alongside `nsc->setFreestream(uNew)` (boost stays the shared `gpu::boostX`). Order: sponge strip before ghost (matches cyl `applyBc`).
- update the `.json`/banner to record `nsstrip`.

Guard: with `nsStrip==0` no `DeviceCompFar` is created and no hook/behaviour changes — sphere Re100 96³ path is identical.

## New / extended unit tests (`tests/test_gpu_comp_nscbc.cpp`)

1. **`openZOnlyFreestreamDrift`** (gates Fix A). `OpenRig` variant with `openY=false, openZ=true` at `nz>1` (ctor tweak/bool param). Uniform freestream, 40 steps, drift `< 5e-6`. Without Fix A the j-edge z-face cells diverge from the wrap-contaminated rhs.
2. **`openZOnlyPulseExit`** (gates Fix A, stronger). Same rig, z-normal Gaussian acoustic pulse; residual `|p−pInf|/(amp·pInf) < 0.2` after ~1.6 z-crossing times (mirror `radialPulseResidual`). Exercises the newly-enumerated z-face edge lines.
3. **`lateralVortexAlongFace`** (gates Fix B). Vortex core ON the y-hi face (`y0 = Ly`), advecting in x with the freestream so its induced `u_n` sweeps through 0 along the face over many steps. Finite everywhere, `|v|/v0 < 0.2` at end. Probes the `u_n≈0` chatter the blend removes.
4. Keep all existing checks and thresholds unchanged; whole suite stays green.

## Deferred (need long GPU runs or separate campaigns — NOT in the implement step)

- **D1 FP32 gauge-pressure differencing** (finding 4): p′=p−pInf or FP64 boundary subtraction; gate with M0.05 tests; re-run 2048² Re2000; prove M0.2 anchors unchanged.
- **D2 wall-aware transverse stencil** (finding 5): `wallA[3]` mask, one-side `ct`/skip wrapped transverse face on walled axes; verify walled anchors bit-stable.
- **D3 LES `nut` into boundary transverse flux** (finding 7).
- **D4 inflow-stiffness CFL guard** (finding 8): fold `etaIn·c/L` into `stableDt` or clamp/document in `make()`.
- **D5 split `betaLat` from `betaTrans`** and scan on six-face `--open`.
- **D6 Long-window six-face validation campaign**: cyl `--open` Re60/Re800 saturated cd/cl/St vs walled-NSCBC + sponge controls, grid-converged; mesh Tier-A shedding anchor in `data/validation/anchors.csv`; Re>800 mesh with `--nsstrip`.
- **D7 x-outflow backflow experiment** (finding 6, rejected-for-now): only if D6 shows a residual growing mode β cannot kill; must regenerate Re60/Re800 anchors.

## Build / test commands for the implementer

```
cmake --build build -j
ctest --test-dir build -R gpu_comp_nscbc --output-on-failure --timeout 300
ctest --test-dir build -R 'gpu_comp$|gpu_comp_ghost|gpu_comp_far|gpu_comp_mesh|gpu_comp_les' --output-on-failure --timeout 600
```

Do not run multi-hour flows in the implement step. NEVER git commit or push.
