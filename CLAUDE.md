# dns — working notes

Full reference: **README.md** (CLI, solvers, BCs, measured anchors). Design history: `docs/archive/`.
Specs: `docs/nscbc.md`, `docs/validation.md`, `docs/viewer-format.md`, `docs/channel.md`, `docs/scene.md`.

## Ground rules

- ONE binary: `build/dns <flow|tgv|hit|channel|bench>`. The old 15 apps are gone. `dns --help` / `dns <sub> --help` are authoritative — read them, don't guess flags.
- `--dry-run` validates + prints the resolved case (and the DNS/LES label) without touching GPU or filesystem. Use it before every real run.
- Parsing is strict: unknown/missing/non-numeric flag = hard error. Exit codes: 0 ok, 1 CLI, 2 config, 3 input, 4 output, 5 no GPU, 6 resource, 7 numerical, 8 non-convergence, 77 skipped.
- `main` = Parse → Validate → Acquire → Run. Phases 1-2 touch neither filesystem nor GPU; phase 3 creates every output before the first `hipMalloc`.
- `std::expected` for anything a USER can cause; `assert` only for what a PROGRAMMER can cause (legitimate only if the validator makes it unreachable). Release is the build that runs — never assert-as-input-guard.
- Build: `build/`, CMake+ninja, Release, `DNS_WITH_HIP=ON`. GPU arch is AUTO-DETECTED (this machine is gfx1201, an RX 9070 XT); wrong arch = HIP `getStatFunc` assertion. Tests: ctest; GPU tests skip (rc 77) without a device. Always run tests with a timeout.
- Long runs: `run_in_background`, act on the completion notification. No Monitor/sleep/until polling loops.
- Artifacts when asked to publish: dark mode; videos colormapped with curl and |velocity|.

## Layering

`cli/config` (pure parse+validate; also `dns/scene`, `dns/remap`, `dns/sgs`) → `bodies/geometry` (sdf + cut sampling; also `dns/curvmesh`, `dns/framemap`) → `gpu/{tunnel,bodyforce,les,cutpoisson,cutflux,derived,comp_channel}` → `gpu/runner_{incomp,comp,channel}` → `output` (RunOutput, the one artefact sink). Preserve it; validated physics constants live at their definitions with the measured bracket recorded.

## Invariants you must not break

- **2D is `nz == 1`, not a code path.** Periodic wrap makes every z-stencil term cancel identically; w stays exactly 0. Never branch z-terms out. Only solvers special-case degenerate axes (hipfftPlan2d, DCT skip, open box faces). Only z may be degenerate; `hz = 1` is validator-enforced. Keep the `*2d` parity tests green.
- **Ghost view is diffusion-only.** `gpu/ghostfill.*` mirror values must never reach convection or leak into the state — measured energy injection. Ghost faces have zero aperture.
- **Tunnel lateral walls are free-slip/Neumann.** Do not reintroduce the Dirichlet `u=uInf` far field (clamped sheared flow, painted a wall vortex sheet).
- **Open/lateral BCs must leave normal velocity free.** Relaxing v→0 or u→uInf laterally is a wall in disguise. In the open tunnel, lateral faces are slaved once per stage BEFORE projection; the post-projection refresh is `refreshOnly=true` (re-slaving broke div u by ~150×).
- **NSCBC is an RHS overwrite** (`setRhsHook`), never a state overwrite — that is why it avoids the deleted `comp_bc`'s acoustic transients. Inflow relax must stay stiff (`etaIn 100`). Outflow needs the Yoo-Im transverse term (`--nscbc-beta`, default 0.5; 1.0 diverges).
- **Sponge is NEVER a default — anywhere, any physics, any mode.** `--far sponge` is an escape hatch / explicit fallback ONLY: it relaxes the freestream in a boundary band that eats the near field. The compressible default is `--far nscbc` (physical characteristic boundary). If a run needs sponge, the user asks for it by name; never reach for it silently. Same for the struct/runner defaults — keep them off sponge.
- **`--far nscbc` alone only to Re800**; beyond, add `--nscbc-strip ~ny/64` (validator enforces).
- `--pot` off, or Q=0 ∧ a=0, is bit-identical (test-pinned). `csLes == 0` is bit-for-bit molecular DNS.
- Deterministic iteration/order wherever output depends on it.

## DNS vs LES labeling

The core scheme's only dissipation is physical ν; it is DNS iff the grid resolves the body BL (~D/√Re). `--les Cs` makes the field FILTERED — statistics depend on grid AND Cs by construction, and body dissipation carries model error (`cdf` uses ν only). If a run needed `--les` to survive, label it LES everywhere (reports, artifacts, memory) and state Cs AND the model (`--sgs smag|sigma`). σ-model (`--sgs sigma`, Cσ ≈ 1.35 — NOT a Smagorinsky Cs): νt from `dns/sgs.hpp`, zero in laminar shear / any 2D state, the transition-capable choice; validator rejects it at nz == 1. 2D + `--les` is a regularized 2D Euler: qualitative only.

## Known gotchas

- Upstream "ripples" near bodies are trapped 6-12h dispersive standing waves, continuously forced by the SUB-CELL boundary layer — NOT staircase noise and NOT a start transient. Measured: `--cut-flux` ~20% (incomp only; comp variant diverges and is validator-blocked), `--penalize` ~15% (HALVES cdf — never for friction physics), `--ramp` quiet start ~0%. The only true cure is resolving the BL (D/√Re ≥ 3 cells); otherwise they are the honest footprint of under-resolution. Do not reach for filters/limiters — dissipation stays ν + explicit SGS.
- **Anisotropic cells break latent isotropy assumptions.** The ghost wall's deep-solid threshold (quiescent vs mirrored) is a physical DEPTH: it must follow the WIDEST local spacing, or the mirror band is under two cells thick along the wide axis, the interior stencil reads quiescent deep cells, and drag inverts (measured, `--focal` Re100 cylinder at fine-cell aspect 1.7: cd 1.49 → −7.6). Isotropic Cartesian grids hide this. Audit any other `min(hx,hy,hz)` used as "the" cell size before putting a body on a clustered mesh.
- **`--focal` cd/cl are NOT mis-scaled.** It looks like they must be: `sampleCut` evaluates the sdf at REFERENCE coordinates while the ghost wall classifies against MAPPED centres. Measured 2026-07-25: the reference body is smaller in physical space by the plateau slope `c·r` (~1/R) and the reference spacings are larger by the same factor, so the frontal area comes out as the solved body's true physical `D·span`, exactly, for every R. Weighting rows by their PHYSICAL extent — the obvious "fix" — makes the area low by that slope (2/3 at R=2, 2/5 at R=4). Pinned in `test_geometry`; the table is in `dns/geometry.hpp`. Published focal numbers stand.
- `gpu_cutcell2d` was red on gfx1201 on 2026-07-23 (nz=8 half, wMax ~1e-6). Re-measured GREEN at 6aaf912 on 2026-07-25, full 54-test sweep passing. Green is the baseline; a red there is a real regression, not a known-red.
- Never cite `comp_bc`-era numbers (module deleted; it regressed real flows).
- Re800 compressible anchors are start-condition-sensitive (quiet-start vs impulsive differ); they need regeneration.
- The three runners each have an unquarantined smoke test (`gpu_runners`): exit 0, finite coefficients, healthy projection, stats rows. No literature band — bands belong in the `val_*` tier. It covers `--far tunnel` WITH walls (the incompressible default), which nothing else exercises end-to-end.
- The nine literature-band `val_*` tests are quarantined behind `-DDNS_VALIDATION_BASELINE=ON` by decision, not failure. Re-measure before citing a band. `val_spindown` is the green canary.
- Blockage excess vs unbounded literature GROWS with Re — bands are repo-calibrated, don't "fix" them to raw Williamson. TGV convergence is oscillatory, not monotone.
- `--pitch`/`--yaw` are DEPRECATED and are NOT aircraft angles. Bake attitude with `build/orient.py pitch yaw roll` and leave both at 0. Viewer needs `camera.up.set(0,0,1)`.
- Ports 8000 and 8080 are taken on this machine.
