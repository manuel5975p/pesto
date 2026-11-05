# Masterplan: mesh-agnostic multiphysics

Goal (user, 2026-11-03): {any fluid model} x {any chemistry model} x {any
mesh / domain} x {any boundary condition}. Example: "LES with air-methane
chemistry on this domain, with those boundary conditions, on this mesh".

Axes, each independent of the others:
- fluid: compressible | lowmach | incompressible (Boussinesq T included)
- turbulence: DNS | LES (Smagorinsky, sigma) | TFLES (reacting) | RANS (SA, SST)
- chemistry: none | C0 premixed | C1 two-reactant | C2/C2b rich CH4/air
  (`--equivalence`, `--sl-law gri30`); incompressible takes reacting scalars
  without heat release (constant density) or refuses by name
- body force: none | gravity
- mesh: any FaceMesh (`--polymesh DIR` or the flags' mesh, O-grid, wedge later)
- boundaries: per patch, a kind from the fluid's table (`--bc FILE`); a kind
  that has no meaning for a module is refused by name, never ignored

## Composition, not pairs

The cross product is too large for one hand gate per pair. Every module
(turbulence model, chemistry model, body force, boundary kind) is a set of
ADDITIVE face/cell terms on the fluid family's one stage, and passes a
GENERIC gate set that the harness runs for every combination it composes with:
1. off = the stage without the module, byte for byte;
2. invariants: freestream / uniform state exact, uniform power-of-two Y exact,
   Z(Y) == carried tracer (two reactants), mass and energy telescope;
3. FvRef (FP64 oracle) of the composed stage <= 1e-13 per pass;
4. rotation / mesh-renumbering invariance (Source/Morton/ColourMajor bitwise);
5. where a structured counterpart exists: same mesh + same dt = rounding.
A combination is live when the harness matrix passes for it; the
docs/fvcomp.md "Combinations" table becomes the generated record of that matrix.
Physics-specific anchors (S_L, cd, Nu) stay as hand gates on top.

Case surface: one case description (mesh + bc table + module choices), the
subcommands become presets of it; the UI writes it (M6).

"Air-methane" in this code is model C2b: one-step F + s O -> (1+s) P with
S_L(phi) calibrated to GRI-3.0 (docs/chem.md). Multi-step chemistry and
variable cp stay out of scope (gamma 1.4 fixed is an invariant).

Rules carried from masterplan_meshsolver: one FaceMesh, no per-mesh fork,
structured hot kernels untouched, off = bit for bit, cross-verification table
in docs/fvcomp.md / docs/fvincomp.md is the record.

## Phases

| Phase | What | Oracle |
|---|---|---|
| M1 | Compressible FV species LES (eddy Fick/Fourier, Sc_t = Pr_t) + TFLES, premixed C0 | structured `kSpeciesStage<Eddy/Thick>` on the uniform periodic box: rounding (X1/X2 pattern) |
| M2 | Compressible FV `--equivalence` (C2/C2b) + two-reactant LES | structured C2 jet (laminar): rounding/order; LES: Z(Y) == carried tracer, FvRef |
| M3 | Low-Mach FV burner: per-face inflow rows, open faces, gravity, two-reactant/`--equivalence`, rim as Wall patch | structured `gpu/lowmach` burner (bunsen2d, rich2d, hydro, openrich) on the same mesh |
| M4 | Low-Mach LES (momentum mu_t + species eddy flux), structured and FV together | structured vs FV rounding; laminar limit bit for bit |
| M5 | Composition: species + gravity (compressible candle on FV), checkpoints on FV, turbulent inflow rows | per pair, its own gate |
| MC | Composition harness: modules as additive terms, generic gate matrix over every (fluid, turbulence, chemistry, force) cell, one case description; refusals become matrix rows | the generic gates above |
| M6 | dns-ui: case = mesh + bc table + physics/turbulence/chemistry pickers, export to the FV route | `--check` round-trip gate |

MC status (2026-11-03): the compressible FV family is done: `dns/fvmodules` (inventory, one
decision per pair), `gpu_fv_matrix` (43 live cells x 4 meshes, the generic gates 1-4),
`gpu_fv_matrix_phys` (a physics number per pair turned on), docs/fvcomp.md "Combinations"
generated. Not done: low-Mach and incompressible FV (their own module axes and stage), triples
(the matrix is pairs closed under requirements), gate 5 per cell (the structured counterparts
stay hand gates: X1, X2, S8), the one case description (M6).
Incompressible FV (mci, 2026-11-03): `dns/fvimodules` (LES smag/sigma, T, gravity live; RANS and
species refused alone by name, species on evidence: issues.md FI1), `gpu_fvi_matrix` (9 live sets x
4 meshes: uniform, rotated, random, O-grid; gates (a)-(e)), the eddy heat flux nu_t/Pr_t of LES + T
(new), `dns conv --solver fv --physics incompressible --les`, `gpu_fvi_matrix_phys` (Ra 1e5 3D:
sigma +3.4%, +3.1%, Smagorinsky +8.7%, +5.6% of the same-grid DNS Nu at 32, 48), docs/fvincomp.md
"Incompressible combinations" generated. Not done: gate 5 for LES (no structured LES counterpart
of this route, FI3).

M2 status (2026-11-03): built on the compressible FV family (docs/fvcomp.md "Air-methane"):
module `C2b` (`FvChem::law`; C2 is C1's stage), 48 harness cells, LES live, TFLES refused by
name; FV vs structured two-reactant interior rounding (R1), 1D order 2, rotated = planar,
sheared order 2, laminar LES sigma = DNS. Not done: a turbulent LES flame (`--turb` on FV),
the compressible burner, two-reactant TFLES.

M4 status (2026-11-03): LES (Smagorinsky, sigma) and TFLES (uniform, dynamic, Charlette; C0)
on both low-Mach solvers, structured and FV, off bit for bit; `dns/lmmodules` (the low-Mach
inventory, one decision per pair), `gpu_lm_matrix` (22 live sets x 3 FV meshes, gates 1-4),
`gpu_lowmach_les` (laminar limit bitwise both solvers; TFLES FV vs structured rounding;
Smagorinsky a stated order-2 scheme gap of mu_t), docs/chem.md "Low-Mach LES". Not done: a
turbulent low-Mach LES (no `--turb` on low-Mach), TFLES of two reactants (refused by decision).

M5 checkpoints status (2026-11-03): `--checkpoint-dt` / `--wall-limit` / `--restart` and
`tools/chem/chain.sh` on every FV flame route (`runFvComp`: premixed C0 / C2 / C2b, LES,
TFLES, compensated, FP64, the jet; `runFvLowMach`: planar, burner with open laterals, rim,
gravity, LES, TFLES), box and polyMesh; the mesh (polyMesh files + `--bc` content) is part of
the identity (exit 3). `gpu_restart_fv` / `gpu_restart_fvlm`: chained = straight byte for
byte, last checkpoints equal, teeth (zeroed low words / warm start p / face flux phi change
the outputs). Off = the runners bit for bit (`comp_bit_identity.sh`). Not done: FV `dns flow`
/ `dns conv` checkpoints (no flag there), a grown `--tt` on the low-Mach FV loop (its equal
steps depend on `--tt`).

M5a status (2026-11-03): compressible FV gravity x open patches through the hydrostatic
ambient's per-face rows (`FvConfig::ambient`, `fvAmbientRows`: p_a = `candleAmbientP` at the
owner centroid; FvRef checks them by name), the candle on `--solver fv` / `--polymesh --bc`
(gravity x C1 live, `gpu_fv_candle` rest / g0 / height; docs/fvcomp.md "Hydrostatic
ambient"). Rest exact in FP64 on uniform, oblique-g and random meshes; FV candle tip
h-independent 7.13 l, -10% vs structured etaIn 1000 (the inflow model's fuel gap). Not done:
the wick (ghost wall) on FV, a stratified (non-isothermal) ambient, low-Mach candle.

M5c status (2026-11-03): turbulence on the FV flame routes and the low-Mach solvers. `--turb`
on the compressible FV box (C0, C2b) and on the low-Mach planar flame, structured AND FV: the
ONE generator (`dns/turb_ic`) at the centroids / MAC faces (IC) and swept past the inflow face
centres as per-face rows (`dns/fvturb`, `TurbSweep`, `setBcRows` / `setInflow` /
`setInflowTurb`; the structured low-Mach tangential inflow its own `kLmMomT` instantiation).
Module `Turb` in both inventories with its pairs; `gpu_fv_turb` (IC rounding 3.7e-14 u', u' 0 =
laminar bytes, X3 FV vs structured rounding), `gpu_lm_turb` (u' 0 bytes, FV vs MAC order 2),
the matrix cells (7 + the lm Turb sets). Physics: S_c and flame surface vs the structured DNS,
3 grids (low-Mach 3e-4; compressible a stated non-shrinking 0.2-1.1% boundary gap, issues.md
M5c2). Not done: `--turb` on a polyMesh, on the low-Mach burner, with low-Mach checkpoints.

M1 and M3 touch disjoint solver files and run in parallel. MC starts once M1
and M3 land: every later module (M2, M4, M5) is written against the harness, so
it composes with everything at once, not with one partner. M6 last.
