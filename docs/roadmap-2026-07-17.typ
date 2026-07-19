#set document(title: "DNS roadmap notes — 2026-07-17", author: "fluid/dns")
#set page(margin: 2.2cm, numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt)
#set heading(numbering: "1.1")
#show heading: set block(above: 1.4em, below: 0.7em)
#show raw.where(block: false): box.with(fill: luma(240), inset: (x: 3pt), outset: (y: 3pt), radius: 2pt)

#align(center)[
  #text(17pt, weight: 700)[DNS roadmap notes]\
  #text(11pt, fill: luma(90))[Session record, 2026-07-17 — Re 800 validation, solver directions, compressible pivot]
]

= Current state (validated)

Incompressible DNS: MAC grid, RK3 (SSP) + pressure projection, KE-conserving skew-symmetric
advection — the only dissipation is the physical $nu$. HIP GPU backend (FP32 fields, FP64
reductions) mirrors the CPU reference 1:1.

- *Sharp viscous wall* (ghost fill, `src/dns/gpu/ghostfill.*`): sub-cell no-slip via mirror
  ghosts fed _only_ to the diffusion term (ghosts in the skew convection inject energy,
  measured +2.3%/300 inviscid steps). Grid-converged: cylinder Re 60 $c_d$ = 1.785 at both
  512#super[2] and 1024#super[2]; the old staircase wall sat 0.066 low.
- *Free-slip lateral tunnel walls* (all four tunnel apps): tangential mirrored (wall vorticity
  $equiv 0$, verified $5 dot 10^(-4)$ max), normal antimirrored, pressure Neumann. The earlier
  Dirichlet $u = u_infinity$ clamp painted a real vortex sheet (wall $|omega|$ 1.2–3.6 vs
  0.016 interior) — do not reintroduce.
- *Latest run* — circular cylinder, $"Re"_D = 800$, 512#super[2], 15 flow-throughs, pure DNS:
  $c_d = 1.832 plus.minus 0.218$, $"St" = 0.257$, $c_l$ rms 1.17 (peaks $plus.minus 1.6$).
  Honesty: at Re 800 the _2D restriction itself_ is the dominant model error (real wake is 3D
  past Re $approx 190$; experiment $c_d approx 1.0$, 2D characteristically 1.8–2.0). Boundary
  layer $approx D\/sqrt("Re") approx 2.3$ cells — just under the 3-cell rule.

= Grounding against reality and literature

Cheapest-first ladder:

+ *Canonical validation suite as ctest targets.* Cylinder Re 100 ($c_d$ 1.33, St 0.164),
  Re 200 (St 0.197), sphere Re 300 ($c_d$ 0.657, $c_l$ 0.069 Johnson & Patel — we measure
  0.706/0.071), 3D Taylor–Green Re 1600 dissipation curve vs the spectral reference.
+ *Blockage correction.* The 8$D$ tunnel over-reads $c_d$ by ~5–10% (Maskell); likely most of
  the sphere 0.706-vs-0.657 gap. Validate at 16–24$D$ or report corrected values.
+ *Convergence triples* (256/512/1024) with Richardson extrapolation; report the extrapolated
  value and the observed order, always.
+ *Discrete KE budget closure* per run: $dif K\/dif t + epsilon_nu + "body drain" - "boundary flux" = 0$
  to roundoff. Would have caught the Dirichlet wall sheet immediately.
+ FP64-field anchor run to bound FP32 drift; 3D spectra ($-5\/3$ band) and enstrophy vs
  published TGV data.

= Cylindrical coordinates — verdict: skip

A native $(r, theta, z)$ MAC grid costs: metric terms that break the exact skew-form KE
conservation, the $r = 0$ axis singularity, $Delta t$ collapse from tiny $r Delta theta$ cells,
and a second Poisson family (FFT in $theta, z$ + tridiagonal in $r$). Everything cylindrical
geometry buys — pipes, annuli, rotating and axisymmetric bodies — the validated cut-cell +
ghost-fill machinery already handles on the Cartesian grid. Only worth revisiting for
rotating-machinery flows where azimuthal periodicity buys spectral accuracy in $theta$.

= Arbitrary geometry: STL / glTF / GLB

Highest value-per-effort item. The cut-cell + ghost machinery needs exactly two callables —
$"sdf"(x,y,z)$ and cut geometry ($lambda$, apertures) — so mesh support is a _loader_, not a
solver change:

+ Triangle soup from STL (trivial) or glTF/GLB via `cgltf`.
+ Signed distance: BVH + closest triangle, sign from angle-weighted pseudonormals
  (Bærentzen–Aanæs) or generalized winding number (Jacobson) for leaky scans.
+ Precompute a narrow-band SDF voxel grid (~2$times$ flow resolution) once on CPU, upload as a
  3D texture; ghost fill and the existing supersampled cut classification (`makeGolfCut`
  pattern: 8#super[3] samples per cell) consume it unchanged.
+ Deliverable: `mesh_tunnel_gpu model.stl --scale --pitch --yaw` — one new app, zero kernel
  changes.

= Body-fitted O-grids (the "X-wing" idea)

Wrap the body outline with the $xi$ index ($eta = 0$ _is_ the wall), march $eta$ outward.

*2D special case is genuinely attractive.* A polygonal cross-section (the $+$ shape) admits a
Schwarz–Christoffel conformal map from an annulus to its exterior. Conformal $arrow.r.double$
scalar metric $J = |f'|^2$, exactly orthogonal grid, and the transformed pressure Poisson stays
_constant-coefficient_: $partial_(xi xi) p + partial_(eta eta) p = J r$ — FFT in periodic $xi$
+ tridiagonal in $eta$. The fast exact solver survives. Contravariant volume fluxes on a
staggered computational grid keep discrete continuity and exact skew-form KE conservation
(Zang et al. 1994). Wall BC is a grid line: no cut cells, no ghost fill, and wall-normal
clustering ($Delta y^+ approx 1$) is free.

*The catch: corners.* Conformal maps are singular at polygon corners — $|f'| arrow 0$ at convex
corners ($Delta t$ collapse), $arrow infinity$ at the four reentrant corners (resolution starves
exactly where the shear layer separates). Fix: round the corners with radius of a few cells
(physical anyway). Non-conformal alternative (Winslow smoothing) improves grid quality but
makes the metric a tensor: variable-coefficient Poisson, CG/multigrid only.

*3D reality.* A single structured O-grid around a full aircraft does not exist topologically;
industry uses multi-block/overset (Chimera, e.g. OVERFLOW) — a large lift (hole cutting, fringe
interpolation, approximate conservation across fringes). Not planned.

Recommended order: STL support $arrow$ co-moving frame + time-dependent SDF (pitching bodies,
nearly free) $arrow$ 2D conformal O-grid mini-solver as a validation cross-check $arrow$
forget native cylindrical.

= Pressure solver: one-shot vs preconditioned, and the iteration ladder

Without a body the projection _is_ one-shot: FFT/DCT $arrow$ divide by discrete-FD Laplacian
eigenvalues $arrow$ inverse transform, machine-exact. With cut cells the operator becomes
$"div"(a dot "grad" p)$ with per-face apertures $a in [0,1]$ near the surface — not
translation-invariant, so no diagonal multiply exists. The uncut FFT inverse is instead used as
a _preconditioner_ inside CG; the perturbation lives on a thin shell, so iterations stay
$approx 40$ independent of grid size.

Reduction ladder (pressure dominates step cost: 3 stages $times$ ~40 iters $times$ one FFT
round-trip each):

+ *Extrapolated warm start* (trivial): $phi_"guess" = 2 phi_(n-1) - phi_(n-2)$, kept per RK
  stage. Expected 20–40% fewer iterations. (Plain warm start is already in.)
+ *Tolerance audit* (free): FP32 advection noise re-injects divergence next step; find where
  the useful residual decade ends.
+ *Deflated PCG* (moderate): the slow modes live on the static cut shell — precompute them once
  (Lanczos at startup), deflate. 40 $arrow$ ~10–15. Works in 2D and 3D.
+ *Capacitance matrix / Woodbury* (2D kill shot): $A = hat(A) + P$ with $P$ of rank
  $m approx$ \#cut faces (~800 for the 512#super[2] cylinder). Factor the $m times m$
  capacitance matrix once (Proskurowski–Widlund); every projection becomes 2 FFT solves + one
  small triangular solve — *exact, zero iterations*. In 3D, $m tilde 30"k"$ surface faces makes
  the dense factor ~7 GB: fall back to deflation or an $cal(H)$-matrix.

Expected: ~3–4$times$ faster 2D steps, 2–3$times$ in 3D.

= Compressible Navier–Stokes — next step

Motivation: acoustics, genuine Mach effects, and a solver with _no elliptic solve at all_
(fully explicit — embarrassingly GPU-friendly; the trade is the acoustic CFL
$Delta t prop h\/(|u| + c)$).

Plan (house pattern: CPU double-precision reference first, GPU port after validation):

- Conservative variables $q = (rho, rho u, rho v, E)$, ideal gas $gamma = 1.4$,
  $p = (gamma - 1)(E - 1/2 rho |u|^2)$. Collocated grid (no MAC needed — no projection).
- Convective terms: Kennedy–Gruber kinetic-energy-consistent split-form two-point fluxes
  (the compressible analog of our skew form; central, non-dissipative, stable for smooth flows).
- Viscous terms: face-based compact stencils, Newtonian with Stokes hypothesis
  ($lambda = -2/3 mu$), Fourier heat flux $k = mu c_p \/ "Pr"$, Pr = 0.72.
- Time: SSP-RK3, acoustic CFL.
- *Stated limitation:* central non-dissipative fluxes — smooth/shock-free flows only. Shock
  capturing (HLLC/WENO hybrid or artificial bulk viscosity) is a later, separate step.
- Validation tests (`test_comp2d`): isentropic vortex advection (Euler; exact solution,
  measured order $approx 2$), small-amplitude acoustic pulse speed vs $c = sqrt(gamma p \/ rho)$,
  low-Mach 2D Taylor–Green KE decay vs the incompressible exact rate $e^(-4 nu t)$, and exact
  conservation of $sum rho$, $sum rho u$, $sum E$ to roundoff.
