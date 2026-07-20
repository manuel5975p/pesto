# Six-face NSCBC/LODI hardening review (agent 2)

Scope: `DeviceCompNscbc` (`src/dns/gpu/comp_nscbc.{hpp,hip}`), wiring in `comp_mesh_tunnel_gpu` (always six-face) and `comp_cylinder_tunnel_gpu --nscbc [--open]`, RK3 seam in `comp.{hpp,hip}`, `comp_flux.hpp`, `comp_far.*`, `test_gpu_comp_nscbc.cpp`.

**Verdict:** math conventions correct; slab enumeration correct. Real exposure is (a) two outflow physics weaknesses (backflow, lateral chattering switch), (b) one latent transverse-wrap bug masked only by the wall-mirror overwrite, (c) a near-total validation gap for the six-face path. Line refs are `comp_nscbc.hip` unless noted.

## Verified correct (no action)
- **LODI amplitudes / lo-vs-hi sign** (:115-155). `dn` = (larger−smaller)/h on both faces (:108-110) → true ∂/∂n. Outgoing/incoming split right: x-lo inflow computes only L1, models L2..L5 (:116-121); x-hi & lateral-hi compute L2..L5, model L1 (:149-151); lateral-lo compute L1, model L5 (:152-155).
- **Primitive→conservative conversion** (:158-171) is exact chain rule; normal LODI (primitive) + transverse `faceFlux` divergence (conservative) summed as two contributions to the same conservative rate. Consistent.
- **Tangent selection** t1/t2 (:111) correct for all three normals.
- **Disjoint slab enumeration** (:36-62) complete AND non-overlapping. x-slab owns full y,z at i∈{0,nx−1}; y-slab owns j∈{0,ny−1} at interior-x, any k; z-slab owns k∈{0,nz−1} at interior x,y. y–z edges at interior-x → y-slab (sums a=1,a=2); corners → x-slab (sums all three). `nb` in `apply` (:202-205) matches kernel guard (:38-40).
- **nz==1**: `make` forces openZ=false (:179); a=2 transverse terms cancel by wrap.
- **RK3 seam**: `rhs` calls `primitives(s)` before the hook (comp.hip:303,323) → `PrimView` in `apply` is live stage state, not stale. Overwrite (`=`, :165-171) correctly discards wrap-contaminated periodic rhs at boundary rows.

## Findings, ranked

**R1 — RISK (physics): x-outflow entropy/shear has NO backflow upwind switch.** `if (a == 0 || s * un > 0)` (:140) forces the "compute from interior" branch for every x face regardless of u_n sign; lateral faces upwind on `s*un>0` but the `a==0` short-circuit exempts the outflow. In a shedding wake, cores crossing i=nx−1 give local backflow (u_n<0) where L2,L3,L4 are incoming yet modeled as outgoing one-sided diffs → spurious entropy/shear injection. Plausible co-cause of the "growing global mode" that forced β=0.5 and of the Re2000 blow-up. Fix: drop the `a==0 ||` for the entropy/shear block so the outflow upwinds like lateral faces; re-run Re800 β-scan (expect required β to drop).

**R2 — RISK (physics): lateral entropy/shear switch is discontinuous → chattering.** At u_n=0 the outgoing branch scales by u_n (→0) but the incoming relax is finite `etaLat·c/L·(ρ−ρ∞)` etc. (:140 vs :144-147), so the RHS jumps as u_n crosses 0. In a vortex street the lateral normal velocity oscillates through 0 along the whole boundary every half-cycle → per-step RHS chatter feeding box modes. Small amplitude but genuine non-smoothness. Fix: blend the branches over |u_n|<εc (smoothstep) or scale relax by max(0,−s·u_n)/(|u_n|+εc). Add a unit test: vortex advecting *along* an open lateral face for many periods, assert bounded |v|.

**R3 — BUG (latent, currently masked): transverse `ct`/`faceFlux` wrap at corner rows when a lateral axis is a WALL.** `ct` (:80-87) only goes one-sided when `openA[d]`. In the walled cylinder app (x-only NSCBC), an x-face corner row (i∈{0,nx−1}, j∈{0,ny−1}) computes the Yoo-Im transverse derivative with a central stencil reaching j−1→wrap ny−1 — across the whole domain to the opposite-wall mirror ghost, not the physical wall gradient; the transverse `faceFlux` at those rows (:94-101) is likewise wrap-contaminated. Currently harmless ONLY because those corner rows are `kFarWallY/Z` ghost rows (comp_far.hip:62-95) re-mirrored from j=1 at each stage start (comp.hip:345-351), discarding the NSCBC rhs there. Remove walls / change hook order / run a periodic-lateral config and it goes live. Fix: pass a `wallA[3]` mask, force one-sided `ct` and skip the wrapped transverse face on walled axes; or assert every lateral axis is {open | downstream-mirrored} and document the dependency.

**R4 — RISK (physics): lateral pressure anchor clamps the near-field dipole.** Lateral incoming acoustic = `sigLat·(p−p∞)·c/L` (:138). Body far-pressure decays as a dipole (~1/r²·½ρu²), so p at the lateral boundary ≠ p∞; anchoring it imposes a residual pressure error → blockage (same class as incompressible `--open` fringe). Sphere Re100 cd 1.10–1.146 vs lit ~1.09 shows it small at that stand-off but it scales with body/box size, not any BC tweak. Not a bug; flag so six-face cd is never quoted as unbounded-exact. Real fix (unimplemented) = potential-dipole-corrected fringe target, not sigLat tuning.

**R5 — RISK: `betaTrans` reused verbatim on lateral faces, unvalidated there.** `betaT` applied identically to outflow and lateral Yoo-Im (:136-139); 0.5 was tuned ONLY for x-outflow Re800 (1.0 blows up). Lateral transverse physics differs. Experiment: split `betaLat`, scan on six-face cylinder `--open`.

**R6 — MINOR (bug): LES eddy viscosity dropped from boundary transverse flux.** Boundary `faceFlux` (:95-96) called with default `nut=nullptr`; interior kCompRhs adds μ_t (comp.hip:310-321). For `--les Cs>0` mesh runs the boundary rows lose SGS transverse diffusion + eddy conductivity — silent scheme inconsistency (small, boundary is near-freestream). Fix: thread `nut`/`kapScale` into the nscbc transverse flux when LES on.

**R7 — MINOR: FP32 headroom at low Mach.** c² and all L amplitudes FP32 with p∞=1/(γM²)≈17.9 at M=0.2 (baseline ~18, perturbations ~1 → ~6 sig digits on the perturbation). Passes the 5e-6 fixed-point test now; margin shrinks as M→0 (p∞∝M⁻²). Consider p−p∞ / u−u∞ amplitude formulation if M<0.1 ever needed.

**R8 — MINOR: no dt-vs-eta stability guard in `make`.** `make` (:182-186) validates etaIn>0 but the relax rate is etaIn·c/L·dt; large etaIn or tiny domain can exceed 1 with no diagnostic. Add a guard or documented ceiling.

## Validation gaps (largest exposure)
`test_gpu_comp_nscbc.cpp` covers only transient residuals on tiny grids with 15–20% thresholds; no saturated statistic.
- **VG1**: no long-window limit-cycle test. Every "exits cleanly" test runs ≪ one shedding period; the failure modes that matter (β drift, box modes, R1/R2 accumulation) are slow. Need cylinder `--open` Re800 to saturation, assert bounded cd/cl σ and steady St over ≥15 ft.
- **VG2**: no Re2000 six-face robustness. x-only already needs `--nsstrip`; six-face lateral-open is *more* exposed (two more soft faces), zero coverage. Test the `--nsstrip` hybrid on the open lateral bands.
- **VG3**: no 3D bluff-body/shedding anchor. Only sphere Re100 (steady). No mesh app entry in `data/validation/anchors.csv` (only incompressible sphere_re300); `test_val_comp_mesh_sphere.cpp` does not exercise NSCBC. Add a Tier-A `comp_mesh_tunnel_gpu` drive with pinned cd tolerance.
- **VG4**: no chattering/sign-reversal test (R2), no backflow-outflow test (R1), no mixed open/wall (openY=1, openZ=0 at nz>1) test (would expose R3).
- **VG5**: the unit `Rig` installs DeviceCompFar walls + NSCBC together (:47-62), so R3's corner-overwrite is exercised only in the benign masked direction; the un-masked path is untested.

## Concrete next steps (ordered)
1. Remove `a==0 ||` exemption for x-outflow entropy/shear (R1); re-run Re800 β-scan. Highest value, smallest change.
2. Continuously blend the lateral entropy/shear switch at u_n≈0 (R2); add the along-face-vortex unit test.
3. Add long-window saturated cylinder `--open` regression + a `comp_mesh_tunnel_gpu` Tier-A anchor (VG1/VG3).
4. Make transverse stencils wall-aware or assert the invariant (R3); thread `nut` into boundary transverse flux for LES (R6).
5. Split `betaLat` from `betaTrans` and scan (R5); document the lateral pressure-clamp blockage as known size-scaling residual (R4).
