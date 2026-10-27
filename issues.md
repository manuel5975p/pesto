# dns — open issues

Current, changing from commit to commit. CLAUDE.md holds the stable
invariants; this file holds what is broken, unproven or undecided NOW.
Remove an entry when it is fixed (the fix lands in the commit message and,
if it creates an invariant, in CLAUDE.md). Numbers are measured, gfx1100.

As of the P4 (full-revolution sphere) pass on 3133831 (2026-10-26,
uncommitted).

## Flow RANS (masterplan `plans/flowrans/masterplan_flowrans.md`)

1. **The explicit LTS march has no steady state on the coarse wedges at
   Re ≥ 1e3 -- for EVERY closure, laminar included** (measured 2026-10-26,
   M 0.2, 1e5 iterations). 32×24: laminar Re 1e3 cd jitter 20%, Re 3e3
   resid 150; SA Re 3e3 diverges (exit 7 at 2.8e4); SST-2003m/Vm Re 1e3
   jitter 18% / 11%. L1 (64×48): laminar Re 3e3 resid 53, Re 1e4 resid 261;
   SA Re 1e4 resid 0.46 (the 2.5e6-iteration run: cd 0.4197 ± 1.1%, θ_s
   12.9°); SST-Vm Re 1e4 resid 30; SST-2003m Re 1e4 diverges (exit 7 at 4.8e4,
   13.6 s). The signature is the FRONT: the stagnation-region wall pressure
   breaks into a period-2 θ sawtooth from the axis outward (L1 Re 1e4 SST at
   1e4 iterations: cp 0.10, −0.67, 1.06, 0.50, … on i = 1..4; laminar Re 1e4
   0.33, 1.48, 2.07, 1.09, −2.48; SA a smooth deficit, cp 0.66), whose
   vortical debris then reaches the potential flow (|Δu| 1.5 at j ≈ 20, θ
   50–70°, μt at its freestream value: the mean flow, not the closure). A
   TIME-ACCURATE laminar L1 Re 3e3 run keeps a clean front (cp 1.15, 1.13,
   1.08, …; its wake sheds, cd 0.18–0.60), and OpenFOAM (SIMPLE, central or
   linearUpwind U) converges on the same meshes (SST L1 Re 1e4: 0.430
   central). L1 SST certifies at Re 1e3 (gate `gpu_sst`), and so does L1
   SA-neg (136600 iterations, 29 s, gate `gpu_sa_wedge`, 2026-10-27), and
   SST nearly at Re 3e3 (remaining 1.7e-4 at 2e5). LTS-ratio cap test: see below. Not an SST
   defect; cause in the LTS march not isolated further. Dissipation is the
   obvious cure and is not allowed (CLAUDE.md). DECIDED 2026-10-26: exit 8
   on the coarse wedges is accepted as the honest outcome; kept here as a
   known limitation, not a work item.
2. **Wedge SST-2003m Re 1e4 L2 has no STABLE steady state on this
   discretization** (measured 2026-10-26). The explicit LTS limit cycle (cd
   0.4088–0.4136, θ 125–145°, j 38–50, period ~250 iterations, k at the
   shear-layer edge swinging ±40%) is robust to `--dt-recompute 1` (cd
   0.409–0.413, jitter 4.5e-3) and worse at `--cfl 0.2` (resid 8–240 after
   3e5). The TIME-ACCURATE URANS (`--ft 0.4`, 5.1e5 steps, 231 s) is unsteady
   in the SAME band: cd 0.404–0.424 from t = 8 to 14.6 D_box/U without decay,
   time mean 0.4115, period ~0.7 D_box/U -- so the cycle is the discrete
   model's own unsteadiness, not a pseudo-time artefact. SST-Vm on the same
   mesh HAS a steady state (cd 0.41882, jitter 6.5e-5, remaining 1.1e-4 at
   the 3e5 cap: certifiable with a larger cap); SA's URANS is still drifting
   onto its steady 0.4248 (mean 0.4265 over t ≥ 12). OpenFOAM SIMPLE
   converges anyway (0.39496 linearUpwind, 0.40589 central U) -- SIMPLE
   solves the steady equations, it does not need the state to be stable. An
   explicit march cannot converge to an unstable fixed point; Newton can,
   but NK does not converge from a cold start on the wedge (3). DECIDED
   2026-10-26: SST-2003m Re 1e4 is reported as the URANS time mean, labelled
   URANS (tier A M 0.2: 0.4115 vs OpenFOAM central 0.40589, +1.4%). Open:
   the M 0.1 URANS leg for tier B (~22 min, not run); a hybrid
   explicit-then-Newton start is not pursued. L1 SST-2003m divergence is
   item 1.
3. **Newton-Krylov (`--steady-scheme implicit`) is slower than explicit
   everywhere it converges, and does not converge from a cold start on the
   Re 1e4 wedge.** Cylinder Re 40 256×96 certifies (TOL 1e-4): laminar 78
   steps / 28.4 s (explicit 3.0 s), SA 73 / 30.1 s, SST 174 / 55.4 s. Wedge SA
   Re 1e4 L2 cold: exit 7 (16 rejections, 71 steps). From the explicit
   state at 1.5e5 iterations: cd 0.42482 in ~10 steps at a residual 30×
   below explicit, but GMRES stalls at the FP32 floor (200 Krylov per step;
   `kNkStallKrylov` = 40 after a stall), rejections reset the armed history,
   W jitters 8.4e-4: no certification in 200 steps (129 s) vs explicit
   cold-start certification in 184 s. The line-SGS is now 13.4 ms per
   application (59 ms at 75f39db); its serial chain is the floor. Wedge SST
   L1 Re 1e3 (2026-10-26): cd 0.49852 (= explicit), resid 2.3e-3 (explicit
   0.17), but exit 8 at the 500-step cap (173 s; W jitter 9.5e-4 at the
   floor) where explicit certifies in 28 s. Candidate
   next steps (not built): an explicit-then-Newton hybrid start, a
   floor-aware GMRES tolerance, PCR lines. Default stays explicit (U3).
4. **M 0.1 runs cannot certify on the Re 1e4 wedge.** The jitter-floored
   stop needs trend + √2 J < TOL for cd/cdf and J ≤ 2⁻¹⁰ for the monitors.
   SA L2 at M 0.1 (2.5e6 iterations, 766 s, settled cd 0.41943): cd J
   1.6e-4, W J 4.7e-3 (15× M 0.2's 3.1e-4; p/q∞ = 143 vs 36). So the tier-B
   M 0.1 leg ends at the cap (exit 8) whatever TOL. Decision needed: a
   Mach-aware monitor floor, a different wake monitor than the whole-domain
   deficit W (its jitter is dominated by the large far-field cells), or
   accept exit 8 + the printed floors for M ≤ 0.1.
5. **OpenFOAM SA ft2 sensitivity exceeds the U4 trigger.** L2: `ft2 true`
   0.41285, `ft2 false` 0.40885 (−0.97%, trigger 0.5%). `sa-noft2` on the
   dns side is a user decision (U4), not built.
6. **Immersed SA steady Re 40 is slow.** 384², 4e5 iterations (311 s):
   cd a slowly damped oscillation (2.059 .. 2.0983, period ~8e4
   iterations), jitter 8e-4, not certified at TOL 1e-4 (correctly). The
   uniform mesh gives LTS no leverage; `--focal` or NK on the immersed path
   (not built) would be the levers.
7. **`sph1e4` steady parity is SA L2 only; L3 not run.** The reference is
   OpenFOAM with central U (DECIDED 2026-10-26, now the `sph1e4.py`
   default): SA tier B −0.35%, θ_s −0.18°, L_r +3.8%. (linearUpwind read
   +1.16%; OpenFOAM's own SST cd moves +2.8–7.6% between the two schemes.)
   SST: URANS mean only (2); small SST parity L1 Re 1e3 +0.41% (`gpu_sst`);
   small SA parity L1 Re 1e3 −1.07% (item 21).
8. **Not built:** heavy immersed channel oracle, focal/sinusoidal MMS,
   momentum-budget tally test; multigrid / alternating lines (U5); P6. P4
   (the full revolution) is built for the explicit march and time-accurate
   runs; not built there: Newton-Krylov (3D wavefront line ordering, G4.1),
   `--les` (the axis cells' filter width), the 3D OpenFOAM export
   (`tools/parity/dnsmesh.py` reads the wedge only), and the heavy
   `val_rans_sphere_3d` test (G1 does not certify inside 10 min, item 11).
9. **URANS turbulence scalars are first order in time.** The point-implicit
   stage quotient (D13) degrades SSP-RK3 to first order in the stiff
   sources: the closed-form SST freestream decay converges at order 1.00 (ω)
   / 0.91–0.98 (k), `gpu_sst` (a); relative error 8e-3 (ω) / 3.9e-2 (k) at
   β ω0 dt = 0.12. The steady fixed point is unaffected. A URANS result
   whose β*ω dt is not small carries this error. DEFERRED 2026-10-26: a
   second-order implicit source update (and its byte-identity break) is not
   built for now; URANS results state the first-order source error.

10. **The full revolution is EXACTLY π-symmetric, so its odd azimuthal
    modes never arise on their own** (measured 2026-10-26). The mesh maps
    onto itself under a rotation by π about the axis to the bit, and the
    FP32 kernels are sign-symmetric: an axisymmetric start keeps planes k
    and k + N/2 bitwise equal (cl == cs == 0.0) for ever. DECIDED
    2026-10-26: the trip is ON by default (`kOgridTripDefault` 0.05 U, the
    bracket in docs/rans.md "The full revolution"); `--ogrid-trip 0` is the
    untripped march bit for bit and is warned at Re ≥ 200 or with `--rans`.
    Remaining risk: the stop does not watch cl, so a seed that grows slowly
    can still be outrun by the stop (at Re 250 the 0.05 trip leaves m = 1 at
    21400 iterations against a symmetric stop at 36200; a slower instability
    has less margin). A cl/cs monitor in the stop (absolute, cl has no
    scale) is not built.
11. **G4.3: SA-neg Re 1e4 on G1 has no certified STABLE steady state inside
    10 minutes; from the wedge the symmetric one certifies in 135 s**
    (measured 2026-10-26, G1 = 128×96×96, 1.2M cells, β 4.5, M 0.2, 3.41
    ms/iteration). Cold start, trip 0.05: exit 8 at 170000 (581 s), cd
    0.42519 ± 1.2e-4 (+0.09% of the wedge's 0.42480), an m = 1 cl
    oscillation growing (1e-3 → 1.5e-2 from 5e4 to 1.6e5 iterations).
    `--init-from` the certified 5° wedge (188 s to make): untripped exit 0
    at 38800 iterations / 135 s, cd 0.42520 -- inside the 5-minute target
    (323 s with the wedge) -- but tripped the same m = 1 growth appears
    (cl 8.7e-3 at 1.7e5, residual 0.13 → 4.0, exit 8 at 584 s), so that
    certified number is the π-symmetry-protected axisymmetric state, not a
    stable 3D one. Its end state (steady asymmetric, as SA-neg at Re 1e3, or
    unsteady) is open; following the m = 1 mode (e-folding ~4e4 iterations)
    to saturation is tens of minutes at G1. No `val_rans_sphere_3d` heavy
    test: there is no certified stable G1 number to pin.
12. **The DNS time march of the full revolution is 150× slower in dt than
    the wedge's, and a per-cell bound does not help** (measured
    2026-10-26, laminar Re 100, `--ramp 0`): dt 2.55e-4 on the 5° wedge at
    128×96, 1.70e-6 at N 64 (3.6e-6 at the start). The binding term is
    the viscous bound at the axis cells' azimuthal width ½ R Δθ Δφ. The
    per-cell form (`rans::LaminarDiffDt`) gives the same dt to 0.8%
    (64×48×64: 1.4210e-5 both at the start, 1.1844e-5 vs 1.1934e-5 after
    2000 steps; 128×96×64 equal to 5 digits) -- with constant μ the h_min
    cell binds either way -- and the limit is real: at 1.5× and 2× the bound
    the march runs 3000 steps, at 3× it diverges within 200 (64×48×64). So
    the approved "per-cell bound on the full revolution" was NOT built: it
    would change nothing (the premise -- the global-extreme pairing -- was
    mine and wrong). URANS at Re 1e4 is not penalized because ν is 100×
    smaller (3.99e-6 vs the wedge's 2.18e-6, the wall's RANS term binding
    both). What would: an implicit azimuthal diffusion on the axis ring, or
    azimuthally coarser axis cells (a mesh change); the inviscid acoustic
    bound still sees the axis width (6.2e-5 vs 6.3e-4 at 64×48). A G1 URANS
    costs ~1e3 s per D/U (2.6e5 steps at ~3.7 ms).

## Test tier (masterplan `plans/verify/masterplan_verify.md` W14)

13. **The RANS/LES/MRF subset misses its 200 s budget: 250.8 s serial,
    225.6 s in one `ctest -L 'rans|les|mrf' -j4` call** (measured
    2026-10-27, gfx1100, 41 tests, all pass; baseline 314.7 s for 40). Every
    test is <= 56 s (`gpu_rans` 55.7, `gpu_axisym3d_run` 40.0, `gpu_rans_sst`
    32.2, `gpu_runners` 31.7, `gpu_sst` 28.9, `hit_mini` 21.8). The plan's
    200 assumed `gpu_rans` G9's plain march dropped and the CPU `hit_mini`
    moved behind `DNS_VALIDATION_BASELINE`; by user decision both stay, made
    cheaper (G9 15.3 s on ny 32, `hit_mini` 3 + 3 eddy turnovers, 21.8 s):
    37 s of the gap. The rest is the plan's arithmetic (it predicted ~185 s
    from the same removals). Later W items add ~115 s (masterplan W14
    "end-state budget"), which puts the serial subset near 365 s; the
    plan's trim order applies then. Remaining levers, not taken: G10's
    perturbed compensated 2.2e5 march on ny 32 (est. ~-13 s if 1.2e5
    iterations suffice there as they do for the plain march; unmeasured; the
    plan keeps it on ny 64), SST-G9 perturbation 5% -> 1% (user decision 7,
    ~-27 s).
    **End state (re-measured 2026-10-27 after the last item, W16): 251.1 s
    serial, inside the 300 s end-state budget; no trim taken.** 47 tests,
    all pass, two gpurun calls (`-R 'gpu_rans|gpu_sst|gpu_sa_wedge'` 121.9 s,
    the rest 129.2 s). The predicted ~365 s did not happen: W8/W9 made the
    RANS tests cheaper (`gpu_rans` 55.7 -> 44.7, `gpu_axisym3d_run` 40.0 ->
    32.4, `gpu_runners` 31.7 -> 26.2, `gpu_sst` 28.9 -> 21.2, `gpu_rans_sst`
    32.2 -> 29.4) and the added tests cost ~37 s (`gpu_sa_wedge` 21.3,
    `gpu_mrf_rotor2` 11.0, the rest < 2 s each). Largest test `gpu_rans`
    44.7 s (<= 75). The 200 s figure stays missed (the overrides above).

## Correctness hygiene (masterplan `plans/verify/masterplan_verify.md` W4)

15. **`RansDiffDt` is unstable on isotropic 3D cells where it binds**
    (measured 2026-10-27, `gpu_rans_mask` M7). The bound is 0.25/(D amax²);
    the compact face Laplacian's checkerboard needs Σ_a a_a². A periodic
    16³ box with uniform μt = ρk/ω (k = ω = 0.1, at rest, so the bound
    binds 6× under the acoustic one) and a 1e-3 T checkerboard at uniform p,
    stepped AT the bound: 2D decays to the FP32 floor (1e-3 -> 3.0e-8 in
    2000 steps, λΔt = −2); 3D grows 10× in 4 steps and saturates at O(1)
    (λΔt = −3 against SSP-RK3's −2.51). Ladder in 3D: 0.5×/0.8×/0.83× the
    bound stable (3.0e-8/3.0e-8/7.0e-7), 0.85× and 0.9× grow (10× at steps 39
    and 8): the threshold is the predicted 2.51/3 = 0.837. Where it bites:
    cells near-isotropic in ≥ 2 axes with a large μt -- 3D immersed RANS
    wakes, the full revolution's outer cells, an extruded O-grid's wake;
    wall-clustered cells have amax² ≈ Σa². No run has been seen to diverge
    from it (the protocol-P cases are byte-identical and stable). NOT fixed
    by decision of the plan (W4 g): the Σ_a a_a² form changes every RANS dt
    and LTS bit and the steady certifications would need re-pinning -- a
    USER DECISION (masterplan §5 item 6).
16. **`--ogrid-trip 0.05` on the 2D cylinder O-grid saturates 1.8× earlier
    than untripped, not the planned 2×** (measured 2026-10-27, laminar Re
    100, 192×96, M 0.2, quiet start, t 143, ~28 s each). |cl| envelope
    reaches 0.068 at t 30.3 vs 73.3 (2.4×), 90% of saturation at 52.7 vs 95.8
    (1.82×), 99% at 71.6 vs 114.6 (1.60×). 0.2 / 0.5 give 90% at 41.2 / 35.1
    (2.3× / 2.7×); every trip reaches the same limit cycle (envelope 0.333,
    cd 1.380). The trip moves the linear growth phase only; the ~40 time
    units of nonlinear approach stay. The trip is opt-in (default 0 on the
    cylinder O-grids), so nothing default moved. W12 kept 0.05: `val_ogrid_cyl`
    averages from t 82 and fits in 27 s, so the 2× was not needed.

## LES wall closure (masterplan `plans/verify/masterplan_verify.md` W1)

17. **Fixed 2026-10-27: LES μt had no wall image and the tallies omitted
    SGS.** μt now takes the RANS images (`setMutHook`; O-grid wall face
    exactly 0.0, was ~0.1 ν; immersed odd image; `--noslip` rows odd, mirror
    rows even), and every compressible tally passes μt (momentum closure
    4e-9..9e-9 relative, was 0.17-0.51 -- `gpu_les_wall`); the incompressible
    SGS tally gained z links and fz (`gpu_les_tally3d`). Every LES run with a
    wall or a closed lateral row moved (no LES anchor exists): 128³ smag
    sphere Re 1e4 cd 0.609 -> 0.734 at t 2.31, 3D incompressible sphere
    0.643 -> 0.675, 2D focal Re 3900 mean 1.228 -> 1.247 (2D LES,
    qualitative). Still open, deferred by decision (masterplan §5 item 8):
    the NSCBC x rows' μt is the raw wrap-differenced value (the hook is the
    natural fix if a measured LES outflow shows it). The immersed image's
    negative staircase-face μt is fixed by W3 (item 19). The LES `stableDt`
    bound reads the solver's own imaged μt since W15 (item 26).

## Immersed μt image (masterplan `plans/verify/masterplan_verify.md` W3)

19. **Fixed 2026-10-27: the immersed μt ghost image made staircase faces
    anti-diffusive.** The odd image μt_g = −a Σw μt_s drove the face total
    μ + ½(μt_g + μt_c) NEGATIVE on 51/128 faces (SA, down to −39 μ) and
    21/128 (SST, −16.5 μ) of a 128² Re 1e5 cylinder with a law-of-the-wall
    layer (y⁺ 30–80); unclamped SST's dt fell to 2e-9 within 200 steps. Now
    μt_g = −min(image, min μt of its fluid face neighbours): 0 faces below μ
    (`gpu_rans_immersed` (e)). It binds on any staircase (41 of 101 ghosts on
    a resolved tilted plane), so every immersed RANS/LES run moved, by
    little: P immersed SA cd −3.3e-5 rel., SST < 2e-6; URANS Re 1000 384²
    mean cd SA |Δ| < 1e-5, SST −1.1e-4; 128³ smag sphere LES −1.2e-3. No
    anchor. Still open, NOT caused or cured by the image: immersed URANS on
    the uniform 256² box diverges at t ≈ 2.6 at Re 1e4 and earlier at Re 1e5
    (exit 7, at the same t with and without the clamp, `--rans sa-neg|sst-2003m
    --ft 10`) -- an under-resolved central scheme with no added dissipation,
    by design; `--focal` or a larger `--n` is the lever.

## LES known answers (masterplan `plans/verify/masterplan_verify.md` W10)

18. **Closed 2026-10-27: the LES `stableDt` bound does not bind where
    measured (the "over-conservative on clustered meshes" older item).**
    The bound IS over-conservative by construction (it pairs the largest
    μt, a coarse cell, with h_min), but it never decided a dt.
    `gpu_les_tgv_comp` compares the LES `stableDt` with a DNS DeviceComp's on
    the SAME state at every sample: identical on 121/121 samples for smag
    0.17 and σ 1.35 on the uniform 32³ TGV and for smag on a ratio-3 focal
    box; the diffusive bound clears the dt taken by ≥ 102× / 79× / 36×.
    Threshold, from `comp.hip` `stableDt`: it binds iff
    (ν + νt,max)·γ/Pr > 0.25·h_min²·w_ac/cfl, i.e. for νt,max =
    (Cs·R·h_min)²|S| at a cell R× coarser than h_min, iff
    |S|·h_min > 0.25 (|u| + c)/(cfl·(γ/Pr)·Cs²·R²) = 66.7 U/R² at M 0.2
    (|u| + c = 6U), cfl 0.4, Cs 0.17, w_ac ≈ (|u| + c)/h_min (the LES audit's
    "66.9/R"). The audit's own evidence
    (dt·wmax = 0.01309 in all 27 rows of the focal Re 3900 LES and DNS runs)
    is a TAUTOLOGY and proves nothing: the stats `wmax` is defined as
    cfl·min(h)/dtStab (`runner_comp.hip:788`). For that case the threshold
    in the fine cells is |S| > 1600 U/D (h_min = D/24) against a boundary
    layer |S| ~ U/δ ≈ 60–200 U/D at Re 3900 -- an estimate, not a
    measurement. Reopen if a run's LES dt ever drops below its DNS twin's.
    Also measured by W10 and NOT true (recorded, not pinned): σ is not
    quiet on the TGV for t < 3 (ε_sgs/ε_visc 1.35 at t 0.5, 2.60 at t 3 --
    the TGV is 3D at once and (Cσ h)²/ν = 112 at 32³), and σ's K is not
    closer to 512³ than DNS's (max|K − K512| 0.0229 vs 0.0146). Numbers:
    docs/validation.md "LES known answers".

## Rotor MRF parity (masterplan `plans/verify/masterplan_verify.md` W5-W7)

14. **Rotor MRF parity is open. dns L1 CT is −0.02392 against central
    simpleFoam −0.01790 (+33.6%) and central rhoSimpleFoam −0.01865
    (+28.2%). CQ is +1.0% and −0.45% against the same two references. The
    OpenFOAM reference is itself
    MRF-zone-sensitive by the same order as the gap but less than it
    (+16.2% / −9.4% CT across dns' blend band); at most about half of the
    gap can be the zone. The refinable two-blade `rotor2` (W7) is NOT AT
    PARITY either, and its dns ladder does not approach foam in CT: L3 is
    +67% above rhoSimpleFoam (Richardson +59%), and CQ crosses foam (−1.7% at
    L3) heading for −7.3% (Richardson). The W6 split ATTRIBUTES the CT gap
    to the BLADE (106–114% of it on every level of both rotors; the hub
    agrees with foam within 6–22% and carries slightly LESS drag), and
    shows that the CQ "agreement" is a cancellation: a convective momentum
    flux through the moving staircase wall (zero on a real wall) is +6.7 to
    +24% of CQ; CQ − CQ_c is 8–15% LOW on every level, so the total CQ is
    EXPECTED (an extrapolation, not a measurement) to fall to about −8.5% as
    that flux vanishes.**
    Measured 2026-10-27 by
    `tools/parity` (`runs/parity/report.md`), laminar Re_D 220, J 0.63, tip
    Mach 0.41, central U everywhere (the parity rule). The pre-W5 rows used
    linearUpwind and are retired as stale.
    - **(A) FIXED 2026-10-27: two-sided ghost faces** (exp_mrf14b,
      `plans/verify/impl_mrfA.md`). Where a section is under ~2 cells thick
      a ghost is face-adjacent to fluid on both surfaces but carried ONE
      image; the far-side fluid cell fluxed against the other surface's p and
      mirrored u, and the tally read that face. Now each such face (ghost
      normal away from the fluid cell AND fluid normal > 60° off it) gets a
      SIDE image about the fluid cell's own surface (`CompGhostTables::side*`,
      `kSideRhs` via `DeviceCompGhost::applyRhs` in the rhs hook, `kSideTally`). Always on;
      bodies without such faces are byte-identical (Re40/60/100 n256 comp
      cylinders, a spinning MRF and a moving-table sphere, the t/c 0.35
      ellipse). Test `gpu_comp_ghost_twoside` (membranes holding a pressure
      jump at rest: pre-fix face p off by 0.5 dp, force 11-18%, flow
      0.6-1.06 dp/(rho c); fixed 2e-6 / 4e-7 / 7e-6).
      rotor2 (4.5 D/U, final; Bc L3 CT −0.01729, CQ Richardson 0.08351):

      | | L1 | L2 | L3 | Richardson |
      |---|---|---|---|---|
      | CT before | −0.05223 (+202%) | −0.03180 (+84%) | −0.02891 (+67%) | −0.02737 (+58%) |
      | CT after | −0.04625 (+167%) | −0.03015 (+74%) | −0.02750 (+59%) | −0.02584 (+49%, p 3.9) |
      | CQ before | 0.09167 (+9.8%) | 0.08465 (+1.4%) | 0.08208 (−1.7%) | |
      | CQ after | 0.08790 (+5.3%) | 0.08658 (+3.7%) | 0.08330 (−0.25%) | not monotone |
      | CQ_c before / after | 0.02018 / 0.01298 | 0.00789 / 0.00734 | 0.00560 / 0.00353 | |
      | blade CT_c before / after | −0.00574 / −0.00149 | −0.00263 / −0.00135 | −0.00147 / −0.00018 | |

      L3 blade p +0.02297 → +0.02586 (foam +0.04840), v −0.04198 → −0.04432
      (foam −0.05613); CQ − CQ_c −8.6% → −4.5%. Chord frame, L3 blade:
      normal pressure CT 0.06334 → 0.06648 (foam 0.08318: −23.9% → −20.1%),
      chordwise pressure −0.04254 → −0.04347 (foam −0.03975). Sections
      (sect14, vs foam L3): L2 (y .54) Cn −27.6 → −6.5%, Cc −10.0 → −2.7%;
      L1 Cn +12.8 → +35.2%, Cc −12.9 → −3.3%; y .24 Cn −81.3 → −83.0%, Cc
      +1.4 → +1.8%. The prototype's y .24 gain (−24%) came from a 5th face per
      section that is NOT two-sided (ghost 0.06 h inside, fluid upstream at
      the same height, n_f · n_g 0.75): re-imaging a near-tangent face moves
      Cn by ~60 points, which is (B), not (A). Not covered: the RANS
      turbulence scalars' fluxes at side faces still read the shared ghost
      (mean flow and mu_t use the side image; the run warns); 2–7 wrong-side
      faces per rotor2 level have no usable sample on their side and stay
      on the shared image (counted and warned).
    - **(B) scatter: FIXED 2026-10-27 by impermeable interface faces
      (`plans/verify/impl_imperm.md`; exp_mrfB, `plans/verify/exp_mrfB.md`;
      bias resolved by exp_mrfC, `plans/verify/exp_mrfC.md`). rotor2 is
      still NOT at parity (L3 CT +47%, CQ -5%).**
      On a smooth t/c 0.35 ellipse vs body-fitted foam, Cn has a placement
      SCATTER (±30 / ±9 / ±0.7% at c/h 20 / 28 / 40).
      - **The "−7…−9% bias" was the REFERENCE's (exp_mrfC).** foam's section
        branch ran `limited corrected 0.33` (laplacian + snGrad) on an O-H
        mesh at up to 53° non-orthogonality: +9.3% Cn, Cc −0.3%, and it does
        not refine away. Unlimited, foam is Cn 0.2432 / Cc 0.3851 (L3; L2
        0.2437 / 0.3854). Against that, the dns offset means are +3.6 / +0.4
        / −0.0% at c/h 20 / 28 / 40 (c/h 56 +1.9%, one offset), and the
        impermeable prototype gives +1…+3% with Cc −0.3…0%. Γ(R) falls
        past R 2 c in foam too (wake), the incompressible cut-cell run
        (tunnel BCs, 24 c tall box) is +1.6%, M 0.1 moves Cn −0.6%, and the
        inflow angle and body agree. Fixed in `tools/parity/suite.py`
        (section `limited = False`). **Every section row scored against
        limited foam is stale**, including (A)'s sect14 Cn/Cc figures
        (sect14 re-scored in (D)) (the
        stored `runs/parity/{A,B}/sect` and `ref/*/sect` foam runs are still
        the limited ones, 69–72° max non-orthogonality: regenerate). The
        rotor branch is now unlimited too, and on rotor2 the limiter turned
        out NOT to be the gap (exp_rotorU, bullet (C) below). Side finding (incompressible, FIXED 2026-10-27,
        #30 below): `gpu::bodyBox` clamped k to [2, nz−3] (the interior is
        1..nz−2), and a mesh body overhanging the z box is not `geo.zSpans`
        (`apps/dns_main.cpp:263`), so its cd/cl missed two layers (×0.5 at
        nz 4).
      - It is the FLOW, not the tally: the far-field circulation tracks the
        tally lift across offsets to 0.003.
      - Pressure image: sample ±0.5 h moves Cn ≤ 0.004; a centripetal
        image +0.004 everywhere.
      - Velocity image: its wall position is worth 0.03 Cn per h.
      - **The scatter is transpiration through the staircase interface
        faces.** The KG face mean with the odd linear image does not
        zero u_n (∝ s² at a no-slip wall), so the body is a net source of
        0.1–1.1% ρUc depending on the offset, concentrated at the LE.
        This is the CT_c / CQ_c "flux through a staircase wall" below.
      - A prototype that zeroes the interface faces' advective flux and
        pressure work (resting Cartesian only, removed again) cuts the
        scatter at c/h 28 to ±1% and puts Cc on foam (+0.0…+0.3%, against
        −1.4…−3.1%); its Cn is +1…+3% on the corrected foam.
      - **LANDED 2026-10-27: impermeable interface faces**
        (`plans/verify/impl_imperm.md`, `DeviceCompGhost::applyRhs`,
        test `gpu_comp_ghost_imperm`). Rest: no advective flux, no
        pressure work; MRF: none relative to the frame, pressure work
        p w_n off the same corner stream potential; moving: rho
        u_wall.n {1, u, e} + p u_wall.n with u_wall at the face centre;
        Cartesian and focal; side faces end on the side image's
        impermeable flux; the tally's fc is exactly 0 at rest and under
        MRF. Fluid mass through a resting/MRF body: 4e-4..1e-5 leak ->
        FP32 rounding (3e-8..1.5e-7 over 50 steps). Measured (HEAD-state
        -> new; refs body-fitted):
        - ellipse c/h 28 (6 offsets) vs unlimited foam Cn 0.2432 / Cc
          0.3851: Cn +0.5% sd 9.0% -> +1.5% sd 2.9%; Cc -2.1% -> -0.0%.
          c/h 20: Cn +3.6% sd 30% -> +4.0% sd 6.3%, Cc -3.2 -> -0.6%.
        - square Re 40 M 0.1 vs simpleFoam 2.184: D/h 24 -0.25% ->
          -0.07/+0.25% (two offsets), D/h 32 -0.88% -> -0.49/-0.33%.
        - rotor2 MRF vs Bc unlimited (CT -0.017090, CQ 0.083313), 4.5
          D/U final: CT +170.6/+77.3/+65.1% -> +120.4/+53.6/+46.9%
          (L1/L2/L3); CQ +5.5/+2.5/-0.3% -> -11.3/-5.6/-5.1%. The CQ
          "agreement" was the cancellation this issue predicted: CQ -
          CQ_c was -10.1/-4.9/-4.3% and the convective part is now 0.
          L3 blade p +0.02477 -> +0.02875 (foam +0.04837), v -0.04432 ->
          -0.04498 (foam -0.05592).
        - Re 60 sponge cylinder (`comp_re60_n256_cd`): the steady state
          was a slow transient in BOTH; it sheds (St 0.174) from t ~ 100
          (new) / ~150 (HEAD state). Window cd 1.9441 -> 1.9522, a
          spurious mean cl -0.0139 on the symmetric cylinder -> 0. The
          +6.9% cross-branch gap is NOT closed (+7.3%).
        - Re 40 cylinder vs body-fitted O-grid: see impl_imperm.md (the
          spurious cl halves; cd within the domain-mismatch band).
      - The rotor2 CT gap after (A) is still +61% at L3 against the
        unlimited reference, carried by the blade's normal pressure force
        (−19.7% vs foam; the limiter was worth 0.4% of it, see (C)).
    - **(C) MEASURED 2026-10-27: the foam limiter is NOT the rotor2 gap**
      (`plans/verify/exp_rotorU.md`). snappy rotor2 meshes: max
      non-orthogonality 65°, but only 0.1–0.3% of faces are above 40°;
      blade first-layer faces mean 10°, layers 2–3 ≤ 36°; the 3–9
      "highly skew" faces are all propeller wall faces. foam restarted
      from the converged limited state with `Gauss linear corrected` /
      `corrected` / `grad(U) Gauss linear` (nNonOrth 1, stable, flat to
      ≤ 5e-5 within 200 iterations; a cold Bc L1 matches to 6 digits).
      |CT| falls 2.2 / 1.5 / 1.2% (Bc L1/L2/L3; B 2.3 / 1.5 / 1.3%) and
      CQ 0.2–0.3%, mostly viscous (blade v −0.05613 → −0.05592, blade p
      0.04840 → 0.04837). The new Bc reference is CT −0.017090 (fine L3)
      and CQ 0.083313 (Richardson). dns post-(A) against it:

      | | L1 | L2 | L3 | Richardson (h_fine,max) |
      |---|---|---|---|---|
      | CT vs Bc unlimited (limited) | +170.6% (+167.4) | +76.4% (+74.4) | +60.9% (+59.0) | +52.2% (+50.4) |
      | CQ vs Bc unlimited (limited) | +5.50% (+5.26) | +3.92% (+3.68) | −0.02% (−0.25) | no fit |
      | blade chord-normal p vs Bc L3 (foam 0.08282, was 0.08318) | −39.4% | −24.2% | −19.7% (−20.1) | dns 0.0710 vs foam 0.0831: −14.6% |
      | blade chordwise p vs Bc L3 (foam −0.03947, was −0.03975) | +29.7% | +8.3% | +10.1% | |

      Still NOT AT PARITY. The 1.2–2.3% CT bias is a quarter of the CT
      band, so `tools/parity/suite.py`'s rotor branch is now `limited =
      False`. The references are `runs/parity/{B,Bc}/rotor2/foam_L*_unlim`.
      The old limited dirs stay in place, marked `stale` in their
      result.json, and `split/rotor2/foam_*_L3_limited.json` keeps the old
      split. The stored **rotor** (rotor1) B/Bc foam runs and its zone
      calibrations are still limited, on 65° snappy meshes; they are not
      rerun, and the expected effect is the same order (1–2% CT).
    - **(D) MEASURED 2026-10-27: lens re-referenced, and the rotor2 gap
      localized** (`plans/verify/exp_rotorR.md`).
      - **sect14 lens vs UNLIMITED foam.**
        - foam (tier A, unlimited) gives Cn 0.0932 / Cc 0.3765 (L3, restarted
          from the limited state, 32–42 c/U). L2 cold is 0.0929 / 0.3771. The
          limited 0.1015 was +8.9% off.
        - Unlimited L3 carries a 2Δt odd-even cl mode (±0.07); its block means
          are stable and agree with the clean L2.
        - dns at HEAD + (A), 100 c/U, mean over offsets: Cn **+43% / −15% /
          −5%** at c/h 20 / 28 / 40 (3 / 6 / 3 offsets). The SE is 14–32%, so
          none of the three is distinguishable from zero. Cc is −3.9 / −0.5 /
          −1.0%.
        - The absolute Cn scatter is sd 0.052 / 0.032 / 0.022. At c/h 20–28
          that is the ellipse's (0.075 / 0.022). Unlike the ellipse (0.002 at
          c/h 40), it does NOT collapse by c/h 40: the sharp edges keep ±20%.
        - y .24 (near-tangent face) is −81%.
        - The (A) sect14 figures above are superseded by these.
      - **rotor2 radial bins (0.05 D, L3, chord frame, total =
        p + v + convective).**
        - The blade CT gap (−0.0111) sits MID-TO-OUTER SPAN: 0.20–0.45 D
          carries 87% of it.
        - Root, hub (dns −7% less drag, against the gap) and tip bin (8%) are
          not it.
        - Per bin it is a near-UNIFORM TOTAL chord-normal force deficit:
          −6…−15% in every bin 0.10–0.50 D, local ΔCn −0.006…−0.043, all
          negative. Chordwise is within ±7%.
        - Blade totals: normal −19.4 / −10.0 / −9.4%, chordwise +11.3 / +4.8 /
          +1.9% (L1 / L2 / L3). The normal deficit is FLAT from L2 to L3.
        - The "−19.7% chord-normal pressure" is the staircase's p/v split:
          dns puts 2.5× foam's normal force into v_n.
        - Blade CT is a residual of normal (+0.097) against chordwise (−0.105)
          thrust. So a 9.4% normal-force deficit is 120% of blade CT, and that
          is the +61%.
      - **Ruled out.**
        - **t/h:** L3 has t/h 7.1–9.2 at every radius, and L2's deficit
          (t/h 5.4–7.1) is the same.
        - **Zone:** dns f = 1/2 at r 0.629 D, ±0.297 D axial, equals foam's
          sharp r 0.629 D, ±0.2965 D. The blend starts 0.054 D outside the
          tip. The STL is identical vertex-for-vertex.
        - **Leak:** the convective CT is ≤ 1.6e-4 per bin. CQ_c is 42% in
          the tip bin: that is torque, not thrust.
        - **Two-sided faces:** they cluster at the root and tip (68 / 70 per
          bin) against 25–28 mid-span.
        - **2D placement scatter:** pooled c/h 20 + 28 lens ΔCn is +0.004 ±
          0.015, against the rotor bins' −0.026 mean.
      - **Open.** A systematic ~9–10% normal-force deficit of the ROTATING 3D
        section.
        - Note: foam's local Cn at r 0.325 D is 0.238, against 0.093 for the
          2D lens at 7° (rotational augmentation), so the 2D section is not a
          proxy for it.
        - Next: rotor2 `--rotor-frame moving` vs mrf (done, (E)); a
          non-rotating 3D straight lens wing at the rotor's c/h, against
          foam (not run).
    - **(E) MEASURED 2026-10-27: discriminators** (`plans/verify/exp_rotorM.md`).
      - **Settling: ruled out.** MRF L2 to 9 D/U is flat from 4 D/U on:
        1-D/U block means CT −0.03022…−0.03030, CQ 0.08648…0.08651. Blade
        total normal −10.0 → −10.3%. The quoted numbers are final snapshots,
        but they are within 0.7% of the window means.
      - **Frame: dns MRF ≠ dns moving, by more than the gap, and they
        bracket foam.**
        - Moving (natural bins) time-mean CT is +0.0134 / +0.0090 /
          −0.0009 ± 0.0011 at L1/L2/L3. MRF is −0.0463 / −0.0303 / −0.0273,
          and foam −0.0171.
        - The difference is 0.060 / 0.039 / 0.027 (1.6–3.5× foam's |CT|) and
          shrinks at observed order ~1–1.5.
        - Moving CQ is −15 / −11 / −5%. Without the convective wall flux,
          torque agrees across frames (L2 CQ − CQ_c −4.9% MRF, −5.3% moving).
        - The split is in the normal force. Blade normal p+v: MRF −30.6 /
          −16.6 / −13.1%; moving −4…−12 / −7.6 / +5.9% (snapshots, ±8%).
          Moving chordwise is −7…−12%, MRF's −1…−3%.
        - So the frozen-rotor equivalence fails inside dns at every level.
          The rotor2 numbers are dominated by an UNCONVERGED frame-dependent
          error of the immersed moving wall, and the flat L2 → L3 MRF normal
          deficit is not a converged bias.
        - The MRF path carries a normal deficit that moving does not, but
          moving errs as far the other way. It has its own first-order terms:
          a convective tally of +0.005…+0.034 CT from the staircase sweeping
          the grid, and swap noise.
        - p = 1.5 L2/L3 extrapolations put both near CT −0.020 (inside
          foam's zone span). That is an extrapolation, not a measurement.
      - **Box, BCs, Re, Mach, gas: matched.**
        - Box faces agree to 1e-6 D, with slip / free-slip laterals.
        - NSCBC inlet/outlet vs fixed U / fixed p; the sink is off on
          `--focal`.
        - Re_D 220, M 0.082219, tip M 0.41, J 0.63, γ 1.4, Pr 0.72, constant
          μ, adiabatic walls.
        - Nothing changed in `tools/parity`.
      - **Zone, blend vs sharp: ≤ 1/3 of the gap.** foam Bc L2 with the zone
        at dns' f = 1 / f = 0 radii (0.554 / 0.704 D) gives CT −0.01997 /
        −0.01639 (+13.6 / −6.8% vs 0.629), CQ −2.6 / +0.8%. dns MRF L3
        −0.0273 is outside that span.
      - **Side finding, RESOLVED 2026-10-27 (`plans/verify/exp_corner.md`):
        thick convex corners are excluded.** Fix A's two tests also fire at
        a sharp convex edge of a THICK body (one face per corner cell row:
        square 4 per layer, diamond 2, spun cube 224 / 138).
        - Measured on a square cylinder (B 10%, 2 sub-cell offsets per D/h)
          against body-fitted simpleFoam (central U, cd 2.184):
          - Re 40: the side image is locally better. Corner-face Cp is
            symmetric, independent of the offset and closer to foam. cd at
            M 0.1 is −0.3 / −0.9% for HEAD and +0.3 / +0.1% for fix A.
          - Re 100: it makes the result offset-sensitive. cd spread 0.7% →
            8.7%, cl_rms spread 3.6% → 48% over a 0.2-cell shift (one offset
            1.80 → 1.96).
        - Now a face stays shared when its ghost is in thick solid: the sdf
          stays negative 2 cells behind it along −n_g, AND a deep cell
          (phi < −2h) lies within 3 cells. Each condition alone also caught
          thin-blade faces (rotor2 L1 186 → 142).
        - Results: thick bodies, square / diamond / cube included, classify 0
          side faces and are byte-identical to HEAD. Pinned in
          `gpu_comp_ghost_twoside` (`thickCorners`).
        - Lens L2 / L1 / y .24 are byte-identical to fix A, and so is rotor2
          L1 (186 faces; `gpu_mrf_rotor2` not re-pinned).
        - rotor2 L2 / L3 lose their blade-TIP edge faces (345 → 261, 456 →
          357; the outer blade is > 4 cells there). CT moves L2 −0.03015 →
          −0.03030 and L3 −0.02750 → −0.02821, i.e. +63.2% vs Bc: half of
          fix A's L3 CT gain is given back. CQ L2 0.08541, L3 0.08303. The
          (A) table above is fix A before this amendment.
        - Open: tip corners of blades under 4 cells keep their side images
          (no deep cell near), and the Re 100 sensitivity is not measured
          there.
      - **Next, none run:**
        - an L4 (over the 300 s cap);
        - an MRF-vs-moving rig where the frozen rotor is exact AND the wall
          has a normal velocity (`gpu_mrf` frame parity uses an axisymmetric
          cylinder, where (ω×r)·n = 0, so it never tests that);
        - a time-averaged `--emit surface` for moving runs.
    - **dns (rotor).** L1 is the 80f147f run (4.58 D/U, D/h_fine 59.5, 4.1M cells,
      131 s). L2/L3 are not run: they cost ~550 and ~2100 s, over the cap.
      rotor1 cannot be grid-refined under 300 s, which is why W7 makes
      rotor2.
    - **rotor2 (W7)**: `--blades 2 --root-chord 0.40 --tip-chord 0.30
      --thickness 0.35 --pitch 1.0`, same flow and box; 4.5 D/U per run.
      Verdict bands fixed before measuring (`cases.toml` `parity_bands`):
      CT 5%, CQ 2% against Bc, on the finest level or its Richardson value
      when the ladder is monotone.

      | | D/h_fine,x | BL cells | cells | wall s | CT | CQ |
      |---|---|---|---|---|---|---|
      | dns L1 (ny 100, ratio 5) | 39.1 | 2.6 | 1.50M | 35 | −0.05223 | 0.09167 |
      | dns L2 (ny 132, ratio 7) | 58.5 | 3.9 | 3.45M | 113 | −0.03180 | 0.08465 |
      | dns L3 (ny 164, ratio 8) | 74.9 | 5.0 | 6.62M | 260 | −0.02891 | 0.08208 |
      | dns Richardson (h_fine,max) | | | | | −0.02756 (p 4.3) | 0.07743 (p 1.7) |
      | Bc L1 / L2 / L3 | | | 115k / 145k / 321k | 50 / 67 / 181 | −0.02065 / −0.01785 / −0.01729 | 0.08502 / 0.08406 / 0.08371 |
      | B L1 / L2 / L3 | | | same | 26 / 34 / 89 | −0.01783 / −0.01514 / −0.01460 | 0.08426 / 0.08322 / 0.08285 |

      Against Bc (CT fine L3, CQ Richardson 0.08351): dns CT +202 / +84 /
      +67%, CQ +9.8 / +1.4 / −1.7%. Two acceptance items fail and are
      recorded, not re-banded: foam L2/L3 CT agree to 3.2% (Bc) / 3.7% (B),
      not 1% (CQ 0.4%; foam's CT Richardson order is 4.7 and puts L3 0.8%
      from its limit); and the verdict. Confounds, both measured smaller
      than the CT gap: dns' zone margins are in h_fine cells, so its zone
      shrinks with refinement (f = 1/2 at 0.694 / 0.646 / 0.629 D; foam's
      is set to L3's), and foam moved from L3's to L1's radius shifts CT
      −5.5%, CQ +0.6%; and 4.5 D/U is not settled to 1e-3 (L3 CT ±2.6%
      oscillation, mean drift ≤ 1% over the last D/U). dns L1 sits at
      t/h 3.6–4.4 with 12 deep-solid cells. The CT gap shrinks with D/h
      (+202 → +67%) but extrapolates to +59%, so it is not a resolution
      artefact of the ladder alone; the W6 split below attributes it. The
      bladed L1 is pinned in ctest as `gpu_mrf_rotor2` (1 D/U, ct/cq and the
      split's pressure / convective / viscous CT at 1%, ~11 s).
      **Verdict after the W6 split: still NOT AT PARITY** (CT +67% at L3,
      CQ −1.9% at L3, with CQ − CQ_c at −8.6%, i.e. the total expected to
      fall toward −8.5% as the convective wall flux vanishes -- an
      extrapolation; the bands are unchanged).
    - **W6 split (2026-10-27): `--emit surface` on the immersed ghost wall
      writes the tally per interface face, split into pressure p s,
      convective ρ u_a,n u and viscous; `tools/parity/split_dns.py` sorts
      faces into hub (r ≤ r_hub + T, |x| ≤ hub half-length + T) and blade
      for dns AND for the per-face forces of the converged foam L3, with ONE
      T per case (the coarsest dns level's √median face area: 0.0160 D
      rotor, 0.0267 D rotor2).** dns runs at 4.5 D/U, final state; foam Bc
      (rhoSimpleFoam) L3, the scored reference (B within 0.01 on the hub).
      Hub-only runs (masterplan step 5) were not triggered: the dns hub is
      within 30% of foam's.

      | | CT total | hub | blade | blade p | blade c | blade v | CQ | CQ_c |
      |---|---|---|---|---|---|---|---|---|
      | rotor dns L1 | −0.02390 | −0.00843 | −0.01547 | −0.00179 | −0.00065 | −0.01303 | 0.03205 | 0.00316 |
      | rotor foam Bc L3 | −0.01865 | −0.00905 | −0.00960 | +0.00890 | 0 | −0.01850 | 0.03218 | 0 |
      | rotor2 dns L1 | −0.05221 | −0.00747 | −0.04474 | −0.00488 | −0.00574 | −0.03412 | 0.09168 | 0.02018 |
      | rotor2 dns L2 | −0.03188 | −0.00752 | −0.02437 | +0.01886 | −0.00263 | −0.04059 | 0.08472 | 0.00789 |
      | rotor2 dns L3 | −0.02900 | −0.00851 | −0.02048 | +0.02297 | −0.00147 | −0.04198 | 0.08215 | 0.00560 |
      | rotor2 foam Bc L3 | −0.01729 | −0.00956 | −0.00774 | +0.04840 | 0 | −0.05613 | 0.08371 | 0 |

      Share of the CT gap (dns − Bc): rotor L1 −0.00525 = hub −12% + blade
      +112%; rotor2 L1/L2/L3 −0.03491 / −0.01459 / −0.01170 = hub −6 / −14
      / −9% + blade +106 / +114 / +109%. Robust to T (0.5–1.5×: rotor hub
      −0.0079..−0.0087 dns vs −0.0087..−0.0092 foam). Within the blade the
      pressure deficit is about twice the gap (rotor2 L3 −0.0254) and is
      partly offset by LESS viscous drag (+0.0142); both shrink with D/h
      (blade p −0.0049 → +0.0189 → +0.0230 vs +0.0484), but on a staircase
      the p/v division of a sloped surface's stress is not the body-fitted
      one, so only the per-part totals are a like-for-like comparison. The
      convective part is a discretization error of the immersed moving wall
      (the mirror's residual normal mass flux, carrying the wall-speed
      momentum through the face; foam's is exactly 0). It decays between
      first and second order (rotor2 CQ_c 0.0202 / 0.0079 / 0.0056 at D/h
      39.1 / 58.5 / 74.9: observed order 2.3 from L1 to L2, 1.4 from L2 to
      L3) and is 12–18% of the CT gap. CQ − CQ_c against Bc is −14.6 / −8.2
      / −8.6% (rotor2) and −10.2% (rotor L1); it is flat over L2 and L3, so
      the total CQ is EXPECTED to tend to about −8.5% as CQ_c goes to 0.
      That is an extrapolation (it assumes CQ − CQ_c stays flat while CQ_c
      vanishes), not a measured number -- the measured parity figure at L3
      is −1.9% -- but it says the CQ ladder's apparent convergence to foam
      is the artefact shrinking. Radially (the split's
      0.05 D bins, `split_dns.py` prints them) the CT gap sits MID-SPAN: dns
      is more negative by 0.0009–0.0014 per bin at r 0.15–0.35 D (rotor L1)
      and 0.0017–0.0033 at 0.2–0.4 D (rotor2 L3), small at the root and of
      either sign at the tip; the torque deficit (without the convective
      part) sits at the TIP bin 0.45–0.5 D: −27% (rotor), −18% (rotor2 L3),
      within a few % inboard. Next attribution steps, none built: (a) the
      section force direction vs radius against foam (a lower effective
      pitch or a thicker effective staircase section both tilt it this way);
      (b) a sub-cell wall treatment that zeroes the interface mass flux on
      a moving wall; (c) `--mrf-zone` (zone confound). Recipe: `dns flow
      ... --emit stats --emit surface` (the rotor argv), then
      `python3 tools/parity/split_dns.py dns CASE RUN_surface.csv --tol T
      --json runs/parity/split/CASE/dns_Lk.json` and, inside gpurun, `...
      foam CASE runs/parity/Bc/CASE/foam_L3 --work DIR --tol T --json
      runs/parity/split/CASE/foam_Bc_L3.json`; `suite.py report` renders
      the tables.
    - **simpleFoam (tier B, M → 0).** L1/L2/L3 CT −0.01883 / −0.01788 /
      −0.01790, CQ 0.03222 / 0.03169 / 0.03171. These are the audit's
      central runs, reproduced to ≤ 0.02%.
    - **rhoSimpleFoam (new tier Bc, at the dns Mach).** L1/L2/L3 CT
      −0.01959 / −0.01862 / −0.01865, CQ 0.03265 / 0.03214 / 0.03218. L2
      and L3 agree to 0.18% / 0.14%. The compressibility increment is Bc/B
      − 1 = +4.2% CT and +1.5% CQ. Bc needs classic SIMPLE: SIMPLEC
      limit-cycles on L3.
    - **Zone sensitivity: |ΔCT| > 2%, so the masterplan's `--mrf-zone`
      follow-up (§5/§6) is triggered.** It is not built. On central
      simpleFoam L2, refinement fixed, the sharp zone moved to dns' f = 1
      radius (0.555 D) gives CT +16.2% and CQ −2.6%. Moved to the f = 0
      radius (0.705 D) it gives CT −9.4% and CQ +1.1%. The single-blade
      frozen rotor is not zone-independent. dns' blended zone (0.555–0.705
      D) and foam's sharp 0.62 D zone are different approximations. But
      the zone shift is smaller than the gap: even with the foam zone at
      dns' f = 1 radius (0.555 D, the setting with the largest foam CT),
      dns L1 is still +15.1% above simpleFoam L2 (−0.0239167 / −0.0207808),
      about +10.6% est. against Bc (scaled by Bc/B at L2). So at least
      about half of the CT gap (log share 52% vs B L2) is not the zone,
      and W6 attribution stays needed. CQ is robust. Ways to close the
      zone part, none built: (a) dns `--mrf-zone` to set the
      zone radius and blend; (b) a zone-independent reference
      (time-accurate rotating mesh / AMI, or `--rotor-frame moving` against
      it).
    - **Recipe** (each run ≤ 300 s): `gpurun -t 300 -m 16G -- env
      PARITY_LOCK_HELD=1 python3 tools/parity/suite.py run rotor --tier
      B|Bc --solver foam --level k` and `... --tier B --level 2 --zone-r R
      --tag zrR`; rotor2 dns `... run rotor2 --tier A --solver dns --level
      k` (`-m 14G` at L3).

## URANS time accuracy (masterplan `plans/verify/masterplan_verify.md` W2)

20. **Found and given an opt-in fix 2026-10-27: the plain FP32 time march
    does not converge in dt -- FP32 update stagnation, the steady march's
    floor ulp(q)/(2 dt) on a time march.** O-grid cylinder SST-2003m URANS
    Re 1e4 (256×96, beta 4, far 20, `--ramp 1 --ft 0.04`), RMS over
    t ∈ [0.05, 0.6] of cd between adjacent cfl rungs 0.4/0.2/0.1/0.05:
    plain 6.5e-4 / 3.0e-3 / 1.03e-2 (DIVERGES: ratios 0.22 / 0.29),
    `--compensated` 2.9e-4 / 7.8e-5 / 2.4e-5 (ratios 3.75 / 3.30, order
    ~1.8; cdf ratios 2.01 / 1.99 = the first-order source quotient). The
    windows [0.6, 1.3] and [1.3, 2.2] agree. Plain final cd 1.7261 →
    1.7288 scattered over 1.726–1.742, compensated 1.7255 → 1.7254; plain
    max |cl| 1e-3 → 4.7e-3 off a symmetric start, compensated < 2e-6.
    `--compensated` (flow compressible incl. O-grid/focal/MRF/rotor, and
    channel) is built and OFF by default, so every default run is bit for
    bit (protocol P + 6 time-march cases cmp-identical, docs/rans.md "Time
    accuracy"); 1.16–1.21× per step on the gate case. **Open:** (a) whether
    it becomes the URANS default (it changes bits) is a user decision
    (masterplan §5 item 2); (b) every plain FP32 URANS/DNS dt study cited so
    far is suspect where |R| dt nears ulp(q)/2 -- none of the measured
    anchors is a dt study. W12's URANS shedding against OpenFOAM (item 24)
    ran `--compensated` with a converging cfl ladder; the plain state on
    the same case moves the late-window means by ≤ 0.5% (≤ 1.9 SE).

## RANS code-to-code references (masterplan `plans/verify/masterplan_verify.md` W13)

21. **SA wedge L1 parity −1.07% is L1 discretization: settled 2026-10-27,
    L2 MEASURED** (`plans/verify/exp_sa21.md`, `plans/verify/impl_batch.md`).
    Tier B against OpenFOAM v2606 SpalartAllmaras (ft2 true, Prt 0.9,
    central U) on the identical β 4.5 rFar 20 mesh, Re 1e3:
    - L1 64×48: cd0 0.47901 vs 0.48419, **−1.07%** (cdp −0.86, cdf −1.63%),
      L_r +7.8%.
    - L2 128×96: dns SA-neg M 0.2 0.494552 (exit 0, 119600 it), M 0.1
      0.489505 (explicit, cap 1.2e6 it, exit 8: the jitter-floored stop
      cannot certify M 0.1, item 4; five 2e5-it block means over 2e5–1.2e6
      agree to 2.5e-6, so the state is settled). cd0 **0.487822 vs
      0.488432: −0.125%** (cdp0 −0.06%, cdf0 −0.29%). The estimate it
      replaces was −0.08% (L1's Mach ratio 1.01336; L2 measures 1.01380).
      θ_s 100.41° (M 0.1) vs 100.56°; L_r0 1.429 vs 1.3745 D (+4.0%; the
      estimate said ~+2%).
    - Laminar Re 100, no model, same mesh: −0.49% L1 → −0.10% L2, friction
      led like SA's (cdf −0.81 → −0.21%). OpenFOAM's S̃ limiter is not the
      cause (Cs 0.3 → 0.1 moves its cd −0.006%).
    - dns's L1 → L2 cd0 move (+1.84%) is twice OpenFOAM's (+0.88%): dns
      carries the larger L1 error on this growth-1.22 mesh.
    - The implicit scheme does not reach M 0.1 here (L1 exit 7 "CFL < 0.1
      after 37 rejections"; L2 exit 8 at 500, cd still drifting).
    `gpu_sa_wedge` keeps pinning the dns L1 state, ungated against
    OpenFOAM; a 1% gate belongs at L2, where the pair now sits inside it.
    Explained, not open.
22. **The TMR flat plate, bump-in-channel and NACA 0012 cases cannot be
    expressed today** (recorded 2026-10-27, not scheduled). The O-grid is
    uniform in polar angle (`ogrid.hpp`: theta_i = −2π i/nx), so no
    arc-length clustering at a sharp trailing edge or a plate leading edge,
    and `--rans` rejects box walls (`--body floor|fence|none`, `--noslip`:
    "need the ghost-row adapter, which is not built"). Needs an arc-length
    O-grid and the wall-row adapter -- a feature (masterplan §6), not a
    verification step. The RANS code-to-code references today are the
    CFL3D 2DFDC channel (`rans_channel1d` O5, SA within 0.5% sublayer-
    normalized) and the OpenFOAM wedge (items 7, 21).
23. **A supersonic `--mach` is now accepted on the implicit steady RANS
    channel only** (2026-10-27, for the Coleman-Kim-Moser M_b 1.5
    scenario, docs/rans.md). Everywhere else the (0, 1) bound stands
    (the KG flux has no shock capturing). The explicit steady march and
    URANS stay rejected at M ≥ 1: their pseudo-time / time transients
    carry y-acoustics that were not measured there. The channel has no
    viscosity law (constant μ against CKM's μ ∝ T^0.7), so the scenario is
    a model-label comparison: Tc/Tw 1.382 vs CKM ≈ 1.38.

## O-grid cylinder (masterplan `plans/verify/masterplan_verify.md` W12)

24. **URANS SST shedding vs OpenFOAM: the −4.2 SE cl_rms reading was
    window noise; the residual is OpenFOAM's far-field radius dependence,
    not dns's -- dns is radius-converged at 40 D, OpenFOAM is not at 80 D**
    (measured 2026-10-27, plans/verify/exp_sst24.md + exp_sst40.md; 2D
    cylinder O-grid 256 angular, Re 1e4, M 0.2, `--ogrid-trip 0.05
    --compensated --cfl 0.6`; OpenFOAM v2606 rhoPimpleFoam on the identical
    mesh, central U, backward, nOuter 2, np 6 meshWave,
    `tools/parity/ocyl1e4.py`). (a) Neither W12 series (the old −0.74%
    / −4.2 SE, t ≥ 45 D/U) was saturated: dns cl_rms 1.1994 (45-66) ->
    1.2061 ± 0.0006 (75-128, 12 periods); OpenFOAM as configured
    (freestreamPressure) never becomes stationary (per-period cd sd
    0.013-0.026). Do not cite a URANS window under ~10 saturated periods.
    (b) OpenFOAM's scatter and even harmonics are its reflecting far p:
    `waveTransmissive` (gamma 1.4, lInf = R, U freestreamVelocity kept)
    gives a clean limit cycle -- the OF reference setting. (c) Far-field
    ladder, both codes, matched near-body cells (ny/beta 96/4, 105/4.345,
    114/4.685 at R = 20/40/80 D: cell sizes within 1.2-2.5% over r < 2 D),
    dns − OF: cd −1.10% / −0.59% / +0.77%, St −1.76% / −0.83% / +0.92%,
    cl_rms −0.32% / +0.53% / +0.77% -- the gap CHANGES SIGN, it does not
    close. dns 20->40->80: cd 1.6923 / 1.6759 / 1.6782, St 0.2282 / 0.2281
    / 0.2293, cl_rms 1.2061 / 1.1947 / 1.1944 (moves ≤ 0.5% past 40 D).
    OpenFOAM: cd 1.7111 / 1.6859 / 1.6653, St 0.2323 / 0.2300 / 0.2272,
    cl_rms 1.2100 / 1.1884 / 1.1853 (still ~1.2% per doubling in cd, St).
    OF lInf 40 -> 20 at R 40 moves ≤ 0.1%, so it is not the transmissive
    anchoring. OpenFOAM at 80 D grows a symmetry-breaking mode from ~70
    D/U (cl_mean -> 0.03, a2/a1 -> 0.018, cd@f ×1.1 per period),
    identical at lInf 20; dns at 80 D has none. Not separated: the tanh
    law coarsens the mid/outer field with R (cell at 20 D: 0.83 / 1.23 /
    1.43 D), so OF's drift is radius OR outer resolution; dns is
    insensitive to both together. dns on the 40/80 D discs carries a slow
    13.8-16 D/U cd modulation (cd sd 0.004-0.010 vs 0.0025 at 20 D;
    period R-independent, so not a far-row acoustic round trip); means are
    stationary under it. Open only as an OpenFOAM-reference question (its
    R -> ∞ value); nothing here points at dns. Earlier (still valid):
    halving OF's dt moves OF away from dns (+0.3/+0.4/+0.6%); OF with U
    also waveTransmissive drifts without a stationary state. Also recorded
    (W12): (i) OpenFOAM PIMPLE nOuter 1 is
    NOT a usable reference here (a 10-step cd sawtooth biasing mean cd +5%;
    nOuter ≥ 2 cures it, maxCo 0.25 and CrankNicolson do not); (ii) the
    plan's 40 D/U run length reads "not saturated" for SST (3 saturated
    periods), 64 D/U is needed; (iii) `--dt-recompute 16` is already the
    O-grid cylinder default (the plan's probe is the default run; its
    opt-out 1 costs 1.28× and moves nothing); (iv) `--cfl 0.8` reproduces
    cfl 0.4's statistics within 0.2% at half the cost -- a default change
    is a user decision, not made.

## Channel LES (masterplan `plans/verify/masterplan_verify.md` W11)

25. **Built 2026-10-27: `dns channel --les CS --sgs smag|sigma`; no channel
    LES statistic is converged or banded.** μt takes the RANS odd image on
    both ghost rows (wall face 0.0 exactly, `gpu_channel_les`); `--les 0` is
    byte-identical. Measured at 32×48×32, β 1.6, M 0.3, Re_τ 180, 40 δ/u_τ
    (≤ 82 s per run), ub⁺ over t 25–40 vs MKM 15.68: no model 16.65 (+6.2%),
    smag 0.17 RELAMINARIZED (u_rms⁺ peak 0.20; first-cell ν_t/ν 3.85, its
    largest anywhere: no wall damping), σ 1.35 18.10 (+15.5%, turbulent,
    first-cell ν_t/ν 1.4e-5). All three windows fail the stationarity rule
    (trend 4.7–7.0 batch SE; the 3 δ/u_τ batch SE is itself optimistic).
    Open: σ's +15% (model vs transient vs Δx⁺ 70 is unseparated), a grid
    where the rule passes (longer runs exceed the 300 s cap on this box),
    and the smag wall damping the channel does not have (van Driest or
    dynamic: not built, a model change). docs/channel.md "LES".

## LES step cost (masterplan `plans/verify/masterplan_verify.md` W15)

26. **Partly fixed 2026-10-27: the compressible LES `stableDt` re-derived the
    whole SGS model in FP64 every step.** Now its viscous bound reads the max
    of the solver's own FP32 μt of the same state (`updateSgsNut`, images
    included). 128³ sphere Re 1e4 NSCBC `--ft 0.05`, gfx1100, ms/step: smag
    5.83 → 4.74, σ 7.12 → 4.93 (DNS 4.26; focal ratio 2 σ 8.71 → 6.22, smag
    7.43 → 6.06). Every LES test case is byte-identical (the bound never binds
    there); where it binds, dt moves by the FP32 rounding of μt,max: 1.05e-7
    (smag) / 3.2e-8 (σ) relative on the `gpu_comp_les` rig. **Acceptance
    missed: smag ≤ 4.5 and σ ≤ 4.8.** The rest of the LES gap is the dt
    pass's own kPrim + μt launch (0.11 / 0.15 ms) and the three stage μt
    launches. Reusing the dt pass's μt in stage 1 (planned) needs the
    entry hook to be the identity on the state, and the Cartesian
    compressible hook runs `kFarSponge` and `--penalize`. So it is
    deferred to W9's `Idempotent` stage hook (built, item 28, but claimed
    only by the O-grid and channel runners: the Cartesian hook is still
    uncertified, so the reuse stays open). Merging the
    stableDt host copies into one was measured at −0.03 ms (0.6%) and
    dropped. The DNS part is mostly the FP64 acoustic reduction
    (`WavePerDt` 0.34 ms/step, 8%), which is bit-fixed by the anchors.

## RANS stage cost (masterplan `plans/verify/masterplan_verify.md` W8)

27. **Partly fixed 2026-10-27: the scalar stage kernels re-derived t/ρ per
    face and neighbour, and SST recomputed `sstCellGrads`.** The μt kernel
    now writes a per-cell cache (`rans::RansCellCache`: φ = t/ρ, SST also
    R and ∇k·∇ω) that the stage reads; `kSstStageCurv` carries
    `__launch_bounds__(blockSize)` (without it the cache version hit the
    192-VGPR cap and spilled 9). Every protocol-P output (17 cases, 61 files)
    is byte-identical. Stage per launch, gfx1100: SST −28…−60%, SA −40%
    (immersed 384²), −27…−43% (3D), but **SA on the small 2D grids only
    −16…−18%** (wedge 20.6 → 17.1 µs, O-grid 18.6 → 15.2 µs): the
    acceptance (≥ 20%) is missed there. Those grids put ~6 waves on a SIMD,
    so the kernel is latency-bound. What is left is face work (ν_f, χ_f
    divisions, computed twice per face) and the pointwise source. Neither
    is a per-cell value, so a cache cannot remove them bit-identically. SST
    immersed ms/it 0.931 → 0.750 (−19.5%, acceptance met). docs/rans.md
    "Stage-kernel cost". The small-grid remainder was then cut by launch
    count instead (item 28).

## Launch count (masterplan `plans/verify/masterplan_verify.md` W9)

28. **Fixed 2026-10-27: the small 2D RANS grids were launch-bound.** The
    wedge ran 45.6 kernels per iteration (39 tiny, ~2.8 µs each) and its
    stage kernels formed all three planes, two of which the hook discards.
    Now: the stage kernels run on the wedge's real plane only
    (`DeviceComp::setStageRange`); `step()` no longer re-applies a stage
    hook that the runner certifies idempotent (`Idempotent::yes`: O-grid
    rows + axis, channel wall); the O-grid rows and the channel wall are one
    launch each (`kOGridRows`, `kChannelRows`); SA's kPrim + μt are one
    (`kPrimSaMutCurv`). Launches/it: wedge 45.6 → 27.6, O-grid 31.5 → 15.5,
    channel 30.4 → 18.4. ms/it: wedge SA 0.270 → 0.192, O-grid SA 0.198 →
    0.149, channel 4×66 SA 0.154 → 0.121 (acceptance ≤ 0.24 / 0.18 / 0.135
    met). Byte-identical: protocol P + extras (31 cases, 113 files) and
    `gpu_axisym` (h). Still open, not built: the immersed/Cartesian runner
    does not claim `Idempotent::yes` (its hook is the ghost fill plus
    `comp_far`, which may be the in-place sponge; the ghost fill's
    idempotence is unproven), so its entry hook and the LES stage-1 μt reuse
    of item 26 remain. HIP graphs measured −4% per launch and are not used.
    docs/rans.md "Launch reduction".

## Thermal parity (`plans/verify/exp_thermal.md`, `tools/parity/othermal.py`)

29. **Thermal parity vs OpenFOAM: equivalent (2026-10-27); two items open.**
    Re 40 O-grid cylinder, M 0.2, identical cells, isothermal wall,
    Sutherland: Nu −0.34% / −0.14% and cd +0.43% / +0.14% against
    rhoSimpleFoam at Tw/Tinf 1.1 / 2.0 (const μ Pr 0.72: Nu −0.35%).
    Reproduced end to end by `othermal.py` (impl_batch.md). It holds ONLY
    with OpenFOAM's missing viscous work added (coded fvOption div(τ·u) on
    e; stock reads Nu 4.42% high at Ec 0.16) and dns Sutherland at
    `--prandtl 0.690335` (OpenFOAM's modified-Eucken κ). The residual
    (Nu 0.1–0.35%, growing to −0.9% at 180°) is unattributed, the size of
    dns' own 2×-grid move (0.12%). Open:
    - the M 0.5 adiabatic recovery reference: dns r 0.77, foam + div(τ·u)
      0.71 and still rising; the foam LTS run is unconverged (cd −3.1%);
    - RANS Prt heat transfer: not compared.

## Incompressible force box and placement (`plans/verify/exp_mrfC.md`)

30. **Fixed 2026-10-27: the incompressible force tally dropped the
    wall-adjacent z layers of a body that reaches them, and `--ny ≠ n`
    put the body off-centre.**
    - `gpu::bodyBox` clamped k to [2, nz−3] while the interior is
      1..nz−2. A mesh overhanging the z box is not `zSpans`, so its cd/cl
      came out ×0.5 at `--nz 4` (×14/16 at nz 16; an EMPTY box at nz 2).
      Now a world box reaching into layer 1 or nz−2 keeps it (the z-spanning
      bodies already did, via `zSpans`). The box wall then enters cs only,
      as it already did for those. `gpu_bodyforce` (3b) pins per-span
      cd/cl/cdf at nz 2/4/16 equal to the 2D tally, to 1e-9; it fails
      before the fix (nan / ×0.5 / ×0.875).
    - The placement put cy = h + L/2 whatever `--ny` was; the compressible
      box centres on ly. `--n 64 --ny 32` put the body centre above the box
      without a rejection. The same held for a sphere's cz at `--nz ≠ n`.
      Now both centre on the interior; the axis with n cells keeps the
      L/2 form, so every ny == n run is bit-identical (`cli_defaults`).
    - No anchor moved: nz == 1 and ny == n are untouched; the
      `cyl_re60_n256_cd` argv at `--ft 0.5` writes a byte-identical stats
      file before and after. The exp_mrfC incompressible Cn/Cc (×2
      hand-corrected there, nz 4, ny 449) was taken off-centre: rerun it
      before citing it.

## Older open items (details in CLAUDE.md)

- Re800 compressible anchors and `sphere_re300_n160_cd`: deferred, need
  regeneration.
- Re60 comp vs incomp cd gap +6.9%: cause unisolated.
- `gpu/comp_far.hip` domain y walls hardcoded adiabatic.
- Rotors: no table cache, no incompressible branch, no dns-ui support.
- dns-ui: a stop issued before a run reaches its time loop is not clean.
- README names gfx1201; this machine is gfx1100.
