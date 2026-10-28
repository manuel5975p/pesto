# dns — open issues

What is broken, unproven or undecided at this commit. CLAUDE.md holds the
stable invariants. Remove an entry when it is fixed (the fix lands in the
commit message and, if it creates an invariant, in CLAUDE.md); git history
keeps it. Item numbers are stable (cited elsewhere); gaps are closed items.
Numbers are measured, gfx1100.

## Flow RANS

1. **Coarse wedges: the explicit LTS march has no steady state at Re ≥ 1e3,
   for every closure, laminar included** (M 0.2, 1e5 iterations). 32×24:
   laminar Re 1e3 cd jitter 20%, Re 3e3 resid 150; SA Re 3e3 exit 7 at
   2.8e4; SST-2003m / Vm Re 1e3 jitter 18% / 11%. L1 64×48: laminar Re 3e3
   resid 53, Re 1e4 resid 261; SA Re 1e4 resid 0.46 (2.5e6 iterations: cd
   0.4197 ± 1.1%, θ_s 12.9°); SST-Vm Re 1e4 resid 30; SST-2003m Re 1e4 exit 7
   at 4.8e4. Signature: a period-2 θ sawtooth in the front wall pressure
   (docs/rans.md "SST: small-case verification"); a global dt (LTS ratio
   capped at 1) and the time-accurate march keep the front clean, and
   OpenFOAM SIMPLE converges on the same meshes. L1 Re 1e3 certifies (SST
   `gpu_sst`, SA-neg `gpu_sa_wedge`); SST L1 Re 3e3 nearly (remaining 1.7e-4
   at 2e5). Decided: exit 8 on the coarse wedges is accepted; the only
   measured cure is dissipation (not allowed). Cause inside the LTS march not
   isolated further.
2. **Wedge SST-2003m Re 1e4 L2 has no stable steady state.** LTS limit cycle
   cd 0.4088–0.4136 (robust to `--dt-recompute 1`, worse at `--cfl 0.2`); the
   time-accurate URANS is unsteady in the same band (cd 0.404–0.424, mean
   0.4115 over t 8–14.6 D_box/U). Reported as the URANS mean: tier A M 0.2
   0.4115 vs OpenFOAM central 0.40589 (+1.4%). SST-Vm is steady there (cd
   0.41882, remaining 1.1e-4 at the 3e5 cap: certifiable with a larger cap).
   Open: the M 0.1 URANS leg for tier B (~22 min, not run).
3. **Newton-Krylov (`--steady-scheme implicit`) is slower than explicit
   wherever it converges, and does not converge from a cold start on the Re
   1e4 wedge.** Cylinder Re 40 256×96, TOL 1e-4: laminar 78 steps / 28.4 s
   (explicit 3.0 s), SA 73 / 30.1 s, SST 174 / 55.4 s. Wedge SA Re 1e4 L2
   cold: exit 7 (16 rejections, 71 steps); from the explicit state at 1.5e5
   iterations: cd 0.42482 in ~10 steps, but GMRES stalls at the FP32 floor,
   rejections reset the armed history, W jitters 8.4e-4: no certificate in 200
   steps (129 s) vs explicit cold-start certification in 184 s. Wedge SST L1
   Re 1e3: cd 0.49852 (= explicit), exit 8 at the 500-step cap (173 s, W
   jitter 9.5e-4) where explicit certifies in 28 s. The line-SGS (13.4 ms per
   application) is a serial chain. Candidates, not built: explicit-then-Newton
   hybrid start, floor-aware GMRES tolerance, PCR lines. Default stays
   explicit.
4. **M 0.1 runs cannot certify.** The jitter-floored stop needs trend + √2 J
   < TOL for cd/cdf and J ≤ 2⁻¹⁰ for the monitors. SA L2 Re 1e4 at M 0.1
   (2.5e6 iterations, 766 s, settled cd 0.41943): cd J 1.6e-4, W J 4.7e-3
   (15× M 0.2's 3.1e-4; p/q∞ = 143 vs 36). Every tier-B M 0.1 leg ends at the
   cap (exit 8) whatever TOL. Decision needed: a Mach-aware monitor floor, a
   wake monitor other than the whole-domain deficit W (its jitter is
   dominated by the large far-field cells), or accept exit 8 + the printed
   floors for M ≤ 0.1. Same at the MRF props (2026-10-28,
   `plans/verify/impl_M1.md`): rotor2 L1 (M 0.082) at 60000 iterations ct
   jitter 1.80e-3 (√2 J > TOL 2e-3), cq 2.7e-4, jet monitor 8.05e-3 (>
   2^-10, trend not monotone); DTMB P1 (M 0.109) ct jitter 0.57 at 16000
   (unlimited LTS), 6.5e-2 at 13500 (`--lts-max 1`). Settle (|ct/ct_∞ − 1|
   < 1e-2 / 1e-3 / 3e-4 for good, rotor2 unlimited): 25550 / 59900 /
   60000+ iterations; certified: never. With `--precond 2` (2026-10-28,
   `plans/verify/impl_M2.md`) rotor2 at 30000: ct trend 2.5e-4 jitter
   3.2e-4, cq 1.3e-5 / 2.2e-5 -- the observables WOULD certify TOL 2e-3;
   the jet monitor (jitter 6.2e-2 relative: its mean ~0.028 sits near zero)
   alone blocks it. That makes the monitor choice the binding part of this
   decision for the props.
5. **OpenFOAM SA ft2 sensitivity exceeds the U4 trigger.** sph1e4 L2: `ft2
   true` 0.41285, `ft2 false` 0.40885 (−0.97%, trigger 0.5%). A dns
   `sa-noft2` is a user decision, not built.
6. **Immersed SA steady Re 40 does not certify.** 384², 4e5 iterations
   (311 s): cd a slowly damped oscillation (2.059 .. 2.0983, period ~8e4
   iterations), jitter 8e-4. The uniform mesh gives LTS no leverage;
   `--focal` or NK on the immersed path (not built) are the levers.
7. **`sph1e4` steady parity: SA L2 only, L3 not run.** Against OpenFOAM
   central U: SA tier B −0.35%, θ_s −0.18°, L_r +3.8% (the ±3% L_r tolerance
   is set at L3). SST: URANS mean only (item 2).
8. **Not built:** heavy immersed channel oracle, focal/sinusoidal MMS,
   momentum-budget tally test; multigrid / alternating lines (U5); P6. On the
   full revolution: Newton-Krylov (3D line ordering), `--les` (the axis cells'
   filter width), the 3D OpenFOAM export (`tools/parity/dnsmesh.py` reads the
   wedge only), the heavy `val_rans_sphere_3d` test (no certified stable G1
   number to pin, item 11).
9. **URANS turbulence scalars are first order in time.** The point-implicit
   stage quotient degrades SSP-RK3 in the stiff sources: order 1.00 (ω) /
   0.91–0.98 (k) on the closed-form SST decay (`gpu_sst` (a)), relative error
   8e-3 (ω) / 3.9e-2 (k) at β ω0 dt = 0.12. Steady fixed points unaffected.
   Deferred: a second-order implicit source update (breaks byte identity).
   URANS results state the first-order source error.
11. **SA-neg Re 1e4 on G1 (128×96×96, 1.2M cells) has no certified STABLE
    steady state inside 10 minutes.** Cold start, trip 0.05: exit 8 at
    170000 (581 s), cd 0.42519 ± 1.2e-4 (+0.09% of the wedge's 0.42480), an
    m = 1 cl growing (1e-3 → 1.5e-2 from 5e4 to 1.6e5 iterations).
    `--init-from` the certified 5° wedge: untripped exit 0 at 38800 / 135 s,
    cd 0.42520 -- the π-symmetry-protected axisymmetric state; tripped, the
    same m = 1 growth (exit 8 at 584 s). The end state (steady asymmetric, as
    SA-neg at Re 1e3, or unsteady) is open; following the m = 1 mode
    (e-folding ~4e4 iterations, 3.41 ms each) to saturation is tens of
    minutes.
12. **The full-revolution DNS time march is ~150× slower in dt than the
    wedge's** (laminar Re 100, `--ramp 0`): dt 2.55e-4 on the 5° wedge at
    128×96, 1.70e-6 at N 64. Binding term: the viscous bound at the axis
    cells' azimuthal width ½ R Δθ Δφ. It is a true limit (1.5× and 2× the
    bound run, 3× diverges within 200 steps, 64×48×64), and the per-cell bound
    gives the same dt to 0.8%. URANS at Re 1e4 is not penalized (ν 100×
    smaller; the wall RANS term binds both meshes), but the acoustic bound
    still sees the axis width (6.2e-5 vs 6.3e-4 at 64×48); a G1 URANS costs
    ~1e3 s per D/U. Remedies, not built: implicit azimuthal diffusion on the
    axis ring, or azimuthally coarser axis cells.

## Correctness

15. **`RansDiffDt` is unstable on isotropic 3D cells where it binds**
    (`gpu_rans_mask` M7). The bound 0.25/(D amax²) misses the compact face
    Laplacian's checkerboard, which needs Σ_a a_a²: λΔt = −2 in 2D (decays),
    −3 in 3D (past SSP-RK3's −2.51). Periodic 16³ box, uniform μt, T
    checkerboard stepped at the bound: grows 10× in 4 steps; 0.83× the bound
    stable, 0.85× grows (predicted 2.51/3 = 0.837). Bites only where μt is
    large on cells near-isotropic in ≥ 2 axes (3D immersed RANS wakes, the
    full revolution's outer cells, an extruded O-grid's wake); no run has been
    seen to diverge from it. Fix (Σ form) changes every RANS dt and LTS bit
    and needs the steady certifications re-pinned: USER DECISION (masterplan
    §5 item 6).

30. **The immersed RANS steady march accepts `--far sponge` and
    `--nscbc-strip`**, whose relaxation runs per stage in the stage hook, not
    per unit local dt: the fixed point then depends on the lts ratios (and on
    dtRef). Rejected for the MRF steady march (2026-10-28); for immersed RANS
    it is unmeasured and still accepted (rejecting it changes accepted argv:
    user decision).

31. **`--precond` is laminar only and loses on the diffusive-bound O-grid.**
    RANS is rejected (the scalars' point-implicit stage has no
    preconditioned form; masterplan K6, SA wedge M 0.2, not run). O-grid
    cylinder Re 40 (2026-10-28, `plans/verify/impl_M2.md`): M 0.1 certifies
    cd 1.55804 in 38000 (plain) / 48000 (K 1) / 42400 (K 3) iterations,
    2.99 / 5.41 / 4.81 s; M 0.05 80000 / 68800 / 59200, 6.26 / 7.63 /
    6.81 s. The wall cells sit on the diffusive bound (design_implicit
    §2.1), and the materialized residual costs +43% per iteration on this
    latency-bound 2D case (+20% at prop scale). A fused stage would remove
    most of it; not built (no win to fuse for here).

## LES

17. **The NSCBC x rows' LES μt is the raw wrap-differenced value** (every
    other wall/mirror row takes the RANS image via `setMutHook`). Deferred by
    decision (masterplan §5 item 8); the hook is the fix if a measured LES
    outflow shows an effect.
19. **Immersed URANS on the uniform 256² box diverges** at t ≈ 2.6 at Re 1e4
    and earlier at Re 1e5 (exit 7, `--rans sa-neg|sst-2003m --ft 10`), with
    and without the μt clamp; the 128² Re 1e5 seeded-layer rig collapses dt
    past 200 steps likewise. An under-resolved central scheme with no added
    dissipation, by design; levers: `--focal` or a larger `--n`.
25. **Channel LES (`dns channel --les CS --sgs smag|sigma`): no statistic is
    converged or banded.** 32×48×32, β 1.6, M 0.3, Re_τ 180, ub⁺ over t 25–40
    vs MKM 15.68: no model 16.65 (+6.2%), smag 0.17 RELAMINARIZED (first-cell
    ν_t/ν 3.85, no wall damping), σ 1.35 18.10 (+15.5%). No window passes the
    stationarity rule (trend 4.7–7.0 batch SE). Open: σ's +15% (model vs
    transient vs Δx⁺ 70 unseparated); a grid where the rule passes (longer
    runs exceed the 300 s cap on this box); smag wall damping (van Driest or
    dynamic: a model change, not built). docs/channel.md "LES".
26. **Compressible LES step cost misses its acceptance** (smag ≤ 4.5, σ ≤ 4.8
    ms/step; 128³ sphere Re 1e4 NSCBC: smag 4.74, σ 4.93, DNS 4.26). The rest
    of the gap is the dt pass's own kPrim + μt launch (0.11 / 0.15 ms) and
    the three stage μt launches. Reusing the dt pass's μt in stage 1 needs an
    idempotent entry hook; the immersed/Cartesian runner does not claim
    `Idempotent::yes` (its hook is the ghost fill plus `comp_far`, which may
    be the in-place sponge; the ghost fill's idempotence is unproven), so both
    its skipped entry hook and the LES reuse remain open. The DNS part is
    mostly the FP64 acoustic reduction (`WavePerDt` 0.34 ms/step, 8%),
    bit-fixed by the anchors.

## Rotor MRF parity

14. **Rotor MRF parity: NOT AT PARITY. The gap is frame-independent and sits
    in the blade's chord-normal force.** Laminar Re_D 220, J 0.63, tip M
    0.41, central U, `tools/parity` (`runs/parity/report.md`).
    - **rotor2** (`tools/propeller.py --blades 2 --root-chord 0.40
      --tip-chord 0.30 --thickness 0.35 --pitch 1.0`; bands CT 5%, CQ 2% vs Bc, fixed before measuring)
      vs rhoSimpleFoam Bc unlimited (CT −0.017090 fine L3, CQ 0.083313
      Richardson), MRF, 4.5 D/U final: CT +120.4 / +53.6 / +46.9%, CQ −11.3 /
      −5.6 / −5.1% (L1/L2/L3: D/h_fine,x 39.1 / 58.5 / 74.9, 35 / 113 / 260
      s). `gpu_mrf_rotor2` pins L1 at 1 D/U (ct 0.040455, cq −0.0738025).
    - **Frames agree.** Moving (`--rotor-frame moving`, freed-cell fill +
      swap ledger) vs MRF, window 3.0–4.5 D/U: CT +5.6% (L1) / +3.6% (L2),
      CQ +15% / +7.3%, converging; vs Bc, MRF L2 +57% / −5.8%, moving L2 +63%
      / +1.1%. The ledger closes fluid momentum to ≤ 2.9e-7
      (`gpu_rotor_swap_ledger`); Lz only to 5–8e-4 of ΔLz (cell-centre arms
      do not telescope). Residual freed-cell sink +0.0083 / +0.0025 CT
      (L1/L2).
    - **MRF azimuth bias at L1**: spread 0.0165 CT over blade orientations 0
      / 22.5 / 45 / 67.5° (ct +0.03833 / +0.03280 / +0.04933 / +0.04025, mean
      +0.04018; cq spread 0.0031). Quote the azimuth mean at L1; L2/L3 not
      scanned.
    - **Where** (`plans/verify/exp_rotorR.md`, measured on the wall before
      the impermeable faces): 87% of the blade CT gap at 0.20–0.45 D, a
      near-uniform total chord-normal force deficit (−6…−15% per 0.05 D bin,
      flat L2 → L3); chordwise within ±7%; hub, root and tip bin not it. Blade
      CT is a small residual of normal against chordwise thrust, so a ~9%
      normal-force deficit is the whole gap.
    - **Ruled out**, each measured smaller than the gap: settling (L2 to 9
      D/U, block means within 0.7%); foam's laplacian limiter (1.2–2.3% CT;
      every reference is now unlimited); the MRF zone (foam Bc L2 with the
      sharp zone at dns' f = 1 / f = 0 radii 0.554 / 0.704 D: CT +13.6 /
      −6.8%, CQ −2.6 / +0.8%); t/h (L2 at t/h 5.4–7.1 and L3 at 7.1–9.2 carry
      the same deficit); box, BCs, Re, Mach, gas (matched); the frame (above).
    - **foam self-consistency**: Bc L2/L3 CT agree to 3.2% (B 3.7%), not the
      1% acceptance (CQ 0.4%); recorded, not re-banded.
    - **DTMB 4119** (MRF, Re_D 800, J 0.833, tip M 0.41;
      `plans/verify/exp_prop.md`). Reference: simpleFoam G2 × F1's A/B ratio,
      CT 0.0233 (±5%+), CQ 0.0612 (±1.2%); every rhoSimpleFoam F1 attempt
      diverged. dns D/h_fine 114 / 154 / 189 (P3 20.7M cells, 1186 s): CT
      0.01615 / 0.01835 / 0.01939, CQ 0.06126 / 0.06005 / 0.05991 → CT
      −14.7%, CQ −2.2% at 189. Ladder extrapolates CT 0.021–0.024: parity
      neither shown nor excluded. Blade bins inboard of r/D 0.40 within 3–5%
      of G2; 0.40–0.45 −9.3% (flat 154 → 189), 0.45–0.50 −34% (still
      rising): 75% of the deficit at r/R > 0.8, t/h_fine 3.3 → 1.7. Hub at
      parity. Azimuth bias at D/h 114: +15° moves CT −10.0%, CQ −1.1%. Torque
      p/v split +29% / −18% vs G2 while total CQ agrees (unisolated).
    - **rotor** (single blade): last measured at 80f147f, before the
      two-sided and impermeable faces; L2/L3 exceed the 300 s cap; its B/Bc
      foam references are still limited (65° snappy meshes). Not re-measured.
    - **Known limitations of the wall**: the RANS scalars at two-sided faces
      read the shared ghost (warned); 2–7 wrong-side faces per rotor2 level
      have no usable side sample (counted, warned); tip corners of blades
      under 4 cells keep their side images, Re 100 offset sensitivity not
      measured there.
    - **Next, none run**: the radial split on the current wall; a
      non-rotating 3D straight lens wing at the rotor's c/h against foam; an
      MRF-vs-moving rig where the frozen rotor is exact AND the wall has a
      normal velocity (`gpu_mrf`'s axisymmetric cylinder has (ω×r)·n = 0); a
      time-averaged `--emit surface` for moving runs; `--mrf-zone` (zone
      radius and blend; triggered by |ΔCT| > 2%); MRF azimuth scans at rotor2
      L2/L3 and DTMB D/h 189; an L4 (over the cap).
    - **Steady frozen rotor** (`--rotor-frame mrf --steady`, 2026-10-28,
      `plans/verify/impl_M1.md`): rotor2 L1 fixed point ct 0.037694, cq
      −0.074022 (identical at `--lts-max` 1/2/4/∞; time march 3.0–4.5 D/U
      mean 0.037608 / −0.074030). No wall-time win: LTS slows the march
      (ct inside 1% after 8100 / 19600 / 23850 / 25550 iterations at R = 1 /
      2 / 4 / ∞; time march 8570 steps, 35.5 s; steady 4.0 ms/it vs 3.69
      ms/step), no certificate in 60000 (item 4). DTMB P1: unlimited LTS ct
      sample sd 0.011 at 12000–16000 (mean −0.01731), R = 1 sd 0.0012 (mean
      −0.01706, still drifting); time march last 0.6 D/U −0.01629. Open: why
      LTS amplifies the pseudo-time oscillation (the fine plateau carries the
      near wake at dtRef; the blend region steps up to R× faster). M2
      (`--precond K`, 2026-10-28, `plans/verify/impl_M2.md`) damps it: K 2
      unlimited settles rotor2 ct to 1e-2 / 1e-3 / 3e-4 after 8200 / 15500 /
      22550 iterations (ct 0.037697, cq −0.074022), 4.94 ms/it (+20%); K 3
      stable, noisier; K 1 diverges (exit 7 < 1000 it), K 1 + `--lts-max 1`
      survives but crawls; K 3 + `--lts-max 1` worse than either. DTMB P1 K 2
      (13500 it, 22.7 ms/it): ct sd 7% (unlimited plain 65%), window means
      −0.01691 / −0.01688 / −0.01684 (flat; R = 1 plain drifts −0.01802 →
      −0.01706); time march −0.01629. Still no wall win over the rotor2
      time march (ct 1e-2 at 8200 it ≈ 40 s vs 35.5 s); DTMB P3 (K5) not
      run. Open: the U_r floor should follow the tip speed, not U∞ (K 1
      diverging at U_r = 0.08 c under a 0.41 c tip).
    - **Recipe** (each run ≤ 300 s): `gpurun -t 300 -m 16G -- env
      PARITY_LOCK_HELD=1 python3 tools/parity/suite.py run rotor2 --tier
      B|Bc --solver foam --level k`, `... run rotor2 --tier A --solver dns
      --level k` (`-m 14G` at L3); split: `python3
      tools/parity/split_dns.py dns CASE RUN_surface.csv --tol T --json
      runs/parity/split/CASE/dns_Lk.json` (T = 0.0267 D for rotor2), then
      `suite.py report`.

## URANS and RANS references

20. **The plain FP32 time march does not converge in dt** (FP32 update
    stagnation, floor ulp(q)/(2 dt)). O-grid SST URANS Re 1e4 gate: plain cd
    ladder ratios 0.22 / 0.29 (diverges), `--compensated` 3.75 / 3.30 (order
    ~1.8); docs/rans.md "Time accuracy". `--compensated` is built and OFF by
    default (1.16–1.21× per step). Open: (a) whether it becomes the URANS
    default (changes bits) is a user decision (masterplan §5 item 2); (b) any
    plain FP32 dt study is suspect where |R| dt nears ulp(q)/2 -- none of the
    measured anchors is a dt study.
22. **The TMR flat plate, bump-in-channel and NACA 0012 cannot be
    expressed.** The O-grid is uniform in polar angle (no arc-length
    clustering at a trailing or leading edge) and `--rans` rejects box walls
    (`--body floor|fence|none`, `--noslip`: no ghost-row adapter). Needs an
    arc-length O-grid and the wall-row adapter (a feature, masterplan §6).
    Code-to-code references today: CFL3D 2DFDC channel (`rans_channel1d` O5)
    and the OpenFOAM wedge (item 7).
23. **Supersonic `--mach` is accepted on the implicit steady RANS channel
    only, and is unproven elsewhere.** The explicit steady march and URANS
    stay rejected at M ≥ 1 (their transients carry y-acoustics not measured
    there); everywhere else the (0, 1) bound stands (no shock capturing). The
    channel has no viscosity law (constant μ vs CKM's μ ∝ T^0.7), so the CKM
    scenario is a model-label comparison (Tc/Tw 1.382 vs ≈ 1.38).
24. **O-grid cylinder URANS SST shedding vs OpenFOAM: open only as an
    OpenFOAM-reference question.** 2D, 256 angular, Re 1e4, M 0.2,
    `--ogrid-trip 0.05 --compensated --cfl 0.6`; OpenFOAM rhoPimpleFoam on
    the identical mesh, central U, backward, nOuter 2, `waveTransmissive` far
    p (its `freestreamPressure` never becomes stationary). Saturated dns
    window t 75–128 D/U (12 periods): cl_rms 1.2061 ± 0.0006. Far-field
    ladder with matched near-body cells (R 20 / 40 / 80 D), dns − OpenFOAM:
    cd −1.10 / −0.59 / +0.77%, St −1.76 / −0.83 / +0.92%, cl_rms −0.32 /
    +0.53 / +0.77% -- the gap changes sign. dns cd 1.6923 / 1.6759 / 1.6782,
    St 0.2282 / 0.2281 / 0.2293, cl_rms 1.2061 / 1.1947 / 1.1944 (≤ 0.5% past
    40 D); OpenFOAM cd 1.7111 / 1.6859 / 1.6653, St 0.2323 / 0.2300 / 0.2272,
    cl_rms 1.2100 / 1.1884 / 1.1853 (~1.2% per doubling in cd, St), and at 80
    D a symmetry-breaking mode from ~70 D/U (cl_mean → 0.03) that dns does
    not have. Not separated: OpenFOAM's drift is radius OR outer resolution
    (the tanh law coarsens the outer field with R). dns at 40/80 D carries a
    slow 13.8–16 D/U cd modulation (cd sd 0.004–0.010 vs 0.0025 at 20 D,
    R-independent period); its means are stationary. Not re-measured against
    this reference: SA-neg. Next: OpenFOAM's R → ∞ value.
29. **Thermal parity vs OpenFOAM: equivalent; two items open.** Re 40 O-grid
    cylinder, M 0.2, isothermal wall, Sutherland: Nu −0.34% / −0.14%, cd
    +0.43% / +0.14% vs rhoSimpleFoam at Tw/Tinf 1.1 / 2.0 (`othermal.py`,
    with OpenFOAM's missing viscous work added and dns at `--prandtl
    0.690335`). Residual Nu 0.1–0.35% (−0.9% at 180°) unattributed, the size
    of dns' own 2×-grid move (0.12%). Open: the M 0.5 adiabatic recovery
    reference (dns r 0.77, foam + div(τ·u) 0.71 and still rising, foam LTS run
    unconverged, cd −3.1%); RANS Prt heat transfer, not compared.

## Older open items (details in CLAUDE.md)

- No anchor for the Re800 compressible cases or `sphere_re300_n160_cd`
  (rows commented out; their tests skip).
- Re60 compressible vs incompressible cd gap +7.3%: cause unisolated
  (ghost-mirror vs cut-cell wall, sponge vs free-slip laterals).
- `gpu/comp_far.hip` domain y walls hardcoded adiabatic.
- Rotors: no table cache, no incompressible branch, no dns-ui support.
- dns-ui: a stop issued before a run reaches its time loop is not clean.
- README names gfx1201; this machine is gfx1100.
