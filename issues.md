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

54. **The immersed RANS steady march accepts `--far sponge` and
    `--nscbc-strip`**, whose relaxation runs per stage in the stage hook, not
    per unit local dt: the fixed point then depends on the lts ratios (and on
    dtRef). Rejected for the MRF steady march (2026-10-28); for immersed RANS
    it is unmeasured and still accepted (rejecting it changes accepted argv:
    user decision).

55. **`--precond` is laminar only and loses on the diffusive-bound O-grid.**
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
    - **`--thin-seal` overshoots** (2026-10-28, `plans/verify/impl_propperf.md`;
      analysis_propperf gate S5 FAILED, S6/S7 not run). P1 D/h 114, parity
      argv, 235 fluid/fluid faces sealed, vs foam G1 (× 0.98828 / 1.00219),
      window 2.58–3.18 D/U: CT 0.01629 → 0.02883 (−33.3% → **+18.0%**, gate
      ≤ 10%), CQ 0.06129 → 0.06374 (−1.1% → **+2.9%**, gate ≤ 2%); tip bin
      0.45–0.50 −74.2% → −0.5% (gate −25%: met). Final-snapshot bins
      0.30/0.35/0.40: +14.4 / +12.8 / +8.7% (snapshot 6% above the window).
      Still rising (blocks 2.5–3.0 / 3.0–3.18: 0.02853 / 0.02932). Same
      pattern as the floored-STL emulation (+7.3% vs foam on that body).
      Open: the sheet model's own body bias (t → 0 below τ) vs the
      remaining p/v split error; side-image stencils of a sealed face may
      take fluid corners from the far side of a sub-cell sheet (unmeasured).
      P1 az 15° and P2m sealed not run.
    - **NSCBC centre table is NOT bit-identical** (`plans/verify/
      verify_propperf.md`): it is on by default for every `--focal` NSCBC
      run. Focal sphere 128×96×96 (with and without `--pot`): stats differ
      from the pre-table binary from the 4th row on at rounding level (cd
      equal to 6 digits; cl/cs ~1e-3 relative). base + table alone ==
      final byte for byte, so the table is the cause, not the seal refactor.
      The P1 short byte match (412 steps, 3 rows) was too short to see it.
      Undecided: keep (focal sphere 2.76 → 2.62 ms/step) or revert.
      14.57 → 13.65 ms/step (re-measured 14.41 → 13.44); the plan's `kNscbcCells` ≤ 0.15 ms/launch gate
      is NOT met (inferred 0.57 ms/launch from the step saving; not
      profiled), so 14 FP64 maps were only ~35% of that kernel. With
      `--dt-recompute 16` (now in `runs/prop/bench/dns.sh`,
      `runs/prop/dns/argv.txt`): 12.25 ms/step full P1 (215.7 s wall vs
      247.6), window CT/CQ 0.01629 / 0.06129 unchanged to 5 digits
      (verified: 1.3e-5 / 1.2e-6 relative). P2m full with dt16
      (`plans/verify/bench_prop_after.md`): 25.37 ms/step, window CT/CQ
      0.01846454 / 0.06007321 vs 0.01846500 / 0.06007321 before (−2.5e-5 /
      0 relative). `runs/prop/dns/argv.txt` is the P3 argv: dt16 there is NOT
      measured at P3 (its CT/CQ are the pre-dt16 run's).
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

## Chemistry (`dns flame`, docs/chem.md)

Measured gfx1100 via simjob, 2026-10-28.

30. **A stationary flame pins to the grid.** With `--inflow-speed 1` at 30
    cells/δ_th (|S_h − 1| ≈ 5e-4) the consumption speed follows the local
    fresh velocity instead of being a flame property: five run variants
    settled anywhere in 0.9995–1.0011 (cfl 0.2 flat at 1.000106, sink 32
    flat at 1.00109). S_L is therefore measured on a SLIDING flame
    (`--inflow-speed 0.98`, window t ∈ [10, 20]). Undecided: whether the
    default `--inflow-speed` should be below 1.
31. **S_c wanders at the 1e-4–5e-4 level as h and M shrink.** Sliding,
    window halves differ 1.2e-5 (7.5 cells), 3.7e-5 (15), 4.7e-4 (30) at
    M 0.01, and 3.1e-4 at M 0.005 (15 cells). Consistent with FP32 pressure
    resolution (ulp(p0)/ρ_u S_L² = 4e-4 at M 0.01, 1.7e-3 at M 0.005), not
    proven. The plan's 1e-4 flatness band was not met; the tests use 1e-3.
    The M 0.005 point also sits +8.3e-4 above the O(M²) fit through M
    0.01/0.02 (1.004144 vs 1.00331), so the ladder's low end is
    FP32-contaminated; M 0.01 is the sweet spot. Inflow variants do not
    help (c15, 40 flame times: `--nscbc-tinflow` and `--nscbc-etain 2` drift
    by 3e-4 where the default holds 8e-5).
32. **Periodic-box FP32 mass drift.** The conservation box (two flames, M
    0.02, 5696 steps) loses mass 5.1e-5 and E + qρY 5.3e-5 relative; the
    non-reacting twin (ρ never sees the species code) 2.0e-5 / 2.9e-5. Both
    negative, larger than a random-walk estimate (~1e-5): a solver-level
    rounding bias of the stage write, unexplained. The fuel tracks it
    exactly (fuel discrepancy = 0.93 × the mass drift = the fuel fraction).
    Over 50 steps every balance holds to 1e-7.
33. **RANS open-face scalar fix runs after the sink** (`comp.hip`, the
    `bnd_.any()` branch, vs `runner_comp.hip`'s rhs hook): the sink writes ρ
    but not ρφ, so φ = t/ρ is not preserved in the sink band. Unmeasured;
    found while placing the species re-carry (which runs BEFORE the sink).
34. **The species sink twin conserves species mass only where Y is uniform
    across the band.** It adds Y_k × the band's continuity term (bounded);
    D4 on ρY itself pushed the fresh Y to 1 + 2e-5 (measured, docs/chem.md).
    Exact in the P1 flame (the bands sit in uniform fresh / burnt gas).
35. **NSCBC carries no species waves** (Sutherland & Kennedy 2003; the
    `--mode bunsen` hook re-adds q ω after the overwrite, below): valid only
    while ω vanishes in the boundary rows (omegaEdge 4.7e-16 measured).
    Same premise, second consequence (verify 2026-10-28): the NSCBC
    `stagePut` OVERWRITES E on the i = 0 / nx−1 cells, discarding the
    `crAt q ω` the species launch added there, while the fuel's −ω survives
    the re-carry. E + qρY therefore leaks q ω per boundary cell — 4.7e-16
    of the peak rate in P1, unmeasurable, but structural: a case burning in
    the boundary rows must re-add q ω after the overwrite (or zero ω there).
    `--mode bunsen` (anchored IN the inflow rows) re-adds it
    (`speciesBoundaryHeat`); the 1D runner does not (keeps its bits).
36. **Not built: Le ≠ 1** (no oracle; `--lewis` unregistered), curvilinear,
    compensated, RANS and LES species (asserted), nz > 1 flames (2D is:
    `--wrinkle`, `--mode bunsen`), the
    low-Mach branch, multi-step chemistry, dns-ui support for `flame`.
37. **The chemistry dt bound never binds in the P1 range** (dtChem/dt
    8–32), so kChemDt = 0.5 is unbracketed by measurement: halving or
    doubling it cannot move S_L.
38. **Grid order 2.37 vs the host scheme's 2.12; DNS − host error offsets
    +1.5e-3 / −2.6e-4 / −1.5e-4 at 7.5 / 15 / 30 cells** (after removing the
    measured O(M²) offset −3.9e-4 at M 0.01). The host Newton (scouting, low-Mach) is not the
    DNS's operator (KG energy flux, compressibility); a C++ discrete oracle
    (plan §3.2, optional) would separate scheme from Mach/FP32 error. Not
    built.
39. **2D yHits are an FP32 inflow-row drift, not an unbounded scheme**
    (measured 2026-10-28, `yMin`/`yMax` columns). 2D wrinkled run (M 0.02,
    8 cells, ny 256, tt 4): Y − 1 ≤ 1.8e-6 (15 ulp), Y ≥ 0, only at i ≤ 14;
    the inflow row drifts +9…+15 ulp (1D: stays exactly 1; sink off: still
    +12 ulp). Convected downstream it lit ~150k cells of a long ny 1280 run
    at the 1e-6 (8.4 ulp) threshold. Mechanism inside the open-row re-carry
    under transverse fluxes not isolated further; not fixed (not material),
    the count threshold unchanged (1D bits).
40. **Bunsen flank consumption speed −2.2% at U_jet 3** (SLflank 0.979 /
    0.978 at 8 / 16 cells; mFlank 0.989 / 0.986), vs +0.3% at U_jet 5
    (16 cells). Grid-independent: stretch/curvature on the shorter cone
    (the fit window sits nearer tip and base) or the window choice;
    unattributed. The cone-angle relation sin α = S_L/U_jet reads +2.1% /
    +3.8% high (16 cells) — the fresh flow decelerating ahead of the flank;
    not separated from stretch by a measurement of the approach velocity.
41. **Bunsen anchors only on a burnt pilot.** Measured at U_jet 5: cold or
    T/Tu 3 coflow, a cold u = 0 rim and slow thin burnt strips blow off;
    burnt coflow (default) and 3 δ_th burnt strips at the jet speed anchor
    (docs/chem.md table). Not built: an immersed ghost-wall burner lip — the
    physical rim stabilisation. The walled species kernels exist now
    (premixed: the chamber; two-reactant: the wick candle) but neither
    bunsen nor the jet uses one. Not measured: anchoring vs U_jet (only 3 and
    5 run), the pilot's minimum width, the pulsed jet's anchoring margin.
42. **Bunsen y is periodic** (coflow W per side by default): the image jets
    confine the expansion. Coflow width sensitivity of the cone angle not
    measured; open y faces (NSCBC openY + species) not built.
43. **The diffusion jet is attached BY CONSTRUCTION.** Its inflow row is the
    Burke–Schumann sheet of Z(y): the edge burns at the exit plane. No lip,
    no lifted or triple-flame base was run. Measured map (W 16, s 1,
    docs/chem.md): edge 1 l goes out at t 0.48; edge 2, edge 1 + 2 l Z_st
    pilots, edge 3 hold at β 5 / Da 1000; β 8 needs Da ~5000 (exit 8 at
    1000). Not mapped: the extinction boundary in (edge, Da, β) at other W,
    s, coflow speeds; a frozen (cold, mixed) inflow with an edge-flame base.
44. **The shared limiter clips every species at any voter's extremum.** It is
    what makes the coupling functions exact (Z vs tracer within the FP32
    walk; per-species 1.226e-3), but a first-order face for all species
    wherever one reactant peaks: the slab-centre Z sits 1.5e-3 off the erf
    at h 0.25. Vote threshold 2^-24 is a choice (noise reactants at ~1e-40
    forced Z 5.0e-3 off with every species voting); a non-voter can move
    < 2^-24 past its neighbours per face (counted in `hits`: 0 in every
    run). Grid order of the two-reactant scheme not measured.
45. **The jet's inflow row runs 0.35 T0 cool at the sheet peak.** The LODI
    row keeps transverse conduction against the entropy relax (etaIn c/L,
    etaIn 100): steady default jet, H = T/T0 − 1 + (q/cp T0)(Y_F − Y_F1 Z)
    is −0.351 at i = 0, 0.206 at i = 1, 0.074 at i = 4, ≤ 0.0095 past the
    band. Not fixed. A stiffer --nscbc-etain (G3a, 2026-10-30): max |H| at
    i = 0 / 1 / 4 0.350 / 0.205 / 0.073 (100) → 0.131 / 0.071 / 0.030
    (1000), 0.85 at i = 0 for 20; it also moves the jet's fuel inflow 7%
    (§68). foam's fixed inlet face: 5e-4. The species pin on that row
    (`speciesInflowPin`) is a consistency choice; its isolated effect was
    not measured. The re-added boundary heat is 0 there (pinned sheet: ω 0).
46. **Outflow anchor σ 4 for the jet is a constant, no flag.** At the
    shared 0.25 the box pressurized to p/p0 1.67 (t 14; cold streams 1.6–1.7
    T0, interior 1.28 T_ad); at 4 the steady jet holds p/p0 1.000 ± 0.003.
    The stiffer anchor's acoustic reflection is not measured.
47. **Diffusion validation gaps.** The flame-sheet POSITION x_f(t) is
    validated only isothermal (the heated layer has no similarity solution:
    ρ²D varies); no counterflow/strained flame vs Cantera; the jet tip is a
    regression pin (51.2 l measured vs the constant-density Burke–Schumann
    70.3 l, which D ∝ T shortens — a variable-density estimate not built).
    Grounded against reactingFoam on the identical mesh (G3a, docs/chem.md):
    agreement to 0.7% in tip, 0.0% in half-width, 0.07% in TmaxIn once dns'
    inflow is stiff (etaIn 1000); at the default the tip is 8.4% long (§68).
    The candle (G3b) is not grounded: foam twin built, flame not run (CPU).
    The heated Tmax extrapolation assumes the Da^(−1/3) law (measured ratio
    2.275, not 2). The jet does not flicker unforced (Re = W/Pr ≈ 22, no
    gravity; gravity exists in `--mode candle`, §49–53); `--pulse` opens
    its tip periodically (fuel out/in up to 0.019).
    y periodic: the products fill the box top (coflow 24 l in the demo), and
    the images confine the jet — s 2 tip 147.4 l at coflow 16 l vs 129.3 l
    at 24 l (the 16 l run predates §48's change; closed tip). Open y faces
    not built.
48. **The jet drops q ω on its outflow row** (§35's leak, chosen): re-adding
    it there ran away when the tip opened into the LODI row (T 4.3 T_ad at
    i = nx − 1, p/p0 0.95–1.12, backflow; s 2, coflow 40 l). Without it, a
    pulsed jet that opens its tip burns its outflow row cold (E + qρY_F
    loses q ω there); the size of that leak is not measured. A burning
    outflow needs NSCBC with species/heat source terms (Sutherland &
    Kennedy 2003), not built.
49. **Candle fuel balance closes to ~5%, not 1%.** Steady W 1 (t 20–30):
    burn/in 1.031 AND fuel out/in 0.017–0.025 (Y_F ~8e−4 leaks past the
    tip into the plume: finite-rate tip leakage); W 0.5 burn/in 0.986,
    W 0.75 1.015. Suspects, not isolated: the species sink twin in the
    inflow band, which holds the flame base (§34: exact only where Y is
    uniform across the band; the candle's base is 2 l of a 10 l flame),
    and the first-interior-face fuelIn estimate at Pe ≤ 1. Not tested
    (no band in `gpu_candle_*`); the tip leak's Da dependence not measured.
50. **The candle box recirculates.** y periodic (12 l per side) with a
    fixed 1 U coflow: the plume entrains more than the coflow supplies, so
    air is drawn DOWN through the top in every run (uOutMin −7.6…−12 U).
    The top takes it as ambient air (ρ_a, v = 0 at etaLat 0.05, Y = air:
    the Hydro NSCBC + `yHi`). Not measured: sensitivity of the tip to
    L_y, coflow speed, box length; whether the relax rate matters. Open
    y faces / an open (pressure) bottom not built.
51. **Candle unsteadiness unattributed.** W ≤ 1 steady; W 1.25 one slow
    ~13 l/U cycle (t 10–30); W 1.5 flaps sinuously with tip pinch-off at
    4.26 / 4.07 / 4.14 l/U (f 0.24 U/l = 0.165× Cetegen–Ahmed 0.479
    sqrt(g/W), 0.25× with D = the flame width). Flame instability vs the
    periodic box's recirculation (§50) not separated (no wider box run).
52. **Candle height law: 4/3 approached, not reached.** H ∝ Q^1.19 over
    W 0.5–1 (local 1.12 → 1.30; Roper's buoyant slot 4/3; momentum 2
    excluded). Every W ≥ 1.25 is unsteady, so no steady flame sits in the
    far-field regime. One grid only (8 cells/l), one g (14), Da 5000, β 5:
    tip convergence and chemistry sensitivity not measured. The wick is
    the inflow plane (attached by construction, §43); the immersed wick is
    `--wick` (W1–W9 below), whose anchoring needed Da 5e4.
53. **Candle NSCBC top (`Hydro`).** The hydrostatic incoming waves are
    validated at rest only (|u| 1.2e−3 U vs 0.67 U without); their
    acoustic reflection and the backflow blend's rate are not measured.
    The rest residual 1.2e−3 U is not attributed (FP32 p rounding: ulp(p0)
    = 2e−3 at M 0.005 is the candidate). The Mach is inflated: buoyant
    Mach 0.17 at the defaults (|u|max 34 U, M 0.005); the M dependence of
    the tip not measured.
64. **The 1D flame's burnt gas heats up downstream of the flame** (found
    against OpenFOAM reactingFoam FP64 on the identical mesh, 2026-10-30,
    docs/chem.md "Against OpenFOAM reactingFoam",
    `plans/verify/impl_foamflame_G1.md`). The flame itself is
    energy-exact: the total enthalpy H = cp T + u_rel²/2 + qY gained from the
    fresh reference column to 2.5–3.5 δ_th behind x_f is +8e-6 / −1e-5 /
    +3e-5 q (7.5 / 15 / 30 cells, sd 1–2e-5; foam +1.1e-5 / +3e-6 / +8e-7),
    T_b/(σTu) − 1 there −8.6e-5 / −1.07e-4 / −8.0e-5 vs the exact −9.6e-5.
    Behind that, at constant p, T climbs a STATIONARY ramp and saturates
    (15 cells, `--downstream 20`, T at ξ = 3/5/7/9/12/15/18 δ_th: 35711.8,
    35713.3, 35713.4, 35714.6, 35716.2, 35716.6, 35716.8; foam flat to
    0.1): H at 8–9 δ_th is +1.24e-4 /
    +8.5e-5 / +2.13e-4 q (foam +2.2e-5 / +6e-6 / +2e-6). Same with the sink
    off (+9.6e-5) or 32 cells (+1.21e-4) at 15 cells; at 30 cells `--cfl 0.2`
    LOWERS it (+1.58e-4 vs +2.14e-4, window [6, 12]) with twice the steps,
    so it is not a per-step FP32 bias. S_L is unaffected (the heating sits
    behind the reaction zone; dns − foam S_L +4e-5 / +1.5e-4 at 15 / 30).
    Unattributed.
65. **The dns fresh pressure is not flat at 7.5 cells/δ_th** (same
    comparison): the time-mean p 3–6 δ_th ahead of the flame sits +0.061
    (1.6% of Δp) above the reference column's, so Δp read at the reference
    column is 1.28e-2 BELOW the measured-state Rankine–Hugoniot jump
    (momentum check p_u − p_b vs m Δu: −1.23e-2, sd 4e-3 over 101 dumps);
    +2.0e-4 / −2.3e-3 at 15 / 30 cells. foam: −3.4e-5 / −7e-6 / +1e-6, p
    flat to 1e-4. Not the 6-digit profile output (0.13% of Δp) nor aliasing
    (0.5 and 0.1 τ dump cadences agree). Candidate: the 16-cell inflow band
    is 2.1 δ_th wide at 7.5 cells and the reference column sits 2 δ_th past
    it; not measured.
66. **The bunsen inflow holds DENSITY, so the fresh jet enters hot** (found
    against reactingFoam, G2a, 2026-10-30, docs/chem.md "G2a",
    `plans/verify/impl_foamflame_G2.md`). The cone raises the inflow
    pressure 0.56% (jet) / 0.33% (coflow) above p0 — both solvers agree
    (foam inlet cells +5.59e-3 / +3.29e-3 p0). The default per-row NSCBC
    entropy target is the row's density, so T = p/ρ enters +5.86e-3 (jet) /
    +3.79e-3 (coflow) high (the box's Tmax, 1.0037 T_ad, sits on the coflow
    inflow row). The oracle at Tu × 1.0059 burns 1.1% faster. Measured, U 5,
    8 cells, [20, 25]: sinRatio 1.04823 → 1.02961 with `--nscbc-tinflow`
    (foam 1.02876), SLflank 1.00695 → 0.99742, mFlank 1.00868 → 0.99823,
    xTip 44.65 → 45.54. Every bunsen "Measured" row in docs/chem.md carries
    it. Not changed (no src edits in that task): decide whether bunsen
    should default to `--nscbc-tinflow` (the diffusion/candle rows hold
    density too: on the jet `--nscbc-tinflow` moves tip 51.20 → 51.52,
    fuelIn 9.617 → 9.665 (G3a); the candle not measured). GROWS with the
    jet speed (G4a/G4b, docs/chem.md "fast jets"): default vs T-held tip
    −2.0% / −6.1% / −12.2% at U_jet 5 / 10 / 15, sinRatio +1.8% / +3.6% /
    +11.5%; T-held is within 0.6–2.1% of foam's tip. Not the speed
    alone: G4c (slot 6, U 20, jet Mach 0.3 as G4b) −8.4% (G4b −12.2%),
    T-held −0.3% vs foam.
67. **Bunsen flame base: dns `--nscbc-tinflow` vs foam, 3.5% wider base,
    0.5–0.6% faster foam flank** (same comparison). At equal flank angle
    (tan α 0.21043 vs 0.21025) the fitted flank's half-width at the exit
    plane is 10.640 (dns) vs 10.267 δ_D (foam, 1.9 cells narrower), so the
    cone is 4.0% shorter in foam (xTip 43.72 vs 45.54); SLflank 0.99742 vs
    1.00392, mFlank 0.99823 vs 1.00304. Candidate: the inflow face itself,
    where the base sits — foam's fixedValue T, Y conduct heat and diffuse
    fuel through the burner face (0.22% of the fuel enters by Fick), dns'
    open face carries no diffusion. Not attributed (needs a foam inlet
    without face diffusion, or a dns lip). foam's window is not quite steady
    (sinRatio halves 1.0263 / 1.0312). NOT the §68 inflow overshoot alone:
    `--nscbc-tinflow --nscbc-etain 1000` (job 682) brings fuelIn/fuelNom
    1.038 → 1.010 but moves xTip 45.54 → 45.00, tan α 0.21043 → 0.21275,
    hw0 10.640 → 10.652 (foam 43.72, 0.21025, 10.267). The base sits where
    §68's row-0 u overshoot peaks (jet edge +2.1–2.4 S_L at etaIn 100,
    ~10× less at 1000), yet hw0 does not move with it: not the base gap.
68. **The per-row NSCBC inflow at the default etaIn 100 overshoots its mass
    flux** (found against reactingFoam, G3a, 2026-10-30, docs/chem.md "G3a",
    `plans/verify/impl_foamflame_G3.md`). Default diffusion jet (W 16, 4
    cells/l, [60, 75]): fuel through face 0|1 9.617 vs foam's 8.959 (fixed
    face; nominal ρ_row u_row Y_row 8.467 + Fick), tip 51.20 vs 47.22 l
    (+8.4%; tip per fuel −1.0%). `--nscbc-etain` 20 / 100 / 1000: fuelIn
    11.115 / 9.617 / 8.989, tip 60.29 / 51.20 / 47.54 (foam −0.7% of the
    last); `--nscbc-tinflow` 9.665 / 51.52 (not the entropy target). The
    convective part (9.27) exceeds the nominal row by 9.5% while the row's
    own T at p0 would carry 8.385: the excess is in the inflow row's u.
    The bunsen rows do the same: G2a fuelIn/fuelNom 1.032 (default), 1.038
    (tinflow), 1.010 (tinflow + etaIn 1000); foam 1.005. So the
    `gpu_diffusion_jet` pin (tip 51.17 ± 5%), the anchoring map and every
    bunsen/jet "Measured" row in docs/chem.md carry an etaIn-dependent
    inflow ~3% (bunsen) / ~7% (jet) above its nominal fuel flow; the candle
    rows (same machinery) not measured.
    **Attributed** (`plans/verify/verify_foamflame.md`, 2026-10-30): the
    STEADY LODI inflow cell (`kNscbcCells<…, Prof>`) has dp/dt = du/dt = 0,
    so L1 + L5 = 2 S_p and L5 − L1 = 2ρc S_u, where S_p, S_u are the
    pressure/velocity rates of everything the overwrite keeps or the hook
    adds on that row (transverse y fluxes; the bunsen's re-added q ω). Hence
    u − u_row = (S_p + ρc S_u) L_x / (etaIn (1 − M²) ρc²): zero only if the
    row has no transverse source. The per-row targets put a T(y) step in
    the row (Burke–Schumann sheet; bunsen cold jet vs burnt coflow), whose
    y conduction S_p = (γ − 1) κ ∂²T/∂y² heats the fuel-side rows. From
    dns' row-0 frames that term alone predicts the fuel excess 0.96 / 0.10
    / 0.97 vs measured 0.89 / 0.10 / 0.90 (jet etaIn 100 / 1000 / tinflow;
    4.1 vs 2.7 at etaIn 20: nonlinear), bunsen 3.8 vs 3.3 (etaIn 100) and
    0.42 vs 0.44 (1000). The parameter is etaIn/L_x, as the formula says:
    `--length 160` (L_x 168 vs 104) at etaIn 161.5 gives fuelIn 9.623 vs
    9.617, tip 51.53 vs 51.20; at etaIn 100 fuelIn 9.983 (excess ×1.50),
    tip 53.39 (jobs 683). Direct: the bunsen nz 8 twin's row-0 |u| is
    +2.1–2.4 S_L above u_row at the jet edge (y 19.3–19.7, +50%), −1.6e-3
    on the axis (prediction +2.7–5.0 at the edge, correlation 0.82 over the
    rows). NOT the flame's back-pressure (a row without transverse source
    returns u_row exactly whatever L1 is), NOT the entropy target, NOT the
    species pin. Options (not decided, no src edit): a stiffer default
    etaIn (error ∝ 1/etaIn; 1000 ran at the same dt), a relax rate per cell
    instead of per box length (the error grows with L_x), or a row without
    transverse conduction.
69. **The P0 eigenvalue is not round-off-determined at σ 2, β 8** (found
    2026-10-30, `plans/verify/verify_foamflame.md`, in the numpy port
    `tools/chem/foamflame/model.py`, same shoot/bracket algorithm as
    `dns/chem.cpp`, n 2e4). The cold-end "too fast" test flips 61 times
    across Λ ∈ [4.7246e8, 4.7263e8]: the port bisects to Λ 472623348.6
    (δ_th/δ_D 1.794901), dns prints 472463262.4 (1.7953) — 3.4e-4 in Λ,
    1.7e-4 in S_L, 2e-4 in δ_th. σ 5 is unaffected (port = dns to the
    printed digits). Moves the σ 2 lesgap / G2b normalisation by ≤ 2e-4
    (below their quoted digits); the C++ shoot itself not probed. The port
    produced a garbage table there before (kept integrating dead lanes in
    record mode; now stops as dns does). **Cause found and fixed by §72**
    (2026-10-31): the cold-boundary difficulty — σ 2 β 8 was no eigenvalue
    of the uncut march at all (Λ ∝ 1/n); with the adaptive cutoff both dns
    and the port give 503116277.2 (δ_th/δ_D 1.72336), n-converged 4e-8.
70. **The round-nozzle cone is 27–43% shorter than the ideal cone, cause
    not separated** (found 2026-10-30, `--nozzle round`, docs/chem.md
    "Round-nozzle Bunsen"). Laminar DNS, 4 cells/δ_th: U 5, D 12 tip 21.5
    vs 29.4 δ_th (ST 1.35); U 10, D 6 tip 17 vs 30 (ST 1.63). Tip curvature
    vs tip resolution (4 cells) not separated: no grid study. The periodic
    jet lattice (one box apart) not probed. No GPU test pins the round path
    (host: profile/cone = slot at r, config rejections); slot bit identity
    holds by construction (the slot entry points are the old functions,
    `test_bunsen`; `gpu_bunsen3d_extrude`, `gpu_bunsen_cone`,
    `gpu_bunsen_anchor` green), no base-vs-branch run byte compare.
71. **At jet Mach 0.2 the diffusion jet's inflow mass flux depends on the
    inflow BC kind** (found 2026-10-30, G5, docs/chem.md "fast jets"). The
    box raises the inflow pressure 7.3% (jet; ∝ γM², G3a ~0.5%). foam's
    inlet (fixedValue T, zeroGradient p) lets ρ_in follow p: fuelIn 9.48;
    dns holds ρ (entropy target): 9.03 at etaIn 1000, 9.83 at 100 (the §68
    overshoot). Tip per fuel agrees to +0.6% / −1.0%; the tips differ
    +4.3% / −5.6%. Not decided which inflow is the intended problem.
    `--nscbc-tinflow` does NOT reproduce foam's inlet: fuelIn 10.70
    (+12.9%), tip 58.33 (+15.2%), tip per fuel +2.1%, at pMean 1.022 — the
    excess is more than the pressure; not separated from §68's overshoot
    (tinflow at etaIn 1000 not run). G6 (slot 8, same M): every dns
    variant within 2.4% per fuel, tips +4.2 / −0.6 / +8.1%. foam's lInf = box outlet
    pressurized the box ×2 at M 0.2 (fixed with lInf = box/4); foam still
    drifts −2.4% over [200, 240].
72. **FIXED (2026-10-31, masterplan real_bunsen S1): the P0 oracle's
    bracket bisected spurious verdicts** (docs/chem.md "The bracket and the
    cold boundary"). Two defects. (a) Non-monotone "too fast": a shot far
    above the root collapses onto p* = λqω/(m cp), where RK4 (h J > 2.785)
    bounced p to a positive cold residual; one such shot on the halving path
    gave Λ a power of two — σ 6.5 Sutherland 298 K printed Λ 16383.95 (2^14,
    δ_th/δ_D 5.13; correct 941442.56, 4.378), σ 6.88 42008.4 (45.5;
    correct 882657.41, 4.507), i.e. CH4's real φ-1.3 σ. 9 (constant λ) + 6
    (Sutherland) of 401 σ in 2..10 at β 8 were hit, more at other β. Fix:
    a stiff step (h J > 1; slow shots peak at 2.2e-3) is too fast; a scan
    of 2^−60..2^20 must show ONE transition (else an error by name); the
    bisection runs on the shipped [lo, 1]/[1, hi] bracket, so every
    well-posed Λ is bit for bit (`flame1d` (g); dry-run diff of defaults,
    bunsen, chamber, diffusion, candle, σ 3, β 6/10, σ 7.5 Sutherland:
    identical; the bit-identity suite — 1D, Sutherland 1D, 3D planar,
    bunsen 2D, round 3D, diffusion, candle, chamber, wick — every output
    byte-identical). (b) The cold-boundary difficulty: where the unburnt
    rate is not negligible (β 8 σ ≲ 2.6), captured fast shots end with a
    positive residual and the uncut march drifted as 1/n. Fix: the adaptive
    cutoff (`kColdCut`); a cut above 1e-4 of the rise is an error by name.
    **Changes a shipped number**: σ 2 β 8 (the lesgap / G2b DL campaign,
    §69) Λ 472463262.4 → 503116277.2 (+6.5%), δ_th/δ_D 1.7953 → 1.72336.
    GPU (15 cells, M 0.01, sliding): old Λ burns at S 0.9747, new 1.0023 —
    the new Λ is the solver's S_L = 1; those campaigns normalised by their
    measured grid S_L (their ratios survive), their δ_th-scaled lengths
    (box, l_t, λ) used 1.7953 (4% off). No other docs row used an affected
    (σ, β, Sutherland) — the Sutherland rows are σ 5 / 7.5 at 300 K
    (unchanged), the round real-burner baseline (job 746) σ 7.5 constant
    (unchanged). Newly REJECTED (were 1/n garbage): β 8 σ ≤ 1.70, β 6 ≤
    2.20, β 4 ≤ 3.25, β 3 ≤ 4.25, β 2 ≤ 6.20. The numpy port carries the
    same fix.
73. **The one-step model cannot match CH4's δ_th/δ_D, nor its rich S_L(φ)
    trend** (masterplan real_bunsen D1/D4, found 2026-10-31). (a) δ_th/δ_D:
    model C0 4.51 (σ 6.88 Sutherland), C2 (two-reactant φ 1.3) 4.71; GRI-3.0
    CH4/air φ 1.3 6.40 (`tests/data/chem/ch4_gri30.csv`; the plan's 8.0
    was from the starred 0.70 mm / 0.0877 mm). Of Re, Pe and D/δ_th at most
    two can be real (mapping M1: Pe 1/1.36 of real with C2's 4.71, Pr_eff
    = ν/(S_L δ_D,model) = 0.52). (b) The C2 model's S_L(φ) at fixed A, q, Ta
    RISES with φ (0.755 at φ 1.05, 1.174 at 1.6, rel. to 1.3) where GRI-3.0
    FALLS (1.634, 0.342): the rate ∝ Y_F Y_O starves as Y_F,rem → 0 toward
    φ 1. The masterplan's D4 ("model and CH4 both burn faster there: right
    sign") is wrong in sign; the rim's edge flame (leaner mixture) will burn
    SLOWER than CH4's. **(b) FIXED by model C2b (S1b, 2026-10-31,
    docs/chem.md "Model C2b")**: ω = A C(ξ) ρ² Y_F Y_O e^(−Ta/T), ln C a
    clamped cubic spline in the fuel/air mixture fraction ξ (the
    Shvab–Zeldovich coupling function) on 26 GRI-3.0 nodes φ 0.5–1.7,
    calibrated by the FP64 oracle (lean branch added), C(jet) = 1;
    `--sl-law gri30` is the `--equivalence` default (`onestep` keeps the
    uncorrected step). S_L(φ)/S_L(1.3) = GRI's at the nodes (4e-15),
    ≤ 0.34% at the held-out rows 0.575–1.575; GPU (`gpu_flame_rich1d_law`,
    `--mix-phi` 0.7 / 1.05 / 1.2, 15 cells) within 0.4% of the oracle.
    STILL OPEN: (a) δ_th/δ_D (unchanged: C only scales the rate); T_ad(φ)
    is the one-step's (−16% at φ 0.6, −6.5% at 1, +10% at 1.6 vs GRI; no
    variable cp), so σ of a lean edge flame is low; the outer sheet now
    burns at C(Z_st) = 5.46 × the jet's A (its Da is the law's); and the
    FP32 coupling-function walk reaches the rate (d ln S/dZ = −7.3 at φ
    1.3: 1D jet −2.3e-3 vs the uncorrected step at 30 cells, Z up to
    1 + 4.6e-4).
74. **Checkpoint / restart built on `runFlame`, `runLowMach` and the FV flame
    runners (M5); its byte rule rests on one struct per runner** (masterplan real_bunsen S2, 2026-10-31, docs/chem.md
    "Checkpoint and restart"). `--checkpoint-dt` / `--wall-limit` /
    `--restart` + `tools/chem/chain.sh`; `gpu_restart_flame` proves the
    byte rule on premixed 1D (+ junk past a checkpoint), `--equivalence`,
    bunsen 2D `--pulse` (+ PNG), round 3D, TFLES 3D, candle, `--turb
    --hold-flame`, a grown `--tt`, SIGINT. OPEN: (a) `--wick` and `--mode
    chamber` (own runners) are rejected by name, not wired; plain `--mode
    diffusion` is covered only through the candle case (same runner). (b)
    `--emit web`/`vtk` are rejected with the flags (the web manifest lives
    in one process). (c) **K12**: every host variable the loop carries must
    be a `FlameLoop` member listed in `fields()`; one added outside it is
    not checkpointed and breaks the byte rule SILENTLY — only the chained
    test catches it (a new mode or stats column needs a case there). (d) A
    checkpoint is pinned to the binary's sha256 (exit 3 on any rebuild): a
    chain must finish on the binary that started it. (e) simjob's timeout
    (SIGTERM, 10 s grace) writes a checkpoint, but the job reads rc 124 and
    `chain.sh` stops; continue with `--resume`. (f) The wall-limit stop
    predicts the next interval from the LAST interval's wall time only
    (an interval that slows down, e.g. frames starting at `--frame-start`,
    can still overrun the simjob cap: keep the interval well under T − W).
    **M5 (2026-11-03): built on the FV flame routes too** (`runFvComp`,
    `runFvLowMach`; docs/chem.md "Checkpoint and restart"). Before M5 the FV
    allow-list refused the three flags by name (`--solver: --checkpoint-dt is
    not built on the FV flame path`); the lift covers every FV flame route,
    each proven by `gpu_restart_fv` / `gpu_restart_fvlm` (chained = straight
    byte for byte, last checkpoints equal, teeth). The mesh is in the
    identity (`mesh` record: polyMesh files + `--bc` content). OPEN: (g) the
    FV loop structs (`FvCompLoop`, `FvLowMachLoop`) carry K12's risk the same
    way: a new FV stats column or mode needs a case in `gpu_restart_fv`. (h)
    The low-Mach FV loop's equal steps to `--tt` make its trajectory depend on
    `--tt`: a restart that GROWS `--tt` continues correctly but is not the
    straight run to the larger `--tt` (no "grow" gate there; the compressible
    FV loop clips only its last step, as runFlame). (i) The mesh identity
    hashes every regular file of the polyMesh directory (not subdirectories,
    not `cellZones` semantics): a harmless edit (a comment) is exit 3 too, by
    design. (j) `dns flow` on the FV path (and every non-flame runner) has no
    checkpoint.
75. **Low-Mach pressure solve (masterplan real_bunsen S3, docs/la.md
    "Separable Poisson"): the FP32 PCG's TRUE residual floors at 1.3e-5 –
    5e-5 of |b|, and the tube's projection costs ~2x the plan's budget.**
    `LmPressure` (apertured PCG, spectral preconditioner) on the annulus
    tube: the recursive residual reaches 1e-6 in 12–16 iterations cold, 6–11
    warm, but the FP64-measured residual stops at 3.3–3.8x the pure FP32
    storage floor (residual of float(x_FP64): 3.5e-6 – 1.3e-5), at or below
    the estimate ε_f32 |A| |x| / |b|; the solution still converges to the
    FP64 one (max error 4e-7 of max at tol 1e-6, 2.5e-6 at 1e-5, 4e-5 at
    1e-4). K5 (EOS drift): after the projection div m − D* carries this
    floor (projected max|div m| / max|div m*| 3e-5 – 9e-5) — mass stays
    exact (telescoping), the EOS does not; S4 must measure eosErr against
    it. K6 / cost: at 384 x 176 x 176 one PCG iteration is ~2.9 ms (spectral
    1.45 + operator 0.45 + Krylov ~1), warm solves 9–11 it = 30–35 ms, i.e.
    ~100 ms/step for three stages vs the plan's ~45 ms (its "1 + 6
    iterations x 2.2 ms"). OPEN: the real warm start (extrapolated phi) and
    tol (1e-5 saves ~2 it/solve) are S4/S8's to measure; deflation of the
    tube-surface modes or a fused Krylov step are the levers if S8 needs
    them. Without a solid the solve is one direct spectral pass (1.45 ms).
76. **FIXED (2026-10-31, S3): `gpu_cutcell2d` failed "w stays zero" with a
    run-to-run varying error (5.3e-6, 2.3e-6, 4.4e-7) and PCG iteration
    count (nz 8: 19 vs 12).** The test warm-started `projectCut` from an
    uninitialised `hipMalloc` phi (`runner_incomp` zeroes it, line 138); a
    fresh process got different garbage. The test now zeroes phi: w is 0.0
    exactly, 16 iterations every run. No solver change.
77. **FIXED (2026-11-02): `amg_setup` failed its wall-time gate** (found in
    S6): "[FAIL] 6M setup (lineRatio 4) <= 3 s (3.094201 s)" twice at -j 8,
    blamed on load. It also failed idle, this code and f892cc6: 3.0 / 3.35 s on
    the Ryzen 3900X (docs/la.md's old 1.83 s was another host). `coarsen` was
    60% of it; now threaded, byte-identical at any worker count (`test_amg`;
    hierarchy hash == f892cc6 on 8 cases). Gate unchanged: 1.92 / 2.31 s alone.
    Threaded, it still lost at -j 8 (2.87 / 4.10 s, other tests on its cores),
    so the test is RUN_SERIAL: a wall-time gate needs the host to itself.

### Low-Mach (`--physics lowmach`, docs/chem.md "Low-Mach" / "Low-Mach Bunsen" / "Low-Mach open laterals" / "Low-Mach gravity" / "Low-Mach rim" / "Low-Mach dt rule, real-burner scouts, renderer"; masterplan real_bunsen S4–S9)

LM1. **FIXED (2026-10-31, S4): the FP32 state dropped the preheat zone's
    sub-ulp increments** — the flame lagged: Shvab–Zeldovich defect growing
    with the step count (4.8e-5 / 3.3e-4 / 7.7e-4 at 7.5 / 15 / 30 cells),
    mass off the tally by 3e-5 – 5e-4, S_L −1.2e-3 grid-independent and
    dt-dependent. ρ and ρY_k are now compensated (Kahan low words,
    `combineWord2`; kernels read hi): SZ ≤ 1.4e-6, S_L dt-independent to
    1e-6, Z exact to 2e-7 (K15: the compressible walk is 1.7e-4). OPEN: m
    and p are plain; in 1D m is constraint-determined, but the ROTATIONAL
    part of m in 2D/3D accumulates in plain FP32 — measure on the S5 jet
    (a vorticity budget) before trusting long 3D runs.
LM2. **The dynamic pressure carries an FP32 floor ∝ 1/dt.** The stage
    targets D read the ulp-quantized T through the diffusion term (k δT/h²)
    and the multiplier p reads them over cr: the p jump's row rms is 0.17% /
    1.0% / 6.5% of the jump at 7.5 / 15 / 30 cells (white in time, not
    acoustic); the window mean is within 1.1e-3 of ρ_u S² (σ − 1). The
    plan's "rms ≤ 1e-6" is not an FP32 number. No effect on the flow found
    (1D: m constraint-determined; multi-D: a gradient, removed by the next
    projection). Lever if p itself is needed: T differences from the
    compensated words in R_T. eosErr 1.0–1.4e-7, divErr/max|D| ≤ 1e-5.
LM3. **Observed order 1.66, not the plan's 1.8**: the MC limiter sets T's
    face value (shared with the species so H = T + (q/cp) Y_F rides one
    formula), and the heat release reads T. The Richardson limit 1.000121
    agrees with the compressible ladder's M → 0 limit 1.000079 to 4e-5; at
    15 cells the two schemes' errors have opposite signs (−0.17% vs +0.34%),
    so the plan's "within 1.5e-3 at 15 cells" is not met (the converged
    numbers are the comparison: `gpu_lowmach_1d`).
LM4. **K3 (the T predictor rides m^(s), the species m^(s+1)) is O(dt²):**
    SZ 8.6e-7 at 15 cells (dt 5.4e-4) but 2.5e-4 on the M1 grid (4 cells,
    dt 1.4e-2). The real-Bunsen runs (S9/S10) must report the SZ column;
    the fallback is a second predictor-corrector pass (2× projections).
LM5. **K1 resolved without S4b**: explicit chemistry with the cp-form
    Jacobian binds 1.43× below diffusion at K 0.5 on the M1 grid and not at
    K 1 (S moves 3.1e-5); the default `--chem-dt` is 1. Point-implicit
    chemistry is not built; re-check if a finer M2 grid or a stiffer rate
    makes chemistry bind.
LM6. **What S4–S8 do not have** (rejected by name): the rim off the
    low-Mach burner (S8's `--rim`; the bunsen is S5's, open laterals S6's,
    gravity S7's -- `--gravity` on the burner only, not the planar flame);
    `--pulse` on the
    low-Mach burner; `--mach`, `--nscbc-*` (no NSCBC, no sink);
    LES/TFLES/`--turb`; with PERIODIC laterals the outflow's BACKFLOW
    still carries the cell's own scalars (S6's ambient backflow +
    stabilization exist only with `--laterals open`, where an ambient is
    defined); the tangential inflow is no-slip (antimirror). ny, nz must be 7-smooth; nx
    is rounded up downstream. dns-ui maps physics 0/1 only (LowMach = 2
    never reaches the editor; this build has no UI, `--check` not run).
LM7. **Cost** (3D planar, 128² × ~210, 8 cells/δ_th): 2.87 ns/cell-step vs
    the compressible's 2.33 at M 0.02, but 526 vs 1536 steps per τ — 2.3×
    cheaper per flame time at M 0.02, ~18× at the true Mach (1D 15 cells,
    10 τ: compressible 480k steps / 95 s, drifting −4e-4 per 2.5 τ in FP32;
    low-Mach ~18.5k steps / ~5 s). **S5: 3D diffusion-bound runs now take
    1.5× the steps** (the 3D positivity bound 1/6 h²/D, LM10), ~790
    steps per τ for that case (not re-measured). Burner (S5): 2D 113k cells
    0.51–0.68 ms/step (4.5–6 ns/cell-step, launch-bound small grid; dt every
    step + the stats rows' host passes), round 3D 573k 2.7 ms (4.7 ns), 4.6M
    18.4 ms (4.0 ns: C2 + flame index + per-step dt over S4's 2.87). The
    bunsen stats row downloads ~10 full fields and loops them on the host in
    FP64: at the final 11.9M cells budget ~0.3 s per row (S9: measure, or
    raise the stats cadence).
LM8. **FIXED (2026-10-31, S5): T rode the shared limiter WITHOUT voting**,
    so where no species votes (every |d2| < 2^-24) its face value was θ = 2
    = the DOWNWIND value. In the `--equivalence` jet core (Y_F 0.07: ulp 7e-9
    below the vote threshold) an odd-even T wiggle grew from rounding (T
    0.47 Tu at x 52 δ_D, t 2.4) and diverged the 2D partially premixed
    burner. T now votes (one θ: H still rides one linear face formula; T
    itself TVD). 1D ladder unchanged to the printed digits; the rh p jump's
    window mean −1.1e-3 → −3.0e-3 of ρ_u S² (σ − 1) (gate widened to 4e-3:
    the burnt gas' ulp noise now first-order at its extrema).
LM9. **G2a: the low-Mach cone agrees with reactingFoam, the compressible
    solver sits 2–3% off both** (docs/chem.md "Low-Mach Bunsen"): low-Mach
    vs compressible M → 0 (M 0.01/0.02 T-held etaIn 1000, M²): xTip −1.9%,
    sinRatio −2.6% (the plan's 1% gate NOT met; the test gates 3%), SLflank
    +0.16%, mFlank +0.04%; low-Mach vs foam (identical 392 × 288 mesh):
    xTip +0.48%, sinRatio −0.44%, θ field L2 4.6e-3 (compressible 0.022).
    Not the sink band (`--nscbc-sink 0`: 44.69 / 1.0484), not the inflow's
    fuel flux (compressible 92.98 = foam's burn; low-Mach 92.44). OPEN:
    which compressible ingredient (LODI inflow row, Mach) tilts its flank;
    the compressible G2a's own foam gap (+2.1% default, +4.2% T-held) was
    already recorded (§66–68).
LM10. **SETTLED (2026-11-01, S9): `--dt-rule cell`, the burner's default**
    (docs/chem.md "Low-Mach dt rule ..."): dt also ≤ 1/max_c (2 C_c + D_c),
    the explicit transport step's per-cell positivity bound (MC ×2 on the
    convective rate, the cell's own diffusivity), K = 1. Bounded (0 hits,
    Tmax ≤ σ_st after the IC transient) on every case below that the old
    default was not (2D rim slot U 6 / 10, 3D pilot round g 0 and real g)
    and on the half-scale real burner; 1.2× (3D) – 1.5× (2D) the old
    default's steps, 1.3× fewer than `--chem-dt 0.5`; K 1.5 is NOT bounded
    (1.6e3 / 1.9e4 hits); chemistry in the sum costs 1.5× and bounds
    nothing more. The planar flame keeps `--dt-rule bounds` (S4 bit for
    bit); `--dt-rule bounds` on the burner = the S8 binary byte for byte.
    History: **4 cells/δ_th 3D is not bounded at the default dt** (S5, round
    nozzle, C2 + pilots): explicit convection + diffusion positivity. At the
    diffusion-bound dt 0.0091 the inner tip overshoots T (7.6–8.1 vs σ_st
    6.98) and undershoots Y_O (−5e-5 .. −7e-3, 2e4–5e4 cells); `--chem-dt
    0.5` (dt 0.0055) or `--cfl 0.3` is bounded (Tmax 6.95, 0 hits); 8 cells
    is bounded at the defaults once the 3D diffusive coefficient is 1/6 (at
    1/4, Σ_a 2D dt/h² = 1.5: T 8.6 at 8 cells). The principled bound is the
    per-cell positivity dt ≤ 1/(2 Σ|u_a|/h_a + Σ 2D/h_a² (+ J)) (MC needs
    2× the upwind CFL); not built (it would move every S4 number). **The
    final case is 3D at 4 cells: S9 must pick the dt rule (or `--chem-dt
    0.5`) before the half-scale scout.** S6 data point (open round, D 6,
    U 3, pilots 1.5 at U, 4 cells, default `--chem-dt 1`): dt 0.0071 is
    CONVECTIVE-bound (cfl 0.8; dtDiff 0.0097, dtChem 0.0116) and Y_O
    undershoots in up to 11156 cells (T stays ≤ σ_st); `--chem-dt 0.5`
    (dt 0.0055) has 0 hits — at U 3 it is the MC convective positivity,
    not diffusion, that the default violates. S7 data point: under
    gravity the same burner at `--chem-dt 0.5` is CONVECTIVE-bound below
    the chemistry dt (dt 0.0032 at real g, 0.0038 at 4x, vs dtChem
    0.0058; the buoyant plume is faster than the jet) and bounded (0 hits,
    t 200): S9's dt rule must include the plume's speed. **S8 data (the
    rim, no pilot)**: 2D rich slot U 10, 4 cells: default (cfl 0.8) Tmax
    8.42 (σ_st 6.98), 2.0e4 hits, Y_F min −9e-3; `--cfl 0.4` 7.27 / 1.9e4 /
    −9.5e-6; `--cfl 0.2` 7.18 / 18 / −2.9e-7; `--chem-dt 0.5` 7.20 / **0** /
    −2.4e-7 (the hits sit in the products near the outflow, not at the
    rim); every U ≥ 6 overshoots at the default (U 6: Tmax 7.14, 6390
    hits; `--chem-dt 0.5` 7.06, 0 -- `gpu_lowmach_rim_anchor` runs it), U ≤
    4 undershoots ≤ 4e-5 only; 8 cells: 0 hits at the defaults. 3D round tube (D 6, U 4, open,
    real g): bounded at the DEFAULT dt (0 hits, Tmax 6.973, dt 0.0092
    diffusion-bound) and at `--chem-dt 0.5`. So `--chem-dt 0.5` is the
    measured safe rule at 4 cells in both; the 2D slot at U ≥ 6 needs it.
LM11. **The partially premixed outer cone is long**: Z_st 0.78 sounds close
    to the jet, but the products of a W 12 / U 5 / 3 δ_th-pilot slot need
    ~(W/2 + pilot)² U/D to dilute to it (~180 δ_th): open at 45 δ_th, still
    growing at 410 δ_D in a 424 δ_D box. `gpu_lowmach_rich2d` uses W 4, U 3,
    pilots 1.5 (outer tip 80 δ_D, closed, steady). The round nozzle closes
    faster (radial dilution); S9's half-scale scout must check xTipOut fits.
    preFrac 0.727 (2D), not 1/φ = 0.769: the base's triple flame and the
    cone's lean-mixing edge carry ξ < 0 (the index's split, not a
    leak: burnRatio 0.99987). **S6: open laterals do not fix it in 2D**
    (LM13); the round nozzle's outer tip is box-independent at ~42 δ_D
    (D 6 δ_th, U 3: periodic 41.84, open 42.57 / 41.99 at clearance 3 / 6).
LM13. **2D partially premixed slot: open laterals LENGTHEN or OPEN the
    outer cone** (S6, docs/chem.md "Low-Mach open laterals"). rich2d's
    burner, window [150, 200]: periodic clearance 6 closes at 80.09 δ_D,
    open clearance 6 at 90.47, open 3 / 2 never (NaN); room air (coflow
    0) never; W 12 / U 5 / pilots 3 never (238 of mdotIn 348 leaves
    sideways). The flame's dilatation pushes gas OUT through the
    laterals (net mdotLat −27 … −238; at clearance ≤ 3 every lateral face
    flows out) and a planar jet entrains weakly, so the coflow air the
    periodic box held beside the jet leaves. The periodic box's closure is
    a confinement artefact too (the coflow air is forced to mix). OPEN:
    the real burner's dilution is buoyancy-driven (S7: the plume
    accelerates and entrains); the slot comparison is not the real case's
    question — the 3D round is (it closes regardless, LM11). Also: an open
    box needs t ~ 100 τ for the starting-jet head to leave (at t 40 the
    box drained mass at −115/τ, every column moving). **S7 (gravity,
    docs/chem.md "Low-Mach gravity", `runs/s7/lm13.sh`): REAL g (0.0241
    S_L²/δ_D, Fr 19.8) does NOT close what the open box left open** —
    clearance 6: 90.47 → 82.40 (periodic 80.09), the lateral outflow ×2.7
    smaller (−32.1 → −11.8); clearance 3 / 2 and W 12: still open. Only
    4–12× real g turns the laterals into a net INflow (+83 … +122), closes
    the 2D cone (62–74 δ_D) and makes it FLICKER (xTipOut swings 17–50 δ_D,
    periods 18–36 τ). 3D round: 42.57 → 42.15 at real g (steady to t 200),
    41.46 at 4×, no flicker. OPEN for S9: the real burner (D 16 δ_th, U 6.5)
    is ~2.7× larger, so buoyancy over its box (u_ff ∝ sqrt(g L)) matters
    more there than in these scouts; the 2D slot stays a poor proxy.
LM14. **Bickley at Re 30 sits 1–9% off the similarity solution; at Re 60
    half of that** (S6, `gpu_lowmach_bickley` runs Re 60): u_c^-3 slope
    0.989 / 0.984, b^1.5 0.954 / 0.976, Qj³ 0.908 / 0.967, dQj/dx
    exponent −0.767 / −0.680 (Re 30 / 60) — the O(1/Re) higher-order
    boundary-layer terms. J is NOT conserved, J + P is (0.55% / 0.36%
    along the box): a wall-issued jet hands momentum flux to the wall
    suction of its entrainment flow (Schneider 1985), P(x-lo) = −0.25 /
    −0.12 J_in; Bickley holds with K = the far-field J. Not measured: the
    round jet's Schlichting solution (the plan did not ask; S9's round
    scouts could use the same jet profiles).
LM12. **The burner IC is not the projected flow**: the first projection
    sets velocities up to ~2× the IC's in the products; Y_F undershoots to
    −8e-4 for t < 5 τ on G2a (clean from 5.2), and mass vs the FP64 tally
    carries a constant −4.7e-8 (2D G2a) / −1.2e-7 (rich) from the first
    rows (the adjustment's FP32 divergence rounding; flat afterwards).
    `--dt-recompute` defaults to 1 on the burner (16 diverged the 4-cell
    partially premixed slot within 50 steps). LM1's vorticity budget is not
    measured; the G2a cone is steady to 1e-5 over [20, 25] (no visible drift
    from the plain-FP32 rotational m on 16k steps).
LM15. **Candle cross-check: the plan's gate (low-Mach tip within 5% of the
    compressible candle's 10.79 l) is NOT met — −34.8%** (S7,
    `gpu_lowmach_candle`, docs/chem.md "Low-Mach gravity"). Low-Mach tip
    7.029 (steady), burn 0.887; compressible default 10.785; its D10
    variant (`--nscbc-tinflow --nscbc-etain 1000`) 8.031 (unsteady 7.36 –
    8.33), burn 1.037. Per unit burned fuel 7.92 vs 7.74 (+2.3%, the test's
    gate). At Pe = 1 the length follows the fuel each INFLOW passes: the
    low-Mach face's convective nominal is 0.405 and Fick through the face
    carries 54% of what burns (the Burke–Schumann sheet sits on the exit
    plane; first-order face gradient over h/2 — grid-dependent, not
    measured on a ladder); the compressible LODI row overshoots (§68). At g
    0 the gap is larger: low-Mach 10.41 vs compressible default 25.31
    (fuelIn 1.105). Compressible-side, found, not investigated: the candle's
    fuel budget does not close (burnRatio 1.12 at etaIn 1000, 1.03
    default) and zMax reads 0.851953 constant in both variants, above the
    rows' max Z 0.772 (a maximum-principle violation somewhere in the
    pinned row / band). OPEN: which inflow is right at Pe 1 needs the foam
    candle twin (G3b, built, not run) or a face-gradient ladder.
LM16. **Gravity is the S7 `Grav` instantiation only; no buoyancy in T**
    (by design: P0 constant, no pressure work; the compressible E term −g
    m_x is O(M²)). The face density is the one the face velocity reads
    (`rhoX`), so a backflow face at x-hi buoys as ambient (0). With
    PERIODIC laterals the ambient ρ_a is still the Tu room air (warned: a
    lattice of plumes; a burnt coflow is then buoyant everywhere). Not
    built: gravity on the planar flame (`--mode premixed`), a tilted g, a
    stratified ambient.
LM17. **The rim is a STAIRCASE, not the plan's sub-cell ghost view** (S8,
    docs/chem.md "Low-Mach rim"): a cell is solid when its centre is in the
    wall (`lmTubeSolid`), every MAC face touching it is closed (m = 0, the
    pressure codes), no-slip by the antimirror of the closed tangential
    neighbour, isothermal by the face conduction ρD(T_w)(T_c − T_w)/(h/2).
    Exact for grid-aligned walls (the slot's lips: the developed channel
    holds to 6.4e-7 U, dp/dx = the discrete curvature to 7e-6, wall shear
    = −dp/dx W/2 to 1e-6; the linear conduction profile to 2.4e-7). The
    ROUND tube's circle is a staircase: first-order wall position, its
    Poiseuille rows (analytic, renormalized to the staircase area) are not
    the scheme's steady state and re-develop in the tube. OPEN for S9: the
    final tube is 57 cells across (wall-position error ≤ h/2 = 0.9% of
    R_i); a sub-cell isothermal/no-slip ghost view (gpu/ghostfill's image
    rule on MAC faces + a cell-centred T instantiation) if the 3D base or
    qRim moves with the staircase phase (shift the axis by h/2 and
    compare). Not covered: the stratified column with a tube in it (S7's
    note) -- the rest case is trivially exact; a tilted/stratified test
    would need a T_w matching the column.
LM18. **The cold rim anchors the model's rich φ 1.3 slot at EVERY admissible
    jet speed** (S8, `runs/s8/map`, 2D, M1 mapping, W 14, lip 1.4, open,
    tt 40): U 1.5 … 20 at 4 cells all anchored over the lip, stand-off
    0.62–0.87 δ_th (4 cells; the base row is a cell centre) → 0.69 at 8
    cells (U 6) = 0.42 mm at M1; no flash-back at U 1.5 (the validator's
    floor), no blow-off at U 20 (cones above U 10 still within their first
    flow-through at t 40: the BASE settles in ~15 τ). `--rim-temp 1.5`
    moves the base < 1 cell (rBase 35.0 vs 37.0) and halves qRim. The
    limits are the MODEL's (one-step quench at Pe 0.6×, D1/K7) -- not a
    CH4 flash-back/blow-off map. OPEN: U < 1.5 (validator), longer runs at
    U ≥ 10, 3D limits; preFrac 0.92–0.98 in 2D (outer cone open, LM13); in
    3D round 0.7998.
LM19. **The apertured PCG costs 2.5× the direct path at the final grid**
    (S8): 392 × 180 × 180 with the rim 171 ms/step vs 69 with pilots (direct
    solve), pcgIt 6–14 per stage solve warm (≤ 15 everywhere measured; 13
    cold), 3D 389k cells 7.4 ms/step (launch-bound). Section 5 at 100 τ =
    11k–20k steps ≈ 1900–3400 s (2 simjobs). Levers (not built): `pcgTol`
    1e-5 (S3: −2 it/solve; an API knob, no flag), deflation of the tube
    modes, a fused Krylov step.
LM20. **The 2D rim slot's IC transient leaves the bounds for t < 10–15 τ,
    at ANY dt** (S9, `runs/s9/dt`): T up to 7.2 (σ_st 6.977) and Y_O above
    the air's (Z down to −1.2e-2) with the cell rule, `--chem-dt 0.5`,
    `--cfl 0.2` and at dt 0.0037 alike; both return inside the bounds by t
    15–20 and stay (yHits counts only Y outside [−1e-6, 1 + 1e-6], so Y_O
    > Y_O2 is not a hit: watch zMin). The 3D round burner shows it 30× weaker
    (scout A: Tmax 7.013 at t 1.7, Z ≥ −1.8e-7). Cause not isolated: the
    unprojected IC (LM12) and its Burke–Schumann products column against
    the air. Harmless for the S10 frames (t ≥ 130); OPEN if an early-time
    field is ever quoted.
LM21. **The real burner's outer cone is ~4× the half-scale one: the plan's
    62 mm box is too short** (S9 scouts, docs/chem.md "Low-Mach dt rule,
    real-burner scouts, renderer"). Half scale (D 5 mm, U 6.5, 1.5 D half
    box, tt 160): inner tip 19.31 δ_th above the rim (11.9 mm), outer 39.88
    (24.7 mm). The inner cone scales with D at fixed U, the outer with the
    jet's volume flux (Roper L_f ∝ Q): full scale ~39 δ_th (24 mm) and
    ~160 δ_th (~99 mm) -- section 5's 62 mm (100 δ_th) would cut the outer
    cone; S10 runs --length 200 (124 mm, 840 × 196 × 196 = 32.3M cells).
    **The lateral box biases the outer tip** (K10): 2 D vs 1.5 D half box
    −3.7% over [140, 160] (−5.0% over [120, 160]) while the 2 D box is still
    swinging ±3% at t 160; inner cone, base, preFrac ≤ 0.1%. Above the
    plan's 3% gate; S10 keeps 1.5 D (2 D doubles the case to 61M cells, ~19
    jobs, and does not settle in the same time) and quotes the outer tip as
    1.5 D-box value with a −4 ± 1% box bias. OPEN: the converged box
    dependence (a 2.5 D scout, tt ≥ 250), Pe 0.74× real (D1) shortening the
    model's outer cone vs CH4's.
LM22. **The outer tip's slow transient** (S9 scouts): after the starting
    head the outer tip overshoots (+5% at t 35), dips (−7% at 48) and
    settles over ~100 τ at half scale, phase-locked to a box-scale lateral
    mass-flux swing (mdotLat ±5e4 → ±3e3, period ~30 τ); the inner cone and
    the base settle by t 15. Full scale (2× the box, 4× the outer cone) is
    expected ~2× slower: S10 runs tt 160 with frames from t 130; its
    xTipOut column decides whether the clip is the settled burner (a
    drifting tip is reported, not hidden).
LM23. **`--emit volume` cannot be restricted to channels**: the low-Mach
    burner writes temp, fuel, oxid, omega, findex, speed every frame; the
    renderer reads omega + findex only. S10 at 840 × 196 × 196 (32.3M
    cells, 129 MB per channel-frame) × 121 frames = 94 GB, of which 31 GB
    the renderer needs; the rest serves `flame_dnsvid` / diagnostics or is
    deleted after. Not built: a `--frame-fields` selector (a viewer-format
    question, docs/viewer-format.md).

### 3D premixed in turbulence (`--nz`, `--turb`, docs/chem.md "3D and turbulence")

Measured gfx1100 via simjob, 2026-10-29. Numbered T1… beside the other
flame branches.

T1. **The Taylor-frozen inflow is not a Navier–Stokes solution.** The box
    is swept in frozen; the face imposes it through the LODI relax, so the
    first ~l_t inside the domain is an adjustment region (pressure, the
    sink band's D4 on mesh-scale content). Inflow-row tracking of a smooth
    solenoidal row set: v, w residual 4.7% / 1.4% of the amplitude at 16 /
    32 cells per wavelength (∝ k²: the transverse fluxes the LODI row keeps
    against etaIn c/L_x); u (acoustic L5, half the rate) 6.0% / 4.4%,
    attributed, not proven, to that rate (`gpu_flame_turb_inflow` (c)).
T2. **The turbulence decays in transit; no forcing.** Re_t ≤ 44 at the
    inflow, a narrow-band spectrum (kp L/2π = 2.4), no inertial range.
    One realization of so few energetic modes is anisotropic: the runs'
    seed-1 box has longitudinal L11 = 1.04 / 0.81 / 0.84 l_t (x / y / z,
    mean 0.90; seed 2 mean 0.78) against the lattice expectation 0.979 —
    seed dependence of S_T unmeasured.
    u' at the flame (stats `uPrime`, the slab x_f − 3 l_t … x_f − 1.5 l_t)
    is a fraction of the inflow u': 0.92 / 0.88 / 2.49 at u'_in 2 / 4 / 8
    (upstream 8 / 16 / 16 δ_th). The IC's fresh gas is
    age 0 everywhere while the stationary state has age x/u_in: the run
    is not stationary before the inflow front has crossed the brush.
    `uPrime`'s slab overlaps the brush's leading edge at high S_T (TPRI
    lesgap): DNS 4 cells seed 2, t ∈ [40, 80], slab [17.9, 23.9] δ_D with
    xLead 18.4 reads u' 1.35 against 0.88 for seed 1 (σ 2 seed 2: 1.41 vs
    0.92). Fixing it changes a default 3D stats column (bit identity):
    not done.
T3. **The plane-mean x_f is the wrong controlled variable.** Measured: a
    skewed brush moved x_f downstream while S_c rose; holding x_f flashed
    the flame back into the inflow band (T2b/T3 of 2026-10-29: S_c 8, I0
    2.2, omegaEdge 0.3–0.43). The controller holds ∫ρY dx since; the 3D
    guard exits 8 on the leading edge.
T4. **P1 turbulence gates: what failed** (docs/chem.md, "3D and
    turbulence", runs R1/U2/U3 at 6 cells/δ_th, window [10, 20]). PASS: I0 =
    1.006 / 1.006 / 1.043 ∈ [0.8, 1.2]; x_I held within 1.4 δ_D; correlations
    within [0.5, 2]× of Peters (0.71 / 0.83 / 0.98). FAIL: (a) not stationary
    — window halves differ 23% / 33% / 17% (gate 7%): S_T still grows at
    u'_flame ~0.9 (developing brush + Darrieus–Landau, σ = 5, 15 δ_th box) and
    falls in U3; tt 20–25 is all the P1 budget allowed (3–4 min each). (b) S_T
    not strictly increasing with the MEASURED u'_flame: R1 0.917 → 1.478, U2
    0.879 → 1.703 (equal u' within 4%; R1's upstream is 8 δ_th, U2's 16: the
    longer transit decays u'_in 4 to u'_in 2's level). (c) U3 Tmax/Tb 1.0114
    (gate 1.01; `Tmax` is the max of the PLANE-MEAN T, so R1/U2's 1.0019 /
    1.0041 bound nothing pointwise — the 3D pointwise peak is not measured)
    and unburnt pockets burning in the outflow band (omegaEdge up
    to 0.386). (d) The plane-mean x_f in U3 wanders −6.2/+5.6 δ_D (x_I is
    the held quantity). (e) The plan's 8 cells/δ_th, 10 + 8 δ_th box did not
    contain the brush (15–17 δ_D past t 13.7, omegaEdge 0.51). Measured
    since (TPRI lesgap, docs/chem.md "Why TFLES misses the U2 DNS"): at σ 5
    the 15 δ_th box (1.15 λ_c) has a DL-unstable box mode (+0.05/τ); seed 2
    is ×1.74 seed 1 (4 cells, mean over [20, 80): 3.325 vs 1.912); 2× width
    is ×1.46 over [20, 40) (3.090 vs 2.118); tt-80 10 τ blocks swing
    1.69–2.30. S_T at σ 5 U2 is a single-realization, box-scale-DL
    quantity. At σ 2 (box mode DL-stable, −0.02/τ) the seed spread is
    still ×1.89 (6 cells, [20, 40]: 2.761 vs 1.459) — the realization, not
    DL alone. Not measured: 6 vs 8 cells in 3D; a statistically converged
    S_T. Next: forced turbulence or an ensemble of seeds.
T5. **3D yHits are the 2D FP32 inflow-row drift (§39), more cells.**
    Hundreds to ~50000 cells per row, magnitude ≤ 39 ulp of 1 above (final
    warning lines: R1 1 + 3.22e−6, U2 1 + 4.66e−6, U3 1 + 1.43e−6); never
    below −1e−6 (yMin ≥ 0).
    The stats CSV prints yMax to 6 significant digits (reads 1): the
    magnitude is the run's warning line. Not material; counted, never
    clamped.
T6. **`kFlameColumns` / `kTurbColumns` are one thread per x column**, a
    serial (k, j) loop of ny nz cells each, at stats rows only (every 200
    steps). Not measured separately; whole-run 3.1 ms/step at 1.43 M cells
    (176 × 90 × 90) and 5.9 ms/step at 2.53 M include them. Restructure only
    if a profile shows > 2% of a step.
T7. **`--nz > 1` exists for `--mode premixed` and `bunsen`** (P3, below);
    diffusion, candle, chamber are 2D.

### LES species and TFLES (`--les`, `--tfles`, docs/chem.md "LES species and TFLES")

Measured gfx1100 via simjob, 2026-10-29.

L1. **FAILED at σ 5 — root cause: the DL box mode, measured; gate
    re-scoped (TPRI lesgap, docs/chem.md "Why TFLES misses the U2 DNS" and
    "LES vs DNS at σ* = 2").** U2 case (u'_in 4, l_t 2.5 δ_th, window [10,
    20]): S_T/S_T,DNS = 0.639 (F 2, 3 cells) / 0.612 (F 3, 2 cells) against
    0.85 / 0.75; modelled area 0.646 / 0.629; brush 0.60 / 0.55; Charlette −
    none −0.0002 / +0.0003. The sensor is not the loss (uniform 1.184 vs
    dynamic 1.186). (1) TFLES stretches the DL dispersion relation by F
    (s_F(k) = s_1(F k)/F, L8): the U2 box (24.0 δ_D ≈ 1.15 λ_c) mode grows
    in the DNS (+0.048 / +0.057 /τ at 4 / 6 cells) and decays under every
    thickening (F 1.5 −0.05, F 2 −0.19, F 3 −0.39 /τ; frame times
    verifier-corrected, docs/chem.md). (2) The missing area
    is RESOLVED, mostly the box mode: shell 1 carries 84% of the DNS front
    variance at t = 20 (rms x_I 6.5 δ_D vs LES 2.9 / 2.0 / 1.2); the
    a-priori Ξ_Δ is 1.024 / 1.042 / 1.093 (F 1.5 / 2 / 3), so a perfect
    efficiency lifts LES2-C from 0.64 to ≤ ~0.66. (3) The σ 5 reference is
    not a statistic (seed ×1.74, width ×1.46, blocks 1.69–2.30, T4); the
    LES/DNS ratio does not converge with a longer window (0.558 / 0.349 per
    seed over [20, 80)). The u'_in 0.25 control could not see DL (rate
    0.05/τ, area excess second order: ~1e-3 expected). The σ 5 gate stays
    FAILED, report-only, pinned by `gpu_tfles_dl`.
    **Re-posed at σ* = 2** (U2 box mode DL-stable: −0.016 / −0.019 /τ at 6
    / 4 cells; σ 3 is not, +0.010 at 6 cells): two seeds, window [20, 40].
    DNS6 S_T 1.459 (s1) / 2.761 (s2); LES2-C 0.831 / 0.511 of it (gate
    0.85), excess 0.461 / 0.233 (0.64); LES4-C 0.759 / 0.418 (0.75), excess
    0.234 / 0.088 (0.39); modelled area 0.851 / 0.559 (LES2), 0.841 / 0.504
    (LES4); brush passes; G1 halves 8.0% / 6.8% (7%); G2 DNS4/DNS6 1.090 /
    0.999. Seed 1 INCONCLUSIVE (G1), seed 2 FAILS with G1, G2 passing: **a
    non-DL gap remains at σ 2.** Its sub-filter share: Ξ(2 δ_th) 1.040 /
    1.073, Ξ(3 δ_th) 1.079 / 1.153; at most S_T,LES Ξ_Δ/S_T,DNS = 0.863 /
    0.548 (LES2), 0.819 / 0.482 (LES4) — again resolved-scale and box-mode
    dominated (shell-1 share 0.70–0.86; DNS rms x_I 6.3–9.0 δ_D in s2 vs LES2
    3.0–4.1): the box mode is linearly stable but turbulence-forced, and
    the thickened front relaxes it 5.8× (F 2) / 11.5× (F 3) faster. LES
    u'_flame is 13–21% below the DNS (not separated). Next TPRI: this gap,
    with these numbers; no Charlette/β/C_k/Colin tuning was done. (The
    premixed label now names `sigma=…` / `Zeldovich beta=…` when not 5 / 8,
    so a σ* run is distinguishable; default labels unchanged.)
    TFLES-regime point (σ 2, l_t 6, box 6 l_t, 4-cell DNS, [10, 26.5]: the
    DNS exits 8 on flash-back at t 26.9): F 2 0.762×, F 3 0.650× ([5, 12]:
    0.936× / 0.898×); that box (64.6 δ_D) is DL-unstable at σ 2 (+0.011 at
    λ 53.9) and stable at F 2 (−0.014): the same bias.
L2. **The Shvab–Zeldovich teeth's absolute gate (1e-2) is not met; the
    floor is the scheme.** A non-reacting stirred box: the molecular DNS
    alone has defect 1.58e-2 (MC-limited species vs central energy flux,
    clipping at stirred extrema; 0.32 when under-resolved). The eddy fluxes at
    Sc_t = Pr_t sit at 1.14e-2 (≤ the floor), at Sc_t 0.45 at 0.197 (×17.3);
    the test gates relative to the floor (`gpu_tfles_teeth`).
L3. **The sensor is a T-GATED max filter (spreads toward colder gas), not
    the plan's symmetric dilation.** A symmetric filter could not meet the
    coverage gate on both sides (host scan on the thickened oracle: covering
    c = 0.01 left Ω 0.03–0.6 at 2 F δ_th past c = 0.99). Constants (Ta/2,
    tanh 50, radius 2.5 F δ_th) are measured on the LAMINAR oracle only; in
    turbulence a cold pocket inside burnt gas takes Ω from its hotter
    surroundings, a burnt pocket in fresh gas takes only its own raw value —
    unvalidated against a sensor-free reference.
L4. **TFLES cost.** The sensor's three gated passes at radius 2.5 F δ_th/h
    (15–19 cells) plus curl/Laplacian: F 1.5 at 4 cells 1.69 ms/step vs 1.28
    LES alone (+32%, 0.58 M cells). A shared-memory tiled filter is the
    candidate if P3 needs it.
L5. **μ_t is ignored in the NSCBC rows' transverse flux and in the open-row
    species/energy face** (open faces carry no diffusion, molecular or
    eddy); the open-row μ_t is a copy of the neighbour row (the §17 wrap).
    Not measured at a turbulent outflow.
L6. **Static Charlette β = 0.5, δ_L = δ_th, Re_Δ on ν_u = Pr;** u'_Δ by
    Colin's operator on the resolved field, not filtered. Dynamic β, Colin's
    efficiency (needs a global Re_t) and a separate Sc_t are not built; the
    plan's quarantined `val_turbflame_les` is not registered (the gate fails,
    L1, also at σ* = 2; numbers in docs/chem.md).
L7. **Colin's operator with (Δ/h)^{1/3} up-scaling under-reads the a-priori
    Ξ_Δ by 2–7% in near-DNS fields (h/η ≤ 1.4) at σ 5:** Ebar 1.00001 /
    1.0004 / 1.024 against Ξ_Δ 1.024 / 1.042 / 1.093 (F 1.5 / 2 / 3, U2
    seed 1, t = 20). Secondary: ≤ 9% of S_T. Not fixed.
L8. **TFLES stretches the Darrieus–Landau dispersion relation by F**
    (s_F(k) = s_1(F k)/F; `gpu_tfles_dl`: s(W_f2)/(s(m_dns4)/2) = 1.00).
    Any box or burner with modes in (λ_c, F λ_c) loses them (λ_c ≈ 21 δ_D
    at σ 5); at σ 2 the DL-stable box mode is still damped 5.8× / 11.5×
    faster at F 2 / 3 (−0.112 / −0.223 vs −0.019 /τ). Inherent to the
    thickened-flame operator; no efficiency function restores
    resolved-scale modes. The same scaling check at σ 2 (both rates
    negative and small: F 2 λ 30 −0.0137 vs DNS 4 λ 15 −0.0194) reads
    1.411, just outside [0.6, 1.4] (verifier, frame times corrected; it
    read 1.397 with every frame stamped 0.5 τ late) — the σ 5 pin is the
    load-bearing one; the σ 2 ratio is fragile, not a counter-example.
L9. **The efficiency does not reach the consumption speed at σ 2.** Ebar
    1.016 / 1.096 (F 2 / F 3, seed 1; 1.037 / 1.147 seed 2) — at F 3 above
    the a-priori Ξ(3 δ_th) 1.079 — yet Charlette − none moves S_T/S_L by
    +0.0012 / −0.0019 and I0 1.0155 vs 1.0144 (F 3). Candidate, not
    measured (E is not in the frames): Ebar is Ω-weighted over the sensor
    region (2.5 F δ_th into the turbulent fresh gas); E > 1 in the preheat
    zone with E ≈ 1 at the reaction layer moves no consumption. A
    flame-normal E (constant across the thickened front) is the textbook
    remedy; not built.

### 3D turbulent slot Bunsen (`--mode bunsen --nz`, docs/chem.md "3D turbulent slot Bunsen")

Measured gfx1100 via simjob 612–617, 2026-10-29.

B1. **The video run is LES of a near-DNS velocity field with a 3×
    thickened flame; no reference.** U_jet 8, u'_in 3, l_t 6 δ_th, h/η 1.06
    at the inflow, TFLES F 3 + Charlette (Ebar 1.013: the efficiency adds
    ~1%). Span-mean S_T/S_L 1.586, A_T/A_cone 1.424, I0 1.113 over t ∈ [18,
    36]; no DNS of this burner exists to compare against, and P2 (L1)
    measured TFLES at 0.61–0.64 of the DNS's S_T in the planar case. The
    plan's U_jet 20 was not run (its cone would not fit at that S_T); the
    plan's scout (1.5 cells, F 5) was replaced by a short scout of the
    final grid (F·cells sets what the scout would mean). At σ 5 the span
    (2W = 48 δ_th ≈ 77 δ_D) admits DL modes TFLES 3 suppresses below ~63
    δ_D (L8): S_T carries that bias. "FEmax 3√3 every row" is NOT a
    turbulence signature: the LAMINAR extruded burner (P3 flags, nz 8, tt 6,
    TFLES 3, Charlette) reads FEmax 5.196 = 3√3, Ebar 1.0048, area 0.9447 vs
    0.9442 for E = 1 (TPRI lesgap, analyst 2) — Colin's operator fires on
    laminar jet shear.
B2. **Tmax/T_b 1.131 at the end, ALL in the inflow band.** The pointwise
    peak grows from ≤ 1.043 (t ≤ 27) to 1.131 (t 35.7); every cell above
    1.03 T_b sits in rows i ≤ 12 at the jet edges (15393 cells in the last
    frame), where the base burns in the LODI rows and the boundary heat is
    re-added (§35); the interior (i ∈ [16, 266)) peaks at 1.025 T_b. Cause
    (turbulent edge rows, the TFLES rate at the face, the re-added q ω) not
    isolated; no 2D or laminar 3D control run; still growing at the end of
    the run.
B3. **The 3D cone measurement changed two 2D rules (3D only).** The burning
    window of the flank crossings is ½(2F − 1) δ_th (TFLES puts the
    reaction zone behind Tmid; the 2D 1-cell window at F 3 dropped every
    flank row and exited 8 at t = 16.1 with burn/in 0.977); the "fewer than
    4 burning flank rows" guard does not stop a 3D run (fit columns NaN). The
    2D bunsen with `--tfles` keeps the 2D window (bits) — it has the same
    latent problem, not measured.
B4. **ST and area use the straight-line cone area** 2 hypot(xTip − xBase,
    W/2) L_z of the SPAN-MEAN Tmid contour; the span-mean tip swings ±7 δ_D
    (sd), so ST (sd 0.12) and area move with it while I0 (sd 0.02) does
    not. I0 1.11 > 1 (P1's planar DNS read 1.006). The cone denominator
    CANCELS in I0 (I0 = burn / ∫|∇c| dV), so it is not a candidate (P3
    verifier). I0 > 1 is not a turbulence signature: the LAMINAR extruded
    burner of `gpu_bunsen3d_extrude` (W 6, U 3, tt 2, transient) reads I0
    1.064 (DNS) and 1.130 (TFLES 2). Candidates, not proven: tip and base
    consumption per unit |∇c| (curvature, the burning inflow rows of B2), E > 1
    (Ebar 1.013, at most ~1%).
B5. **The box × J(y) is not plane-mean free** (the z-mean of each y row of
    the box is not zero): an inflow mass-flux fluctuation, an acoustic
    source not measured. The coflow carries no turbulence; the turbulence
    decays over the ~14 τ_F transit to the tip (u' at the flame not
    measured; `uPrime` is planar-only).
B6. **Span means are host loops** over five downloaded fields (5 × 19 MB)
    at every stats row; the run's 16.3 ms/step includes them and the
    41 GB of volume frames (not separated). A GPU span-mean kernel is not
    built.

### Wick candle (`dns flame --mode candle --wick`, docs/chem.md "Wick candle")

Measured gfx1100 via simjob, 2026-10-28. Numbered W1… beside the chamber's.

- W1. **The fuel rate is FIXED, not tied to the heat flux into the wick.**
  The flame delivers qWick = 0.041–0.044 Q q at 8 cells/l (2.7 Q cp T0: the
  vapour's sensible heat T0 → 2 T0 plus ~1.7 cp T0 ≈ 0.5 MJ/kg at T0 300 K,
  paraffin's latent heats' range — a consistency, not a validation), but
  0.022–0.028 at 12 cells (W6). A wick model (ṁ = qWick / (L_v + cp ΔT),
  wick T from a pyrolysis balance) is not built; the candle body takes
  0.12–0.14 Q q (conduction into the T0 wax top under the flame base) and
  nothing melts.
- W2. **Soot model S1 is one equation and appearance-tuned.** Ta_f 50 T0,
  Ta_ox 60 T0 from literature brackets, Da_f 10 / Da_ox 2e5 tuned so the
  soot burns out at the tip (scouting table in docs/chem.md); no
  nucleation/growth split, no PAH, no thermophoresis, no molecular
  diffusion, no radiation (a real candle radiates ~20–30%: absent, so the
  flame runs hotter than one with radiation would). f_v peak ~1.2 ppm (real
  candles 1–10 ppm). Smoke point and smoking threshold not mapped.
- W3. **Tmax 0.875 T_ad** (4.375 T0 at the defaults, t 20–24) against the
  slot candle's 0.977: heat losses to the isothermal body (0.125 Q q) and
  wick (0.041 Q q) are the candidates; the strain at the base not
  separated.
- W4. **Anchoring needs Da 5e4 at g 14** (5000: lift-off and exit 8 at t 3;
  2e4: base wandering 0.6–1.6 l; `--wick-temp 3` at 5000: base 0.125 l). No
  map in (Da, g, Q, T_wick); the base sits 0.75 l above the wax (row
  quantized, flat) — a real candle's base is ~1–2 mm below the wick top's
  level; here it is 1.25 l BELOW the wick tip.
- W5. **hits ~8000 fluid cells** (Y_O above Y_O2 by up to ~2e-5, Z down to
  −6e-6): the FP32 rounding walk of the long M 0.005 run, the slot candle's
  start shows the same (37488 at t 1, issues §39 mechanism). Counted, never
  clamped.
- W6. **Wall heat fluxes are grid-dependent; the flame is not.** 8 vs 12
  cells/l at t 4–6: height 5.988 / 5.973 l, half-width 1.110 / 1.100, Tmax
  4.360 / 4.354 T0, soot formed 0.052 / 0.051 Q — but qWick 0.044 / 0.022 and
  qBody 0.122 / 0.143 Q q (the staircase wall's conduction on a 6- vs
  9-cell wick; 4 fallback ghosts at 12 cells). Two grids only.
- W10. **Not strictly steady.** The flame height cycles slowly (5.99 →
  5.43 → 6.13 → 6.36 l over t 4–24, 8 cells); t ∈ [20, 24] halves agree to
  1%. The slot candle's slow box cycle (§50, §51) is the suspect; the
  periodic box's recirculation (min top u −8 → −4.5 U) is still relaxing.
- W7. **The wick is uniform T_wick, no char tip, no glow**; the body's top is
  flat at T0 (no melt pool, no meniscus); the body floats 1 l above the inflow
  band (the coflow passes under it). The render shows its top 2.5 l only.
- W8. **The render's absolute scales are assumptions**: T0 = 300 K, slab
  depth one cell = 0.4 mm, E(m) 0.26 — they set colour temperature (T) and
  nothing else after the automatic exposure; the chemiluminescence gain is a
  display choice (one-step chemistry has no CH*); white balance 1900 K.
- W9. **Tip / Z read the gas-phase mixture fraction** (Y_F, Y_O): soot is
  outside it; at the tip soot carries < 1% of the fuel element.

### Chamber (`dns flame --mode chamber`, docs/chem.md "Chamber")

Measured gfx1100 via simjob, 2026-10-28, defaults (4 cells/δ_th) and the
6-cell demo. Numbered C1… so they merge beside the other flame branches.

- C1. **Thermo-acoustic limit cycle, mode not identified.** A self-excited
  oscillation at f = 2.95 (4 cells) / 2.93 (6 cells) per flame time grows
  after t ≈ 35 (burn-rms log growth 0.075 / 0.052) and saturates at 4 cells
  (burn rms 6.7%, plenum p rms 0.36%). 1D two-temperature longitudinal
  modes bracket it (2.74 open exit, 3.14 closed exit); hot transverse 2.33.
  No pressure mode shape is output (frames: temp, fuel, omega), no geometry
  or --inject-speed scan. Whether the default should be a quiet operating
  point is undecided.
- C2. **Tmax above Tb in the holder's base recirculation**, growing with
  the C1 amplitude: 1.046 Tb (4 cells, saturated) and 1.013 Tb (6 cells,
  t = 75); ≤ 1.0011 before the oscillation. Not compression of IC gas (the
  inert check bounds that at 1.0013 × Tb (p/p0)^(2/7), and the IC gas is
  gone by t = 30). Candidates, unseparated: the species/energy scheme
  mismatch (MUSCL ρY vs central E) on a moving front at 4–6 cells/δ_th; the
  staircase wall's energy flux (viscous work at a face whose mirror
  velocity is not zero).
- C3. **The ambient beside the nozzle lips sits 4.3% above p0** (both
  resolutions). The outflow's soft pressure anchor (σ 0.25) and the jet are
  the suspects; unmeasured. Chamber ratios are quoted against the MEASURED
  pAmb, never against p0.
  Superseded for nozzle, choking and thrust by the FV reacting nozzle (docs/nozzle.md "Reacting", masterplan_combnozzle); not fixed. Thrust is never quoted from --mode chamber.
- C4. **Nozzle total-pressure drop 1.45× the isentropic one** (Cd 0.83,
  1.448 → 1.470 from 4 to 6 cells), attributed to the viscous exit at
  u_e He/ν_b ~ 100; no Reynolds or resolution study beyond the two points.
  Superseded for nozzle, choking and thrust by the FV reacting nozzle (docs/nozzle.md "Reacting", masterplan_combnozzle); not fixed. Thrust is never quoted from --mode chamber.
- C5. **FP32 injector rounding**: the inert fuel residual grows steadily
  to 3.3e-5 of the inventory by t = 8, and ≤ 178 fresh cells per row read
  Y − 1 ≤ 4.3e-6 (yHits). Attributed to the absorption of the small
  per-stage source increment into an O(1) FP32 state, not proven; an FP64
  per-cell injection tally would separate it.
- C6. **Adiabatic no-slip walls cannot quench**: wall flash-back is
  possible in principle. Not observed (injector-band burn ≤ 1e-15 of the
  total through t = 80 at 3 S_L); the flash-back / blow-off guards never
  fired in any run, so their thresholds (1e-3, 0.5) are unbracketed. No
  --inject-speed scan; isothermal chamber walls not built.
- C7. `fx` is the whole fluid → body force, the exterior coflow drag
  included; thrust is not separated. Exit-plane quantities are cell-centre
  sums over the last cavity column (x = xe − h/2), not face fluxes (ṁ
  closes to 0.1% regardless).
  Superseded for nozzle, choking and thrust by the FV reacting nozzle (docs/nozzle.md "Reacting", masterplan_combnozzle); not fixed. Thrust is never quoted from --mode chamber.
- C8. **6-cell energy-balance entry unverified** (docs/chem.md table).
  The 4-cell entry was quoted as 1.0011 but the labelled quantity
  q burn / (h0 − ṁ_in cp Tu), simple mean over t ∈ [40, 80], is 0.9989
  (verified rerun, stats rows byte-identical to the implementer's over
  t ≤ 40); 1.0011 is its inverse. The 6-cell 0.9959 was not rerun (GPU
  budget), so its definition is unknown: 0.9959 or 1/0.9959. Either way
  within 0.5%.
- C9. **Saturation at 4 cells is approximate.** Burn rms over [t0, 80]
  still rises 6.49% → 6.64% → 6.70% for t0 = 65, 70, 75 (plenum p rms
  0.344% → 0.357% → 0.361%); the quoted 6.7% / 0.36% are the [75, 80]
  values. tt 80 does not show a flat envelope.

### Shock capturing (`--shock`, docs/shock.md)

Plan: `plans/masterplan_shockflux.md`. [R] = its NumPy review harness, no GPU.
Baseline at 644e596: `dns bench --physics compressible --n 256` 20.24 ms/step,
1.46 GiB; `dns flame --mode chamber --tt 3` 0.42 ms/step (gfx1100, `--bench`).

- S1. **KG is not pressure-equilibrium preserving.** `rm ua em` with
  `e = E/rho` gives `p/(gamma-1) (rho_a + rho_b)(1/rho_a + 1/rho_b)/4 !=
  p/(gamma-1)` at a density jump. Density ratio 10, uniform p, u, 200 cells,
  KG alone, max dp/p [R]: M 0.02 sharp 3.93e-2, tanh 2 cells 5.24e-4; M 0.3
  sharp 0.51, tanh 2 7.6e-3; M 1 sharp diverged, tanh 2 4.25e-2; M 3 sharp
  diverged, tanh 2 0.10. A supersonic contact kills KG with no shock in
  sight. The pressure sensor rescues the sharp case (M 1: 0.23, M 3: 0.59),
  stability rather than accuracy. Bears on C2 (Tmax above Tb) and every flame
  run. A PEP flux (KEEP-PE, Shima, memory) would fix it at the root but moves
  every anchor: not in this cycle.
- S2. **Outflow rc 7, not understood.** `dns: stable dt is not finite and
  positive (the state diverged)`, exit 7, from starting vortices at the
  outflow. From the brief: `--inject-ramp 4 --coflow 12` fails; `--downstream
  80 --inject-ramp 4 --coflow 6` passes; the sigma 10 chamber default
  diverges. `--shock` is masked on the rows and no evidence ties this to a
  shock. Candidates, unseparated: NSCBC at a vortex crossing, the coflow
  ratio, `--inject-ramp`, sink. Not a threshold to tune. Measured with `--shock
  sense` (central solver, step 3): `flame --mode chamber --expansion 10 --tt 60`
  exits 7 with `dns: chamber state is not finite at t=14.79 (diverged)`, not the
  brief's dt text; the open-row |u_n|/c reads 0.36 at t 13.7, 1.45 at t 14.74,
  max sensed cells 30, then 111, then 148 at t 14.78, all but 2 outside the
  cavity. `flame --mode chamber --tt 60` (sigma 5) runs, open-row Mach max 0.383.
  **Step 9's reruns, the corrected flags (A8): base `--mode chamber --expansion 10
  --cells-per-delta 6 --inject-speed 4`.** `--inject-ramp 4 --coflow 12 --tt 30`:
  none rc 7, `dns: dt collapsed -- the state diverged` at t ~4.98, the original
  failure, reproduced; hll rc 7, `dns: supersonic flow on the open row at t=16.59
  (|u_n|/c = 1.01): the NSCBC boundary is subsonic only (docs/shock.md)`, 530 x 228,
  1753 cells (1.45%), so `hll` carries it 11.6 flame times further into the guard. `--downstream
  80 --inject-ramp 4 --coflow 6 --tt 80`, 770 x 228: none rc 0; hll rc 0, 332 cells
  (0.19%), Me max 0.696 vs 0.728, exit-column local Mach max 0.741, 16 cavity cells
  off Exit. Base alone (the sigma 10 default at 6 cells, tt 60): none rc 7, `dns:
  stable dt is not finite and positive (the state diverged)` at t ~9.66; hll rc 7,
  `dns: supersonic flow on the open row at t=9.614 (|u_n|/c = 2.27): the NSCBC
  boundary is subsonic only (docs/shock.md)`. So `--shock hll` does not cure the
  outflow divergence, as the plan expected.
  **Separate datum, sigma 5, 4 cells, 364 x 152, NOT the original failure:**
  `--inject-ramp 4 --coflow 12`: none rc 0; hll rc 7, `dns: supersonic flow on the
  open row at t=52.28 (|u_n|/c = 1): the NSCBC boundary is subsonic only
  (docs/shock.md)`. Under `sense` (no D, the none solver) the open row never reads
  1 at stats cadence: max 0.953 at t 9.18, 0.762 in t 48-56. So at t 52 the sonic
  row is `hll`'s trajectory, not inherited there; it had reached 0.95 at t 9 without
  D. Stats cadence, so a shorter excursion is not excluded. Mechanism not separated.
  Sigma 10, 4 cells: `--expansion 10`: none rc 7, `dns: chamber state is not
  finite at t=14.79 (diverged)`; hll rc 7, guard at t=12.95, |u_n|/c = 1.01.
- S3. **The staircase at M > 1.** Input to the geometry cycle. M 3 over
  blocked cells, `kWallRhs` rule, no ghost image [R]: exact p2/p1 vs mean wall
  p: compression 10 deg ny 60 2.054 vs 2.096 (18% marked), ny 120 1.992 (9.5%);
  compression 5 deg ny 60 1.454 vs 1.694 (25%); expansion 10 deg ny 60 0.431
  vs 0.631 (14%), ny 120 0.461 (8.6%); expansion 5 deg ny 60 0.668 vs 0.808
  (13%). Every stair is a shock or fan source. A diverging nozzle is the
  expansion row: +46% at ~50 cells of wall, +7% at ~100. Gentler walls are
  worse per stair. A CD nozzle needs >= 100 cells along the diverging wall, or
  a body-fitted mesh. Where the CD nozzle lives (FV vs structured curvilinear)
  is raised at the geometry cycle: masterplan_shockflux section 11.
- S4. **Gate 5's first bracket was the wrong sensor's.** Masterplan section 3.2
  quotes the moving-shock ringing of `S3` with 6-neighbour dilation
  (`Cfg.dil='cell'`, FP64: Ms 1.2 0.033, 1.5 0.045, 2 0.027, 3 0.017, 6 0.010);
  decision 5 ships axis dilation. Host port (`shockflux_weak`, FP32): 0.0607,
  0.0061, 0.0092, 0.0150, 0.0127; the review harness at `dil='axis'`, FP32:
  0.0608, 0.0061, 0.0092, 0.0152, 0.0128. Ms 1.2 missed the first band 0.06.
  Not a port bug. Amendment A3: the user set the Ms 1.2 band to 0.07 with the
  axis brackets in gate 5; no other band moved. Not a device re-baseline.
  **Amendment A5 (user decision): FP32 gates are control-relative.** Rerun: gate
  15 passes, near-vacuum rungs at 2 x the rung's host FP32 vs FP64 gap. Gate 22
  passes: controls are the same model, no --shock, a smooth wave driven at the
  shock run's own dt sequence, equal steps. LES mass 3.23e-5, E 8.70e-6 vs
  control 3.63e-5, 1.81e-5 (ratio 0.89, 0.48, 5290 steps: the LES dt is ~10x
  smaller); plain 2.83e-6, 1.01e-6 vs 3.71e-6, 1.98e-6 (0.76, 0.51, 503 steps);
  compensated 1.01e-8, 5.20e-9 (absolute, control 2.34e-10, 2.74e-10). The cause
  of the plain FP32 drift stays UNPROVEN (FMA, the second write on a sensed cell
  are candidates, neither tested); the controls show only that the untouched
  solver drifts as much at the same steps.
  **Device numbers, step 4, first run: gates 15 and 22 failed as first written.**
  Gate 15, N 200, device vs host FP32 twin (relative L1 rho; band 1e-5): Sod
  5.97e-7, Lax 3.85e-6, 1-2-3 2.86e-6, blast 3.23e-6, M3 1.02e-6, p ratio 1e5
  3.16e-6, coll 1.82e-7 pass; the near-vacuum rungs fail: u +-5 2.33e-5, +-8
  2.42e-5, +-16 5.75e-5 (rho min ~1.5e-3). The host's own FP32 vs FP64 on those
  rungs: 1.83e-5, 2.13e-5, 3.68e-5, so the band sits under the FP32 rounding
  scale of a near-vacuum state. Step counts equal, rho, p > 0 everywhere.
  Gate 22, periodic double shock, FP32, 400 x 4, t 0.1 (band 1e-6): plain mass
  2.83e-6, E 1.01e-6 (prototype 3.1e-7, 3.7e-7: 9x, so section 8's re-baseline
  stops); LES (Cs 0.17, mu 0) 3.23e-5, 8.70e-6; compensated 1.01e-8, 5.20e-9
  pass. Host FP32 twin on this IC 9.11e-8, 1.64e-7. Control, no --shock, a
  smooth wave at the same step scale, 0 sensed: device KG mass 3.93e-6, host
  1.02e-6. So the plain device stage write drifts this much on its own: the
  correction's rhs is conservative (compensated 1e-8) and the plain FP32 stage
  (FMA, two writes on a sensed cell) is what the band meets. Cause not proven.
  Passed at step 4: gate 16 Shu-Osher N 400 5.26e-6; 19 Quirk vmax/S 1.19e-5,
  spread 0, standing M 5 7.9e-5; 20 vortex N 160, 320 0 sensed, bytes equal,
  order 1.995, N 80 1.224 x KG; 21 contact, M 1 .225, M 3 .593; 26 extrusion
  and permutation bitwise.
- S5. **Wrinkled flame v_max: 9.03e-4** (5.56e-4 past t 1), `--shock sense`,
  `--turb 1 --turb-length 1 --cells-per-delta 4 --ny 24 --nz 24 --upstream 6
  --downstream 4 --tt 3` (M 0.01, default; 0 cells sensed). In burning cells
  2.38e-4 (rerun after A1: the same). One short box; not a developed brush.
- S6. **3D CFL bracket on the device.** 64^3, every-step dt, hot from step 0: blast
  and diagonal Sod from rest run at cfl 0.4 AND 0.5 (rho min 0.0614, 0.125; 0.0616,
  0.0821), so the review's "0.5 fails all 5" (N 48, FP64) did not reproduce on
  these 2 cases. Upper bracket unmeasured on the device; the cap stays 0.4.
- S11. **Gate 29 FAILED on both boxes; the plan stopped at step 9.** Long box as
  written (A8), `flame --mode chamber --expansion 10 --inject-speed 8 --downstream
  80 --inject-ramp 4 --coflow 6 --shock hll --tt 20`, 524 x 152: rc 7, `dns: stable
  dt is not finite and positive (the state diverged)` at t 10.76, not the guard's
  text; open-row Mach 0.727 on the last row and its max; pMin 1933 -> 938 over the
  last 4 rows, rhoMin steady 0.1415; 14 cavity cells off Exit (band 0) from t 4.69.
  The throat does NOT choke: Me max 0.649, exit-column local Mach max 0.681,
  p0Noz/pAmb max 1.693. Cell not located on the long box; the same signature as
  the short box.
  Short box, first form, located with a temporary per-step diagnostic (removed):
  under `hll` the jet's head reaches x-hi and the state fails ON THE OUTFLOW ROW:
  i 363 (last row), sink band, no wall within 2, ambient region, not lip, not
  stair; local |u|/c 0.92, 1.03, 1.10 at the row at t 5.35, 5.60, 5.82, pMin 273
  at (363, 51) at t 5.8228; first NaN at t 5.8233, read at (0, 50) across the wrap.
  The guard samples |u_n|/c at stats cadence: 0.906 was its max. Without D, under
  `sense` (the choke refusal bypassed for this run only), it dies EARLIER and
  ELSEWHERE: t 3.764, (323, 73..75), x 127, in the jet in the ambient, 25 cells
  short of the sink, rho 5e-4, p < 0, first bad field p. So `hll` does not cause
  the failure: it carries the jet head 1.4 flame times and ~90 cells further,
  into the row, and the outflow takes over (S2). Off-Exit cavity cells: i 185, 186,
  j 65-67 and 84-86, the mirror pair at the nozzle wall 1-2 columns upstream of
  the lips, wall within 2, omega/peak ~1e-16, T/Tu 10.6-10.9: burnt gas at the
  contraction's last stair, no flame there. They are the lip's neighbours as the
  exit speeds up, bytes 7 to 77 at the first rows. Mass balance read only at rc 0.
  Long box located (step 9b, temporary per-step diagnostic, removed): ON THE ROW,
  (523, 46) of 524 x 152, x 205.52, y 18.26, no wall within 2. Row |u_n|/c max
  0.725 at t 10.704, 0.727 at t 10.761 with p at that cell already 881 (pMin 1934
  before), then 817, 744, 660, 557, 424, 203 one step apart, t 10.7608 to 10.7640;
  local Mach 0.959 at 424, 1.835 at 203 (row |u_n|/c 1.547); first NaN t 10.7645,
  read at (0, 45) across the wrap. The collapse STARTS BELOW M 0.8, the bottom of
  A9's blend: `Super` is the shipped row until the cell is already falling. Whether
  A9 cures it is unmeasured: gates 29, 29b not run (S13).
- S13. **Step 9b stopped at gates 33 and 34.** Build, guard and A9's kernel done
  (`kNscbcCells<..., Super>`, `shockRowGuard` at the dt cadence). Gate 32 bytes, 13
  rows, sense, quiet: pass. Gate 35 pass: left behind Ms 2 1.116e-2, Ms 3 1.595e-2
  of the jump, sink off 1.366e-2, 1.680e-2, shipped row 2.925e-2, 4.302e-2. Gate 30
  pass. Gate 33: all four pulses leave at <= 8.2e-4; FAILS its teeth at M 3 entropy,
  Super 8.171e-4, shipped 8.159e-4: rig has `pInf = p0`, so an entropy pulse leaves
  Lin 0 and the true L1 0, and the two rows are the same row by construction; the
  8e-4 is the interior's dispersive tail, before-exit error 7.0e-3, not the row.
  Report-only rerun at pInf/p 0.8, 1.25: Super unchanged to all digits, shipped
  3.8e-2 to 9.6e-2, 51 to 484 x worse: the teeth live in a back pressure the jet
  does not match, not in the pulse. Gate 34: fails at N 400 M, 2.74e-5 against 2 x
  held 7.40e-6; N 200 passes. Held M error is not monotone, 2.66e-4, 7.00e-5, 7.40e-6,
  1.88e-5 at N 100, 200, 400, 800, and held p falls to 5.6e-7 at N 800: FP32 noise
  under a higher-order held row, while the row is first order on purpose (Super M
  2.27e-4, 5.80e-5, 2.74e-5, 1.06e-5; p 6.39e-4, 1.63e-4, 4.79e-5, 2.62e-5).
  Shipped row 1.85, 2.66 in M at N 200, 400. Both gates need a ruling: neither band
  nor rig was changed after the failure. Ruled (A10): gate 33 pass, 7.517e-4,
  1.732e-4, 8.171e-4, 1.584e-4; Super at pInf/p 0.8, 1.25 unchanged, shipped 51 to
  484 x. Gate 34 pass, order N 200 to 800 M 1.23, p 1.32. The extrapolated reference
  quoted in A10 and S13's first form, M 3.2e-4, 1.6e-4, was NOT that row: a test bug,
  the rig's march died after one step (`dt 0 at t=0.0006997 step 1`, the stage hook
  acts on a stage's input, so stableDt read rows 0, nt - 1 with the wrap's flux, E < 0)
  and nothing checked it. Fixed: M 1.697e-3, 8.609e-4, 4.220e-4; p 8.321e-3, 4.217e-3,
  2.109e-3 at N 200, 400, 800.
- S14. **Gate 29 FAILED on the Super build: a SUBSONIC row collapse, A9 is not the
  fix.** Long box unchanged: rc 7, `dns: stable dt is not finite and positive (the
  state diverged)` at t 10.76, Me max 0.649, MexitMax 0.681, p0Noz/pAmb 1.693, 58
  cells 0.073%, no choke; the failure time and numbers of the build before A9. Trace
  of the row's lowest-p cell (temporary diagnostic, removed): no row cell has Mn > 0.8
  at t 10.7634 (Mn max 0.756); 2 cells at t 10.7640, the last step. The cell is NEAR-STAGNANT on the row with the jet's flow
  running ALONG it: (523, 43), y 17.08: t 10.61 Mn 0.011, v -8.15, rho 1.01, T/Tu
  1.09; t 10.71 Mn 0.001, v -12.59, rho 1.60, T/Tu 0.70. Then (523, 46), y 18.26: t
  10.748 Mn 0.059, v -6.66, rho 6.88, T/Tu 0.130, p/p0 0.89, the first row cell under
  0.9 p0. T/Tu 0.13 is far below any fluid in the box (Tu is the coldest): an
  unphysical heavy cell built on the row, p about 1.1 p0 throughout, the neighbour
  i 522 at 1.21 p0. From it the cell empties outward: Mn 0.11, 0.17, 0.23, 0.34, 0.45,
  0.76 at p/p0 0.80, 0.67, 0.58, 0.46, 0.37, 0.24 (t 10.7528 to 10.7634), then Mn 1.56,
  M 1.85, p/p0 0.11 at t 10.7640, one step before the NaN. Lin 5e2 against the true
  L1 3e4, the transverse term 100 x the relaxation.
  Reading (not proved): rho and T pile up on a row where un ~ 0 turns off L2, L3, L4
  (all proportional to un), the sink and D do nothing there (D masked, sink under
  the overwrite), and the row's only y transport is the central flux. No `Super`,
  no term past M 0.8 reaches it. A design decision, not a tuning. **That reading is
  superseded by S15**: the row's lowest-p cell is the end of it, not its start.
- S15. **The row collapse is BACKFLOW through x-hi: L2's downwind difference, not a
  contact sliding along the row. `kShockRow` not built (A12 step 2 stopped).** Trace
  of the row's COLDEST cell from t 0, long box (temporary diagnostic, removed): row
  T/Tu 1.04 at t 8.53, then 0.82, 0.62, 0.475, 0.36, 0.28 at t 8.82 to 9.22, 0.13 at
  9.54, 0.012 at 10.69; that is the mirror pair j 37..45 and 105..110, y 14.7..17.9
  and its image. Backflow all the way, rate of rho per unit time against L2's:
  t 8.80 un -8.15, inner -8.02, rho 1.38 (inner 1.21), L2 +2.62, transverse -0.61;
  t 9.21 un -5.95, rho 4.14 (3.33), +2.99, -0.23; t 9.81 un -4.00, rho 16.5 (13.5),
  +1.83, -0.69; t 10.6 un -1.41, rho 72.2 (54.3), +0.99, +1.53. Measured e-fold of T
  from the list above, 0.38 t: rate 2.6. So L2 = un (c^2 dr - dp) at un < 0 is a
  DOWNWIND difference, d ln(rho)/dt = |un| (rho - rho_in)/(h rho): the row cell runs
  away from its neighbour and the backflow carries it in (the inner cell tracks it).
  comp_nscbc.hip's "Deliberately NO backflow model" comment measured a rig to 4 x
  freestream swirl; here backflow of 1.4 to 8 S_L is sustained over 2 time units,
  against coflow +6, and rho reaches 72. Why the flow turns inward there is not
  measured. S14's cell (523, 46) is where the heavy cold pool's p finally lets go.
  Row sensor, stride along the row, on the same run: max 0.0010 at t 9.02 when T/Tu is
  already 0.475, 0.0061 at t 9.74 (T/Tu 0.09), first reaches th1 0.02 at t ~10.08,
  T/Tu 0.05. Does NOT fire in time; a density term or a lower th1 was ruled out, and
  HLL on transverse faces would not touch an L2 runaway in any case.
  Gate 36(a), `test_gpu_shock outflow_contact_a`, 96 x 152, h 0.392589, p0 1785.71,
  mu 0.72, sink 16, slab T/Tu 10, 60 cells deep, u 0, 5 transits, no adjustment: V 20
  (M 0.4 on the cold gas): collapse, row T/Tu 0.009, p/p0 0.063, `dt 0 at t=11.84
  step 12496`, first T/Tu < 0.5 at the row (i 95) t 9.08, p/p0 0.965, Mn -0.273: in
  BACKFLOW, as above; gap 20 control min T/Tu 0.908, no collapse. V 10 (M 0.2): no
  collapse, row T/Tu 0.846, p/p0 0.837, control 0.918. The rig's sensor fires at t
  0.013 only because the IC is a one-cell jump, peak 0.71. So 36(a)'s signature holds
  at M 0.4, not at M 0.2, and its un < 0 at the onset points at S15's mechanism too.
  `kNscbcCells` stays as A9 left it; a backflow form for L2 to L4 is a "Not edited"
  decision and CLAUDE.md's "no backflow model" invariant is measured against it.
- S16. **Gate 36(b) FAILED on its bitwise clause; the shipped re-carry is not bitwise
  under a transverse jump.** A13's blend built in `Super`. On the V 20 rig (DNS,
  `--shock hll`, uniform Y 1/2, chamber hook order), 10 transits: rho, p > 0, row
  T/Tu min 0.720 (shipped 0.009, collapse): the collapse is cured there. But row p/p0
  min 0.124, rho 0.036, box p/p0 min 0.098, unexplained, no band on it; 66.3% of row
  samples in backflow, 0.36 of a box at rho 1 taken in. `rhoY == Y rho` on the row: off
  on 176409 of 206720 row samples, max |Y - 1/2| 1.766e-6, about 30 ulp, first at
  step 2 to 3 on i 95 only, 1 ulp (rhoY 1.99b6a8p-5 vs 1.99b6aap-5), at j beside the
  contact. NOT the blend: the same rig on the shipped row is off from step 3, 87919 of
  95000, max 2.220e-6 at 3 transits, with sink off, shock off and mu 0 each too.
  `kSpeciesOpenFix`'s `y / rs * oRho` holds Y to the bit only while `y` is exactly
  half of `rs`; a row with a strong transverse mass flux breaks that by rounding.
  Which rounding is not isolated. Gate 23 stays green: its rows see no transverse
  jump. CLAUDE.md's "exact to the bit through every stage, hook and band" is
  untested against this case. `kSpeciesStage:247` already carries `yIn` on backflow
  at x-hi, and the chamber's `yIn` is 0, so ambient Y needs no kernel change.
  `rTgt` on the chamber is `rhoI` = 1 (with `--nscbc-tinflow`, p/(R Tu)): cold ambient.
  Ruled: gate 36(b) is control-relative, row |Y - 1/2| <= 2 x shipped at 3 transits,
  where both rows live (the shipped row collapses in transit 4): 1.766e-6 against
  2.220e-6, pass. OPEN: `kSpeciesOpenFix` is left alone this cycle, and CLAUDE.md's
  "exact to the bit through every stage, hook and band" is NOT true on this rig.
- S17. **Gate 29 on the blend build: the row collapse is cured; two clauses and the
  mass clause fail, none on the row.** Long box, DNS + `--shock hll`, tt 20: rc 0,
  t end 20, at most 224 cells sensed (0.28%; 90 at stats rows). Row (temporary
  diagnostic, removed): backflow 12 cells at t 8.1, un min -2.53; 30 at t 9.1, -5.37;
  none from t 10.1 on, which reads as the jet head's passage, not sustained
  entrainment (S15's reading), unproved. Row rho max 1.21 (was 72), T/Tu min 1.05
  (was 0.012), row Y 0 to 1.6e-3: bounded, near ambient. `Super` acted: row Mn max
  1.034 with 6 cells M >= 1 at t 17 to 18.5 (openMach 1.03), 18 cells above 0.8.
  pMin/p0 stays >= 0.87: rig's dip to 0.124 does not appear.
  Throat: Me max 0.791, MexitMax 0.824, p0Noz/pAmb max 2.54 at t 15.9, 2.04 at t 20:
  NOT choked, and unsteady: p0Noz/pAmb 1.81 at t 12.9 rises to 2.54 in 3 t, then falls.
  Clause 1, burning: 36 sensed cells at most, t 13.4 to 15.4, 0 elsewhere. Clause 2,
  cavity past Exit's 3 columns: 2 to 12 cells from t 12.9, all at i 182, 183, 4 and 5
  columns upstream of Exit 187. All wall <= 2, region Cavity, mirror pair j 63..66 and
  85..88: the contraction's last stairs. The sensor reads the stair, not the flame:
  at t 12.86, Y 0, T/Tu 9.6, omega 0: (183, 64) byte 2, vx 0.0202, p[i-2..i+2] 3162
  3104 3077 2767 3246. From t 13.4 unburnt gas reaches those stairs (Y 0.1 to 0.7,
  T/Tu 3 to 9.6) and burns ON them, omega/peak 0.01 to 0.76, bytes 5 to 255, vx 0.02
  to 0.21, vy 0.007 to 0.08; (186, 65) p[i-2..i+2] 3204 3197 3514 2655 1649. Sensed
  burning cells, samples by column i 182..188 over the run, listing capped at 10 per
  row: 8, 9, 18, 21, 17, 6, 2; of all listed, none off a wall.
  Mass clause, as coded mdotNoz/mdotIn over the late half: 0.819, mdotNoz 120 to 174
  against 188.4: the cavity is not steady at tt 20, p0Noz moves 20% in 5 t. That clause
  assumes a steady exit it does not have here; no storage column to close it with.
  Y outside bounds: 1 to 249 cells, i >= 38 on the diagnostic, never on the row.
  Inject 4, tt 20, both exit 0: `yHits` none max 68 mean 14.9, hll max 56 mean 14.0:
  `--shock hll` does not add them.
  A14: the burning and window clauses stay UNMET (a real limit of D on a staircase;
  the fix is a body-fitted wall). The mass clause became the exact budget: domain mass
  change against the integrals of inflow, injection and outflow. It needed the flux
  through the END FACES: `1.5 row - 0.5 inner`, what a one-sided LODI row telescopes
  to; the row's own rho u read -1.68e-3, this -9.4e-4 of 3392 injected.
- S18. **The chamber chokes, on the long box, and blows off.** DNS + `--shock hll`.
  p0Noz/pAmb 2.5 at exit Mach 0.82 is RESOLVED: the sonic line is past the geometric
  throat, which is the exit column i 187, 20 cells. At t 16.2, inject 8: throat MThroat
  0.820, isentropic p0/pa 2.37, static p/pa 1.56, so the exit is not at ambient: the
  jet expands past it, `MJet` 1.38. The lip's stair contracts the jet (a vena
  contracta); the axis is M 0.78 at the exit and passes 1 about 5 columns on. So the
  flow is sonic downstream of the exit well before `MexitMax` reads 0.98.
  Inject 12: never choked, exit 8 at t 27.4. Inject 16: choked, longest span 5.13 from
  t 26.9, 24.4 in all; exit 8 at t 53.97. Inject 20: 2.07 from t 18.07, exit 8 at t
  20.14 (at tt 20, 1.93). Inject 24 is past `--inject-speed`'s cap, 20. Each exit 8 is
  the same guard, `dns: blow-off: 0.502 of the injected fuel leaves through the nozzle
  at t=20.14 (lower --inject-speed or raise --holder-height)`, not the solver, and
  raising the holder made it earlier: 16, `--holder-height` 7: t 24.37. So no settled
  or periodic state was reached: UNMET. The late chamber is a pressure surge of ~8 t
  with outNoz/inj 0 to 0.5, and at inject 16 p0/pa keeps climbing, mean 5.73, t >= 40,
  range 4.83 to 7.34, against the burnt choked value 5.2 from mdot (48 per unit
  throat height, sqrt(R T_b) 133, /0.685).
  Jet structure, inject 16, t 50: axis M to 2.56, a Mach disk ~65 columns past the
  exit, axis M 0.94, p/pa 0.29 to 2.75 across it, then a second cell to M 2.11, sensed
  cells per 10 columns past the exit 122 25 0 9 76 66 177 58, then 2 45, 26 53, 54 at
  the later cells. Inject 20, t 20:
  M 2.26 to 1.04, three cells. Sensed: inject 16, t >= 40, 1.24% mean, 2.1% peak.
  Row: openMach max 2.0 (inject 16): `Super` carries it.
- S19. **Gate 31's sensed rig is UNMET.** Quiet `hll`, `dns bench` 256^3: 22.64 against
  20.92 ms/step, +8.2%, pass; 1.49 GiB. Default chamber 0.56 against 0.47, +19%, on a
  case that is not quiet (lip, S7). Rig with 9.38% sensed, band +40%: +82.1%, +131% on
  a rerun (off 10.27, 10.51; hll 18.70, 24.28: the inviscid baseline is half of
  `dns bench`'s and the hll number moved 30% between runs), viscous pair +42.7%, 20.20
  to 28.83. Budget: ~1.7 ms/step quiet overhead, ~6.7 ms/step `kShockFaces` over 3
  stages at 2 evaluations per face. Levers: sense + mask in one launch cannot be done
  (the mask reads neighbours' bytes, a race in one launch), the zero-sensed skip acts
  on quiet runs only. A per-face buffer would halve the evaluations, at memory and
  launches beyond the plan: not built.
- S12. **The chamber is not hot from step 0.** Default chamber, `hll`: hot at t
  4.371, not 0 (A4, gate 27 state step 0). Under `sense` 58 cells read at t 0.027;
  under `hll` 0 at that row with dt already moved at ~t 0.3: D fired between the
  16-step cadence checks and damped the transient, so dt stayed on the default
  cadence through it. Sigma 10: hot at 0.0151. Plan 5.5 accepts this latency.
- S7. **Gate 14 failed: the plan stopped at step 3 by its stop rule.** Device,
  `--shock sense`, M 0.02, 4 cells (FP32): planar sigma 5, 10 v_max 8.98e-4,
  2.14e-3 (t 0..20, 0 cells sensed; prototype 3.12e-3 is the sigma 10 line). The
  chamber's cavity reads 4.07e-2 at sigma 5 (t <= 60), 5.83e-2 at sigma 10 (t < 10)
  against the stop value 0.0133. A design decision, not a threshold edit.
  **Where**: exactly 2 cavity cells above th1 from t ~3 on, the mirror pair at
  the nozzle lips, last column of the cavity: i 187, j 66 and 85 (x 75.08, the
  exit x_e 75.28; region Exit). Peak list at t 60, sigma 5: 3.80e-2 those two,
  then 1.48e-2 at (188, 65), (188, 86) in the ambient beside the lip. The first
  version of this entry blamed the blocked read: WRONG, see the outcome below.
  With no solid within +-2 on either axis, the cavity's largest
  v is still at the lip column: i 185, j 66, x 74.28: 1.35e-2 at sigma 5 (t 60),
  1.40e-2 (t 20), 1.43e-2 at sigma 10 (t 12), so the band fails there too, by
  1.6-7.7%. **Upstream of the contraction (x < x_n), whole stencil, the
  flame's own number: 3.31e-3 at sigma 5 (t 60), 1.31e-3 (t 20), 2.56e-3 at
  sigma 10 (t 12)**: the prototype's 3.12e-3 holds, end-of-run snapshots only.
  So the flame does not trip the sensor; the lip's subsonic corner does, at Me
  ~0.4 (sigma 5). IC transient, t < 1: 58 cells sensed (66 at sigma 10), 2 (12)
  in the cavity, max 3.39e-2 (3.10e-2). Under `hll` the two lip cells would
  run at sigma 0.45 (v 3.80e-2) to 0.96 (v 5.83e-2) on their windows' faces,
  every step, inside the nozzle's subsonic flow.
  Options for the user (masterplan step 3): the stop rule's own (S2's `kap`
  gate or a Mach gate plus triggers, or a higher th1 paid in weak-shock
  quality), or read the gate in the flame only and mask the blocked read
  (sigma 0 when a stride is blocked, or the lip's cells), which is not in the
  plan. Steps 4-11 not run. Code past step 3 was written and removed unrun.
  Sensor cost, `--bench`, `--shock sense` (raw v kept): `dns bench` n 256 21.59
  vs 20.45 ms/step (+5.6%), 1.55 vs 1.46 GiB; chamber 0.45 vs 0.43 ms/step.
  **Outcome after A1, A2 (blocked axis reads nothing): gate 14 (a) passes, (b)
  fails, the plan stopped again at step 3.** Whole run, t >= 1, stats cadence:
  (a) burning cells v_max: planar 8.98e-4, 2.14e-3; chamber sigma 5 4.18e-3
  (t <= 60), sigma 10 6.83e-3 (to the rc 7 at t 14.79), band 0.0133. (b) cavity
  cells sensed: 2 at both sigma; band 0. A1 changed neither lip reading: cavity
  v_max 4.0735e-2 and 5.8278e-2, the same digits as before it, same cells:
  (187, 66), (187, 85). The reason: the wall tip at the lip is ONE staircase cell,
  (187, 65) (`_mask.bin`: solid at j 65 only at i 187; i 185..189 fluid on j 66).
  Only the y axis is blocked; the x stride p[185], p[187], p[189] is whole and
  reads across the exit plane, from inside the nozzle into the ambient. So the
  reading is a true second difference of the pressure at the sharp lip of a
  subsonic jet (Me ~0.4), not the first difference S7 first claimed. A1's stated
  premise does not hold for these cells. Under `hll` they would hold bytes of
  sigma 0.45-0.52 (sigma 5, v 3.80e-2..4.07e-2) and up to 0.96 (sigma 10), on
  their windows' faces, every step: derived from v, `hll` was not run.
  **Gate 14 (b) was rewritten after it failed** (masterplan amendments A2, A4):
  its first form, 0 sensed in the whole cavity, failed on these 2 lip cells; it
  now excepts the Exit column: 0 cavity cells off Exit past t 1, at most the 2
  lip cells above th1. The gate moved to the failure; the failure stands as
  measured. New gate 12b (bounds fixed before any `hll` run) decides the lips.
  **A1 was reverted** (masterplan section 1): it changed neither lip reading
  and cost near-wall faces (S8). The centre substitution of 5.3 is back; gate 14
  rerun on it passes as rewritten: same (a) and (b) numbers as under A1.
- S8. **Gate 24 passes after A1's revert and A7's rewrite of its test.** A7:
  "crosses" is the exact passage, incident and reflected, from the exact shock
  speeds: within +-4 cells of travel of each, any step with a jump > 10% needs a
  nonzero window byte. 6 cases, unsensed 0 (faces crossed 16, 645; 16, 713; 16,
  716). The first form, largest jump over the whole run, was rewritten after it
  failed on 1 plateau-ripple face. History: A1 reverted after this entry's
  failure; gate 24 then failed on 1 face, the test's proxy, not a missed shock. On the reverted
  kernel, unsensed near-wall faces at 30 deg: Ms 1.5 1 of 667, Ms 3 0 of 717, Ms 6
  0 of 717; 0 deg 0 of 16 each. Plateau -4.3e-4, -1.17e-3, -2.93e-3 (0 deg), -7.8e-3,
  -8.3e-3, -9.3e-3 (30 deg), |fc| exactly 0. The one face, L (230, 47), y face, d
  -0.0023: the test takes each face's largest jump over the WHOLE run; this
  one's, 0.147, is at t 0.204, long after the incident passed (~0.11), in the
  plateau: p 3.89 3.85 3.989 3.749 4.299 | solid, an odd-even wall ripple, v 0.013.
  During the incident's own passage its top jump is 0.087 and its window byte
  255. The gate's near-wall clause and test are unchanged; it stays FAILED as
  coded. Whether "crosses" means the incident's passage is the user's call.
  **The A1-era unexplained case, jump 1.1, LL with a whole y stride, byte 0: no
  timing bug, no kernel bug.** Bytes and jumps are read from the same state in
  the same step (`shockBytes` on the state the p comes from). Face L (208, 85), Ms
  6, t 0.02664: p at(-3..1) 24, 19.7, 14.36, 10.88, 5.179; bytes 146 0 0 0 0. LL
  = at(-1): v = |5.179 - 28.72 + 24|/(5.179 + 28.72 + 24) = 0.0079: the smeared
  shock is a nearly LINEAR ramp over the stride, and the sensor is exact-zero on
  linear by design. The jump-of-1.1 estimate of 0.2 assumed a one-face step. Only
  at(-3), at the ramp's top, reads the curvature. A blocked cell with the centre
  substitution reads the ramp as a first difference, which is what A1 removed.
  First form of this entry, under A1 (all numbers below):
  Ms 1.5, 3, 6 onto an immersed plane, sub-cell offset, ghost wall,
  device. Passes: plateau p -5.3e-4, -1.14e-3, -2.93e-3 (0 deg), -7.7e-3, -8.1e-3,
  -9.1e-3 (30 deg, 256^2), band 2%; |fc| exactly 0 everywhere; 0 deg: all 16
  near-wall faces sensed. Fails: at 30 deg, near-wall faces the shock crossed
  (jump > 10%) with sigma_f 0 at their largest jump: 48 of 714, 37 of 717, 19 of
  717. They are y faces one cell off the wall (d -0.002 to -0.005, h 0.0039), jumps
  0.12 to 1.1. Window read directly from the mask: on a typical y face R and RR
  (and often L) have a solid within +-2 on both axes, so they read nothing; LL
  keeps a whole y stride, x blocked, and read below th1 at that moment. One x face
  had 2 whole-stencil cells and a 12% jump, below th1. So A1's "the cell 2 away
  ... hands its byte to every near-wall face" does not hold on a slanted staircase:
  the byte that would see the jump is the blocked cell's. Cause stated from the
  sampled faces, not all 104 were inspected. The run stays stable and the plateau
  is right; those faces run bare KG with `kWallRhs` beside them. Gate unchanged, a
  design question. The closure, imperm, twoside tests with `enableShock` pass unchanged;
  `gpu_nscbc_order`, `gpu_comp_sink` pass. Gate 22's zero through-wall mass reads
  this rig's |fc| == 0.
- S9. **Gate 25 redefined (A6): "in reach" had no definition.** First form,
  blast slab p ratio 3, NSCBC + sink, 320 x 8: rows bitwise equal to off for 409
  steps of 1177, then 1 ulp (1.49e-8) with the nearest sensed cell 12 rows off a
  face: a difference made anywhere reaches a row at finite stencil speed, so
  bitwise equality over a run proves nothing about the mask. A6's form passes: one
  stage from one state with strong jumps at the open rows, testForce vs off, rows
  0 and n-1 bitwise equal on all fields, rows 2, n-3 differ (teeth). Blast through
  x-hi finite, open-row Mach max 0.616, max sensed 5.94%.
- S10. **OPEN, solver defect without --shock: rhoY != Y rho at a solid.** Gate 23
  under A6 passes: no solid, rhoY == Y rho bitwise (0 mismatches, 753 steps,
  NSCBC rows, sink, species sink); with a solid, max |rhoY - Y rho|/rho 9.76e-7
  (154631 mismatched cell-steps) vs no-shock control 9.65e-7 (131216), same 771
  steps and dt, band 2 x. The control's magnitude is the defect: 9.65e-7, first
  at step 1, i 217, j 11, the block's face, worst at step 389, i 217, j 13. Walled
  species kernel vs `applyRhs` rho at a wall face. Not caused by `kShockFaces`;
  `kWallRhs`, `applyRhs` and the walled species kernels were not touched.

## FaceMesh / FV path (`docs/mesh.md`, `docs/fv.md`, `plans/masterplan_meshsolver.md`)

56. **The FaceMesh and the CPU FP64 FV oracle (its GPU family: item 78).**
    Built and pinned: polyMesh reader/writer,
    `fromPolyMesh` / `fromCurvMesh`, orderings, `CellGraph`, `dns mesh`
    (`plans/verify/impl_meshsolver_p1.md`); `fvflux.hpp` / `fvgrad` /
    `FvRef` (`plans/verify/impl_meshsolver_p2.md`, gates in docs/mesh.md).
    Open by plan: no scalar/species rows, no MRF frame flux, no LTS march; one
    `FvBc` per patch (no per-face profiles). Scheme limits, stated: `Far` is a 1D characteristic
    state (no transverse terms), `Outflow` fixes p (reflects acoustics),
    `Inflow` is subsonic only, `Slip` keeps the normal viscous stress.
    Unproven: the KG flux with no dissipation on snappy 2:1 meshes beyond
    one stage (U1 is P4/P7's measurement); the G4 order is shown on a smooth
    sinusoidal mesh only (random-perturbed meshes: checkerboard and KE
    gates, no MMS).
    Deferred by plan: `wedge` patches and axis prisms in `fromPolyMesh`, and
    revolved structured meshes (`axisFaces`, `MirrorZ`) in `polyOf` (P8);
    decomposed meshes (`processor*`: run reconstructParMesh), non-conformal
    couplings, rotational cyclics — all rejected by name. `meshQuality`
    needs a `fromPolyMesh` mesh (on a `fromCurvMesh` one `srcCell` is
    `Grid::idx`). Which cell order the GPU path uses (Morton vs colour-major)
    is P4's measurement. The `mB` gate (`facemesh_mb`) is local only
    (`runs/` is git-ignored); its checkMesh log was an 8-rank run whose face
    count (2 622 899) includes processor faces — the reconstructed mesh has
    2 600 459, which the gate pins.
    OPEN (found M5b, 2026-11-03, not investigated): `FaceMesh::polyOf` of a
    SHEARED CurvMesh with a y-cyclic wrap writes broken top-row cells
    (skewness 103, non-orthogonality 89 deg). `gpu_restart_fv`'s poly case
    writes its sheared mesh in C++ (port of `tools/chem/fv_flame_mesh.py`)
    instead. Suspect: the wrap's node identification under shear.
    FIXED (verifier P2): the `FvRef` diffusive dt took `max(1, γ/Pr)`, which
    misses momentum's 4/3 normal stress: a longitudinal u checkerboard on
    100:1 cells grew ×1.78e6 in 60 steps at Pr 2 and 7 (λΔt −2.67). Now
    `max(4/3, γ/Pr)` (FV-only `fvDiffusivityFactor`; the structured
    `Transport::diffusivityFactor` and every structured dt are unchanged).
    `test_fvref` E6: decays at the bound at Pr 0.72/2/7, grows ×4.1e3 at
    1.3× (no slack).

57. **`gpu/la` (P3): its first runner customer is the incompressible FV
    path (P5b, item 59); NK and every structured path still do not use it.** Gates
    LA1/LA2/LA-perf/Shim pass (`docs/la.md`; report
    `plans/verify/impl_meshsolver_p3.md`). Open by plan: device
    assembly into `EllView` slots (P5b/P6/P7), coloured line-GS (lines are
    line-Jacobi), `strongLines` seeded from wall faces. Measured, not fixed:
    restarted FGMRES(30–60) STAGNATES on the Pe 1e3 recirculating cell
    (128², upwind) with Jacobi / MC-SGS / lines (3000 it at 2e-3–7e-4) where
    BiCGStab converges (651–1325 it) — P5b's momentum solves should default
    to BiCGStab (OpenFOAM's `PBiCGStab` choice) or larger restarts until
    AMG; MC-GS in natural order costs 1.86 SpMV per sweep vs 1.09 in
    colour-major numbering (P4's global order decision). Unproven: the
    perf numbers on a polyhedral mesh (gated on the 7-point 100³ graph only);
    the CUDA backend RUN (compiled and linked, no NVIDIA device here).
    P3 review defects fixed with P5: `LineSmoother::make` now rejects a line
    set that is not a permutation (out of range / a row twice). Still open
    from that review: Krylov with N == 0 leaves a sticky launch error,
    `KrylovOptions::restart` above the basis is clamped silently,
    `BlockEll::upload` compares nnz but not columns.

58. **`gpu/la` AMG (P5); first customer: the P5b pressure equation (item 59)** (`docs/la.md`, report
    `plans/verify/impl_meshsolver_p5.md`). Gates: refresh bit for bit vs the
    host Galerkin; LA3 (Poisson 12/16/15 it 64²→1024², aspect 100 6/21 vs
    iso 15); LA4 (FFT-PCG 14 vs AMG-PCG 15, p equal to 9.4e-8); Setup 1.83 s
    at 6M rows. On OpenFOAM's own pressure matrix (identical fields to 1e-14)
    AMG-PCG needs 6–7 iterations to OpenFOAM's relTol 0.01 residual where
    serial GAMG needs 6–8 V-cycles. Measured, not fixed: the cycle is
    LAUNCH-bound on coarse levels (~3 µs per launch; a B_G1s iteration is
    5 ms, most of it coarse-level launches) — hipGraph capture is the lever,
    not built (the library launches on the null stream); the coarsest dense
    inverse is one 1024-thread block (1.4 ms at ≤ 128 unknowns, 218 ms at
    ~1000: keep `coarseUnknowns` small; since the performance round the
    register-resident kernel at ≤ 128: 0.51 ms at m = 114, FP64-bound on one
    workgroup); K-cycle + Jacobi coarse levels is
    POOR on convective block systems (B3 recirculating cell FGMRES 194 it vs
    90 with GS coarse) — the defaults are the pressure's. The pressure
    matrix of a real SIMPLE iteration (rAtU-weighted, non-orthogonal
    correction in the source) is now exercised by P5b (item 59: scaled-V +
    2 Jacobi sweeps beat the K-cycle there, 34.9 vs 45.2 ms/it on F1).
    Unproven: AMG on HIGH-aspect
    layered (BL) meshes beyond the synthetic aspect-100 box (the local snappy
    meshes mA/mB/mC carry ~3.5 of 5 prism layers, max aspect 16.6–20.1, and
    converge in 18–22 it — nothing above aspect ~20 is tested on a real mesh);
    FP32 AMG preconditioning an FP64 solve; block AMG with the
    masterplan's physical strength `|ṁ| + ρν|S|/d` (gates use |block|_F of
    the values); the CUDA RUN.

59. **Incompressible FV path (P5b, `docs/fv.md`): `dns flow --polymesh CASE`
    = OpenFOAM v2606 simpleFoam (SIMPLE / SIMPLEC, laminar, MRF) on
    the identical polyMesh, FP64 on the GPU.** Gates (report
    `plans/verify/impl_meshsolver_p5b.md`): IF1 four blockMesh cases with
    tight solves equal simpleFoam to ≤ 1.1e-12 of scale at every compared
    iteration (`gpu_fv_simple`); IF2 snappy sphere Re 100 cd 8.8e-13; prop
    F1/G1 + MRF thrust 4.3e-6 / 5.3e-6, torque 3.7e-8 / 2.0e-9 at 600 / 700
    iterations; IF3 8.2–8.7× simpleFoam np 12 to a matched residual on F1
    (708k), 7.9× wall on G1, but only 1.4–1.5× on the 66k sphere (launch /
    host-sync bound; CORRECTED count: ~1470 dispatches + ~71 synchronous
    copies per iteration on rotor2 L1, not "~40 launches + ~15 syncs").
    PERFORMANCE ROUND (`plans/verify/impl_performance.md`, docs/fv.md
    "Performance round"), bit-identical to the FINAL outputs: rotor2 L1
    8.69 → 3.96 ms/it, L2 10.11 → 4.99, L1 SST 14.58 → 7.18, F1 32.02 →
    22.07 (vs simpleFoam np 6: 5.4× / 5.9× / 6.8× / 10.9× per iteration,
    4.4× / 4.8× / 7.3× / ≈ 10.2× to a settled CT/CQ); 905 dispatches + 21
    syncs per L1 iteration. Not done (USER decisions,
    `plans/masterplan_performance.md` §6): non-parity levers (stronger p
    smoother, FP32 AMG, Anderson), a re-swept foam baseline (np / renumber),
    the integer-ticket one-launch dots and the single-workgroup coarse tail;
    hipGraph (needs a stream through gpu/la). Remaining L1 active iteration:
    GPU busy ~4.2–4.6 of ~5.6 ms (launch latency + 21 syncs), the dense
    coarsest inverse 0.5 ms of it. Semantics moved (failure paths only, no
    bit of a successful run): a singular p AMG is reported at its first use
    in an iteration (an iteration whose p solves all start converged builds
    no hierarchy); a PCG solve converged at r0 no longer evaluates M r, so a
    non-finite M r there no longer errors. Not built: PISO / PIMPLE (pimpleFoam, `ddtCorr`),
    (RANS: built in phase BL, item 61), `fixedFluxPressure` / `totalPressure` / rotating-wall BCs,
    `leastSquares` gradients, `pRefPoint`, restarts from a non-zero time,
    `phi` output with cyclic patches — all rejected by name (`limited`
    laplacian / snGrad and several MRF zones: item 60). Unproven: the
    symmetryPlane patch uses per-face normals where OpenFOAM uses the plane's
    mean normal (equal to rounding on a planar patch; a warped "plane" would
    differ); a mesh with two faces between one cell pair (the ELL accumulates
    them, untested); the CUDA backend RUN.
    FIXED (verifier FINAL, measured on the cavity; fix round 1): (a)
    `constant/fvOptions` / `system/fvOptions` were never opened, so a case
    with a source ran WITHOUT it at exit 0 (simpleFoam applies it: a uniform
    `vectorSemiImplicitSource` moved p 6.7e-2, U 2.2e-3 of scale) — now
    rejected by name on presence (exit 3, `fvcase` C2); (b) forces
    `e1`/`e3` were accepted and ignored: with `origin` + `e1 (0 1 0)` and no
    `CofR`, simpleFoam writes force.dat in the rotated frame (total
    (2.639e-5, 1.495e-4, 0)), dns wrote the global one — now rejected
    without `CofR`; with `CofR` (OpenFOAM reads neither `origin` nor the
    axes) CofR now wins over a stray `origin` (it was the last of the two);
    (c) `adjustPhi`'s `totalFlux` added the boundary |phi| where OpenFOAM's
    `sum(mag(phi))` is internal faces only — now internal only (thresholds
    only; `gpu_fv_simple` IF1a closed cavity unchanged).
    FIXED (re-verify FINAL, fix round 2): a forces object without `CofR`
    and without `e1`/`e3` (`origin` only, or neither) ran at exit 0 in the
    global frame, but simpleFoam v2606 stops on both (`No entries of the
    type (e1, e2) or (e2, e3) or (e3, e1) found`, axesRotation.C:196;
    `Entry 'origin' not found in dictionary ...functions/forces1`) — now
    `CofR` is required (exit 3, `fvcase` C2 pins both forms). Every prop
    case carries `CofR`. Kept deliberately: an EMPTY `fvOptions` (header
    only) is rejected on presence although simpleFoam runs it source-free
    (conservative, no parity risk).

60. **FV MRF = OpenFOAM `MRFZoneList` (phase MRF, `docs/fv.md` "MRF").**
    Several zones zone by zone (a zone/zone face gets both frame fluxes, as
    OpenFOAM), inactive zones, `omega constant`, `nonRotatingPatches`
    literal + regex, MRF-included faces on every non-empty patch of a zone
    cell; `limited [corrected] ψ` laplacian / snGrad. Gates (report
    `plans/verify/impl_meshsolver_mrf.md`): IF1e/IF1f vs simpleFoam ≤ 3.9e-13
    of scale with tight solves (`gpu_fv_simple`); rotor2 L1/L2 (laminar
    Re 220, the parity suite's meshes, simpleFoam + MRF np 6, 1500 it) CT
    ≤ 1.3e-11, CQ ≤ 2.4e-13 relative, both schemes, settling at the same
    iteration (± 1; F1 CQ ± 2); DTMB F1 converged (1200 it) CT 2.1e-9, CQ 1.2e-9.
    Phase FINAL re-ran all prop cases cold vs simpleFoam np 12 (`--bench`,
    `plans/verify/impl_meshsolver_final.md`, docs/fv.md "Final prop
    parity"): rotor2 L1/L2 CT ≤ 1.1e-11, F1 2.7e-9; wall to a settled CT/CQ
    2.6–2.9× (rotor2) and 8.4× (F1) vs simpleFoam np 12, but np 6 is
    foam's faster setup on every mesh: rotor2 L1 2.1×, F1 ≈ 6.9×
    (extrapolated from a 200-it np 6 bench). The
    old rotor2 parity table (`runs/parity/report.md`, structured dns vs
    foam Bc) is a DIFFERENT comparison (immersed staircase rotor vs
    body-fitted) and stays as it was. Speed (performance round, item 59):
    rotor2 L1 / L2 now 4.4× / 4.8× simpleFoam np 6 to a settled CT/CQ
    (3.96 / 4.99 ms/it, was 2.1–2.3× at 8.6–10.3), F1 ≈ 10.2× (was ≈ 6.9×);
    still launch / host-sync bound on rotor2. Not built:
    overlapping MRF zones, MRF zones touching cyclic faces (both rejected),
    time-dependent `omega`, `makeAbsolute` (unused by simpleFoam). A
    `nonRotatingPatches` selector matching no patch is an ERROR here
    (OpenFOAM ignores it), a zero axis likewise (OpenFOAM: Ω = 0 with phi =
    0 on every boundary face of the zone).

61. **FV RAS + boundary layers (phase BL, `docs/fv.md` "RAS", report
    `plans/verify/impl_meshsolver_bl.md`).** kOmegaSST and SpalartAllmaras
    as simpleFoam runs them, OpenFOAM's wall functions, meshWave /
    exactDistance wall distance, bounded scalar transport. Gates: IF-T
    (3 blockMesh cases, tight solves) ≤ 1.6e-10 of scale at it 100 / 400
    (`gpu_fv_simple`); wall distance = simpleFoam's to ≤ 5e-12
    (`fvwalldist`, local `fvwalldist_rotor`); BL-a TMR-like plate (tutorial
    `turbulentFlatPlate` at 1/4 resolution, limitedLinear k / omega) cf
    ≤ 2.6e-4, profiles 2.2e-5 U, drag 5.7e-6; BL-b rotor2 L1 + snappy prism
    layers, SST + MRF, nu 5e-4 (y+ mean 0.85): CT 2.3e-8 / CQ 1.3e-9 of serial
    simpleFoam, 12.1 s vs 39 s (np 6, `--bench`). Against simpleFoam np 12
    (phase FINAL) CT is 2.0e-6 off: that is foam's decomposition (np 12 vs
    serial foam moves CT 2.0e-6, nut 4.3e-3 of scale; dns = serial foam).
    Open:
    (a) **snappy layer coverage on rotor2 is poor**: 79% of blade faces
        extruded but 1.52 layers on average (41% of the target thickness) —
        snappy truncates the stacks around every unextruded spot; no tried
        cheap setting (buffer −1, nOuterIter, medial / feature angles,
        relaxed tet quality) raised it. A finer blade surface level or a
        different mesher is the lever, not solver work.
    (b) **No steady state at y+ ~ 10–20** (nu 2e-5 / 5e-5 / 1e-4 on this
        mesh): simpleFoam's CT oscillates ±3% (SIMPLEC or SIMPLE, layers or
        not); central U diverges at nu 2e-5. The parity gate therefore runs
        at nu 5e-4; the unsteady regime is compared on means only (report).
    (c) `limitedLinear` transients are not reproducible bit-wise (rounding
        noise sets the limiter on uniform fields; CLAUDE.md) — converged
        cases only.
    (d) Not run: DTMB F1 with layers + SST (optional gate c), SA on a 3D /
        MRF case, `nutk`/`omegaWallFunction` in the log region on a 3D mesh
        (the plate covers Spalding + binomial there in 2D).
    (e) Unproven: exactDistance on WARPED wall faces (the faces' own fan
        triangulation vs simpleFoam's triSurface triangulation: equal on
        planar faces only); meshWave on meshes with cyclics (rejected);
        the CUDA backend RUN.
62. **FV FP32 is the default (`--fv-precision fp32|fp64`, docs/fv.md
    "Precision", report `plans/verify/impl_fv_fp32.md`).** FP64 mode is the
    previous solver bit for bit (bit_perf 14 cases + the four full-length
    FINAL cases); non-FV dry-runs and GPU runs byte-identical. FP32: rotor2
    L1 / L2 / SST / F1 2.63 / 2.30 / 3.79 / 8.18 ms/it (FP64 3.98 / 4.99 /
    7.19 / 21.88), CT ≤ 1.7e-6 of FP64 laminar, 1.25e-5 SST (its end jitter
    is 3.0e-5), CQ ≤ 8.8e-7, settle iteration equal ± 1; reruns
    byte-identical. Open:
    (a) **The floors are a measured choice, not a derived bound.** The
        task's "smallest floor that never stalls a solve to maxIter" is
        degenerate: no solve stalled at any floor (0 … 3e-6), FP32
        estimates keep falling below the true floor. The constants are the
        first decade above the tail noise of rotor2 L1 / SST / F1 (U ≤
        7.9e-7, p ≤ 6.8e-7, k 2.2e-8); another mesh's noise (normFactor
        scaling, a larger ‖x‖/‖b‖) could sit above 1e-6 — then its tail
        keeps iterating on noise (slower, not wrong). No per-case
        adaptive floor.
    (b) **The turbulence floor moves the nut FIELD**: rotor2 SST nut
        6.8e-3 of scale (max-norm) vs FP64 at 1e-7 (k 2.1e-4, U 1.2e-4);
        FP32 with no floors is 4.8e-5. CT / CQ only within the floor's
        jitter. The 400-iteration sst_plate transient: nut 1.03e-2 vs FP64
        with or without floors (gpu_fv_simple band 3e-2) — cause not
        isolated (before the wall-function constants became float it was
        1.5e-4 with floors, 1.0e-2 without: a transient that amplifies
        rounding, not a floor effect).
    (c) Small meshes stay launch / sync bound: the float kernels alone buy
        1.52× on F1 (no-floor ms/it) but ~4% on the first 357 L1
        iterations; the rest of the FP32 whole-run gain is the floor
        ending the active phase early. The coarsest dense inverse is still
        FP64 in `Amg<float>` (0.5 ms per refresh on L1).
    (d) Not built / not run: FP32 on the CUDA backend RUN (the shim build
        `build/cuda_shim` compiles and links dns + the FV tests; nothing
        ran there), the flat plate BL-a and the IF2 sphere / G1 in FP32, an
        FP64 defect correction around FP32 solves.
63. **Coupled p-U prototype (plan_perf_5 P0a + P0b, docs/fv.md "Coupled
    prototype"): EXPERIMENTAL, opt-in `--fv-coupled N`, FP64, laminar.**
    P0a: SIMPLE's `iterate()` is now a composition of stage methods
    (bit_perf 14 cases byte-identical FP32 and FP64, reruns identical).
    P0b: inexact Picard on simpleFoam's own fixed-point residual, FGMRES(30)
    with a linearised SIMPLEC step. G-JAC ≤ 4.5e-14 (cavity, mrf2, channel,
    rotor2 L1); G-FP0: CT moves ≤ 6e-13 per outer from a 1500-it SIMPLE
    state. Converged CT / CQ vs SIMPLE FP64 and simpleFoam: L1 ≤ 6.8e-11 /
    2.3e-12. FP64 loop to settle 1e-3: L1 0.83 s (η 0.1) / 0.45 s (η 0.3)
    vs SIMPLE 2.17 s; F1 (hybrid 30, η 0.1) 9.47 s vs 11.88 s. Open:
    (a) **KILL GATE NOT PASSED on F1** at the measured configuration: FP32
        projection 0.87–1.13× vs SIMPLE FP32 5.32 s (L1 1.77–1.97×, η 0.3
        3.3×). F1 at η 0.3 is a MODEL (1.57–2.0×), not a run. Decide on one
        F1 η 0.3 run before any P1 work.
    (b) F1's C_K is 6.2 ms FP64 (plan model 1.35 ms FP32): CGS2 over up to
        30 vectors of 4n = 2.83M doubles; K̄ 77 per outer to settle (292
        on the first outer of the hybrid start). Restart length, Eisenstat–
        Walker η, a cheaper first outer and FP32 Krylov vectors are unmeasured.
    (c) F1's CT cannot be certified to solver tolerance: both references
        (SIMPLE FP64 and simpleFoam, 1200 it) still drift (−6.2e-10 per
        iteration); the coupled fixed point (probe ≤ 1e-12 / 1e-10) is
        +2.8e-6 from them, inside the extrapolated SIMPLE limit's bracket.
    (d) G-JAC is not exact with freestreamVelocity / freestreamPressure
        patches (duct 1.2e-4 / 3.8e-4: their value fractions follow Ub in
        the FD). G-PAR, G-FOAMRESTART, FP32, RAS, the CUDA backend: not built.

78. **Compressible FV family (`docs/fvcomp.md`, `gpu/fv_comp`): explicit DNS/LES
    on any FaceMesh, FP32 default, cross-verified against the structured solver.**
    Device vs FvRef, trajectory vs `Comp` / `DeviceComp`, grid ladders: the table
    in docs/fvcomp.md. Open:
    (a) **Plain FP32 wanders on a settled flow.** Re 40 O-grid 128 x 64: FV cd
        1.5665-1.5764 for ever, period ~250, total mass pulsing 2.7e-4; FP64
        flat to 6 digits (1.571630); structured plain FP32 1.5650-1.5665. Updates
        below ulp(q)/2 round away. `--compensated` (off by default, as on the
        structured path) or FP64 fix it. USER decision: should FV FP32 default
        to compensated? It costs 1.02x per step (0.186 vs 0.182 ms, 256 x 128).
    (b) CLOSED: the settled cd plateau FV - structured (+6.0e-3) was a bug in
        `Far`: the outgoing Riemann invariant read `c_P`, so the warm wake's
        entropy acted as a pressure source of relative size M. Now the
        isentropic `c` of p. Gap -1.43e-3, -4.13e-4, -1.6e-4 (96, 192, 384; order
        1.79, 1.37; 384 at the job cap, t 59), Re 100 mean cd -9.4e-5. The whole
        structured row on FV gives the same cd to 7e-5. docs/fvcomp.md "Far gap".
        Left: no R -> infinity limit extrapolated, 384 not flat, the gap at 80 / 160 D
        M 0.1 is -1.2e-3 / -4.0e-3 at n 192 (order 2.1, 2.4; h1 grows with R).
    (h) FIXED (2026-11-02), seen while gating, none caused here. `gpu_sa_wedge`,
        `gpu_sst`, `gpu_precond` gated `dns flow --steady`'s stop iteration, a first
        passage of noise through TOL: SA 111000-185400, SST 122400-231600 under a
        1e-6 nudge of M or Re, cd within 1e-4. Wedges now gate the first `remaining`
        < 3e-4 (98600-113000 <= 1.3e5, 104400-124200 <= 1.4e5), cap 2e5 -> 4e5.
        M 0.05 K 1 sat on the jitter floor at TOL 1e-4 (one nudge not within 4e5):
        TOL 3e-4 there, 43000-68200 under the same 1e5 cap. The wedge's L_r read 0
        on 1/3 of its stats rows (first axis cell at jitter around 0): fixed in
        `recircLength`, pinned on every settled row. `gpu_la_perf`: f892cc6 = now on
        gfx1201 (1.269 ns/cell, 0.79-0.81 ms), the 0.8 ms gate was gfx1100's: bounds
        per device (`laPerfGate`). `amg_setup`: see 77.
    (g) Ladders stop at 3 levels by decision (no GPU job over 5 minutes); every
        quoted order rests on those. Re 100 St difference shrinks at order
        ~1 on rungs that are not asymptotic (St moves 4% between the last two);
        cl rms non-monotone; immersed vs FV convergence NOT measured (25.6
        cells/D still falling at t 51). Dropped rungs: docs/fvcomp.md.
    (c) Far is 1D characteristic: no transverse term, no sink, no NSCBC. A
        box on `--solver fv` has none of the structured far-field machinery.
        Re > 800 unmeasured.
    (d) Not built, rejected by name: revolved O-grids (P8), immersed bodies on
        a FaceMesh, MRF, `--temp`, volume / vtk / slices frames, checkpoints,
        per-face profiles. RANS and `--steady`: item 84. Species: 78b.
    (e) Unproven: KG with no dissipation on snappy 2:1 meshes past Re 40
        (`cyl2d_snappy` stable, coarse), 3D snappy, LES on a real wall, the
        tangential part of the FV wall face gradient (the owner's LSQ value,
        structured: exactly 0 on the image): second order on cd measured, wall
        stress profiles not compared; Sutherland + LES; CUDA run (shim covers
        every call used, no device); Morton vs other cell orders for speed.
    (f) `reference.area` is a user number on 3D polyMeshes; no check against the
        wall's projected area.

78b. **Species on the FV family (`dns flame --solver fv`, `--polymesh`, docs/fvcomp.md
    "Species").** Premixed one-step C0 on any FaceMesh. Interior rounding-equal to the
    structured kernels (S8), planar rotated 5e-8, non-orthogonal order 2. Open:
    (a) **FV vs structured S_L end to end is not closed below 1e-3.** Boundaries differ:
        NSCBC + sink hold rho, `Far` holds entropy, T_u differs (1.3e-3 at M 0.05, 7.5
        cells), S_L follows at dln S/dln T ~ 1. Read at a matched T_u, -4e-5, -9e-6, +8e-4
        at u_in 0.98; +3.4e-4, +7.8e-5, -5.3e-4 at u_in 0.9: the finest rung changes sign,
        moves 3e-4 with `--nscbc-sink`, and sits inside the structured window span
        (~1e-3). No order claimed. Cause of that span not isolated.
    (b) **Plain FP32 at M 0.01 scatters S_L ~1e-3, both solvers**, non-monotone in h and
        dt (FV FP32 1.003167 at dt 9e-5, 1.003884 at 1.8e-4, FP64 1.003472; 30 cells
        1.001623, FP64 1.000437). Every quoted structured S_L at M 0.01 carries it.
        `--compensated` matches FP64 to 6 digits on FV; the structured flame has no such
        mode. USER decision: compensate the structured species path?
    (c) Not built, rejected by name: bunsen, candle, chamber, wick,
        `--turb` (`--equivalence`: 78e), the jet's `--pulse` (checkpoints: built, M5, item 74). Species LES and TFLES: 78d. No bluff-body or
        wall-anchored flame on the FV path has been run: a wall's zero-gradient Y and
        adiabatic T are unit-tested only (S1, S5). Diffusion C1: item 78c.
    (d) `Far` + `Outflow` is an acoustic half-pipe: fine for a planar flame. Unproven on a
        burner: no sink, `Outflow` clamps p.

78c. **Diffusion flame on the FV family (`dns flame --mode diffusion --solver fv`,
    `--polymesh`, docs/fvcomp.md "Diffusion flame").** Model C1 on any FaceMesh: shared
    limiter, per-face rows, a fixed inflow face. Numbers: the table there. Open:
    (a) **Inflow + Outflow rings.** The start's p/p0 +-5% decays with an e-fold ~30
        l/U, no growth at these Da. No absorber: `Far` rows sag u (fuelIn -11%). A jet
        reads past t ~100; `gpu_fv_flame_jet` reads [60, 80] and carries the ring.
    (b) **A 30 deg sheared jet diverges late**: FP64, 3 cells/l at t 308.7, 4.5 at t
        155.9; 2 cells/l survives t 160, not run past it. A cell-to-cell sawtooth (p 3.7
        vs 569) on the lean side of the sheet at x ~6, not a boundary; blow-up time ~
        h^1.7; constant-coefficient symbols of the Laplacian and the stress operator are
        stable to 45 deg, so it needs the variable state. Cause NOT isolated. Rotated and
        wavy show nothing. Reproduce: `tools/chem/fv_jet_ladder.py --cells 4.5 --tt 160
        --meshes shear`. Tip -0.700, -0.342 (order 1.77) stand for 2, 3 cells/l.
    (c) Inflow rows pin a FACE, structured a CELL: tip offset first order, +0.725, +0.490,
        +0.352 vs etaIn 1000, limit +0.05 by a line through 3 rungs. The two-reactant
        interior is rounding-equal to `kSpeciesStage2` on a periodic box (78e R1); not on the jet.
    (d) Not measured (3 levels, 295 s): 6 cells/l: `tools/chem/fv_jet_ladder.py --cells 6
        --tt 160 --meshes rot`.

78d. **Species LES and TFLES on the FV family (`dns flame --solver fv | --polymesh --bc`
    with `--les --sgs`, `--tfles`, docs/fvcomp.md "Species LES and TFLES").** Premixed C0:
    eddy species flux, thickened flame (uniform / dynamic sensor, Charlette), the sensor's
    filter walking faces. FvRef 1e-13 every pass; FV vs structured on one periodic 3D box
    rounding over 300 steps; rotated = planar, sheared order 2. Open:
    (a) **Rotated 30 deg at 3.75 cells/delta_th: S_L -1.08e-5** (7.5, 15: < 1e-9). The
        uniform sensor gives -9.2e-6: not the walk. Cause not isolated (the coarsest h of
        any FV flame row).
    (b) X1's dynamic-sensor rows sit 1.3-2.1 x their own FP32 distance (uniform, LES:
        0.6-1.2 x); inside the 3 x bound, excess not isolated.
    (c) Not built, rejected by name: TFLES of two reactants (`FvRef::make`, the
        inventory); two-reactant LES is solver-live since MC (`gpu_fv_matrix`, Z == tracer),
        spelled on the CLI by `--equivalence` (78e); the jet stays 2D only;
        `--turb` on the FV route, LES on a 2D mesh (box:
        `--nz >= 8`; polyMesh: exit 3 at the run), a thickened boundary face. Unproven:
        a turbulent FV LES flame (no synthetic inflow on the route), TFLES on a wall-
        bounded mesh (edge layers and the walk's stop are unit-gated only, L1 cyl2d).
    (d) Not measured: a 30-cell rung of the TFLES ladder (`tools/chem/fv_flame_ladder.py
        --cells 30 --tt 28 --tfles 2 --box-up 16 --box-down 12 --meshes rot,shear`).
    (e) FIXED (MC, `gpu_fv_matrix` (d)): the sensor walk took the first of exactly tied
        sub-faces in CSR order, so on split faces it depended on the cell numbering
        (box_refined ColourMajor vs Source, TFLES dynamic: 2.3e-3 of the state in 3 steps).
        Ties go to the lowest srcFace. Uniform meshes have no tie: unchanged there.

78e. **Air-methane on the FV family (`dns flame --equivalence --solver fv | --polymesh --bc`,
    docs/fvcomp.md "Air-methane", masterplan_multiphysics M2).** C2 = C1's stage, C2b =
    `FvChem::law` (`kFvSpCellPhi`, `ChemJ2Phi`), module `C2b` of `dns/fvmodules`; LES live,
    TFLES refused. FP64 S_L order 2 about the M^2 offset, Z exact; rotated = planar, sheared
    order 1.8-2; interior rounding vs `kSpeciesStage2` (R1). Open:
    (a) **R1 at the flame's A diverges in all three solvers** (the tanh slab is not a flame:
        its preheat zone explodes at constant volume, T 2 Tb, Y_O < 0, chaotic); the gate runs
        A / 10. A gate on an equilibrium IC (the oracle profile) at the real A not built.
    (b) **Structured vs FV at 30 cells: 0.996512 vs FV FP64 1.000123.** The structured FP32
        Z walk through C (docs/chem.md "Model C2b"); FV FP32 reads the same 0.996539, FP64 and
        `--compensated` do not walk. Plain FP32 C2b at >= 30 cells is walk-limited on both.
    (c) Not built: `--turb` on the route (exit 2), a turbulent LES flame, TFLES of two
        reactants, `--gravity` on the flame (open patches), a lean `--mix-phi` on FV and a
        wavy mesh (not measured), the Bunsen burner (compressible: structured only too).
    (d) Not measured (3 levels): a 60-cell rung (`tools/chem/fv_flame_ladder.py --cells 60
        --tt 20 --dth 4.70964 --meshes rot,shear,wavy --extra "--equivalence 1.3 --expansion
        6.88 --visc sutherland --tinf 298"`).

79. **Time-accurate incompressible FV (`--solver fv`, `--bc`, docs/fvincomp.md).**
    Built: `gpu/fv_incomp`, `gpu/runner_fvincomp`, `dns/fvincomp`; DNS, LES;
    wall, slip, inflow, outflow, convective, far, cyclic, empty; `dns tgv
    --solver fv`, `dns flow --solver fv --ogrid`, `dns flow --polymesh DIR --bc`.
    Every ladder rests on 3 levels (user decision); no order is converged. Open:
    (a) **Re 40 tunnel cd FV vs MAC: cause isolated, one question left.** Same
        mesh + same BCs vs simpleFoam is the oracle (docs/fvincomp.md "Same mesh").
        O-grid: U rms order 1.76, 1.30, cd diff −8.9e-3, −3.4e-3, −1.9e-3: the
        projection and face flux hold. The gap was: the tally (p_P + G.r, +1.2e-2
        to +2.0e-2, moved with dt: dropped), the wall stress (Laplacian alone +2.9e-2
        vs simpleFoam at N 128: OpenFOAM's whole wall stress taken, budget closes to
        4e-4), MAC itself (2.08833, 2.08091, 2.0654 at N 128, 256, 512, last
        doubling −1.56e-2), the outlet (convective vs p = 0: −1.05e-2, flat).
        FV now 2.04109, 2.04452, 2.0432 (N 256 not flat, ~2.042), span 3.4e-3; FV −
        MAC −4.72e-2, −4.10e-2, −3.77e-2, order 0.35, 0.29: it shrinks because MAC
        moves. OPEN: does MAC reach ~2.04? N 1024 not measured (~40 min), limit
        estimate ≤ 2.05, so a gap ≤ 0.4% is not excluded; first-order cut wall.
        Re 100 row predates the wall stress and tally, not re-run.
        On the tunnel mesh the FIELD difference vs simpleFoam does not shrink (rms
        1.4e-3, max 1.6e-2 at the ring's corner cells, 43°, volume ratio 1.67 at
        every N): a mesh property, cd within 1.9e-3 past N 128 with no trend.
    (b) **CLOSED: T6 cavity 8.6e-2.** Lid-corner singularity, proved by ladder:
        all-cell max 1.01e-1, 9.41e-2, 9.10e-2 at N 24, 48, 96 (order 0.1), a fixed
        2 h_24 away 2.03e-2, 6.17e-3, 4.99e-4 (order 1.7, 3.6), rms there order 3.
        T6 gates outside 2 cells of a lid corner, SAME 5e-2 bound: 2.03e-2 passes.
        The 8.6e-2 also held a kick from (c). Corner cell moves with c, stated:
        no bound on it.
    (c) **CLOSED: cavity 3.2e-2 move.** Not the PCG, not noise: a short step.
        `march` clamped its last step to the report time; a step of f dt on a
        steady state moves U by (1 − f) 3.24e-2 (4.07e-2 with the wall stress), any
        f << 1, because phi holds the last c's Rhie-Chow term and the predictor
        forgets it; p grows as 1/f; 1e-12 dt diverged the next steps (p 8e10 in
        the warm start). Reproduced on demand, T7. The two runs had different
        report times, so different last steps: settings differed, determinism
        stands. Fix: equal steps to an end time, runner and test; `step` precondition.
        The runner's written fields had carried the same kick.
    (d) Random meshes: order 1 (ν 0.05), 0.4–0.65 viscous-dominated, not 2. Convective
        ½ is forced by skew-symmetry; AlongD on a non-smooth mesh also limits.
        Re-measured after (a)-(c): unchanged, 1.20, 0.95 at ±0.1 h; K −47.9% in t 29
        on 32² ±0.25 h. Neither fix touches a periodic interior. No redesign started.
    (e) The projection is dissipative: K −48% in t 30 on 32² ±0.25 h random, inviscid;
        2.8% of peak on 192³ TGV. No growth seen, not proved.
    (f) MAC vs FV TGV −dK/dt difference is non-monotone on the three levels.
    (g) Not built: implicit viscosity (`dt ~ h²/ν` makes wall clustering expensive),
        restart, frames, immersed bodies, revolved O-grids, `--ramp`, `--steady`.
        Unproven: the CUDA backend (no nvcc here, these files never compiled there);
        `--far`-type outflow beyond a smoke run; `outflow` backflow stability.

80. **simpleFoam on the structured meshes (`dns flow --solver simple`,
    `docs/fvstructured.md`).** Built: uniform, `--stretch`, `--focal`,
    channel, O-grid 2D / 3D, wedge, full revolution, laminar, SA, SST, MRF;
    `--export CASE`. Open:
    (a) OpenFOAM absent: no generated case has been run by simpleFoam. The
        `vectorSemiImplicitSource` spelling (`sources`, `injectionRateSuSp`)
        and its sign are written from memory of v2606; Poiseuille pins the
        solver's reading, not OpenFOAM's. First check when OpenFOAM is back.
    (b) OpenFOAM's `wedge` patch is not built: the wedge's sides are `symmetry`,
        a `cos(dphi/2)` difference in the face value. `Uz` is solved, so a
        wedge (and any 3D case with a zero component) never certifies a
        `--steady` stop: its residual is rounding over rounding (0.25).
    (c) O-grid cd vs structured `--ogrid`: a STATED MODEL GAP. At 20 D it is the
        far model: U clamped on the far circle takes simple central from -0.0228
        to +8.1e-4 of structured (n 256; two rungs, no order). Structured 160 D =
        80 D at n 128 was a grid coincidence: h1 grows with R at fixed ny; steady
        stop = time march; n 192, 256 fall to 1.5080, 1.5072. OPEN: structured -
        simple plateaus 0.005 at 80, 160 D (n 256), and at 160 D M 0.1 -> 0.05 moves
        structured -3.1e-3 at every n (20 D: -1e-4). Cause not isolated. FV with
        clamped U rings, no steady number. docs/fvstructured.md "Reading the gaps".
        The cut-cell solver (10% above) is a different domain. Finer rungs: "Not
        measured".
    (d) `cd` from wall faces is first order on an O-grid (zeroGradient wall p,
        one-sided wall stress): the ladder at beta 2 is non-monotone (1.5392,
        1.5392, 1.5408 at n 256, 512, 1024) and Richardson at order 2 lies.
    (e) MRF: `--nz > 1` and revolved meshes rejected; the zone is r < D, a
        sharp OpenFOAM zone, not the structured blend.
    (f) Immersed bodies on Cartesian meshes have no export (rejected): every
        `--stretch` / `--focal` case here is body-free. `dns flow --solver
        simple` on `--body` needs `--ogrid` or an imported polyMesh.

81. **`dns conv` on a FaceMesh (`docs/fvcomp.md` "Gravity", `docs/fvincomp.md`
    "Temperature", `gpu/runner_fvconv`, `dns/fvconv`).** Both models, `--solver fv`
    and `--polymesh DIR --bc FILE`, one route (C6: binary polyMesh == box, bytes).
    Ra 5e4 2D, FV vs structured same model: Boussinesq -0.7%, -0.19%, +0.10%,
    compressible -0.7%, +0.11%, +0.25% on 48, 96, 192 x H/2 grids; closed -0.15%.
    Cross 4.7% (15% band, not wider). Open:
    (a) Compressible FV gravity is WELL BALANCED now (`fvGravityFace`, docs/fvcomp.md
        "Gravity"): T linear in the height + hydrostatic p is a fixed point on any mesh,
        FP64 4.4e-14 Uff on random ±0.3 h (was 1.1e-2), nuFlux 1.000000 (was 0.61), no
        reference state. Left: the FP32 floor, 2.1e-5 to 3.9e-5 Uff (frozen `eps p/h`,
        two-word state, FP32 fluxes). Not exact: Sutherland, a curved adiabatic wall, any
        nonlinear T(h): truncation. Random Ra 5e4 192 x 96 reads 3.61 vs 3.75 on 96 x
        48: a different roll state suspected, not shown; uniform 192 at t 120 not
        measured (`dns conv --solver fv --nz 1 --n 192 --ny 96 --rayleigh 5e4 --tt 120`).
    (b) FP32 Boussinesq |div| floors at 1.4e-4 (96), 4.6e-4 (192): `4 eps gross` with
        `c B_f` in the flux. Nu unaffected to 4 digits. Gate on FP64.
    (c) `--hot-patch`, open faces under gravity, 3D Nu: not built / not measured.
        48 x 24 x 24 runs are smokes (`f3d`). Not measured: `dns conv --solver fv
        --physics incompressible --nz 48 --n 96 --ny 48 --rayleigh 5e4 --tt 40`.
    (d) Non-orthogonal Ra 5e4 is one mesh (sine 0.05), no ladder. Not measured: `dns
        conv --solver fv --n 192 --ny 96 --nz 1 --rayleigh 5e4 --tt 40 --mesh-perturb
        0.05`, both physics.
    (e) Height bins on a non-planar mesh smear: read `nuBotF`, `nuTopF` there.

84. **RANS on the compressible FV family (`docs/fvcomp.md` "RANS", `dns/fvrans.hpp`,
    `gpu_fv_rans`): SA-neg, SST-2003m, SST-Vm on any FaceMesh, URANS and `--steady`
    (local time stepping).** Numbers: the table in docs/fvcomp.md. Open:
    (a) The channel is driven through `FvCompConfig::bodyForce` only: no CLI flag,
        no `dns channel --solver fv`. Not built.
    (b) Far scalars switch sharply on the boundary velocity's sign; the structured
        far row blends with a tanh band and holds a ghost value. A stated scheme
        difference, not measured alone.
    (c) Isothermal `omega_w`, `rho_w`: the FV wall face reads the EOS at T_w, the
        structured ghost the mean of two cells. Same limit, in the channel's O(h^2).
    (d) `--steady` on an external FV flow: certified stop NOT measured. Re 1e3
        cylinder 128x50 SA: FV and structured both exit 8 at 1e6 iterations, cd
        jitter 3-5%, no steady wake. Only the channel fixed point is measured (from
        the oracle, LTS, FP64). LTS drifts the mass, no projection built. A cold
        start and FP32 compensated on the channel: not run.
    (g) O-grid cdf FV - structured shrinks slowly (+1.5e-5, +1.1e-5, +7.6e-6): no
        order established on 3 rungs; 4e-4 relative at the finest. SST-2003m cd at
        128x50 disagrees 10% (+7.85e-2; y+ max 1.7), the finer pairs 1.4e-3, 4.7e-4,
        and SST cd is non-monotone in h: not asymptotic. Finer: `tools/fv/comp_ladder.py
        --re 1e4 --ft 0.2 --levels 1024:386 --beta 4 --rans sst-2003m --compensated
        --limit 1800` NOT measured.
    (e) SA's negative-branch grad rho coefficient: unverified on any path (docs/rans.md).
    (f) Not built on FV: implicit steady, `--precond`, RANS + MRF, wall functions,
        a per-patch inflow turbulence value (one freestream phi for every open
        patch), Spalding-style y+ columns (stats carry mutMax, floorHits).

85. **The compressible FV modules compose through one inventory** (`dns/fvmodules`,
    `docs/fvcomp.md` "Combinations", `gpu_fv_matrix`, `gpu_fv_matrix_phys`). Live since MC:
    species + gravity, gravity + LES, gravity + bodyForce, species + bodyForce, two
    reactants + LES (solver level; physics numbers in the docs table). Open:
    (a) Refused by physics, each needs a model, not a gate: species + RANS (mean
        reaction-rate closure), RANS + gravity (buoyancy production), LES + `--steady`.
        By code: species or gravity under LTS (`kFvChem`, `kFvGravity` take the bare cr;
        a `Lts` instantiation each + a fixed-point-vs-lts gate would turn them on).
    (b) No CLI spelling for most live pairs: `dns conv` has no `--les`, nothing spells
        `bodyForce` with gravity, the FV jet is 2D (`--nz 1`) so its LES is unreachable.
        The case surface (masterplan M6) is the route; until then the solver-live pairs
        are gated at the solver only.
    (c) `--les` + `--steady` on the compressible polyMesh route was ACCEPTED before MC (an
        LTS LES run, labelled DNS); now exit 2 by the inventory's text. The structured-mesh
        FV route already refused it. No pinned case used it.
    (d) The matrix is pairs (closed under requirements), not triples: e.g. LES + gravity
        + bodyForce is live by composition and ungated as a triple. Precision runs inside
        every cell; steady LTS cells skip the FvRef march (FvRef has no LTS) and mass.
    (e) The TFLES dynamic sensor's T-gate is a switch: an exactly tied state flips under
        rounding (tiedSensor measures it per mesh); the matrix runs untied states.
    FP32 RANS on the channel fixed point: 84(d).

86. **The candle on compressible FV** (`docs/fvcomp.md` "Hydrostatic ambient",
    `gpu_fv_candle`). Gravity x open patches and gravity x C1 live through the ambient's
    per-face rows. Open:
    (a) FV vs structured tip -9.5% / -10.2% at 6 / 8 cells/l against etaIn 1000 (FV h-flat
        7.133). The FV fixed inflow face carries 5.4-5.9% less fuel than the pinned structured
        cell (W 1: 8 cells across the slot, tanh edges 4 cells); Roper's Q^(4/3) explains
        -7.4%. The ~3% residual is not attributed (top: Far row vs NSCBC + sink + lateral relax
        blend; structured flaps below 8 cells/l, so no h trend). Against the structured
        DEFAULT (etaIn 100) the gap is the known inflow overshoot: 39% more fuel at W 1
        (10.19 vs 7.13).
    (b) The jet's fixed-p Outflow as a candle top: an organ pipe the flame sustains (p/p0
        +-0.7%, backflow -20 U, Y_O 3e-5 over Y_O2 in ~4000 cells counted as hits, extinction
        at 4 cells/l). The box candle uses a Far top; a polyMesh `type = "outflow"` top is
        still accepted under gravity (its rows are the ambient's) and rings: not refused, not
        recommended. The Y_O overshoot itself (virtual-upwind MC is not strictly bounded in
        multi-D) is counted, not fixed.
    (c) Not built: the wick (`--wick`: an immersed ghost wall, no FaceMesh twin), a
        stratified (non-isothermal) ambient column, the low-Mach candle, `--les` on the candle,
        the structured candle's `--pulse`. `dns conv` keeps its closed layer by decision.
    (d) Plain FP32 quiescent candle max|u| 6.3e-4 U (Far top; 1.2e-3 with the Outflow top) =
        p's FP32 rounding (p = T0 = 28571 at M 0.005); compensated 6.5e-5, FP64 8e-13. Not a
        boundary error.

### Low-Mach FV (`dns flame --physics lowmach --solver fv | --polymesh`, docs/fvincomp.md "Low-Mach")

LF1. **The low-Mach march's energy balance closes at order 1 in dt, on both solvers.** R_T
    reads the stage's input m, continuity its output m. FV measures it (`eDefect`, L3: 7.7e-5,
    3.9e-5, 1.9e-5 relative at dt, dt/2, dt/4 on a burning front; the semi-discrete balance is
    rounding). Structured has no energy tally; L7 says it is the same algebra. Open: whether a
    transient low-Mach quantity (Darrieus-Landau growth) carries the same order. Not measured.

LF2. **Explicit diffusion on a non-orthogonal mesh.** The rule's sum is `|S|^2/(S.d)`, conservative
    and unproved: the Σ `corr.S` of the compressible rule diverged at 67 deg, dt 2x above stable; the
    new rule costs 4.3x in dt there (shear + wavy + rotate, 30 cells: dt 3.2e-5, a 30-cell rung past
    295 s). No sharper bound measured. The compressible FV flame's dt is acoustic and has not been
    tried under the same stress at M >= 0.05.

LF3. **The FV step costs 6 to 14x the structured one** (512 x 4: 1.8 ms FP32, 4.3 ms FP64 at tol
    1e-3, vs 0.31 to 0.44). Launch-bound. Nothing tuned: FP32's 3 iterations a step is the floor stop.

LF4. **Not built, rejected by name**: the 3D slot and the round nozzle, the burner's
    `--profile-dt`, turbulence on a polyMesh or the burner (M5c: the planar box is built), a closed box, gravity on the planar flame (checkpoints: built, M5, item 74),
    `--rim` on a polyMesh. A plain outflow (no open patch in the run) keeps its backflow zero
    gradient both ways, no gate drives it. Built (M3, 2026-11-03): two reactants C2 / C2b
    (rounding vs structured, docs/fvincomp.md table), the 2D slot burner (per-face rows,
    `--dt-rule cell`), LF5; open faces, gravity, the rim (steps 3-5): LF6-LF8. LES and TFLES (M4, both
    low-Mach solvers): LF9-LF11.

LF5. **FV vs structured burner is a scheme gap, not rounding** (G2a and rich2d at 4, 6, 8 cells,
    docs/fvincomp.md table): collocated vs MAC momentum (L8) and the cell velocity's definition
    (FV (rho u)/rho, structured the mean of the face m/rho_f) differ. Every cone number's gap
    shrinks from 6 to 8 cells (order 0.8 to 2.5; G2a xTip +3.2e-3, +1.5e-3, +1.1e-3), FV(FP64) at 6
    cells moves it < 1e-4. Not asymptotic: G2a mFlank -1.93e-3, -2.12e-3, -1.34e-3 (grows 4 to 6),
    rich2d preFrac non-monotone; 3 levels by decision, no limit claimed. The FV conv dt rate is the
    face MEAN per axis, LmConvRate the larger face: binary dt sequences differ (stated). The FV
    burner step is 3.4 to 9.7x the structured one (the .json ms/step, startup included: G2a 4 / 6
    / 8 cells FV32 6.2 / 8.1 / 7.7 vs 1.3 / 2.4 / 0.8 ms; rich2d 8 cells 13.7 vs 3.4; FV PCG ~40
    iterations a step at FP32 tol 1e-5). Not tuned.
    The polyMesh burner ran (`--wavy 2`, G2a 6 cells: xTip +1.2e-3 vs the box); one rung only.

LF6. **FV open faces vs structured: the OUTFLOW PARTITION plateaus without gravity** (G2a
    `--laterals open`, 4/6/8 cells, docs/fvincomp.md table): the cone gap shrinks (xTip +3.9e-3,
    +2.5e-3, +2.2e-3), the split of the outflow between x-hi and the laterals does not (mdotLat
    -9.2e-2, -1.03e-1, -1.15e-1 of 37, 0.25-0.31%, grows); with `--gravity 0.0964` it shrinks at
    order 1.2-1.5 (+0.57, +0.35, +0.23). uLatMax (+1.9e-2, 1%) is flat in both: a pointwise
    extreme. The API ladder (O4, a transient to t 0.25) shows the same quantity: lateral outflow
    FV - MAC -1.4e-2, -2.0e-2, -1.1e-2 of 0.13-0.21, no smaller without the corner columns
    (+2.2e-2, -2.7e-2, -1.6e-2): the corners alone are not it. Cause NOT isolated. Structural
    differences left: an open face's predicted flux is
    its CELL's (phi* = S . (rho u)*_P, zeroth order) where the MAC face integrates its own
    momentum; the corner cells where an open face meets x-hi (two p = 0 faces); the normal
    momentum flux phi_b (u_P . n) vs m_b^2/rho_f. Fixed on the way (measured): species upwinded
    by the stage input's flux sign (Y_O -2.6e-5), traction-free tau_nn = 0 instead of the
    structured zero normal-stress gradient (API l1 distance grew with h). Not measured: the
    clearance-6 rich2d open slot (S6's [150, 200] window is past 295 s), Bickley on FV.

LF7. **FV gravity: hydrostatic rest exact on the box, a residual on non-orthogonal meshes**
    (G2: |rho u(g) - rho u(0)| of the free-fall scale 4.3e-16 box, 5.2e-5 sheared 21 deg, 3.1e-6
    random 0.3 h; gated at gpu/lowmach's 1e-4): the deferred `kg` and the walls' image `dpB`
    balance the face force only to the correction's order. The plume similarity gate
    (gpu_lowmach_plume, 192 x 128 to t 450) and a gravity run on a polyMesh read from disk: not
    measured (API meshes in memory only).

LF8. **FV rim = the structured staircase on the same cells; its corners are scheme numbers**
    (R1: the max FV - MAC rho u sits at the tube's inflow corner and stays 9.8e-2 of 2.9 at 3
    levels, the lid-corner case; 0.5 off the corners and l1 shrink at order ~1.1). The 2D slot's
    lips are grid-aligned, so the staircase IS the body-fitted wall; a curved tube (the round
    nozzle, a polyMesh tube generator) is not built. Anchor case (rich slot U 6, open 6, T_w 1):
    anchored at 4, 6, 8 cells, the base on the structured row or one row off (6 cells until t
    39.5), qRim -0.57%, -0.36%, -0.22%. FV cost there: 8 cells 268 s vs 92 s structured (the
    whole run, 2.9x; FV PCG ~85 iterations a step).

LF9. **Low-Mach Smagorinsky: FV vs structured is a scheme gap of mu_t, not rounding** (M4 X2,
    docs/chem.md "Low-Mach LES"): mu_t differences the cell velocity, (rho u)/rho on FV, the mean
    of the face m/rho_f on MAC. Order 2 (1.2e-2, 3.1e-3, 7.7e-4 of max mu_t at h, h/2, h/4); the
    1D march's rho gap 1.0e-6, 7.2e-8, 2.4e-8 reaches the L7 rounding at h/4. TFLES is rounding
    (X1, the T and Y algebra never reads u); sigma on a planar flame is 0 in both. The structured
    gradient's end cells take themselves as the missing neighbour, FV's least-squares reads the
    boundary value: stated, not unified.

LF10. **The laminar flame's LES shift of S_L is small and not monotone in h** (Smagorinsky 0.17,
    3D planar, 4/6/8 cells: +1.8e-4, +8.1e-4, +4.2e-4 structured; FV +1.9e-4, +8.4e-4 at 4/6, the
    solvers agree to 3e-5): the DNS S_L itself moves non-monotonically there (1.00065, 0.99252,
    0.99484). No order claimed; 3 levels. FV sigma on the planar flame is the DNS to 7 digits but
    not bitwise in FP32 on the Morton mesh (mutMax 9.4e-9: D_sigma of row-to-row rounding); the
    device-level gate (L1, 64 x 4 x 4) is bitwise.

LF11. **Low-Mach LES not measured / not built**: a turbulent low-Mach LES-vs-DNS statistic beyond M5c5's single grid,
    the round-nozzle 3D Bunsen under LES (144 x
    144 cross-section past 295 s); FV TFLES dynamic at 8 cells and FV 3D LES at 8 cells (past
    295 s). Refused by decision (dns/lmmodules): TFLES with C2/C2b, with open laterals, LES and
    TFLES with the rim. The 3D slot Bunsen is z-invariant, so sigma is exactly 0 there
    (bitwise the DNS): only Smagorinsky moves it (xTip -0.13% at 4 cells).

### Turbulence on the FV and low-Mach routes (`--turb`, M5c; docs/fvcomp.md "Turbulence", docs/chem.md "Low-Mach turbulence")

M5c1. **`--turb` on a polyMesh is not built** (exit 2 by name): the point evaluation
    (`dns/fvturb`) is mesh-generic, but the box's periods (L_y, L_z), its lattice
    truncation (n_d) and the frame (x along the flow from the inflow plane) are the flags'
    box. A polyMesh needs them from the mesh (its cyclic periods, a slab width as h, a
    lateral basis): undecided, no case asks for it yet.

M5c2. **The turbulent inflow is a stated gap between the compressible solvers, never
    measured on its own**: structured = the NSCBC inflow CELL relaxed to the box's FP32 lerp
    at x = h/2 (etaIn 100) behind a 16-cell sink band; FV = the Far FACE set to the exact sum
    at x = 0, no sink. The physics window (t in [0.5, 2]) is the IC's turbulence (the inflow's
    reaches the flame after ~13 flame times, past a 295 s job at 6 and 8 cells). A
    swept-inflow statistic (u' at a fixed distance downstream of the inflow, FV vs structured)
    is not measured. Measured in that window (u'/S_L 4, l_t 2 delta_th, M 0.02): FV - structured
    S_c -0.22%, -0.84%, -1.14% and area -0.05%, -0.18%, -0.28% at 4, 6, 8 cells/delta_th: NOT
    shrinking; the laminar flame differs -0.64%, -0.47%, -0.43% in the same window (the M 0.02
    start transient through NSCBC + sink vs Far faces); `--compensated` moves FV by < 1.1e-4.
    The low-Mach pair (same face conditions on both solvers) agrees to 6.7e-4, 3.4e-4, 2.6e-4 on
    the same case, so the turbulence path itself is consistent; the compressible cause is not
    isolated further (candidates: the 16-cell sink bands, the NSCBC inflow cell, the acoustic
    start).

M5c3. **Low-Mach `--turb` with checkpoints is refused** (exit 2): the loop carries the sweep
    X in its checkpoint fields (`LowMachLoop::sweep` when turb), the FV low-Mach runner has
    no checkpoint at all; no restart byte gate was run for the structured one. The
    compressible structured --turb restart (gpu_restart_flame) is untouched.

M5c4. **Low-Mach FV vs MAC under turbulence is a scheme gap, order 2** (L2: 7.7e-4, 2.0e-4
    of rho at 4, 8 cells/delta_th, t 0.4): collocated vs MAC velocity, and the tangential
    inflow rows sit on the structured y / z edges but on the FV face centres. Two levels
    (the third, 12 cells, 72^2 x ~180, not run in the gate). On S_c (docs/chem.md table, u'/S_L
    4) the gap is 6.7e-4, 3.4e-4, 2.6e-4 at 4, 6, 8 cells: below the grid change of S_c.

M5c5. **Physics numbers for the M5 pairs beyond C0 DNS are single-grid** (4 cells,
    docs/fvcomp.md / docs/chem.md tables): LES smag / sigma, TFLES, Charlette, C2b. C2b has
    no structured compressible partner (two-reactant --turb is not built there): its
    cross-solver partner is the low-Mach route (compressible FV M 0.02 vs low-Mach FV and MAC).

M5c6. **Found, not chased: the compressible laminar TFLES flame differs 10% between the
    structured and the FV solver in the 3D M 0.02 start window** (`--tfles 2`, 4 cells/delta_th,
    8^2, t in [0.5, 2]: S_c 1.13224 structured vs 1.01787 FV; the low-Mach structured 1.10592).
    No `--turb` in it. The FV vs structured TFLES gates (X1, gpu_fv_flame_les) are a periodic
    box and a 1D flame: none covers this open 3D start. The turbulent pairs are read as S_c /
    laminar S_c (1.0222 vs 1.0238) and the flame surface (1.4e-4).

M5c7. **Low-Mach LES under turbulence: FV - MAC S_c -9.0e-3 (Smagorinsky), -5.5e-3 (sigma) at 4
    cells**, the LF9 cell-velocity gap of mu_t, larger with resolved turbulence than on the
    planar flame (where sigma is 0). One grid; not laddered.

### Incompressible FV combinations (`dns/fvimodules`, docs/fvincomp.md "Incompressible combinations")

FI1. **Species on the incompressible FV stage: refused by decision, on evidence** (C0, C1, C2b
    refused alone, exit 2 where a flag can name them). No oracle: neither structured
    incompressible solver (gpu/incomp MAC, gpu/boussinesq) carries a scalar beyond T, so the
    generic gate 5 has no counterpart. No consumer: every flame model is calibrated by its
    heat release (S_L, T_ad), which constant density removes; the variable-density flame on
    the same mesh view is DeviceFvLowMach. A passive tracer already exists: the T row at
    betaG 0 (gpu_fvi_matrix (a): momentum bitwise the run without it; conv T energy 0 to
    rounding). Reopen if a constant-density mixing case (a Z field, a scalar plume) gets a
    reference: then a second T row (kFaceT/kBndT/kGatherT, a per-patch value table) with Z ==
    tracer, no new kernel shape.

FI2. **RANS on the incompressible FV stage: refused** (SA, SST refused alone, `--rans` exit 2 by
    the inventory's text). The family's RAS is simpleFoam (gpu/fv_simple + gpu/fv_ras, OpenFOAM
    v2606 verbatim, IF-T gate 1e-9); URANS on the projection stage would restate the closures
    with a second wall treatment. Not built, no consumer.

FI3. **LES of the Boussinesq layer has no structured counterpart** (`dns conv --solver fv
    --physics incompressible --les`): the structured conv solvers carry no SGS model, so gate
    5 is open for LES x T and LES x gravity; their physics numbers are LES vs DNS on the same
    FV solver (gpu_fvi_matrix_phys). FV LES vs the MAC LES (gpu/les, `dns flow` only) is two
    MODELS, not a scheme gap: gpu/les is the gradient form div(nut grad u) with a lambda
    fluid-fraction damping, FV the symmetric stress nut (g + g^T).S; no shared periodic route
    exists (MAC LES runs only around a body in runner_incomp). Not measured.
    The interior nuFlux column is resolved + molecular (no SGS heat flux); the wall columns
    are exact (walls carry no nut_f).

### Converging-diverging nozzle (`docs/nozzle.md`)

Plan: `plans/masterplan_cdnozzle.md`, amendments at its top. T1 holds: device vs rhoCentralFoam
at sigma 1 to 3.4e-13 of scale on every cell, shock and wedge included.

- Z1. KG vs MUSCL in smooth flow, band vs limiter at a shock: T2 is a gap, order quoted. **G23's
  order >= 1 is UNMET** on two levels: A vol L1 p 0.59, wall 0.53; B 0.70, 0.74; every 160 band met. The
  gap past x 1.5 is a flat offset, see Z4. 640 foam cut: dns alone there, Cd 0.995697, 0.995727, 0.995741,
  Cf 1.440293, 1.440341, 1.440369 at 160, 320, 640.
- Z2. Wall `p_P` first order in both solvers; viscous inlet face absent (`Total`), foam has one.
- Z3. `Slip` vs `wedge` rotation: `cos(dphi/2)` on laminar gradients.
- Z4. Foam's non-settling at vanLeer, cause not measured (A3 E4). Measured here on case A: in the
  supersonic cone foam's p0 error is -1.6e-3, -9.7e-4 at 160, 320 (order 0.74) and its T0 error -3.4e-4,
  -2.4e-4; dns -3.4e-4, -6.9e-5 (order 2.3), T0 5.5e-6, 1.7e-6. So the flat +2e-3 wall offset past x 1.5
  is mostly foam's. Its cause inside foam (Euler in time on a low-dissipation flux?) is not proved.
  **One row reproduces it** (masterplan_combnozzle R12, A12): nc A, 1 x nx, rhoCentralFoam Kurganov vanLeer to t 2.4 s:
  cone p0 error -1.71e-3 / -1.39e-3 (t 2.1 / 2.4) at 200, -1.21e-3 / -9.0e-4 at 400 (order ~0.6); T0 -3.8e-4 / -2.9e-4,
  -2.8e-4 / -2.0e-4 (2D: -3.4e-4, -2.4e-4 at 160, 320); not settled (p moves 1.2e-3, 2.0e-3 of p0 between t 2.1 and 2.4,
  growing with nx). Upwind (sigma 1) on the same row: steady to 6e-15, T0 exact, p0 -1.3e-2. dns on the row: CfVac
  order 2.0 to 2.1e-6 (G2.5). So Z4 is foam's own 1D behaviour at vanLeer: no axis focus, no 2D. The 800 rung and B (x_s)
  were cut (CPU budget).
- Z5. Axis focus: a pointwise axis M is no parity number. B exit axis M gap 9.6e-3, 1.40e-2 at 160, 320: it
  grows; vanLeer vs Minmod 1.2e-2, 4.0e-3. Cause open.
- Z6. Post-shock wall p0 3.0% gap [A1]: unexplained, not re-read by a gate.
- Z7. LTS on `Total`: stall [R] vs A2's 2.3e-12 on Riemann: open, `--steady` refused meanwhile.
- Z8. Structured steady odd-even: gate 34's order 1.23 may be the same mode: unproven. No structured D3.
- Z9. Foam's one-row L1 p 2e-3 vs dns 4e-5 [A2 E11]: unexplained, not dns's. Same item as Z4 (R12): foam vanLeer on
  one row does not settle, its p moves 1.2e-3 of p0 over t 2.1 to 2.4 s at 200 cells (2.0e-3 at 400).
- Z10. **Closed by G3.3** (masterplan_combnozzle P3, A14-A16): on the poly1d flame at M 0.01 and 0.042, 7.5 / 15 / 30
  cells, the line sensor reads 0 cells on every stats row, and D3 moves S_L by -3.0e-3, -3.4e-4, -4.2e-5 (order 3.1,
  3.0), below a third of the d4-0 error at every rung; the Shvab-Zeldovich defect is at or below the d4-0 one. Shock
  x C0 is live; C1, C2b stay refused (no gate, P8).
- Z11. Line sensor on tets, snappy cells. **Measured, refusal built** (masterplan_combnozzle R8, A11, docs/nozzle.md
  "Sensor"): unlined cells (an active face with no opposite at cos <= -0.5; wedge axis prisms exempt) are 0 on every hex
  mesh, cyl2d_snappy, and A1's prism75, tet75, tetdual (1.99, 1.99, 7.40 lines per cell): Kuhn tets and union-jack prisms
  have a <= 60 deg opposite on every face, so the sensor reads zigzag lines there and D3 acts along them. A regular tet
  is unlined: `--shock` refuses it by count (exit 3). Open: D3 / the sensor along 60 deg zigzag lines is ungated (P5).
- Z12. Total inlet under sonic backflow diverged once [A3]; no guard built, exit 7 by `stableDt`.
- Z13. Hoop term: one-column model, u_r only, Pr 2; G24 is the proof: 0.37 uniform, 0.58 tanh of the bound, on
  w = 0 only. `d_z` in the formula, not `|d|`: amendment A5.
- Z14. `FvRaster` on a wedge: built, PNG frames of `nozzle_W` show the z = 0 plane, read by eye once; no gate.
- Z15. Plume box: next cycle, placed in masterplan_combnozzle P7 (after viscous walls). Needs `Far` p, t, u; S1 contact;
  foam domain ladder; RANS for real separation. Until then a supersonic exit is ambient-proof by construction (G2.7: state
  and row bytes equal across p_a, `exitSuper` 1); `_thrust.csv` flags `exitSuper < 1`.
- Z16. FP32 D3 and FMA: G25 reads compensated -1.6e-8 mdot, 0.000 cells x_s vs FP64, plain -2.0e-5, on the
  planar B at t* 150; FMA itself not separated.
- Z17. **The chamber rings past t* 1800 on the `nc` wedge**: out/in - 1 reads +-3e-3 at single samples, a mean of
  800 samples reads 9.1e-6. Exit mdot flat (the throat is choked). Amplitude, period, driver not measured; `Total`
  reflects and lambda |u.S| damps little at M 0.1. T2 is on time means for this reason as well as Z4. Cd itself
  creeps until t* ~1500 (0.99596 at 300).
- Z18. **G22 passes by 0.04%**: Richardson d R^2 0.02426, -3.0% of (g + 1)/96 at 200, 240, 320 x 40, with
  A1's 0.0253 on another contour. Not a margin to rely on.
- Z19. **An MPI_ERR_TRUNCATE killed foam at 160 x 48 np 12** at its first step (`method simple`, n (6 2 1)); 320 x
  96 ran. Cause not looked at; level 160 ran serial.
- Z21. **T3 is unsettled.** G28 UNMET: wall L1 p 1.1e-3, 2.0e-3 at 160x48, 320x96, Cd gap 2.4e-4, 2.5e-4, tau 16%, 10%;
  a foam mean window of 17 t* against the ring (dns window halves 1.1e-3). G29 UNMET: foam at t* 76 is not settled, Cd
  0.99982 vs dns 0.993 at t* 70, 0.9889 at 140. Rerun both to t 0.6 with a 0.3 s window before reading a scheme gap off
  them: ~50 foam minutes at 320. Foam's vanLeer stage on the 5 deg wedge dies at maxCo 1 (sigFpe), runs at 0.3.
- Z20. **The exit row of a boundary layer.** `riemann` at p_b 0.05 sets p on the layer's subsonic faces: wall p
  14% below foam's zeroGradient in the last cells (a real change of problem, not a gap). T3 runs `extrapolate` from
  rest + ramp: it starts, ungated as a start. A viscous nozzle's production exit needs a plume box (Z15) to mean
  anything.
- Z22. **T2's shock position and axis M do not close.** B x_s gap 0.0109, 0.0124 at 160, 320 (`0.18 h, 0.41 h`: h halves,
  the length grows 13%); exit axis M gap A 7.5e-4, 1.6e-3, B 9.6e-3, 1.4e-2, both rising. Only volume, wall, mdot and
  A's Cf fall (B's Cf rises 3.6e-5, 2.6e-4). dns x_s 3.0716, 3.0696, 3.074: not monotone; foam 3.0825, 3.0820 at 160,
  320. A flat gap is a bug or a stated model gap: unattributed. Needs foam at 640 (cut).
  **One row** (masterplan_combnozzle R12, G2.6, A10): dns x_s 3.08025, 3.08188, 3.08240 at 200, 400, 800 vs the quasi-1D
  3.08380 (-0.07 h, -0.08 h, -0.12 h, settled from t* 50; order 0.88, 0.46: UNMET on order), Fexit -1.3e-5, -3.2e-6,
  -5.9e-7 (order 2.0, 2.4). The 2D dns x_s sits ~1e-2 upstream (3.0716, 3.0696 at 160, 320): ten times the one-row
  error at a like h, so the 2D offset is not the row algebra; it is 2D (the shock foot, a curved shock) or the 2D
  scheme. Foam's one-row B was cut (CPU budget).

### Reacting nozzle (`docs/nozzle.md` "Reacting", `plans/masterplan_combnozzle.md`)

P0, the 0D reactor (docs/chem.md "P0: the 0D reactor"; amendments A1–A3):
- N1. **G0.2 UNMET as specified**: N 2000/4000/8000 orders 2.203, 2.583, finest |ΔY| 3.57e-4 vs order
  ≥ 2.8, 1e-9. The exact-arithmetic SSP-RK3 recurrence has the same numbers (device gap 1.2e-10 to
  4.85e-9, counted as G0.2c): the ladder sits in the thermal runaway (J dt ≈ 2 at t_e/2000). Order 2.904,
  2.961 at 16000/32000/64000 (G0.2b); the 1e-9 needs N ~ 6e5 by truncation, while the device's rounding
  gap to the exact recurrence grows with N (1.2e-10 at 2000, 4.85e-9 at 64000, ~5e-8 extrapolated to 6e5):
  1e-9 is not attainable in FP64 at any N on this ODE.
- N2. **G0.3 UNMET as specified**: the full burn at t_e/8000 ends at Y −4.617e-3, T 3.19e-3 off T_end;
  that dt is 6.6x the solver's own `chemDt` (kChemDt 0.5) at T_end, where the floored RK3 overshoots
  past Y 0 and stops reacting. A scalar twin gives −4.61669e-3. Inside the bound (t_e/64000, G0.3b):
  T_end 2.8e-13. No solver change: `dns flame` never steps past chemDt. A fixed-dt API caller can.
- N3. **G0.5 UNMET**: chemFoam's `constantProperty volume` is an O(dt) split (its chemistry ODE
  integrates T at constant pressure inside each step; h is reset to u0 + p_old/rho + int Qdot after):
  max T error 0.495, 0.391 at t_e/2000, /8000, ignition delay 2.53e-3, 7.06e-4 (order 0.92). And seulex
  (the plan's ODE) aborts on the runaway in every case: the T row of `StandardChemistryModel::jacobian`
  is zero. With RKDP45 the constant-pressure reactor matches the cp quadrature to 3.3e-10 (G0.5b):
  the janaf-constant map is exact; a constant-volume foam reference needs another solver.

P1, the 1D flame on one polyMesh (docs/chem.md "P1"; amendments A5–A7): G1.0–G1.7 PASS, every control
fails as required. Open:
- N4. **The oracle's fresh-state sensitivity and the measured one disagree.** `Oracle.sens()` (A, Ta, q, cp,
  lambda fixed) gives dln S/dln T_u = 2.335, dln S/dln p_u = −0.5; docs/fvcomp.md's structured NSCBC pair read
  ~1.0 per unit ln T at M 0.05 (rho held: 2.335 − 0.5 = 1.84 would be the oracle's). Not resolved; G1.4 passes
  at either value (δS_fresh ≤ 2.8e-5 vs band/4 ≥ 2.5e-4).
- N5. **dns − foam on one mesh does not shrink at a clean order** (+1.38e-3, +4.6e-4, +2.3e-4: ratios 3.0, 2.0).
  Inside G1.2's band at every rung (4× below the smaller own error at 30 cells); the 30-cell gap is of the size
  of foam's missing viscous work (G1 reading) and of dns' O(M²) Richardson spread. No 60-cell rung (budget).
  Measured (N21, `fv1d_vw`): the work is a constant +1.49e-4 of it; with it the gap reads +3.10e-4 / +8.6e-5 (15 / 30).

P2, the minimal CD nozzle and the thrust ledger (docs/nozzle.md "Thrust", "Quasi-1D"; amendments A8-A13):
- N6. **G2.3 UNMET as specified (v == 0.0 bitwise on one row)**: the wall face's S_y and the axis face's differ by an ulp
  per cell (2.2e-19 to 8.7e-19 absolute, ~2e-16 relative; OpenFOAM's face fan on the slanted quad), so v is a rounding
  field: max |v|/c 1.4e-17, 7.2e-18, 7.1e-17 (parabola 200, 400, 800), <= 1.9e-16 on nc A, B and the twin (G2.3b,
  band 1e-12, counted). The x faces' S_y is exactly 0.0. A 1e-9 x-jitter of one axis node gives 6.7e-12 (the control).
  No fix: the geometry is OpenFOAM's by invariant.
- N7. **G2.6 UNMET on order**: one-row x_s within 0.12 h of the quasi-1D shock at every rung, settled, but the (p1 +
  p2)/2 crossing's sub-cell offset (-0.07 h, -0.08 h, -0.12 h) is not monotone in h: orders 0.88, 0.46 vs >= 0.9.
  The post-shock integrals converge at order 2 (Fexit, wall). Stop rule: recorded, P4 proceeds (its nozzle is smooth).
  `--shock-d4 0` (reported): x_s 3.08016, 3.08190, 3.08241, Fexit 1.5740346, 1.5740521, 1.5740563: D3 moves x_s <= 1e-4.

P3, Shock x C0 (docs/nozzle.md "Reacting"; amendments A14-A16): G3.1-G3.4 PASS, G3.5 reported; R6's fallback not
needed. Open:
- N8. **Two plan controls were dead and are replaced once (A14).** G3.1's "D3 applied to rho Y itself" cannot break a
  power-of-two uniform Y: d(rho Y) = d(rho)/4 exactly, so `Y_f F_KG + l d(rho Y)` IS `Y_f F_final` bit for bit
  (measured: bitwise). Replaced by "species on the KG mass flux" (the order-swap hazard): breaks. G3.2's structured
  `D Y_f(sign m_KG)` stays in [0, 1] on the Sod step (min Y 0.0: D and m_KG share their sign at the contact); live on
  the Mach 3 piston (-1.08e-2, a 8% margin over -1e-2), which carries the gate's control.
- N9. **Closed (P4)**: `[thrust].station` feeds `_thrust.csv`'s `p0b T0b CdB CfBvac CfB` (`fvStation`). The route's
  smoke moved from `gpu_fv_flame_shock` to `gpu_fv_nozzle_react` (a `[nozzle]` flame now needs a throat).
- N10. **G3.3's control reads S_L on [2, 4], not [7, 14].** HLL on every face (force sigma 1) thickens the flame ~3x;
  past t ~ 4 its preheat reaches the probe's fresh reference slab (2 delta_th from the inflow), whose rho Y normalises
  S_L, and the probe reads ~2e8. Over [2, 4] the flame is inside the guard and S_L 1.631 vs 1.010 (shift 0.62 vs the
  needed 1.8e-2). The force runs' late S_L is a probe artefact, not a speed.

P4, the reacting quasi-1D nozzle (docs/nozzle.md "Reacting"; amendments A17-A22, P4b A23-A26). G4.1-G4.4, G4.6 (p0b),
G4.7, G4.8 PASS at 15 cells; at 30 cells G4.2-G4.4, G4.6 (p0b), G4.8 PASS, G4.1 and G4.6 (mdot) UNMET, G4.5 unevaluated
(twin cut); every control run fails as required. Open:
- N11. **Rig corrections forced by the plan's own rules and cost** (A17): Lcon 4 y* at eps_c 6.3 slopes to 64 deg and R8
  refuses 1400 one-row cells (the axis face's partner, the wall, at cos -0.43); eps_e 6.25 puts the viscous dt at 2.5e-5
  tau (rho_e 0.052 rho_0b): 3.1e6 steps per 80 tau, past the job cap. The rig is y* 50 delta_D, Lcon 6, Ldiv 4, eps_e 2.
  An eps_e 6.25 reacting nozzle needs an implicit (or local) viscous term, or a coarser nozzle half.
- N12. **The neutral flame drifts at sigma (u_d - S_c)** (A18): the throat holds the burnt volume flow. The plan's 0.98
  S_L design drifted -0.1056 S_L (8.4 delta_D in 80 tau, preheat into the reference slab); u_d = S_L,num: -0.0040.
  R5's "~0.02 S_L" sizing is wrong by sigma; the kinematic gate holds at both (G4.3).
- N13. **G4.5 Cd at the band** (P4b, A24): CdB - Cd_twin -1.10e-4 over the plan's [40, 80] (UNMET), -9.91e-5 over the
  ring-cancelling [20, 80] fixed before the 30-cell run (inside 1e-4 by 9e-7, less than the twin's own unsettled drift
  1.7e-5 between its [30, 40] and [60, 80] windows: not robust); CfBvac -6.9e-6 / -4.2e-5 passes. The flame half is
  h-independent (CdB 0.998976 -> 0.998977 at 30 cells); the 30-cell twin was cut (~13 min GPU, viscous dt), so G4.5 at
  30 is unevaluated. The viscous loss itself (twin - inviscid) is Cd -9.24e-4, CfVac -2.94e-3 at Re* 736.
  **CLOSED (combclose K5, K6)**: the IC slug (N15). Fixed IC, compensated FP64, rule window: dCd (throat flux) +1.05e-6
  at 15 (W [50.7, 140], k 4), -8.0e-7 at 30 (W [57.6, 80], k 1) against the 30-cell twin by grid sequencing (Cd
  0.9990744066, CfVac 1.4546626487, Re* 737.0612 = the 30-cell rig's y*; the analysis' 736.2747 took the 15-cell y*);
  dCf +1.5e-6 / +2.1e-6.
- N14. **The Total-fed choked chamber rings, h-independently**: u_u rms 1.0e-3 to 2.6e-3, p0b rms 2.8e-5, CdB per-row rms
  2.1e-4, spectral peaks 20, 30, 15, 60 tau, IDENTICAL at 15 and 30 cells (P4b): a physical oscillation of the chamber
  + nozzle, not a scheme ring; not growing over 80 tau; mode not identified (no p frame on one row). Window means of
  40 tau carry its phase (+-1e-4 in Cd); 60-tau windows cancel its peaks. The CLI twin from rest rang +-1e-3 in Cd at
  60-80 t* (A20); from the q1d IC it settles.
  **Identified (combclose K3): ONE damped linear acoustic mode, a start transient, not sustained.** The "peaks 20, 30,
  15, 60 tau" are the DFT bins of a 60-tau record (withdrawn). `nozzle_react.py modes` (linear quasi-1D network, Total
  inlet, compact moving flame, non-compact cosine contraction, throat regularity): mode 1 P 22.680 tau, e-fold 33.80 at
  15 cells (22.700 / 34.00 at 30); fitted on uFresh: p15 [50.2, 80] 22.417 / 31.66, p30 22.452 / 32.02, q15 [50.2, 140]
  22.382 / 31.95. Controls: p-node inlet e-fold 57.70 (the Total inlet radiates ~42% of the damping), compact choked
  nozzle P 7.283. Its old amplitude was the IC slug's (N15); windows now span integer periods from t_e + 10 (2.4).
  Fixed-IC runs (G4.11 pass on all three): K5 15 cells 22.324 / 32.53, K6 30 cells 22.384 / 33.63 (model 22.700 /
  34.00), K7 gamma 1.2 22.438 / 33.71 (model 22.802 / 35.73); u_u amplitude 1.7e-3 vs 3.4e-3 old IC. Not a defect: a
  damped start transient the window rule cancels. Open only as a cost: a 1-period window (30 cells, tt 80) leaves
  ~2e-5 of storage in G4.6 mdot.
- N15. **G4.6's mdot UNMET at 15 AND 30, does not fall** (P4b, A24): (mdotOut - rho_b x_f' A_c) vs rho_u S_c A_c -2.25e-4
  / -2.23e-4 over [20, 80], -1.37e-4 / -1.41e-4 over [40, 80] (order ~0). The composition side closes (the ledger's burn
  rate vs rho_u S_c A_c +7.5e-6); the gap is A19's drift storage model: the ledger's exact chamber mass change dM/dt
  -0.0819 / -0.0845 vs (rho_u - rho_b) A_c x_f' -0.0870 / -0.0897 (6% over, 1.7e-4 of mdot), also with x_f' from the fuel
  mass. Cause of the extra storage not identified (not the ring's phase: 60-tau windows).
  **Cause found (combclose):** the IC. `fvReactingNozzleIC` laid the chamber with `premixedPoint` (low-Mach in energy:
  burnt T0 +1.7094e-3 over T0b); that entropy slug crosses the throat at t_e 40.2 tau, inside [20, 80] and [40, 80]
  (mdotOut dip -7.0e-4, the ring re-excited), and its leaving density deficit is the "extra storage". Fixed in K2 (the
  chamber on the Rayleigh line, `fvref_react` R5, G4.12a 2.7e-10 on the first row); the gated rerun is K5. Same cause
  for N13's Cd.
  **CLOSED at 15 cells (combclose K5)**: fixed IC, compensated FP64, tt 140, rule window W [50.716, 140] (k 4 of the
  fitted P 22.324): G4.6 mdot +1.475e-6 (= burn -3.68e-7 + storage +1.84e-6, identity 2e-16), G4.5 dCd +1.05e-6 (throat
  flux), dCf +1.49e-6, G4.12 b dip -1.05e-4 (old -7.08e-4); contraction storage over W -2.7e-6 of mdot. Legacy [20, 80]
  / [40, 80] mdot -6.7e-5 / -4.3e-5 (old -2.20e-4 / -2.10e-4 through the same estimator): the remainder is the ring's
  phase, which the rule's integer periods cancel. Both gates now hold by ~70x (mdot) and ~100x (Cd) inside 1e-4.
  30 cells (K6, W [57.619, 80], k 1): mdot +2.126e-5 = burn -5.1e-7 + storage +2.18e-5 (one period's ring residual,
  contraction storage -1.7e-5), legacy [20, 80] -6.5e-5. "Falling with h" is replaced by the band at each rung (2.5):
  burn -3.7e-7 / -5.1e-7 sits at the estimator floor.
- N16. **Re-diagnosed (P4b, A23): the reacting G4.1 growth is the FP64 STATE's rounding, not the ledger.** The ledger is
  now compensated (`fvLedgerAdd`, two-sum; `gpu_fv_nozzle_ledger` N16: 1e5 steps 3.9e-15 flat, the plain hi word
  2.6e-13), and the reacting residual did not change (8.36e-14 / 8.38e-14 old / new at 31 639 steps): res_mass grows
  +7.0e-15 per step at 15 cells and +7.2e-15 at 30, linear, one sign, independent of h and dt: each stage write rounds
  the same way every step on a slowly drifting state. res/S' tends to that loss over |dI| per step (1/dt): 8.7e-13 at
  15 (t 80), **3.51e-12 at 30: G4.1 UNMET at 30**. A fix needs a conservative FP64 state update (a compensated FP64
  state, refused today): not built.
  **CLOSED (combclose K4): compensated FP64.** Cause pinned (analysis_combclose_2 A2): the plain stage-3 write
  `ca a + cb b + cr r` (cb = R(2/3)) compiles to `fma(cr, r, fma(ca, a, RN(cb b)))`, which rounds `ca a + RN(cb b)` on a
  lattice of spacing ulp/3 before the increment: `test_stage_bias` (host, std::fma, binary128) +0.1112 ulp per write on
  mantissas [1, 1.5), -0.0000 on [1.5, 2); the increment form +0.0003. `--compensated` under `--fv-precision fp64` = the
  double-double state (`gpu/word2d.hpp`, the `Comp` kernels' R = double instantiation; docs/fvcomp.md "Precision: the
  plain FP64 write is biased"). N16b (`gpu_fv_nozzle_react`, 15 cells, tt 3, 31 639 steps): max r 2.07e-16, flat (last
  10% of rows 1.57e-16 vs rows 10-20% 1.82e-16); plain control 1.48e-13. Off = bytes (bit subset vs dns.BASE, the plain
  `[thrust]` rig vs dns.K3). Cost +3.1% per step (0.3076 vs 0.3170 ms/step, 15-cell reacting rig, `--bench`). The
  full-length gated runs (K5/K6) run compensated. K5 (15 cells, tt 140, 1 473 931 steps): G4.1 2.07e-16, G4.1f flat
  (last 10% / rows 10-20% 0.87); the old plain q15 9.79e-13. K6 (30 cells, tt 80, 3 369 062 steps): 1.65e-16, flat
  0.76 (plain P4b 3.51e-12): G4.1 at 30 MET.
- N17. **G4.7 partial**: the start perturbation is x_f + 2 delta_th (`--upstream 14`) only; "p0 IC + 1%" has no knob (none
  built for a test). The flame stays 3.20 delta_D downstream to the end; window mdot, CfBvac, S_c 4.4e-5, 2.2e-6, 3.8e-7.
  **CLOSED (combclose K5, G4.7b; R8)**: "p0 IC + 1%" is replaced ONCE by the old-IC start (q15: the same rig with a
  1.71e-3 entropy slug and 2x the ring amplitude), compared on the fixed-IC run's W [50.716, 140]: mdotIn -3.45e-3,
  CfBvac +2.25e-6, S_c -7.76e-7 vs 2 x span 0.18, 2.2e-4, 6.0e-5. (+1% would be u'/u ~ 0.17, 50x today's ring, ~125 tau
  more to settle.) No knob built.
- N18. **The viscous one-row v is a field**, max |v|/c 6.9e-3 in the twin (P_my/P_mx 8.7e-4 in the flame run; the flame
  route outputs no v): the slip wall's least-squares gradient carries the normal stress into y. Quasi-1D only inviscid.
- N19. **Cut** (masterplan_combnozzle section 6; P4b A25): the eps_c 13.2 row and the gamma 1.2 row (G4.9) by GPU budget;
  G4.9's dead control ("off >= 4%": 0.56% at eps_e 2) replaced once by "CfBvac vs the gamma-1.4 closed oracle misses by
  >= 1e-3" (expected 8.1e-3), not run. The reactingFoam `transonic yes` scout (G4.10) ran 5 CPU min and did NOT settle:
  t 7.8 of 80 tau, mdotIn 31.13 vs mdotOut 30.64 (1.6% transient); stopped. The 30-cell rung ran (A24).
  **combclose:** the gamma 1.2 row RAN (K7, 15 cells, compensated, tt 140, W [50.255, 140] k 4): G4.1-G4.6 pass (dCd
  +8.3e-7, dCf +8.5e-7, mdot -1.3e-6, p0b +8.1e-7, G4.2 +2.1e-5), G4.11/G4.12 pass; control CfBvac vs the gamma-1.4 oracle
  +5.51e-3 >= 1e-3 fails as required (docs/nozzle.md "The gamma 1.2 row"). eps_c 13.2 CUT BY DECISION (R9): the M^2
  scaling of the Rayleigh loss is the oracle's closed form and G4.6 p0b holds to 7.5e-4 of it; entry for a later cycle
  = a 1D hll ladder plus a foam rung at M 0.02. G4.10 is DECOMPOSED (R10): the flame vs reactingFoam (G1.2; G1.8 at M
  0.042 hll, on foam + viscous work, N21), nozzle + Total + riemann vs rhoCentralFoam (T1), the coupling vs the closed
  oracle + the inert twin (G4.5/G4.6); an end-to-end foam run is not run (~60-90 CPU min for a ~1e-3 converged number).

combclose (`plans/masterplan_combclose.md`, amendments B1-):
- N20. **Every plain stage write `ca a + cb b + cr r` with cb = R(2/3) is biased** (masterplan_combclose R12, N16's
  mechanism): it rounds `ca a + RN(cb b)` on a non-uniform lattice before the increment. FP64 measured: +0.11 ulp per
  write on mantissas [1, 1.5) (`test_stage_bias`), +0.03-0.09 u|q| per step in every conserved row on the reacting nozzle
  (res_mass +7e-15 per step). FP32 (`kFvGather<float, false>`, the structured `stagePut` plain path): the same form,
  `R(2/3)` rounds UP in float, so the sign likely flips; ~0.1 ulp_f32 per step, the order of the known plain-FP32
  wander: UNMEASURED (a closed-box plain vs compensated mass-drift probe would settle it). Not fixed: the plain writes
  are the bit-identity invariants of every shipped row; `--compensated` (FP32 and, since K4, FP64) is the fix where a
  ledger or a settled number needs it.
- N21. **CLOSED (B4): the converged dns − reactingFoam O(M²) gap is stock reactingFoam's missing viscous work** (K1,
  docs/chem.md "P1" "At the nozzle's Mach"). v2606 `reactingFoam/EEqn.H` (h + K, −dpdt, laplacian(alphaEff, he),
  Qdot) has no div(τ·u); dns' E flux carries u_f·(τ_f·S) (`fvMeanFlux`). Every other term matches here (stress,
  sensible h with equal Cp, Le 1, K, dpdt, constant μ). In the flame u_x ∝ T_x, so (4/3)μ u u_x is extra conduction
  (4/3)Pr(γ−1)M² T/T_u the species do not get (effective Le up to 1 + 3.4e-3 at M 0.042): stock foam is SLOWER.
  Measured, same poly1d files, M 0.042, dns `--shock hll` FP64: dns − foam stock +2.620e-3 / +2.593e-3 (15 / 30), with
  the work as an h fvOption (`writer.VISCW`, `fv1d_mn_vw`) +2.099e-5 / +1.379e-5; foam's Richardson (p 2) 0.988374 →
  0.9909475 vs dns S∞ 0.9909473. Finite-Mach oracle (`tools/chem/mach_flame_oracle.py`, steady compressible NS ODE):
  0.990884 with, 0.988268 without, difference 0.265 γM²(σ−1) (measured 0.26); fresh-state corrected 0.990938 vs dns
  S∞. M 0.01 (`fv1d_vw`): the work moves foam +1.49e-4 (oracle 1.50e-4), dns − foam +3.10e-4 / +8.6e-5 (stock
  +4.59e-4 / +2.34e-4): P1's 0.29 γM²(σ−1) is the same term. G1.8 PASSES on foam + the work; the stock gap is a stated
  MODEL gap of stock reactingFoam (no viscous heating), printed beside, not gated. Any dns-vs-reactingFoam comparison
  at finite Mach (the reacting nozzle's foam twin, G4.10) quotes foam + the work or states the gap (~0.27 γM²(σ−1) in
  S_L for this flame).
- N22. **CLOSED (combclose K8): the FV `dns flow --steady` label misstated the state.** The driver compensates a
  `--steady` march in FP32 only (`c.compensated || (compSteady && fp32)`, unchanged since M1); the RANS label printed
  " compensated" for ANY `--steady` (a plain FP64 state: wrong since 644e596, kept through K4) and the non-RANS label
  printed nothing on a compensated FP32 laminar `--steady` state. Both now read `fvCompStateCompensated(c)`
  (config.hpp), the ONE rule the driver runs (`test_fv_comp`: six label/state checks; the RANS FP64 and laminar FP32
  ones fail on the old code). Label text moves on exactly: RANS `--steady --fv-precision fp64` without `--compensated`
  ("FP64 compensated" -> "FP64") and laminar `--steady` FP32 without `--compensated` ("FP32" -> "FP32 compensated"), in
  the `_stats.csv` / `_thrust.csv` comment line, the `--dry-run` label and the run's stdout. Physics bytes unchanged:
  `comp_bit_identity.sh` vs dns.BASE 55 of 56 rows byte-identical, fvpolysteady differs in line 1 of `_stats.csv`
  only (json, mask, data rows equal); an FP64 SA-neg O-grid `--steady` (old vs new binary) differs in line 1 only.
- N23. **CLOSED (combscale S1): `shockHot` was carried and never checkpointed on the FV flame route**, and the
  `[thrust]` ledger was refused with checkpoints. `runFvComp`'s `--shock hll` sticky dt cadence lived in a local whose
  comment said "checkpoints are refused on this route" (false: `dns flame --polymesh --shock hll --checkpoint-dt` passed
  `resolveCheckpoint`), so a restart after the first sensed cell would have re-entered the default cadence. It is now
  `FvCompLoop::shockHot`, record `loop.shockHot` only under the auto cadence (runner_flame's pattern); no CLI-reachable
  teeth (the nozzle senses 0 cells). Same commit: the ledger's device words and `loop.thrust.*` are checkpointed, the
  device `load` moved after `setLedger(true)` (it re-zeroed the words), the `bc.sections` record names a changed
  `[nozzle]`/`[thrust]`; `gpu_restart_fv_thrust` (docs/nozzle.md "Thrust").

## Older open items (details in CLAUDE.md)

- No anchor for the Re800 compressible cases or `sphere_re300_n160_cd`
  (rows commented out; their tests skip).
- Re60 compressible vs incompressible cd gap +7.3%: cause unisolated
  (ghost-mirror vs cut-cell wall, sponge vs free-slip laterals).
- `gpu/comp_far.hip` domain y walls hardcoded adiabatic.
- Rotors: no table cache, no incompressible branch, no dns-ui support.
- dns-ui: a stop issued before a run reaches its time loop is not clean.
- README names gfx1201; this machine is gfx1100.
- **Host phase** (`dns/parallel.hpp`, bit-identical; `runs/prop/curv/`,
  `--bench` short runs): DTMB 4119 P1 50.7 → 8.0 s, P3 156.0 → 20.8 s.
  Merged with the NSCBC table + `--dt-recompute 16` (`plans/verify/
  bench_prop_after.md`): host P1 7.2 / P2m 13.3 / P3 20.9 s; P1 end to end
  247.6 → 172.3 s measured, P2m 635.1 (extrap.) → 478.5 s measured.
  Serial P3 split: body IC 122.1 s (one mesh-sdf call per cell; the older
  "dominated by `CurvMetrics::build`" was wrong), metrics 16.7, MRF
  classify 9.6 (10 threads), `sampleCut` 4.3. Left at P3: body IC 12.6 s on
  20 threads. A bbox short-circuit (cell > 3h outside the body AABB ⇒ f = 1)
  would skip most calls but equals the full sweep only where the mesh sign
  test is right far from the body (non-watertight meshes) — not built.
