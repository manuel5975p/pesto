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
    band. Not fixed; a stiffer --nscbc-etain not measured on the final
    code. The species pin on that row
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
- C4. **Nozzle total-pressure drop 1.45× the isentropic one** (Cd 0.83,
  1.448 → 1.470 from 4 to 6 cells), attributed to the viscous exit at
  u_e He/ν_b ~ 100; no Reynolds or resolution study beyond the two points.
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

## FaceMesh / FV path (`docs/mesh.md`, `docs/fv.md`, `plans/masterplan_meshsolver.md`)

56. **Phases 1–2 only: the FaceMesh and a CPU FP64 FV oracle exist, no GPU
    solver runs on it.** Built and pinned: polyMesh reader/writer,
    `fromPolyMesh` / `fromCurvMesh`, orderings, `CellGraph`, `dns mesh`
    (`plans/verify/impl_meshsolver_p1.md`); `fvflux.hpp` / `fvgrad` /
    `FvRef` (`plans/verify/impl_meshsolver_p2.md`, gates in docs/mesh.md).
    P2 open by plan: no GPU FV kernels (P4), no scalar/species rows, no MRF
    frame flux, no eddy viscosity, no LTS march; one `FvBc` per patch (no
    per-face profiles). Scheme limits, stated: `Far` is a 1D characteristic
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
