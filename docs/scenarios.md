# Scenarios — cheap runnable recipes

One page of "interesting fluid scenarios" a user can run as-is (masterplan
`plans/verify/masterplan_verify.md` W16). Every argv below passes `--dry-run`
on the 2026-10-27 build (checked 2026-10-27, W16); every number comes from a run
another verification item already made — this page adds no code and no run.
gfx1100 (RX 7900 XTX), Release `build/`. Box units unless stated (O-grid
cylinder/sphere: D = π/4, U = 1, so St = f·D and `--ft` counts 40 D).

Conventions:
- **Wall time** is the measured run (stepping plus setup, lock wait
  excluded). "long" = over 60 s; the hard ceiling is 300 s.
- **Peak VRAM: not recorded.** None of the source runs logged it and this page
  adds no GPU run. All of these grids are ≤ 6.6 M cells.
- **Label** is the one the run prints (dry-run `label` line, progress lines,
  stats header). Quote it with the number.
- **Run directory**: `$S` = the verification session's scratchpad
  (`/tmp/claude-1000/-home-manuel-Documents-pesto/c3bc040d-2132-4633-a610-50a9da9c0d9c/scratchpad`),
  which is ephemeral; the durable record of each run is the cited
  `plans/verify/impl_W*.md` and docs section. `runs/parity/` is in the repo.
- Always `--dry-run` first; replace `--out` with your own prefix.

| # | scenario | label | wall | source |
|---|---|---|---|---|
| 1 | O-grid cylinder Re 40 steady / Re 100 shedding | DNS | 3.2 s / 21 s | W12 |
| 2 | O-grid cylinder Re 1e4 URANS shedding vs OpenFOAM | RANS SST-2003m / SA-neg, URANS | 96–191 s (long) | W12 |
| 3 | Compressible TGV Re 1600, DNS / smag / σ vs 512³ | DNS / LES | ≤ 0.5 s each | W10 |
| 4 | Channel Re_τ 180, DNS / smag / σ | DNS / LES | 52 / 74 / 82 s (long) | W11 |
| 5 | Supersonic isothermal channel M_b 1.5 (CKM 1995) | RANS SA-neg, implicit steady | < 1 s | W13 |
| 6 | Full-revolution sphere Re 250, steady m = 1 wake | DNS, steady | inside a 40 s test | P4, W14 |
| 7 | Two-blade MRF rotor `rotor2` vs OpenFOAM | DNS [MRF frozen rotor: an APPROXIMATION] | 35 s (L1) | W7, W6 |
| 8 | Axisymmetric SA-neg sphere Re 1e3 vs OpenFOAM | RANS SA-neg, steady, wedge | 29 s | W13 |

## 1. O-grid cylinder: Re 40 steady and Re 100 shedding (DNS)

    build/dns flow --physics compressible --body cylinder --ogrid --mach 0.2 \
        --emit stats --n 256 --ny 96 --re 40 --steady 1e-4 --out re40
    build/dns flow --physics compressible --body cylinder --ogrid --mach 0.2 \
        --emit stats --n 192 --ny 96 --re 100 --ogrid-trip 0.05 --ft 3.8 --out re100

- Grid: body-fitted O-grid 256×96 (Re 40) / 192×96 (Re 100) × 1, far row
  20 D (the default). Labels `DNS -- steady (local time stepping:
  pseudo-time, not time-accurate)` and `DNS [O-grid trip 0.05 U]`.
- **Re 40**: 31200 iterations, 3.2 s. cd 1.56656. Divided by the MEASURED
  M 0.2 / 20 D factor 1.0382 (grid, Mach and far-row studies,
  docs/validation.md "O-grid cylinder: literature and the Re 40 offset"):
  1.5089, −0.86% from Dennis & Chang 1.522.
- **Re 100**: 21 s. Window t ∈ [82, 121.4], 7 periods: St 0.1679
  (Williamson 0.164–0.166), cd 1.3800 (2D DNS 1.33–1.40), cl' 0.3333
  (0.32–0.34); first/last-period cl' 0.3323 / 0.3331 (saturated).
- `--ogrid-trip 0.05` is what makes 3.8 suffice: the 2D cylinder O-grid is
  untripped by default and then reaches the same limit cycle ~1.8× later;
  the trip shortens only the linear growth (issues.md 16).
- Test: `val_ogrid_cyl` (both runs, 26.7 s, labels `validation;literature`).
  Runs: `$S/impl/W12/re40*` (+ the grid/Mach/far study `re40_*`),
  `plans/verify/impl_W12.md`.

## 2. O-grid cylinder Re 1e4: URANS shedding vs OpenFOAM (long)

    build/dns flow --physics compressible --body cylinder --ogrid --n 256 --ny 96 \
        --ogrid-beta 4 --ogrid-far 20 --re 1e4 --mach 0.2 --ogrid-trip 0.05 \
        --rans sst-2003m --compensated --ft 1.6 --emit stats --out sst
    # SA: --rans sa-neg; 2x cheaper: add --cfl 0.8

- Grid 256×96, β 4, 64 D/U. Label `RANS SST-2003m -- URANS (time-accurate)
  [O-grid trip 0.05 U]` (SA: `RANS SA-neg -- ...`), compensated state
  (`--compensated`, issues.md 20).
- Wall: SST 191 s (781682 steps, 0.244 ms/step); SST `--cfl 0.8` 96 s
  (390842 steps, means within SE of cfl 0.4: cd / cl_rms / St 1.6989 /
  1.2190 / 0.2314); SA 171 s of stepping (780090 steps × 0.220 ms; W12
  quotes 195 s, its log 286 s including lock wait). Run health: y⁺
  max 0.93–1.12 (the dry-run's 1.76 is an estimate), 0 SST floor hits.
- Report a URANS number as a window mean ± batch SE (docs/rans.md
  "Reporting a URANS result").

| | window D/U (periods) | cd | cl_rms | St |
|---|---|---|---|---|
| dns SST-2003m | 27.7–62.3 (8) | 1.6985 ± 0.0069 | 1.2164 ± 0.0084 | 0.2314 ± 0.0019 |
| OpenFOAM kOmegaSST | 45.7–63.5 (4) | 1.6945 ± 0.0130 | 1.2084 ± 0.0020 | 0.2249 ± 0.0035 |
| Δ | | +0.24% | +0.66% | +2.92% |
| dns SA-neg | 45.2–66.0 (5) | 1.7383 ± 0.0054 | 1.3581 ± 0.0047 | 0.2406 ± 0.0003 |
| OpenFOAM SpalartAllmaras | 49.6–66.4 (4) | 1.7372 ± 0.0043 | 1.3626 ± 0.0028 | 0.2391 ± 0.0015 |
| Δ | | +0.06% | −0.33% | +0.63% |

- Every Δ is within 1.7 SE. Open: on the common late window t ≥ 45 the SST
  cl_rms gap is −0.74% at −4.2 SE, unattributed (issues.md 24).
- OpenFOAM on the identical mesh (rhoPimpleFoam, PIMPLE nOuter 2, np 6,
  ≈ 3 min per model): a 1-step `--emit mesh` run, then
  `python3 -I tools/parity/ocyl1e4.py case MESH.bin DIR sst|sa --end 66.5 --np 6`
  and `python3 -I tools/parity/ocyl1e4.py run DIR --timeout 288`.
- Runs: `$S/impl/W12/{sst_c04,sst_c08,sa_c04,foam_sst,foam_sa}`,
  docs/validation.md "O-grid cylinder URANS shedding vs OpenFOAM",
  `plans/verify/impl_W12.md`.

## 3. Compressible Taylor–Green vortex Re 1600: DNS, Smagorinsky, σ (LES)

    ctest --test-dir build --timeout 180 -R gpu_les_tgv_comp -V
    # or directly: build/test_gpu_les_tgv_comp

- There is no `dns` argv: `dns tgv` is the incompressible solver and takes
  no `--les` (that plumbing was dropped by decision, masterplan §6). The
  test binary IS the recipe; it prints every number below.
- M 0.1, Re 1600, 32³, cfl 0.4, to t = 12. Labels DNS / `LES smag Cs=0.17` /
  `LES sigma Csigma=1.35`. 32³ is under-resolved by design: model
  behaviour, not a converged LES.
- Wall 0.20 / 0.45 / 0.50 s per run; the test (plus a focal dt probe) 2.0 s.
- Reference: incompressible 512³ DNS (`data/validation/tgv3d_512.csv`), peak
  −dK/dt 0.012994 at t 9.03.

| run | peak −dK/dt | t_peak | vs 512³ | max\|K − K512\| |
|---|---|---|---|---|
| DNS | 0.009380 | 10.74 | −27.8% | 0.0146 |
| LES smag 0.17 | 0.011813 | 6.42 | −9.1% | 0.0250 |
| LES sigma 1.35 | 0.011632 | 6.86 | −10.5% | 0.0229 |

- The models fix the peak VALUE, not the curve: both peak ~2.5 time units
  early. σ is NOT quiet for t < 3 on the TGV (ε_sgs/ε_visc 1.35 at t 0.5).
- Runs: `$S/impl/W10/tgv_run{1,2,3}.log`, docs/validation.md "LES known
  answers", `plans/verify/impl_W10.md`.

## 4. Channel Re_τ 180: DNS vs Smagorinsky vs σ (LES, long)

    build/dns channel --n 32 --ny 48 --nz 32 --stretch 1.6 --mach 0.3 \
        --re-tau 180 --kick 0.1 --tt 40 --profile-dt 1 --out dns
    # LES: add --les 0.17 (smag) or --les 1.35 --sgs sigma

- Grid 32×48×32, stretch 1.6: y⁺₁ 2.08, Δx⁺ 70.7, Δz⁺ 23.6; 40 δ/u_τ,
  ≈ 5.0e5 steps. Labels `DNS`, `LES smag Cs=0.17 (filtered: grid- and
  Cs-dependent)`, `LES sigma Csigma=1.35 (filtered: ...)`.
- Wall 52.3 / 74.2 / 82.0 s (0.10 / 0.15 / 0.16 ms/step).
- Window t 25–40, 5 batches; reference MKM ub⁺ 15.68:

| model | ub⁺ | vs MKM | state |
|---|---|---|---|
| none | 16.647 ± 0.004 | +6.2% | turbulent, u_rms⁺ peak 3.31 |
| smag 0.17 | 16.47 ± 0.14 | meaningless | RELAMINARIZED (u_rms⁺ 0.20): no wall damping, max ν_t in the first cell |
| σ 1.35 | 18.103 ± 0.026 | +15.5% | turbulent, u_rms⁺ peak 3.44 |

- No window is stationary (|trend| > 4.6 SE on every run): a result, not
  a band. nutWall == 0.0 exactly every row (odd wall image).
- Runs: `$S/impl/W11/man/`, docs/channel.md "LES", issues.md 25,
  `plans/verify/impl_W11.md`. Test: `gpu_channel_les` (0.8 s).

## 5. Supersonic isothermal channel, M_b 1.5 (Coleman, Kim & Moser 1995)

    build/dns channel --n 4 --nz 1 --ny 128 --re-tau 192.63 --mach 1.517 \
        --rans sa-neg --steady 1e-6 --steady-scheme implicit --out ckm

- One broadcast column (v = 0, no shock): a supersonic `--mach` is accepted
  ONLY on the implicit steady RANS channel (issues.md 23). Label `RANS SA-neg
  -- steady (implicit Newton pseudo-time: not time-accurate)`, constant μ
  (CKM use μ ∝ T^0.7).
- 27 iterations, < 1 s. The flags are the CKM point M_b 1.5, Re_b 3000
  (oracle-solved); the nominal `--re-tau 220 --mach 1.5` is another point.
- Tc/Tw 1.3816 (CKM ≈ 1.38); Re_τ on wall values 224.5 (CKM 222); the van
  Driest slope over 30 < y⁺ < 0.3 Re_τ is 3.07 vs 3.04 at M_b 0.2 (oracle
  ny 128; raw u⁺: +17%) — CKM's collapse.
- Runs: `$S/impl/W13/ckm/`, docs/rans.md "Scenario: the supersonic
  isothermal channel", `plans/verify/impl_W13.md`.

## 6. Full-revolution sphere Re 250: the steady m = 1 wake (DNS)

    build/dns flow --physics compressible --body sphere --ogrid --ogrid-far 20 \
        --n 64 --ny 48 --nz 32 --re 250 --mach 0.2 --ogrid-beta 3 \
        --steady 1e-3 --steady-steps 300000 --emit stats --emit surface --out ax3r

- 64 meridional × 48 radial × 32 azimuthal. Label `DNS -- steady (local time
  stepping: pseudo-time, not time-accurate) [full revolution 32: odd-mode
  trip 0.05 U]`. The trip is the default; `--ogrid-trip 0` certifies the
  exactly π-symmetric (m = 1-unstable) state instead.
- 89200 iterations. Not timed alone: it is one of the runs of
  `gpu_axisym3d_run` (40.0 s total).
- cd 0.70690, cl −0.0647 (sign from the transient), |cs| < 1e-3 |cl|;
  untripped cd 0.69099 (`--steady 1e-4 --ogrid-trip 0`). Johnson & Patel 1999
  (context): steady planar-symmetric wake, cd ≈ 0.70, cl ≈ 0.062.
- Source: docs/rans.md "The full revolution", `gpu_axisym3d_run`
  (`plans/verify/impl_W14.md` for the TOL 1e-3 cut).

## 7. Two-blade MRF rotor `rotor2` vs OpenFOAM

    build/dns flow --physics compressible --body tests/data/rotor2.stl --robust \
        --n 150 --ny 100 --nz 100 --dr 6 --focal auto --focal-ratio 5 --focal-blend 1 \
        --rotor-mach 0.41 --rotor-axis x --rotor-frame mrf --mach 0.082219 \
        --re 220 --kick 0 --ft 0.217058 --emit stats --out rotor2_L1
    # the suite form (levels, OpenFOAM tiers, report):
    # PARITY_LOCK_HELD=1 python3 tools/parity/suite.py run rotor2 --tier A --solver dns --level 1

- Laminar Re_D 220, J 0.63, tip Mach 0.41, 4.5 D/U. L1: 150×100×100
  (1.50 M cells), focal D/h_fine 39.1, BL 2.6 cells (warned
  under-resolved), 12 deep-solid cells. Label `DNS [MRF frozen rotor: an
  APPROXIMATION]` — exact only for an axisymmetric outer flow.
- `tests/data/rotor2.stl` is byte-identical to the suite's `body.stl`.
  Wall 34.9 s (L1), 112.7 s (L2, ny 132 ratio 7), 259.6 s (L3, ny 164
  ratio 8).
- Reference: rhoSimpleFoam (tier Bc), central U, same STL. CT/CQ below are
  the suite's (`runs/parity/report.md`); the stats file's own `ct`/`cq`
  columns carry the opposite sign.

| | CT | CQ |
|---|---|---|
| dns L1 / L2 / L3 | −0.05223 / −0.03180 / −0.02891 | 0.09167 / 0.08465 / 0.08208 |
| dns Richardson | −0.02756 | 0.07743 |
| OpenFOAM Bc L3 (CQ: Richardson) | −0.01729 | 0.08351 |
| verdict (bands CT 5%, CQ 2%, fixed before measuring) | +59% | −7.3% → NOT AT PARITY |

- The W6 split puts the CT gap on the blade (106–114% of it), not the hub;
  the CQ near-agreement at L3 is a cancellation with a convective flux
  through the moving staircase wall (CQ − CQ_c −8.6% at L3). issues.md 14.
- A 1 D/U pin of L1 (`--ramp 0.4 --ft 0.054512`, 10.9 s) is ctest
  `gpu_mrf_rotor2`.
- Runs: `runs/parity/A/rotor2/dns_L{1,2,3}`, `runs/parity/{B,Bc}/rotor2/`,
  `runs/parity/report.md`, `runs/parity/split/rotor2/`,
  `plans/verify/impl_W7.md`, `plans/verify/impl_W6.md`.

## 8. Axisymmetric SA-neg sphere Re 1e3 vs OpenFOAM (wedge)

    build/dns flow --physics compressible --body sphere --ogrid --nz 1 --n 64 --ny 48 \
        --re 1e3 --mach 0.2 --ogrid-far 20 --ogrid-beta 4.5 --rans sa-neg \
        --steady 1e-4 --steady-steps 200000 --emit stats --emit surface --out sa

- Wedge 64×48 (5°). Label `RANS SA-neg -- steady (local time stepping:
  pseudo-time, not time-accurate) [axisymmetric wedge 5 deg: ensemble mean,
  no swirl, no 3D wake modes]`.
- 136600 iterations, 29.4 s. cd 0.485412, θ_s 99.50°, L_r 1.269 D, μt/μ max
  9.86.
- OpenFOAM SA (ft2 true, central `linear` U, same mesh, 16 s via
  `tools/parity/sph1e4.py`): cd 0.484191 → tier A +0.25%. Tier B (dns
  Mach-extrapolated with an M 0.1 run, 86 s): −1.07%, outside the planned 1%
  gate (issues.md 21).
- Test: `gpu_sa_wedge` (29.5 s) pins the dns state. Runs:
  `$S/impl/W13/wedge/`, `plans/verify/impl_W13.md`.
