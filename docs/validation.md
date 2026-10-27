# Validation suite

Automated tests asserting the solver against external reality (analytic
solutions, published benchmarks) and against the repo's own converged numbers.
Complements the verification tests (parity, convergence order, conservation),
which check the code against itself.

## Tiers and labels

Every validation test carries the ctest label `validation` plus its truth
class(es), and `SKIP_RETURN_CODE 77`:

| label        | truth source                                                    |
|--------------|-----------------------------------------------------------------|
| `analytic`   | closed-form solution or an already-validated analytic pipeline  |
| `literature` | published, cited value with a blockage-justified tolerance band |
| `anchor`     | repo's own converged numbers (`data/validation/anchors.csv`)    |
| `heavy`      | Tier B: registered only with `-DDNS_HEAVY_VALIDATION=ON`        |

- **Tier A** (default `ctest`): the tier the table below describes. In the
  default build only **two** of its rows are actually registered —
  `val_spindown` (the green canary) and `val_comp_cyl_re60`, which RUNS since
  its two anchors were regenerated on 2026-07-29 (it used to exit 77, every
  check being anchor-gated). The other Tier-A rows are quarantined; see below.
- **Tier B** (`cmake -B build -DDNS_HEAVY_VALIDATION=ON`): heavy runs,
  `RUN_SERIAL`, ~23 min GPU total. Most Tier-B rows are quarantined too.
- **Tier C**: manual anchor regeneration + optional studies, documented below;
  never run by ctest.

Anchor-gated checks print `[DEFER] anchor '<key>' missing` while the anchor row
in `anchors.csv` is commented out; a test whose checks are ALL anchor-gated
skips (rc 77). Deferred by decision on 2026-07-17; `cyl_re60_n256_cd` and
`comp_re60_n256_cd` were regenerated on 2026-07-29 (Tier C below), the
remaining three rows are still deferred.

### The `DNS_VALIDATION_BASELINE` quarantine

The nine literature-band tests are **registered only** with
`cmake -B build -DDNS_VALIDATION_BASELINE=ON`:

`val_cyl_re60`, `val_cyl_re100`, `val_tgv3d_64`, `val_comp_mesh_sphere`
(Tier A) and `val_cyl_re200`, `val_cyl_blockage`, `val_tgv3d_128`,
`val_sphere_re300`, `val_mesh_sphere` (Tier B, which also needs
`DNS_HEAVY_VALIDATION=ON`).

This is a **decision, not a failure** (CLAUDE.md): the bands are repo-calibrated
against the confinement argument below and must be RE-MEASURED before being
cited. Do not undo the quarantine, and do not uncomment anchor rows, without
that measurement.

Their `add_executable` targets are built unconditionally even when zero of the
tests are registered. That is deliberate — it keeps the quarantined sources
compiling against the current library instead of rotting.

## The confinement (blockage) calibration argument

All tunnel apps confine the body between FREE-SLIP lateral walls, i.e. symmetry
planes: the run is a periodic ARRAY of bodies with 8D (dr=8) spacing, not an
unbounded flow. Free-slip periodic-image confinement accelerates the bypass
flow and confines the wake far harder than the open-jet/wind-tunnel corrections
(Maskell ~+9% at this area ratio) that literature values are corrected to.
The repo's own measured excesses over unbounded literature at beta = 12.5%:

| quantity            | Re 60      | Re 100     | Re 800          |
|---------------------|------------|------------|-----------------|
| cd vs unbounded     | +26% (1.785 vs 1.42) | +29% (1.730 vs 1.34) | --   |
| St vs unbounded     | +12% (0.152 vs 0.1358) | +20% (0.196 vs 0.164) | +30% (0.257 vs ~0.198) |

The excess grows with Re (stronger wake, same images). Literature bands in
Tier A are therefore one-sided-wide: the unbounded value is a hard FLOOR
(confinement only adds drag and raises St), the ceiling admits the measured
confinement excess plus margin. The tight literature claims live at dr = 16
(beta = 6.25%) in `val_cyl_blockage`, where the images are twice as far away.

## Per-test table

Measured on gfx1100, 2026-07-17. "runtime" = full ctest wall time.

### Tier A (default; total ~190 s GPU)

| test | truth source | expected | asserted | why this tolerance | runtime |
|------|--------------|----------|----------|--------------------|---------|
| `val_spindown` | analytic: spin-down of solid-body rotation in a no-slip circle, u_th = sum c_k J1(lam_k r/R) exp(-nu lam_k^2 t/R^2), c_k = -2 Omega R/(lam_k J0(lam_k)); series self-verified in-test (c_k vs quadrature < 1e-6, IC L2 reproduction 0.05%) | KE(4)/KE(0) = 0.2020; u_th profile | KE ratio +-3%; profile L2 <= 4% | first-order cut wall at R/h = 49 => O(h/R) ~ 2% KE error expected; measured 2.7% / 0.0008. A 20% nu perturbation shifts the truth KE ratio by ~19% => tolerance bites | 6 s |
| `val_cyl_re60` | Williamson (1988) St(60) = 0.1358; Tritton/Henderson cd floor 1.42 | St 0.1521, cd 1.818 | St in [0.130, 0.156]; cd in [1.42, 1.90]; cl_rms > 0.01; >= 5 periods | ceiling +15% per the confinement table above (measured +12%); floor -4% discretization; run-to-run repeat was bit-identical | 87 s |
| `val_cyl_re100` | Williamson (1988) St = 0.164; Henderson cd 1.33-1.35 | St 0.1963, cd 1.730 | St in [0.157, 0.205]; cd in [1.33, 1.82] | ceilings +25%/+35% per the confinement table (measured +20%/+29%, excess grows with Re); floors = unbounded values | 69 s |
| `val_tgv3d_64` | anchor: repo curve `ours_tgv64.csv`; literature: 512^3 spectral reference `spectral_Re1600_512.gdiag` (dissipation peak 0.01286 at t = 8.97) | K/eps curves; peak 0.01182 at t = 8.34 (-8.1%) | curves within 2e-3 / 2e-2; peak time [8.0, 10.0]; value [-40%, +5%] | same binary => near-bitwise vs own curve (measured 6.6e-6 / 1.0e-5; slack for compiler/ROCm drift); 64^3 under-resolves the peak => trend band only, tight claim at 128^3 | 0.3 s |
| `val_ogrid_cyl` (2026-10-27, gfx1100) | (a) Williamson (1988) St(100) 0.164-0.166, 2D DNS cd 1.33-1.40, cl' 0.32-0.34; (b) Dennis & Chang (1970) cd(40) 1.522 | (a) 192x96 `--ogrid-trip 0.05`, window t [82, 121.4] (7 periods, scatter 2e-4): St 0.1679, cd 1.3800, cl' 0.3333; (b) 256x96 `--steady 1e-4`: cd 1.56656 (exit 0, 31200 it), offset-corrected 1.5089 | (a) St in [0.160, 0.172], cd in [1.33, 1.42], cl' in [0.30, 0.36], >= 6 periods, first/last-period cl' within 2%; (b) cd / 1.0382 within 2% of 1.522, cd within 1e-3 of 1.56656 | (a) the masterplan's bands, set before measuring; body-fitted wall, 20 D far row, 15.7 cells in D/sqrt(Re): St +1.2..2.2%, no blockage excess to admit; (b) 1.0382 = the MEASURED M 0.2 / 20 D factor ("O-grid cylinder" below), 2% = the literature spread (Fornberg 1.498); the 1e-3 pin is 25x the run's cd jitter (3.6e-5) | 27 s |
| `val_comp_cyl_re60` | anchor only (comp cd has no clean unbounded literature value at this confinement) | cd 1.9441 (dead steady: flat to 1.5e-5 in the window, repeat run bit-identical); cd_comp - cd_incomp = +0.1255 | cd vs `comp_re60_n256_cd` +-1%; cd_comp - cd_incomp in [0.06, 0.20] | anchors regenerated 2026-07-29, so the test runs; the offset band is the measured value +-50% and is a CROSS-BRANCH check, not an M^2 correction (adjustment 6 below) | 21 s |

### Tier B (`-DDNS_HEAVY_VALIDATION=ON`; total 1362 s GPU)

| test | truth source | measured | asserted | why | runtime |
|------|--------------|----------|----------|-----|---------|
| `val_cyl_re200` | 2D-DNS consensus St 0.196-0.201 (Henderson 1997; experimental wake is 3D at Re 200, so 2D computations are the truth class) | St 0.2237 (20 periods, scatter 0.000), cd 1.6957 | St in [0.183, 0.235]; cd in [1.32, 1.78] | 512^2, D = 64 cells, BL ~ 4.5 cells: honest DNS; ceilings admit the measured confinement excess (St +13%, cd +27%, in line with the calibration table) | 298 s |
| `val_cyl_blockage` | Maskell blockage trend + Williamson St at weak confinement | dr16: cd 1.4730, St 0.1749 (26 periods); dr8: cd 1.7054, St 0.1967 | dr16 cd in [1.33, 1.55]; dr16 St in [0.156, 0.180]; cd(dr16) < cd(dr8); St(dr16) <= 1.01 St(dr8) | halving beta roughly halves the measured excess (cd +29% -> +9%, St +20% -> +6.6%); floors stay at Williamson/Henderson - 5% | 630 s |
| `val_sphere_re300` | Johnson & Patel (JFM 1999): cd 0.656, in-plane |cl| 0.069, St 0.137 | cd 0.7427, cl_y -0.0265, cl_rms 0.0064, St gate not met | cd in [0.657, 0.77]; |cl| in [0.01, 0.10]; St in [0.12, 0.15] gated on cl_rms > 0.01 and >= 5 periods | 160^3, D = 20 cells: coarse BL + 8D blockage => cd +13%; only the y-projection of the planar-symmetric lift is tallied and the wake plane drifts, so the |cl| floor asserts non-axisymmetry only; the shedding oscillation is weak near onset (J&P dCl ~ 0.016), so the St band rarely arms | 360 s |
| `val_tgv3d_128` | as `val_tgv3d_64` | curves K 1.06e-07 / eps 6.28e-07; peak 0.013580 at t = 8.62 (+5.6%) | curves 2e-3/2e-2 vs `ours_tgv128.csv`; peak time 9.0 +- 0.7; value [-10%, +10%] | convergence to the spectral peak is oscillatory (see convergence row above); 512^3 sits at +1.2%, so +-10% brackets the physical claim at 128^3 | 1 s |
| `val_comp_cyl_re800` | anchor only (2D Re800 = model, not reality) | -- | cd 1.71 +-3%, St 0.2445 +-3% vs anchors | SKIPS rc 77 until anchors regenerated | skip |
| `val_mesh_sphere` | analytic-sphere pipeline itself (validated by `val_sphere_re300`): generated unit-sphere STL via `--body sphere.stl` vs the analytic `--body sphere`, matched Re300/n96/dr8 | cd 0.7099 vs 0.6851 (3.6%) | windowed cd (last 3 ft) within 5% | the gap is geometry-pipeline discretization at D = 12 cells (supersampled sdf band vs exact overlap), NOT faceting: subdiv-4 and subdiv-5 STLs give identical cd to 1e-4; aperture qRef differs -0.15% from the exact disk | 73 s |

TGV dissipation-peak convergence (repo curves vs spectral 0.012858 at t = 8.97):
64^3 0.011820 (-8.1%), 128^3 0.013580 (+5.6%), 256^3 0.012298 (-4.3%),
512^3 0.013010 (+1.2%) -- oscillatory, not monotone from below.
| `hit_mini` (extended, CPU, label `slow;validation;literature;les`, not `heavy`) | Kolmogorov constant ~1.5 (Pope 2000), corrected by Pope's model spectrum for this resolution | C = 0.938 (8 samples, 3 + 3 Te; 0.942 at the former 6 + 6) | C = <E(k) k^(5/3) eps^(-2/3)>, k in [4,10], in [0.7, 1.6] | at 64^3 with h = 1.5 eta the band reaches k*eta = 0.26..0.65 (dissipation rolloff): the model spectrum (C = 1.5, beta = 5.2, c_eta = 0.4, f_L for L = k^1.5/eps) predicts a band mean of 0.78; bottleneck lifts the measured value to ~0.94. Band = [0.9x, 2x] model | ~22 s |
| `gpu_hit_mini` (GPU twin through `dns hit`, FP32, labels `les;fast`) | as `hit_mini` | C = 0.947 (24 spectra, 6 + 6 Te, seed 7); eps/eps_f 0.978, u' 0.972, maxDiv 8.5e-6 | the same eps / u' / C bands; maxDiv < 1e-4 (FP32) | same case as the CPU test, the binary's own forcing and averaging; eps_f, T_e recomputed in box units by the test | ~2 s |

### RANS channel vs MKM (a MODEL vs DNS comparison; `rans_channel1d`)

Not a `val_*` test: the comparison runs on the FP64 oracle (the GPU's own
discrete operator, `docs/rans.md`) in milliseconds, and the GPU is pinned to
the oracle (`gpu_rans`) rather than to MKM. Truth: MKM `chan180.means` /
`chan395.means`, ub⁺ by trapezoid over the published profile (15.679 at
Re_τ 178.12, 17.545 at 392.24; the headers carry no Re_b).

| closure | Re_τ 178.12, ny 64 | continuum | Re_τ 392.24, ny 96 | continuum | asserted |
|---|---|---|---|---|---|
| SA-neg | 15.8069 (+0.82%) | 15.8403 (+1.03%) | 17.5596 (+0.08%) | 17.6226 (+0.44%) | within 5% (provisional ceiling); own value pinned 1e-6 |
| SST-2003m | 15.6332 (−0.29%) | 15.1289 (−3.51%) | 17.8881 (+1.96%) | 17.2213 (−1.84%) | same |
| SST-Vm | 15.5986 (−0.51%) | 15.0941 (−3.73%) | 17.8455 (+1.71%) | 17.1785 (−2.09%) | same |

"continuum" = the Richardson limit of ny 192/384/768 (stretch 2,
2026-10-27); THE MODEL-vs-DNS NUMBERS TO CITE. SA-neg is 2nd order and its
test grids are within 0.2–0.4% of the limit. SST is FIRST order in ny (the
TMR wall ω = 60ν/(β1 d1²) moves with the first cell; ladder pinned in
`rans_channel1d`, docs/rans.md "The FP64 oracle"): at the test grids the
wall-ω error (+3.3% at ny 64, +3.9% at ny 96) has the opposite sign to the
model's, so the "SST agrees with MKM to 0.3–2%" reading of the first table
columns is retracted: SST-2003m is −3.5% / −1.8% below MKM, SST-Vm −3.7% /
−2.1%. The 5% band is a ceiling on model error, not a calibration; a DNS
grounder on this solver (docs/channel.md) has not been run.

### RANS channel vs CFL3D (code to code; `rans_channel1d` O5)

SA-neg against NASA TMR's CFL3D SA solution of the 2D fully-developed
channel (2DFDC, file `sa_u+y+KM_cfl3d.dat`), oracle at Re_τ 1e5, ny 1024,
stretch 5: u⁺ at y⁺ 30/100/300/1000 is +0.73..+0.77% raw and
−0.44..−0.48% after removing the file's own sublayer slope 0.988 (u⁺/y⁺ ≠ 1
in its sublayer: a ~0.6% u_τ offset in its wall units); flat (spread 0.04%)
and grid-independent (ny 512 / 2048 within 0.06%). Asserted: normalized ≤
0.75%, raw ≤ 1.2%, spread ≤ 0.1%. The GPU binary equals the oracle there to
2.5e-8 in ub (`gpu_rans_implicit` I5b). Table in docs/rans.md.

The planned `val_rans_channel` (CLI cold start `--steady TOL` → exit 0) is NOT
in ctest: the FP32 steady march cannot reach any TOL that certifies the mean
(the update floor, `docs/rans.md`), so the run is exit 8 by design. It is a
manual check there (ub within 0.06% of the oracle after 3–4e5 iterations).

### Sphere Re 100 on the revolved wedge vs OpenFOAM (`sph100`, identical mesh)

Code-to-code, not literature: OpenFOAM v2606 `simpleFoam` (incompressible,
`linearUpwind` on U -- measured before central U became the parity
reference, see `sph1e4` below; SIMPLE consistent) run ON THE dns MESH — `dns flow
--emit mesh` writes `_mesh.bin`, `tools/parity/dnsmesh.py` turns it into a
`constant/polyMesh` (axis cells become prisms, `checkMesh` OK), and
`tools/parity/sph1e4.py case|run|read|dns` drives and reads both sides. Tier
B: dns cd0 = (4 cd(M 0.1) − cd(M 0.2))/3 against the incompressible solver.
Both dns runs are the explicit LTS march stopped at its plateau (measured
before the jitter-floored stop, exit 8 at the cap; docs/rans.md); OpenFOAM to
`|q − q_final| ≤ 1e-4` settle. Wedge 5°, rFar 20 D, β 3 (gfx1100, 2026-10-26):

| level (nθ × ny) | dns cd M 0.2 | dns cd M 0.1 | dns cd0 | OpenFOAM cd | Δ | θ_s dns M0.1 / OF | L_r dns cd0-extrap / OF |
|---|---|---|---|---|---|---|---|
| L1 64 × 48 | 1.09280 | 1.08709 | 1.08519 | 1.09053 | −0.49% | 126.54 / 127.15° | 0.877 / 0.840 D |
| L2 128 × 96 | 1.09492 | 1.08918 | 1.08727 | 1.08830 | −0.095% | 126.71 / 126.89° | 0.869 / 0.860 D |
| L3 256 × 192 | 1.09543 | 1.08967 | 1.08775 | 1.08789 | −0.013% | 126.74 / 126.83° | 0.867 / 0.865 D |

Gate `sph100` (|ΔCd| ≤ 0.5% at L2): PASS, and |ΔCd| falls monotonically
L1 → L3. θ_s within 0.2° and L_r within 0.3% at L3. The dns compressibility
correction is not small (cd falls 0.5% from M 0.2 to M 0.1), so a tier-A M 0.2
comparison would read +0.7%. Δφ: a 2.5° wedge at L2 moves cd by +5.8e-4
(1.09556). OpenFOAM wall times 10 / 75 / 617 s serial.

### Sphere Re 1e4 RANS on the revolved wedge vs OpenFOAM (`sph1e4`, identical mesh)

Same harness, β 4.5, models per masterplan §8: dns `RANS SA-neg -- steady
(local time stepping: pseudo-time, not time-accurate) [axisymmetric wedge 5
deg: ...]` against OpenFOAM v2606 `simpleFoam` `SpalartAllmaras { ft2 true;
Ct3 1.2; Ct4 0.5; }`, `Prt 0.9` (OpenFOAM's SA is NOT SA-neg: it bounds ν̃
≥ 0 and lacks TMR's ∇ρ term; both vanish in tier B except the bound), and
`kOmegaSST` against `RANS SST-2003m`. OpenFOAM serial (its `exactDistance`
wall distance does not decompose), 3000 iterations, settled to 1e-4 within
245–712. gfx1100 / host CPU, 2026-10-26:

| | dns M 0.2 | dns M 0.1 | dns cd0 | OpenFOAM | Δ (tier B) |
|---|---|---|---|---|---|
| SA L2 128×96 cd | 0.42480 (exit 0, 606000 it, 184 s) | 0.41943 (settled; exit 8 at 2.5e6, 766 s) | 0.41764 | 0.41285 (59 s) | **+1.16%** |
| SA L2 θ_s | 87.40° | 87.64° | | 87.59° | +0.05° (M 0.1) |
| SA L2 L_r / D | 1.2206 | 1.2065 | 1.2018 | 1.2566 | −4.4% |
| SA L1 64×48 cd | 0.4197 ± 1.1% (no steady state) | 0.4264 ± 0.8% | (0.4286) | 0.39282 (11 s) | (+9%, not a converged comparison) |
| SST L2 cd | 0.4088–0.4136 (limit cycle) | -- | -- | 0.39496 (70 s) | -- |
| SST L1 cd | diverged (exit 7) | -- | -- | 0.39964 (15 s) | -- |

- Tier B at L2 is +1.16% (the plan's tolerance max(1%, ½ · OpenFOAM's L1 →
  L2 change of 5.1%) = 2.6%: PASS at L2; the "non-increasing L1 → L3" clause
  cannot be read: dns L1 has no steady state and L3 was not run). θ_s
  within 0.05°. L_r −4.4% (the ±3% tolerance is set at L3).
- The dns Mach correction is −1.3% from M 0.2 to M 0.1 (tier A at M 0.2
  would read +2.9%).
- OpenFOAM `ft2 false` at L2: 0.40885, −0.97% from `ft2 true` -- above the
  U4 trigger (0.5%): a `sa-noft2` dns variant is a user decision, not built.
- dns time to solution, SA L2 M 0.2: 79 s to cd within 1e-3 of its plateau,
  184 s to the certified stop (TOL 1e-4, explicit LTS, 0.30 ms/iteration).
  OpenFOAM: 59 s serial for 3000 iterations, settled to 1e-4 in 413.
- SST and L1: no dns steady state (docs/rans.md, issues.md 1-2).

**The reference depends on OpenFOAM's U scheme** (measured 2026-10-26,
`sph1e4.py case ... --u-scheme linear`: central `Gauss linear` on U, the
scheme that matches dns's dissipation-free KG; everything else identical).
OpenFOAM SIMPLE converges with it on every case below (settled to 1e-4 in
555–911 iterations):

| case (identical mesh) | OpenFOAM linearUpwind cd / θ_s / L_r | OpenFOAM central cd / θ_s / L_r | dns |
|---|---|---|---|
| SA L2 Re 1e4 | 0.41285 / 87.59° / 1.2566 D | 0.41911 / 87.82° / 1.1576 D | cd0 0.41764: **−0.35%** (vs +1.16% against linearUpwind); θ_s 87.64° (M 0.1); L_r 1.2018 (+3.8%) |
| SST-2003m L2 Re 1e4 | 0.39496 / 86.69° / 1.4005 D | 0.40589 / 87.14° / 1.2014 D | no steady state; URANS M 0.2 time mean 0.4115 (t 8–14.6 D_box/U, band 0.404–0.424), θ_s 86.82° / L_r 1.264 D at the final instant |
| SST-2003m L1 Re 1e4 | 0.39964 / 87.18° / 1.2578 D | 0.43000 / 87.53° / 0.9095 D | no steady state (closure-independent at L1, issues.md 1) |
| SST-2003m L1 Re 3e3 | -- | 0.43739 / 93.54° / 0.9530 D | M 0.2 0.44261 at the 2e5 cap (remaining 1.7e-4), θ_s 93.00°, L_r 1.0139 D |
| SST-2003m L1 Re 1e3 | 0.47129 / 100.02° / 1.4890 D | 0.48872 / 100.77° / 1.0674 D | **cd0 0.49070: +0.41%** (M 0.2 0.49851 exit 0 in 114600 it / 28 s; M 0.1 0.49265 settled, exit 8 by the M 0.1 floor); θ_s 100.30° / 100.51°; L_r 1.1656 / 1.1498 D (+7.7% at M 0.1) |
| SA L1 Re 1e3 (2026-10-27) | -- | 0.48419 / 100.24° / 1.1407 D (3000 it, settled 1e-4 in 327, 13 s) | **cd0 0.47901: −1.07%** (M 0.2 0.48541 exit 0 in 136600 it / 29 s; M 0.1 0.48061 settled, exit 8 by the M 0.1 floor); θ_s 99.50° / 99.68°; L_r 1.2691 / 1.2391 D (cd0-extrap 1.2291: +7.8%) |

The SST-2003m L1 Re 1e3 row is the small gate `gpu_sst` (b); the SA row is
`gpu_sa_wedge`, which pins the dns state and PRINTS the OpenFOAM comparison
without gating it: −1.07% is outside the 1% the masterplan (W13) set for that
gate before measuring (issues.md 21). The
"OpenFOAM SST near-wake k 2.3× below dns" seen against linearUpwind is the
scheme: with central U the two agree within 10–30% near the body. A parity
number names the U scheme from now on; the SA row's +1.16% above stands
for the linearUpwind reference it was measured against.

**Decisions (2026-10-26).** The gate reference is OpenFOAM with CENTRAL U
(`sph1e4.py` default `--u-scheme linear`); linearUpwind is kept only to
reproduce the rows measured against it. SA L2 tier B: −0.35%, PASS.
SST-2003m at Re 1e4 L2 has no stable steady state (the URANS is unsteady in
the same band as the pseudo-time cycle), so it is compared on the URANS TIME
MEAN, labelled `RANS SST-2003m -- URANS (time-accurate)`: tier A at M 0.2
0.4115 vs OpenFOAM central 0.40589 = +1.4% (the M 0.1 URANS leg for tier B,
~22 min, not run). The coarse wedges' exit 8 (issues.md 1) is accepted as
the honest outcome: the only cure measured is dissipation.

### O-grid cylinder: literature (`val_ogrid_cyl`) and the Re 40 offset (W12)

`dns flow --physics compressible --body cylinder --ogrid --mach 0.2`, far row
20 D (default β 3), gfx1100, 2026-10-27. **Re 100** (192×96, h1 = D/157,
15.7 cells in D/√Re, `--ogrid-trip 0.05`, the run to t 121.4 box = 155 D/U
in 21 s): St 0.1679, cd 1.3800, cl' 0.3333 over t ∈ [82, 121.4] (7 periods,
period scatter 2e-4); against Williamson's St 0.164-0.166 that is +1.2..2.2%,
inside the masterplan's bands (set before measuring). The envelope is within
0.5% of its limit from t ≈ 80 (issues.md 16: 99% at 71.6).

**Re 40 steady** (explicit `--steady 1e-4`, every run exit 0), the offset
the test removes before reading Dennis & Chang:

| study | runs | cd | reading |
|---|---|---|---|
| grid, 20 D, M 0.2 | 192×72 / **256×96** / 384×144 | 1.5663 / **1.56656** / 1.5668 | converged (0.03%) |
| Mach, 256×96, 20 D | M 0.2 / 0.1 / 0.05 | 1.5666 / 1.5580 / 1.5579 | M 0.2 is +0.55% over M → 0 |
| far row, M 0.2, h1 matched (≈ 5e-3 box) | 20 D / 40 D (ny 124, β 3.25) / 80 D (ny 150, β 3.55) | 1.5666 / 1.5354 / 1.5240 | Richardson order 1.45, R → ∞ 1.5174 |

So cd(20 D, M 0.2) = 1.0382 × cd(R → ∞, M → 0) = 1.0382 × 1.5090, and
1.5090 is −0.86% from Dennis & Chang 1.522 (Fornberg 1.498: +0.7%). The
20 D far row is a +3.3% blockage on cd at Re 40, the largest of the three
offsets; the test gates the corrected value at 2% (the literature spread) and
pins the raw 1.56656 at 1e-3. Runs: 2-9 s each.

### O-grid cylinder URANS shedding vs OpenFOAM (`ocyl1e4`, identical mesh, W12)

Code to code, same method (tier A): dns `flow --physics compressible --body
cylinder --ogrid --n 256 --ny 96 --ogrid-beta 4 --ogrid-far 20 --re 1e4
--mach 0.2 --ogrid-trip 0.05 --rans sst-2003m|sa-neg --compensated --ft 1.6`
(66.5 D/U including the ramp; labels `RANS SST-2003m -- URANS
(time-accurate) [O-grid trip 0.05 U]`, `RANS SA-neg -- URANS ...`), against
OpenFOAM v2606 `rhoPimpleFoam` `kOmegaSST` / `SpalartAllmaras { ft2 true }`
(Prt 0.9 both, TMR inflow k = 2.25e-7 U², μt/μ 0.009, ν̃ = 3ν; wall ω =
60ν/(β1 d1²) from dns's d1) ON THE dns MESH (`tools/parity/dnsmesh.py`'s 2D
O-grid layout, `checkMesh` OK, non-orthogonality 1.7e-6°). OpenFOAM: backward
in time, central `Gauss linear` U, maxCo 0.5, PIMPLE nOuter 2, 6 MPI ranks
(wall distance meshWave: exactDistance fails in parallel), impulsive start
from the freestream plus dns's trip vortex at the same cell centres;
`tools/parity/ocyl1e4.py case|run|read|dns`. Both read by the same
`stats()`: the window starts at the first period after which the per-period
cl swing changes < 2% between consecutive periods for good (≥ 4 periods, else
"not saturated"); values are batch means over whole periods, SE = std /
√n. y⁺ max 0.93-1.12 (dns), 0 SST guard hits; OpenFOAM bounds k ≥ 0 every
step (min 0) and ν̃ ≥ 0 (SA-neg does not).

| | dns window (periods) | OpenFOAM window | cd | cl_rms | St | cl swing |
|---|---|---|---|---|---|---|
| SST-2003m dns | 27.7-62.3 D/U (8) | | 1.6985 ± 0.0069 | 1.2164 ± 0.0084 | 0.2314 ± 0.0019 | 1.7166 ± 0.0078 |
| kOmegaSST OpenFOAM | | 45.7-63.5 (4) | 1.6945 ± 0.0130 | 1.2084 ± 0.0020 | 0.2249 ± 0.0035 | 1.7048 ± 0.0025 |
| Δ (SE) | | | +0.24% (+0.27) | +0.66% (+0.91) | +2.92% (+1.64) | +0.69% (+1.44) |
| SA-neg dns | 45.2-66.0 (5) | | 1.7383 ± 0.0054 | 1.3581 ± 0.0047 | 0.2406 ± 0.0003 | 1.9013 ± 0.0063 |
| SpalartAllmaras OpenFOAM | | 49.6-66.4 (4) | 1.7372 ± 0.0043 | 1.3626 ± 0.0028 | 0.2391 ± 0.0015 | 1.8936 ± 0.0058 |
| Δ (SE) | | | +0.06% (+0.16) | −0.33% (−0.82) | +0.63% (+1.00) | +0.40% (+0.89) |

**On the windows of the rule, no disagreement exceeds 3 SE.** The largest is
the SST St (+1.6 SE), and it is window-sensitive: the dns SST cycle
overshoots (swing 1.757 at 32-36 D/U) and then lengthens its period (4.18 →
4.39 D/U), which the 2% swing rule admits. Over a COMMON late window (t ≥ 45
D/U; dns 3 periods, OpenFOAM 4) the SST row reads dns cd 1.6882 ± 0.0014,
cl_rms 1.1994 ± 0.0008, St 0.2276 ± 0.0008 against the OpenFOAM values above:
−0.37% / **−0.74%** / +1.2% (−0.5 / **−4.2** / +0.7 SE). That cl_rms gap is
small but resolved by the late window's tight SEs, so it is OPEN (issues.md
24), not banded away; SA is unchanged (its window already starts at 45). The SE are batch means of 4-8 correlated
periods: an estimate, not a confidence interval. Saturation takes 28 D/U
(dns SST), 45 (dns SA), 46-50 (OpenFOAM, from its impulsive start): the
plan's 40 D/U would have held 3 saturated SST periods (the first 40-D/U run
read "not saturated", swing still 1.738 → 1.757 → 1.728).

**OpenFOAM settings matter more than the closure here** (measured: restarts of
the SST case at 30 D/U, 1.5 D/U each, `ocyl1e4.py` docstring). PIMPLE
nOuter 1 at maxCo 0.5 rings a 10-step cd sawtooth (HF rms 0.34 against a
mean 1.19) that biases the window-mean cd by +5% (1.192 vs 1.133); maxCo
0.25 (0.22) or CrankNicolson 0.9 (0.28) do not cure it, nOuter 2 or 3 do
(0.063, cd 1.133, identical) and Euler does (0.011) at first order. The
first serial nOuter-1 run (32 D/U in 285 s) is discarded. Cost: OpenFOAM 6
ranks 195 s (SST) / 176 s (SA) for 66.5 D/U, 24400 steps at maxCo 0.5
(dt ≈ 2.7e-3 D/U); dns 191 s / 195 s for 7.8e5 acoustic steps (0.244 /
0.220 ms/step, compensated) -- at `--cfl 0.8` (below) 96 s.

**URANS dt and cost on this case** (SST-2003m, compensated, same argv):

| run | steps | ms/step | wall | cd | cl_rms | St | paired max \|Δcd\| / \|Δcl\| vs cfl 0.4 |
|---|---|---|---|---|---|---|---|
| cfl 0.4 (default) | 781682 | 0.244 | 191 s | 1.6985 ± 0.0069 | 1.2164 ± 0.0084 | 0.2314 ± 0.0019 | -- |
| cfl 0.6 | 521122 | 0.245 | 128 s | 1.6986 ± 0.0069 | 1.2174 ± 0.0085 | 0.2314 ± 0.0019 | 3.9e-4 / 1.3e-3 |
| cfl 0.8 | 390842 | 0.245 | 96 s | 1.6989 ± 0.0068 | 1.2190 ± 0.0083 | 0.2314 ± 0.0019 | 1.2e-3 / 3.5e-3 |
| cfl 0.4, `--dt-recompute 1` | 781682 | 0.313 | 245 s | 1.6985 ± 0.0069 | 1.2164 ± 0.0084 | 0.2314 ± 0.0019 | 3.5e-4 / 1.4e-3 |
| cfl 0.4, plain state | 781729 | 0.217 | 170 s | 1.6952 ± 0.0037 (window 40.3-62.1, 5) | 1.2016 ± 0.0025 | 0.2286 ± 0.0005 | trajectory decorrelates: rms Δcl 0.17 over the window |

The means move by ≤ 0.2% of cl_rms and 0.02% of cd from cfl 0.4 to 0.8, far
inside the SE, and the paired trajectories stay within 3.5e-3 in cl (a
swing of 1.7). **Recommended URANS `--cfl 0.8` on the O-grid cylinder**
(half the cost); the default stays 0.4 (a default change is a user
decision). `--dt-recompute 16` IS already the default on the compressible
cylinder O-grid (the plan's probe "at 16" is the default run); its opt-out
`--dt-recompute 1` costs 1.28× per step and moves nothing beyond the paired
1.4e-3. The plain FP32 state (issues.md 20) decorrelates from the
compensated trajectory during the growth phase; over the common late window
(t ≥ 45) it reads cd 1.6924 ± 0.0017 / cl_rms 1.2012 ± 0.0018 / St 0.2287 ±
0.0009 against compensated 1.6882 / 1.1994 / 0.2276 (+0.25% / +0.15% /
+0.5%, ≤ 1.9 SE): on this statistic the floor is below the window SE.

### The full revolution vs the wedge (P4, `docs/rans.md` "The full revolution")

Code-to-itself: the full-revolution O-grid (`--nz N`) against the 5° wedge
at the SAME meridional mesh, both the explicit LTS march stopped by the
jitter-floored rule, M 0.2, rFar 20 D (gfx1100, 2026-10-26). In exact
arithmetic an axisymmetric 3D state IS the wedge of Δφ = 2π/N (`gpu_axisym3d`
(c): N 16 vs a 22.5° wedge agree to 4e-5 in cd, 1.4e-5 per plane), so the
N ladder is the wedge's own Δφ convergence -- the polygonal body's frontal
area is sin(Δφ)/Δφ of the circle's -- and G4.2 (|Δcd| ≤ 0.5%) is a statement
about which N is fine enough.

**G4.2, laminar Re 100, β 3, TOL 1e-4** (every run exit 0; cd relative to
the wedge at the same (nθ, ny)):

| mesh | wedge 5° | N 16 | N 32 | N 64 | N 128 |
|---|---|---|---|---|---|
| L1 64×48 cd | 1.09281 | 1.07635 (−1.51%) | 1.08930 (−0.32%) | 1.09257 (−0.022%) | 1.09339 (+0.053%) |
| L1 θ_s / L_r | 126.27° / 0.898 D | 127.65° / 0.855 | 126.56° / 0.889 | 126.29° / 0.898 | 126.22° / 0.900 |
| L1 iterations / wall | 57400 / 6.8 s | 25400 / 4.2 s | 22200 / 4.2 s | 47600 / 13 s | 139000 / 84 s |
| L2 128×96 cd | 1.09493 | | 1.09143 (−0.32%) | 1.09470 (−0.021%) | |
| L2 θ_s / L_r | 126.43° / 0.890 D | | 126.72° / 0.881 | 126.45° / 0.890 | |
| L2 iterations / wall | 75800 / 41 s | | 37200 / 69 s | 62000 / 131 s | |

- **G4.2 PASS** at L1 and L2 for N ≥ 32 (−0.32%); at N 64 the 3D cd is
  within 0.022% of the wedge, θ_s within 0.03°, L_r within 0.1%.
- **nφ convergence is second order in Δφ**: L1 successive differences
  +0.01295, +0.00327, +0.00082 (ratios 3.96, 3.99). Richardson from N 32 / 64
  at L2: cd(Δφ → 0) = 1.09579, against the wedge's own 5° / 2.5° Richardson
  1.09577 (2.5° wedge 1.09556) -- the two limits agree to 2e-5.
- The N 16 body loses 2.55% of its frontal area and 1.51% of its drag: cd is
  not proportional to the polygon's area.
- Cost: 0.118 ms/iteration on the wedge (launch-bound), 1.65 µs per thousand
  cells and iteration in 3D from 2e5 cells up (L1 N 64 0.270 ms, L2 N 64
  1.35 ms at 8.0e5 cells). The stop certifies after a number of iterations
  set by the monitors' jitter (W, the limiting series: averaging N planes
  lowers it, so N 32 stops before the wedge).

**Small RANS and the m = 1 mode, L1 64×48, N 32** (β 4.5 for RANS, β 3
laminar; TOL 1e-4, every run exit 0, 0 guard hits; "trip" = `--ogrid-trip
0.05`, the default since 2026-10-26; "3D N 32" = `--ogrid-trip 0`; measured
before the stop's lateral observables -- with them the tripped SA-neg run
certifies at 128400, the SST-2003m one at 146600, the Re 250 one at 108800,
same values):

| case | wedge 5° | 3D N 32 | 3D N 32, trip |
|---|---|---|---|
| RANS SA-neg Re 1e3 cd / cl | 0.48541 / 0 | 0.48229 (−0.64%) / 0 | **0.49196 / −0.0585** (steady m = 1) |
| — θ_s / L_r | 99.50° / 1.2691 D | 99.63° / 1.2650 | 98.66–101.00° / (off-axis) |
| — iterations, ms/it | 136600, 0.215 | 119200, 0.393 | 116800, 0.395 |
| RANS SST-2003m Re 1e3 cd / cl | 0.49851 / 0 | 0.49511 (−0.68%) / 0 | 0.49511 / 1.2e-5 (decays) |
| — θ_s / L_r | 100.30° / 1.1656 D | 100.45° / 1.1623 | 100.45° / 1.1623 |
| — iterations, ms/it | 114600, 0.239 | 95200, 0.463 | 95000, 0.466 |
| laminar Re 250 cd / cl | 0.69343 / 0 | 0.69099 (−0.35%) / 0 | **0.70690 / −0.0647** (steady m = 1) |
| — θ_s / L_r | 112.31° / 1.5644 D | 112.55° / 1.5546 | 110.83–114.36° / 1.3825 |

- At N 64: SA-neg Re 1e3 untripped 0.48518 (−0.047% of the wedge; L_r
  1.2690 D, θ_s 99.51°), tripped 0.49557 with cl +0.0598 (N 32: −0.0585 --
  the sign of the lift is the transient's choice, its plane the trip's);
  laminar Re 250 tripped 0.70999, cl −0.0652 (N 32: 0.70690 / −0.0647).
- The untripped 3D state is the wedge's (to the polygon), for every closure:
  it is exactly π-symmetric and so cannot leave axisymmetry through m = 1.
- Tripped, the laminar Re 250 sphere takes the steady planar-symmetric wake
  of Johnson & Patel 1999 (context only: cd ≈ 0.70, cl ≈ 0.062 there;
  here 0.707 / 0.065 on L1 × 32): the 3D solver reproduces the first
  bifurcation of the sphere wake, which no wedge can.
- RANS SA-neg at Re 1e3 is m = 1-unstable on this mesh (it takes the same
  kind of asymmetric steady wake, cl −0.0585), SST-2003m is stable. A
  symmetric SA-neg 3D number at Re 1e3 is therefore an UNSTABLE fixed point
  the exact symmetry protects -- report the tripped one.

**G4.3, RANS SA-neg Re 1e4 on G1** (128×96×96, 1.2M cells, β 4.5,
explicit LTS, 10-minute cap, 3.41 ms/iteration):

| start | trip | outcome | cd | lateral force |
|---|---|---|---|---|
| cold | 0.05 | exit 8 at 170000 / 581 s | 0.42519 ± 1.2e-4 (+0.09%) | \|cl\| 1e-3 → 1.5e-2 (5e4 → 1.6e5 it), growing |
| the certified 5° wedge (`--init-from`) | 0 | **exit 0 at 38800 / 135 s** | 0.42520 (+0.094%) | 0 (π-symmetric) |
| the certified 5° wedge | 0.05 | exit 8 at 170000 / 584 s | 0.42520 | ±1e-3 → 8.7e-3 at 1.7e5, growing |

θ_s 87.38–87.42° (wedge 87.40°), L_r 1.2214 D (1.2206), y⁺max 0.33, 0 hits
on every row. The wedge start puts the certified number inside the
5-minute target (323 s with the wedge's own 188 s), but the tripped runs
show that number is an m = 1-UNSTABLE axisymmetric state: G4.3 has no
certified stable 3D steady state. cd is blind to it (cd is even in m = 1);
the full revolution's stop now carries `cd+cl` / `cd+cs` so the tripped run
is exit 8, not a certificate (docs/rans.md "The full revolution",
issues.md 11). Reported as found, not damped.

### LES known answers (`gpu_les_smag` (8), `gpu_les_tgv_comp`; masterplan W10)

Measured 2026-10-27, gfx1100. Labels: LES smag Cs 0.17 / LES sigma Cσ 1.35;
the TGV rows at 32³ are under-resolved by design (that is the regime an SGS
model exists for), so they are model behaviour, not a converged LES.

**Model magnitude** (`gpu_les_smag`, < 1 s). On linear fields the FP32
central gradient is exact, so νt is pinned to its closed form. The code's
convention, now test-pinned: `|S| = sqrt(2 S_ij S_ij)` (u = a·y gives
|S| = a, not a/√2), `Δ² = (hx hy hz)^(2/3)` (hx·hy at nz == 1), and on a
curvilinear mesh the cell's own `vol^(2/3)` (`vol/hz` in 2D); the
compressible slot holds μt = ρ·νt.

| case | grid | worst error |
|---|---|---|
| incomp smag, u = 3y | 32² ; 16×32×8 (anisotropic) | 0.02 ulp |
| incomp sigma, fixed rank-3 G (D_σ = 0.017803) | 16×32×8 | 1.10 ulp vs FP64 `sigmaDsigma` |
| comp smag, checkerboard ρ ∈ {1, 2} | 32² ; 16×32×8 | 0.98 ulp |
| comp sigma, same G | 16×32×8 | 0.10 ulp vs FP64 |
| comp focal smag (x[/z] ratio 3, vol ratio 3 / 9) | 32×16×1 ; 32×16×32 | 1.97 / 1.92 ulp |
| comp focal sigma (on the FP64 discrete gradient) | 32×16×32 | 37 ulp = 2.7e-6 rel (band 3e-5) |

Bands: 4 ulp; focal sigma 3e-5 relative, because the mapped gradient is
formed in FP32 and D_σ amplifies its rounding where σ2 ≈ σ3. A host FP32
replica of `sigmaDsigma` is 13 ulp off the device (contraction and rounding
order) while the device sits within 0.1–1.1 ulp of FP64, so FP64 is the
reference. The Smagorinsky ε_sgs/ε_visc band of the original plan
([1.6, 2.4]) encoded one |S| convention and was replaced by these pins.

**Compressible TGV** (`gpu_les_tgv_comp`, ~2 s): M 0.1, Re 1600, 32³,
cfl 0.4, t = 12, against the incompressible 512³ DNS
(`data/validation/tgv3d_512.csv`, peak −dK/dt 0.012994 at t 9.03).

| run | steps | peak −dK/dt | t_peak | vs 512³ (band ±3 pp) | max\|K − K512\| | K(12) |
|---|---|---|---|---|---|---|
| DNS | 1715 | 0.009380 | 10.74 | −27.8% | 0.0146 | 0.0685 |
| LES smag 0.17 | 1680 | 0.011813 | 6.42 | −9.1% | 0.0250 | 0.0415 |
| LES sigma 1.35 | 1680 | 0.011632 | 6.86 | −10.5% | 0.0229 | 0.0432 |

- σ μt ≡ 0.0 at t = 0 (exact: w = 0 makes det g = 0 in FP32).
- Smagorinsky ε_sgs/ε_visc(0) = 1.97759, device vs host FP64 closed form
  3.9e-8 relative (band 1e-3). ε = Σ m |S_dev|², m = μ or μt.
- **Not true, not pinned:** "σ stays quiet for t < 3" — the TGV is 3D from
  the first step (w ∝ t) and (Cσ h)²/ν = 112 at 32³, so σ's ε_sgs/ε_visc is
  1.35 at t 0.5, 2.01 at t 1, 2.60 at t 3 (smag 1.99 / 2.07 / 2.91); both
  peak at t 5.4 (smag 4.52, σ 3.94). Also not true: "σ's K is closer to 512³
  than DNS's" — both models over-dissipate early and peak ~2.5 time units
  before the reference; DNS under-dissipates and peaks late, so its
  max|K − K512| is the smallest. The models fix the peak VALUE (−28% → −9 to
  −11%), not the curve.
- The incompressible 32³ DNS peaks 30.6% low (LES audit, `dns tgv --n 32`);
  the compressible M 0.1 DNS 27.8% low.

**LES dt bound** (same test, every sample of every LES run): the LES
`stableDt` equals the `stableDt` of a DNS DeviceComp on the same state
exactly (121/121 samples, uniform smag and σ, and a ratio-3 focal box with
smag), and the diffusive bound clears the dt taken by ≥ 102× (smag),
79× (σ), 36× (focal). See issues.md "LES known answers".

**LES dt bound cost (W15, `gpu_comp_les`).** Since 2026-10-27 the bound reads
the max of the solver's own FP32 μt of the state (`updateSgsNut`, images
included). Before, an FP64 functor re-derived the model in the reduction. On a
32³ 3D TGV state (w and ρ modulated), the FP32 max agrees with the host FP64
evaluation of the deleted functor to 5.7e-8 (smag 0.17), 4.3e-8 (σ 1.35),
1.05e-7 (smag 5) and 3.2e-8 (σ 40) relative. At the non-binding Cs the LES
dt equals the DNS twin's exactly. At Cs 5 / 40 it binds and equals the
formula on the FP32 max to 1e-14; the FP64-functor dt differs by that
rounding. Every LES run byte-compared is identical: 128³ sphere smag and σ,
open-box smag, ratio-2 focal σ, and channel smag and σ. Step cost on the 128³
sphere, Re 1e4, NSCBC, `--ft 0.05`, gfx1100, in ms/step:

| run | before | after |
|---|---|---|
| DNS | 4.26 | 4.24 |
| smag 0.17 | 5.83 | 4.74 |
| σ 1.35 | 7.12 | 4.93 |
| focal ratio 2 smag | 7.43 | 6.06 |
| focal ratio 2 σ | 8.71 | 6.22 |

The plan's targets were ≤ 4.5 (smag) and ≤ 4.8 (σ); both are missed (issues.md
item 26).

## Tolerance adjustments vs the original plan

Recorded per the rule "widen only with written justification":

1. `val_cyl_re60` St ceiling 0.150 -> 0.156 (+10% -> +15% over Williamson).
   Measured 0.1521 (+12%), period scatter 1.5%, repeat run bit-identical: a
   clean, reproducible signal. Cause is physics, not discretization (repo grid
   study: cd changes only -1% from 256^2 to 512^2): free-slip periodic-image
   confinement, consistent with the +26% cd excess at the same Re and the
   +30% St excess at Re 800. See the calibration table above.
2. `val_cyl_re100` St ceiling 0.177 -> 0.205, cd ceiling 1.70 -> 1.82.
   Measured St 0.1963 (+20%), cd 1.730 (+29%); the plan's ceiling assumed the
   Re60 excess, but the excess grows with Re (table above). Floors unchanged:
   unbounded literature remains the hard lower bound.
3. `val_cyl_blockage` dr16 St band 0.164 +-5% -> [0.156, 0.180]. Measured
   dr16 St 0.1749 (+6.6%): halving beta roughly halves the St excess (+20% ->
   +6.6%), still outside +-5%. The band keeps the Williamson floor and admits
   the measured residual confinement + margin; the direction checks
   (cd(dr16) < cd(dr8), St(dr16) <= St(dr8)) carry the Maskell-trend claim.
3b. `val_cyl_re200` St ceiling 0.212 -> 0.235, cd ceiling 1.70 -> 1.78.
   Measured St 0.2237 (+13% over the 2D consensus, scatter 0.000 over 20
   periods) and cd 1.6957 (+27%): the same confinement excess as Re60/Re100
   (calibration table); the old cd ceiling left 0.3% margin over a
   deterministic value.
3c. `val_tgv3d_128` peak band [-20%, +5%] -> [-10%, +10%]. The plan assumed
   monotone under-prediction; the repo's own curves show oscillatory
   convergence with 128^3 at +5.6% and 512^3 at +1.2% (table above). The run
   reproduces the anchor curve to 6e-7, so the overshoot is the scheme, not a
   regression.
3d. `val_sphere_re300` |cl| floor 0.03 -> 0.01, St gate cl_rms 0.005 -> 0.01
   with >= 5 periods. The app tallies only the y-projection of the lift and
   the wake-plane orientation drifts (measured cl_y = -0.0265 vs J&P in-plane
   0.069); the floor now asserts exactly what is invariant to orientation:
   a non-axisymmetric wake. The near-onset oscillation (measured rms 0.0064,
   J&P dCl ~ 0.016) is too weak for a reliable zero-crossing St, so the St
   band arms only on an established signal.
3e. `val_mesh_sphere` cd agreement 3% -> 5%. Measured 3.6%; a subdiv-5 STL
   (4x finer faceting) reproduces the subdiv-4 cd to 1e-4, isolating the gap
   to the supersampled-sdf cut vs exact-overlap geometry at D = 12 cells
   (O(h/D) ~ 8% first-order walls); aperture qRef accounts for only -0.15%.
4. Spin-down tolerances NOT widened: measured KE 2.7% (< 3%), profile L2
   0.0008 (< 4%).
5. `hit_mini` compensated-spectrum band [1.0, 2.4] -> [0.7, 1.6]. First run
   measured C = 0.976 (4 samples; 0.942 with 8): NOT a bug -- the planned
   floor assumed an inertial range, but at 64^3 the k in [4,10] band reaches
   k*eta = 0.65, and Pope's model spectrum predicts a band mean of 0.78 at
   this resolution. Recentred on the model prediction ([0.9x, 2x]); a
   factor-2 normalization error still fails.
6. `val_comp_cyl_re60` cd_comp - cd_incomp band [0.005, 0.06] -> [0.06, 0.20],
   and the claim it carries changed. The old band asserted a Mach-squared
   correction: "small, strictly positive". Both halves are refuted by the
   2026-07-29 regeneration (all `--far sponge --n 256 --re 60 --ft 20`,
   trapezoid cd; the incompressible reference is the `--far tunnel` cut-cell
   run of `val_cyl_re60`, cd 1.8187):

   | M    | cd     |
   |------|--------|
   | 0.20 | 1.9441 |
   | 0.10 | 1.9666 |
   | 0.05 | 2.0016 |

   cd FALLS with Mach, so the compressible branch's own M -> 0 limit sits
   FURTHER above the incompressible 1.8187, not closer — a Mach correction
   cannot be what separates them. Nor is it an unresolved wall converging
   away: at n 512 (D = 64 cells) the compressible run gives 1.9659 while the
   incompressible branch drops ~1% from 256^2 to 512^2, so the gap widens.
   What is left is what the two branches do NOT share — the ghost-mirror wall
   vs the cut-cell wall, and sponge vs free-slip-tunnel laterals. That
   attribution is NOT isolated (no run holds one of the two fixed), so the
   check makes no causal claim: the band pins the two branches to each other,
   which is the one thing neither single-branch anchor can catch, at the
   measured value +-50%.

## Tier C — manual procedures

### Anchor regeneration

Anchors live in `data/validation/anchors.csv`
(`name,value,tol_rel,config,date,note`); pending rows are commented out.
Regenerate ONLY with an intended physics change, then record the change in
CLAUDE.md. For each key, run the exact `config` command from the CSV row from
the repo root, e.g.:

```sh
build/dns flow --body cylinder --far tunnel --n 256  --nz 1 --re 60  --dr 8 --ft 12 --fdt 1e9 --out /tmp/a60
build/dns flow --body cylinder --far tunnel --n 512  --nz 1 --re 60  --dr 8 --ft 12 --fdt 1e9 --out /tmp/a60b
build/dns flow --body cylinder --far tunnel --n 1024 --nz 1 --re 60  --dr 8 --ft 12 --fdt 1e9 --out /tmp/a60c
build/dns flow --body cylinder --far tunnel --n 512  --nz 8 --re 60  --dr 8 --ft 12 --fdt 1e9 --out /tmp/a60z
build/dns flow --body cube     --far tunnel --n 256  --nz 1 --re 300        --ft  6 --fdt 1e9 --out /tmp/cube
build/dns flow --body sphere   --far tunnel --n 160  --re 300 --dr 8 --ft 12 --fdt 1e9 --out /tmp/s300
build/dns flow --body sphere.stl --far tunnel --n 96 --re 300 --dr 8 --ft 12 --fdt 1e9 --out /tmp/ms
build/dns flow --physics compressible --body cylinder --far sponge --n 256 --re 60  --mach 0.2 --ft 20 --fdt 1e9 --out /tmp/c60
build/dns flow --physics compressible --body cylinder --far sponge --n 512 --re 800 --mach 0.2 --ft 15 --fdt 1e9 --out /tmp/c800
build/dns flow --physics compressible --body sphere.stl --far nscbc --open --n 96 --nz 96 --re 100 --mach 0.2 --ft 12 --out /tmp/cms
```

The cube line was `--nz 8` while `--body cube` meant a 45°-rotated square prism
spanning the full depth — a quasi-2D diamond cylinder with eight spanwise cells,
which by the 2D invariant is the `--nz 1` case it now names. A cube is z-finite
as of 2026-07-26 (side `D/√2`, `dns::sceneCube`), so at `--nz 8` that same
command would place a body less than one cell tall. Nothing measured is
invalidated — the cube has no published anchor — but a 3D cube run is a NEW
case and needs `nz` chosen so the side spans cells, not the old `--nz 8`.

The two `--far sponge` lines are NOT an endorsement of sponge: they name the
boundary condition the existing anchor values were measured under, explicitly,
because an anchor must be reproduced under the conditions that produced it.
`--far sponge` remains an escape hatch, never a default (CLAUDE.md).

then compute the statistic named in the row from `<prefix>_stats.csv` with the
same estimators the tests use (`tests/validation_util.hpp`): trapezoidal
`windowMean` over the row's window for cd, hysteresis zero-crossing `strouhal`
for St (compressible: St = f * pi/4). Uncomment the row with value + date.
Run-to-run repeatability measured 2026-07-17 was bit-identical at 256^2, so
tol_rel 0.01 is safe; widen to 0.015 only if drift is observed and note it here.

#### Regenerated 2026-07-29 (gfx1201)

| anchor | value | tol_rel | notes |
|---|---|---|---|
| `cyl_re60_n256_cd` | 1.818652 | 0.01 | 4.5 min; shedding; reproduces the 2026-07-17 value 1.818 — the recomposition did not move it |
| `comp_re60_n256_cd` | 1.944129 | 0.01 | 21 s; steady, cd flat to 1.5e-5 over [30, 62.8] and the same value to 6 digits over [95, 127.6] |

Both repeat runs were byte-for-byte identical in `_stats.csv`, so tol_rel 0.01
stands. `comp_re800_n512_{cd,st}` and `sphere_re300_n160_cd` are still deferred;
their owning tests keep skipping. Regenerating the Re60 pair was a measurement,
not a physics change, so nothing else moved — but note it retired a claim, see
tolerance adjustment 6.

2026-07-30, the absorption-boundary defaults flip (`--nscbc-sink` 16 on for
`--far nscbc` where defined; `--nscbc-etalat` 0.25 → 0.05; docs/nscbc.md):
verified against this suite with NO adjustment needed. `comp_re60_n256_cd` is
measured under `--far sponge`, which the sink never touches — the anchor test
passed unmodified after the flip. The nscbc-side effect of default-on was
measured directly before flipping: Re60 nscbc cylinder mean cd 1.786575 →
1.786622 (+0.003%), open Re60 cylinder 1.803413 → 1.803361 (−0.003%) — an
order below tol_rel. Both are trajectory changes, not new physics; a run
wanting the old bare boundary bit-for-bit passes `--nscbc-sink 0`
(`--nscbc-etalat 0.25` restores the old lateral rate).

### Convergence / blockage sweeps (optional, not automated)

- Cylinder cd/St grid triple:
  `dns flow --body cylinder --far tunnel --n {256,512,1024} --nz 1 --re 60 --dr 8 --ft 12`
  (anchors 1.803/1.785/1.785 pre-date this suite; see CLAUDE.md).
- Blockage sweep:
  `dns flow --body cylinder --far tunnel --n 512 --nz 1 --re 100 --dr {8,16,24} --ft 16`.
- TGV: `dns tgv --n {256,512} --re 1600 --ft 20` vs
  `data/validation/tgv3d_{256,512}.csv` and the spectral reference.
- Kim & Lee Re550 impulsive-start plot: manual, see rejected list (delta ~
  sqrt(nu t) is sub-cell early on; qualitative overlay only).

## Rejected benchmarks (and why)

- **Taylor-Couette**: no longer blocked, not yet written. The solver feature it
  was waiting on -- an INHOMOGENEOUS (rotating) ghost wall -- landed with the
  rotor: `DeviceCompGhost` images the velocity about `u_wall = omega x r` when
  built with a `WallVelocity` (README §12.3), and `gpu_rotor_wall` already pins
  a spinning circle's torque against the Stokes value and its driven flow
  against the wall speed. The incompressible `DeviceGhostFill` still mirrors
  homogeneous no-slip only, so a Taylor-Couette case has to be a compressible
  one. Spin-down covers the same physics with a stationary wall meanwhile.
- **Stokes/Oseen sphere Re 1-10**: dt collapses (diffusion-limited) and the
  1/r far field makes 8D confinement a 20-50% uncorrectable bias.
- **Kim & Lee impulsive start Re550**: boundary layer delta ~ sqrt(nu t) is
  sub-cell for the times that matter.
- **Lamb-Oseen / vortex dipole**: redundant with the TGV2D analytic test and
  the compressible isentropic-vortex test.
- **Acoustic Gaussian pulse**: existing acoustic test suffices; upgrade is
  low-priority.
