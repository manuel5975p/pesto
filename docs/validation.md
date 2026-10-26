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

### Tier A (default; total ~163 s GPU)

| test | truth source | expected | asserted | why this tolerance | runtime |
|------|--------------|----------|----------|--------------------|---------|
| `val_spindown` | analytic: spin-down of solid-body rotation in a no-slip circle, u_th = sum c_k J1(lam_k r/R) exp(-nu lam_k^2 t/R^2), c_k = -2 Omega R/(lam_k J0(lam_k)); series self-verified in-test (c_k vs quadrature < 1e-6, IC L2 reproduction 0.05%) | KE(4)/KE(0) = 0.2020; u_th profile | KE ratio +-3%; profile L2 <= 4% | first-order cut wall at R/h = 49 => O(h/R) ~ 2% KE error expected; measured 2.7% / 0.0008. A 20% nu perturbation shifts the truth KE ratio by ~19% => tolerance bites | 6 s |
| `val_cyl_re60` | Williamson (1988) St(60) = 0.1358; Tritton/Henderson cd floor 1.42 | St 0.1521, cd 1.818 | St in [0.130, 0.156]; cd in [1.42, 1.90]; cl_rms > 0.01; >= 5 periods | ceiling +15% per the confinement table above (measured +12%); floor -4% discretization; run-to-run repeat was bit-identical | 87 s |
| `val_cyl_re100` | Williamson (1988) St = 0.164; Henderson cd 1.33-1.35 | St 0.1963, cd 1.730 | St in [0.157, 0.205]; cd in [1.33, 1.82] | ceilings +25%/+35% per the confinement table (measured +20%/+29%, excess grows with Re); floors = unbounded values | 69 s |
| `val_tgv3d_64` | anchor: repo curve `ours_tgv64.csv`; literature: 512^3 spectral reference `spectral_Re1600_512.gdiag` (dissipation peak 0.01286 at t = 8.97) | K/eps curves; peak 0.01182 at t = 8.34 (-8.1%) | curves within 2e-3 / 2e-2; peak time [8.0, 10.0]; value [-40%, +5%] | same binary => near-bitwise vs own curve (measured 6.6e-6 / 1.0e-5; slack for compiler/ROCm drift); 64^3 under-resolves the peak => trend band only, tight claim at 128^3 | 0.3 s |
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
| `hit_mini` (extended, CPU, label `slow;validation;literature`, not `heavy`) | Kolmogorov constant ~1.5 (Pope 2000), corrected by Pope's model spectrum for this resolution | C = 0.942 (8 samples) | C = <E(k) k^(5/3) eps^(-2/3)>, k in [4,10], in [0.7, 1.6] | at 64^3 with h = 1.5 eta the band reaches k*eta = 0.26..0.65 (dissipation rolloff): the model spectrum (C = 1.5, beta = 5.2, c_eta = 0.4, f_L for L = k^1.5/eps) predicts a band mean of 0.78; bottleneck lifts the measured value to ~0.94. Band = [0.9x, 2x] model | 45 s |

### RANS channel vs MKM (a MODEL vs DNS comparison; `rans_channel1d`)

Not a `val_*` test: the comparison runs on the FP64 oracle (the GPU's own
discrete operator, `docs/rans.md`) in milliseconds, and the GPU is pinned to
the oracle (`gpu_rans`) rather than to MKM. Truth: MKM `chan180.means` /
`chan395.means`, ub⁺ by trapezoid over the published profile (15.679 at
Re_τ 178.12, 17.545 at 392.24; the headers carry no Re_b).

| closure | Re_τ 178.12, ny 64 | Re_τ 392.24, ny 96 | asserted |
|---|---|---|---|
| SA-neg | 15.8069 (+0.82%) | 17.5596 (+0.08%) | within 5% (provisional ceiling); own value pinned 1e-6 |
| SST-2003m | 15.6332 (−0.29%) | 17.8881 (+1.96%) | same |
| SST-Vm | 15.5986 (−0.51%) | 17.8455 (+1.71%) | same |

SA-neg is grid-converged at these resolutions to ~0.2% (2nd order). SST is
FIRST order in ny (the TMR wall ω moves with the first cell): its limit is
15.13 (−3.5%) at Re_τ 178.12, so the ny-64 agreement is partly grid error —
cite the limit. The 5% band is a ceiling on model error, not a calibration;
a DNS grounder on this solver (docs/channel.md) has not been run.

The planned `val_rans_channel` (CLI cold start `--steady TOL` → exit 0) is NOT
in ctest: the FP32 steady march cannot reach any TOL that certifies the mean
(the update floor, `docs/rans.md`), so the run is exit 8 by design. It is a
manual check there (ub within 0.06% of the oracle after 3–4e5 iterations).

### Sphere Re 100 on the revolved wedge vs OpenFOAM (`sph100`, identical mesh)

Code-to-code, not literature: OpenFOAM v2606 `simpleFoam` (incompressible,
`linearUpwind` on U, SIMPLE consistent) run ON THE dns MESH — `dns flow
--emit mesh` writes `_mesh.bin`, `tools/parity/dnsmesh.py` turns it into a
`constant/polyMesh` (axis cells become prisms, `checkMesh` OK), and
`tools/parity/sph1e4.py case|run|read|dns` drives and reads both sides. Tier
B: dns cd0 = (4 cd(M 0.1) − cd(M 0.2))/3 against the incompressible solver.
Both dns runs are the explicit LTS march stopped at its plateau (exit 8; the
integrals jitter at ~1e-5 and never certify, `docs/rans.md`); OpenFOAM to
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

`sph1e4` (SA-neg / SST-2003m at Re 1e4) has NO dns side yet: the explicit LTS
march on the wedge does not reach a steady state (cd oscillates between −5.5
and +6.2 over ~1.5e6 iterations on SA L1; SST L1 diverged at 4.5e5, exit 7;
`docs/rans.md`), and the Newton-Krylov march that was to fix it failed its
gates (B0). The OpenFOAM half of the harness is built and checked
(`sph1e4.py case <mesh> <dir> sa|sst 1e4`).

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
