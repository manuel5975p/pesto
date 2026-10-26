# dns — open issues

Current, changing from commit to commit. CLAUDE.md holds the stable
invariants; this file holds what is broken, unproven or undecided NOW.
Remove an entry when it is fixed (the fix lands in the commit message and,
if it creates an invariant, in CLAUDE.md). Numbers are measured, gfx1100.

As of the SST small-case pass on 85eaa8f (2026-10-26, uncommitted).

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
   central). L1 SST certifies at Re 1e3 (gate `gpu_sst`) and nearly at Re
   3e3 (remaining 1.7e-4 at 2e5). LTS-ratio cap test: see below. Not an SST
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
   SST: URANS mean only (2); small SST parity L1 Re 1e3 +0.41% (`gpu_sst`).
8. **Not built:** heavy immersed channel oracle, focal/sinusoidal MMS,
   momentum-budget tally test; multigrid / alternating lines (U5); P4; P6.
9. **URANS turbulence scalars are first order in time.** The point-implicit
   stage quotient (D13) degrades SSP-RK3 to first order in the stiff
   sources: the closed-form SST freestream decay converges at order 1.00 (ω)
   / 0.91–0.98 (k), `gpu_sst` (a); relative error 8e-3 (ω) / 3.9e-2 (k) at
   β ω0 dt = 0.12. The steady fixed point is unaffected. A URANS result
   whose β*ω dt is not small carries this error. DEFERRED 2026-10-26: a
   second-order implicit source update (and its byte-identity break) is not
   built for now; URANS results state the first-order source error.

## Older open items (details in CLAUDE.md)

- Re800 compressible anchors and `sphere_re300_n160_cd`: deferred, need
  regeneration.
- Re60 comp vs incomp cd gap +6.9%: cause unisolated.
- `gpu/comp_far.hip` domain y walls hardcoded adiabatic.
- Rotors: no table cache, no incompressible branch, no dns-ui support.
- dns-ui: a stop issued before a run reaches its time loop is not clean.
- DNS/LES `stableDt` LES bound over-conservative on clustered meshes.
- README names gfx1201; this machine is gfx1100.
