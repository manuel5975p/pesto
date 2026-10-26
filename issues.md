# dns — open issues

Current, changing from commit to commit. CLAUDE.md holds the stable
invariants; this file holds what is broken, unproven or undecided NOW.
Remove an entry when it is fixed (the fix lands in the commit message and,
if it creates an invariant, in CLAUDE.md). Numbers are measured, gfx1100.

As of the flow-RANS fix pass on 75f39db (2026-10-26, uncommitted).

## Flow RANS (masterplan `plans/flowrans/masterplan_flowrans.md`)

1. **Wedge SA-neg Re 1e4 at L1 (64×48) has no steady state.** Explicit LTS,
   M 0.2, 2.5e6 iterations (551 s): cd 0.4197 with a 1.1% jitter, resid
   0.08, θ_s 12.9° (a spurious front recirculation: cp 0.58 at the front
   stagnation point against 1.01), exit 8. L2 (128×96) certifies. OpenFOAM
   SA on the identical L1 mesh: 0.3928, steady. The central KG scheme has no
   damping for the under-resolved front; cause not isolated further.
2. **Wedge SST-2003m Re 1e4 has no steady state at L2.** Explicit LTS, 1e6
   iterations: a pseudo-time limit cycle in the near wake (θ 125–145°, j
   38–50: u swings 0.28 within 100 iterations), cd 0.4088–0.4136 (jitter
   4.3e-3, lag-1 acf −0.90), resid ~1, exit 8. OpenFOAM SST on the same mesh
   is steady (0.3950). L1 SST diverges (exit 7, 4.8e4 iterations, both CFL
   widths). Whether the limit cycle is a URANS-physical axisymmetric wake
   mode or a KG/LTS artefact is unmeasured. Blocks the SST half of `sph1e4`.
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
   application (59 ms at 75f39db); its serial chain is the floor. Candidate
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
7. **`sph1e4` parity is SA L2 only.** Tier B +1.16% (pass), θ_s +0.05°, L_r
   −4.4%. L1 (no dns steady state) and L3 (not run: L2 takes 3 min at M 0.2,
   L3 is ~4× the cells and the axis dt) remain; SST has no dns steady
   state.
8. **Not built:** heavy immersed channel oracle, focal/sinusoidal MMS,
   momentum-budget tally test; multigrid / alternating lines (U5); P4; P6.

## Older open items (details in CLAUDE.md)

- Re800 compressible anchors and `sphere_re300_n160_cd`: deferred, need
  regeneration.
- Re60 comp vs incomp cd gap +6.9%: cause unisolated.
- `gpu/comp_far.hip` domain y walls hardcoded adiabatic.
- Rotors: no table cache, no incompressible branch, no dns-ui support.
- dns-ui: a stop issued before a run reaches its time loop is not clean.
- DNS/LES `stableDt` LES bound over-conservative on clustered meshes.
- README names gfx1201; this machine is gfx1100.
