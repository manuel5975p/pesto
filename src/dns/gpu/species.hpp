// Species transport and single-step chemistry kernels of `dns flame`
// (docs/chem.md), uniform mesh only. GPU-internal header (HIP TUs only; it
// reads PrimView/ViscView from internal/comp_flux.hpp). DeviceComp calls these
// from stage() when enableSpecies ran; nothing else does.
//
// Scheme. rho Y_k rides EXACTLY the mass flux faceFlux puts in continuity --
// mdot = Real(0.5)(rho_a + rho_b) * Real(0.5)(u_a + u_b), same expression and
// operand order -- with the face value MC-limited MUSCL, upwind on the sign of
// mdot: Y_f = Y_up + MC(Y_up - Y_upup, Y_down - Y_up)/2. A uniform Y then gives
// F_Y = Y F_rho, and for a power-of-two Y every stage value is exactly Y times
// the density's (gpu_flame_passive pins it bitwise). The limiter is
// deliberate, bounded scalar dissipation on the species alone (the RANS
// precedent); the mean flow stays central. MC, not van Leer: measured on the
// host discrete flame, van Leer's order is 1.75 and MC's 2.12 (docs/chem.md).
// Diffusion is Fick with rho D = mu_f / Pr (Le = 1), mu_f the face's own
// Sutherland value, central.
#pragma once

#include "dns/gpu/comp.hpp"
#include "dns/gpu/internal/comp_flux.hpp"

namespace dns::gpu::species {

// The fixed per-run arguments of the species kernels.
struct SpeciesView
{
    int nY = 0, fuel = 0;
    int openX = 0;
    Real invPr = 0;
    Real q = 0;                 // heat release per unit fuel mass
    Real yIn[kMaxY] = {};
    const Real* Y[kMaxY] = {};  // the cell pass's Y_k = rhoY_k / rho
    const Real* omega = nullptr;
};

// The cell pass of state s: Y_k = rhoY_k / rho into Y[k], and
// omega = arrheniusRate(lnA, Ta, rho, Y_fuel, T) into omega (T the live
// primitive). One launch. lnA == -inf writes omega == 0 exactly.
void cellPass(const GpuGrid& g, const DeviceCompState& s, const Real* T,
              Real lnA, Real Ta, int fuel, Real* const Y[kMaxY], Real* omega);

// The species stage: out.y[k] = ca ya + cb yb + crAt(c) (-div F_k - delta_kF
// omega), plus the heat release E += crAt(c) q omega on the SAME omega (fused
// into this launch; nothing in between reads E). Run after kCompStage and
// before the rhs hook. Plain (non-compensated) form only.
void stage(const GpuGrid& g, ViscView vv, PrimView q, const StageCombine& cmb,
           const SpeciesCombine& sc, const SpeciesView& sv,
           DeviceCompState& out);

// The open-x-face re-carry (kOpenScalarFix's plain branch on y[]): on the
// cells i = 0 and nx-1, rhoY_k = rhoY_k / rho_own * rho_out with rho_own =
// ca a + cb b + crAt M, M the species' own mass divergence there.
void openConsistency(const GpuGrid& g, PrimView q, const StageCombine& cmb,
                     int nY, DeviceCompState& out);

} // namespace dns::gpu::species
