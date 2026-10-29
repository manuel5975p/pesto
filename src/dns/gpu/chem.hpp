// Chemistry diagnostics of `dns flame` (docs/chem.md): the explicit
// chemistry dt bound and the deterministic per-x-column sums the flame stats
// are built from. GPU-only header.
#pragma once

#include "dns/gpu/comp.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/transport.hpp"

#include <vector>

namespace dns::gpu {

// Fraction of the inverse chemical rate the step may take. 0.5 is far from
// binding at M <= 0.02 (measured dtChem/dt 8-32: the acoustic bound is the
// smaller); UNBRACKETED -- it never binds, so no measurement can move it
// (issues.md §37).
inline constexpr double kChemDt = 0.5;

// FP64 chemistry dt bound kChemDt / max_c J_c, J_c = A e^(-Ta/T)
// (1 + q Y Ta / (cv T^2)): the fuel's own consumption rate plus the thermal
// runaway rate of the heat release, per cell from the conserved state. The
// solver's stableDt stays byte-identical; the runner takes the min. Two-
// reactant step (SpeciesSetup::ox >= 0): J_c = k rho (Y_O + s Y_F + q Y_F Y_O
// Ta/(cv T^2)).
// Pre: d.nSpecies() > 0, s.nY == d.nSpecies().
double chemStableDt(const DeviceComp& d, const DeviceCompState& s,
                    ReduceScratch& r);

// The thickened flame's diffusive dt bound (dns flame --tfles): 0.25 h^2
// rho_min / (diffusivityFactor (mu(T_max) max FE + max mu_t)) -- stableDt's
// thermal form with the species/energy diffusivity F E mu + mu_t (FP64
// reductions; mu_t only with LES). stableDt stays byte-identical; the
// runner takes the min. Pre: fe = the DeviceTfles FE field of s (g.size());
// d.les() => d.mut() holds s's mu_t (stableDt(s) just ran).
double tflesStableDt(const DeviceComp& d, const DeviceCompState& s,
                     const DeviceScalar& fe, ReduceScratch& r);

// Deterministic FP64 per-x-column sums over (j, k), one thread per column,
// fixed order: out[i * kFlameSums + q] for q =
//   0 rho, 1 rho u, 2 T, 3 rho Y_fuel, 4 omega, 5 E + q rho Y_fuel, 6 u, 7 p,
//   8 count of cells with Y_fuel outside [-1e-6, 1 + 1e-6],
//   9 min Y_fuel, 10 max Y_fuel (Y = rhoY/rho in FP64 from the FP32 words:
//     the overshoot MAGNITUDE the count hides).
// Pre: d.updateChem(s) ran (primitives and omega of s are live).
inline constexpr int kFlameSums = 11;
hipError_t flameColumnStats(const DeviceComp& d, const DeviceCompState& s,
                            DeviceBuffer<double>& scratch,
                            std::vector<double>& host);

// The 3D flame's per-x-column sums (nz > 1 stats), same contract and order as
// flameColumnStats: out[i * kTurbSums + q] for q = 0..2 sum u, v, w; 3..5 sum
// u^2, v^2, w^2; 6 sum |grad Y_fuel| (central, wrap in y and z, 0 on the
// open x rows i = 0, nx - 1) -- the co-area integrand of the flame surface.
// Pre: d.updateChem(s) ran (primitives and the Y cache are live).
inline constexpr int kTurbSums = 7;
hipError_t turbColumnStats(const DeviceComp& d, DeviceBuffer<double>& scratch,
                           std::vector<double>& host);

} // namespace dns::gpu
