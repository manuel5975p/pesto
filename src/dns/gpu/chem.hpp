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
// solver's stableDt stays byte-identical; the runner takes the min.
// Pre: d.nSpecies() > 0, s.nY == d.nSpecies().
double chemStableDt(const DeviceComp& d, const DeviceCompState& s,
                    ReduceScratch& r);

// Deterministic FP64 per-x-column sums over (j, k), one thread per column,
// fixed order: out[i * kFlameSums + q] for q =
//   0 rho, 1 rho u, 2 T, 3 rho Y_fuel, 4 omega, 5 E + q rho Y_fuel, 6 u, 7 p,
//   8 count of cells with Y_fuel outside [-1e-6, 1 + 1e-6].
// Pre: d.updateChem(s) ran (primitives and omega of s are live).
inline constexpr int kFlameSums = 9;
hipError_t flameColumnStats(const DeviceComp& d, const DeviceCompState& s,
                            DeviceBuffer<double>& scratch,
                            std::vector<double>& host);

} // namespace dns::gpu
