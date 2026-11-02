// Host side of the low-Mach reacting FV solver (docs/fvincomp.md "Low-Mach"): the
// model, the --bc table of `dns flame --physics lowmach --polymesh`, the uniform
// box, the tables gpu/fv_incomp's DeviceFvLowMach uploads beside FvIncompSetup.
// FP64, FaceMesh numbering, pure (no GPU, no filesystem), deterministic.
//
// Box units as gpu/lowmach: rho_u = S_L = R = 1, delta_D = 1, Tu = P0 = 1.
#pragma once

#include "dns/facemesh.hpp"
#include "dns/fvcomp_config.hpp"
#include "dns/fvincomp.hpp"
#include "dns/transport.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace dns {

// Fuel, plus an optional passive species (the gates' witness).
inline constexpr int kFvLmMaxY = 2;

// The model of gpu/lowmach on a FaceMesh: rho T = prGas, one irreversible step on
// species 0, Le = 1 (rho D = mu / Pr). lnA = -inf: no reaction.
struct FvLowMachModel
{
    Transport tr;
    double prGas = 1, lnA = 0, Ta = 0, qcp = 0;
    int nY = 1;
    bool tally = true; // the FP64 mass, species and energy tallies (a host read per stage)
};

// The --bc file of the route: the incompressible patch table plus each inflow's
// mass fractions, by patch name.
struct FvLowMachBcs
{
    FvIncompBcs flow;
    std::map<std::string, std::array<double, kFvLmMaxY>> y;
};

// The tables of a low-Mach run beside FvIncompSetup.
struct FvLowMachTables
{
    std::array<std::vector<double>, kFvLmMaxY> bY; // [nB] fixed mass fraction
    std::vector<std::uint8_t> bYfix;               // [nB] 1 on inflow faces
    // [nI] bit 0: the owner, bit 1: the neighbour has no second neighbour through
    // this face (a boundary face sits behind it, within 60 deg): upwind there the
    // shared limiter goes central, sharedThetaOf's first-side rule.
    std::vector<std::uint8_t> first;
    std::vector<double> rUb;                       // [3 nB] the inflow's rho u
};

// spec (docs/fvcomp.md's file) on this route: inflow fixes m = rho_b u_b, T, y; a
// wall its T (isothermal) or no flux; outflow p = 0, scalars zero gradient both
// ways. U, Tu: scales of the file's ratios. Error (exit 3), by name: `far` (no
// characteristic state), outflow `p`, `tback`, `burnt`, `y`, an inflow without `y`,
// no outflow at all (a closed box needs P0(t), not built).
std::expected<FvLowMachBcs, std::string> fvLowMachBcs(const FvBcSpec& spec, double U, double Tu);

// Pre: s = buildFvIncompSetup(fm, b.flow), bindFvIncompBcs done, prGas the EOS's.
FvLowMachTables buildFvLowMachTables(const FaceMesh& fm, const FvLowMachBcs& b, const FvIncompSetup& s, double prGas);

// The flags' box: x-lo `xlo`, x-hi `xhi`, y cyclic, z cyclic or `empty` at nz == 1,
// Morton order: the structured low-Mach grid's cells. Error: the FaceMesh's.
std::expected<FaceMesh, std::string> fvLowMachBox(const Grid& g);

// The box's --bc table: xlo inflow along +x, xhi outflow.
FvBcSpec fvLowMachBoxBc();

// Initial cell state, FP64: rho, rho Y_k [nC], rho u [3 nC], p [nC]. ry1 empty: 0.
struct FvLowMachIC
{
    std::vector<double> rho, ry0, ry1, ru, p;
};

// The flame T(s) at every centroid, s = (C - origin) . dir: rho = prGas / T, Y =
// (sigma - T)/(sigma - 1), rho u = m dir, p = m^2 (sigma - T). Pre: t > 0.
FvLowMachIC fvLowMachFlameIC(const FaceMesh& fm, const std::array<double, 3>& dir, const std::array<double, 3>& origin,
                             double sigma, double m, double prGas, const std::function<double(double)>& t);

} // namespace dns
