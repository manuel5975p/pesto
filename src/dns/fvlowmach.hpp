// Host side of the low-Mach reacting FV solver (docs/fvincomp.md "Low-Mach"): the
// model, the --bc table of `dns flame --physics lowmach --polymesh`, the uniform
// box, the tables gpu/fv_incomp's DeviceFvLowMach uploads beside FvIncompSetup.
// FP64, FaceMesh numbering, pure (no GPU, no filesystem), deterministic.
//
// Box units as gpu/lowmach: rho_u = S_L = R = 1, delta_D = 1, Tu = P0 = 1.
#pragma once

#include "dns/chem.hpp"
#include "dns/facemesh.hpp"
#include "dns/fvcomp_config.hpp"
#include "dns/fvincomp.hpp"
#include "dns/fvles.hpp"
#include "dns/sgs.hpp"
#include "dns/tfles.hpp"
#include "dns/transport.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace dns {

// Fuel, plus an optional passive species (the gates' witness) or the oxidizer.
inline constexpr int kFvLmMaxY = 2;

// The model of gpu/lowmach on a FaceMesh: rho T = prGas, one irreversible step on
// species 0, Le = 1 (rho D = mu / Pr). lnA = -inf: no reaction. ox == 1 (nY 2): the
// two-reactant step F + s O (model C2, arrheniusRate2, oxidizer sink -s omega);
// law.n > 0 adds C2b's ln C(xi) (phiRateLn). ox < 0: species 1 is passive.
struct FvLowMachModel
{
    Transport tr;
    double prGas = 1, lnA = 0, Ta = 0, qcp = 0;
    int nY = 1;
    bool tally = true; // the FP64 mass, species and energy tallies (a host read per stage)
    int ox = -1;
    double stoich = 1;
    PhiRateLaw<double> law;
    // Open faces (FvLowMachTables::bOpen): beta of the inflow face pressure -beta rho_a
    // u_n^2/2 (1: the total-pressure inflow; 0 an API knob of the stability gate).
    double backflow = 1;
    // Gravity (gpu/lowmach's addBuoyancyLm on faces): the acceleration vector; 0 = the
    // solver without it, bit for bit. rho_a = prGas / FvLowMachTables::ambT.
    std::array<double, 3> gravity{};
    // LES (M4, gpu/lowmach's LowMachSetup::cs): Cs or C_sigma (sgs), Delta^2 =
    // FvIncompSetup::delta2, prt = Sc_t = Pr_t; 0 = the kernels bit for bit.
    double cs = 0;
    SgsModel sgs = SgsModel::Smagorinsky;
    double prt = kLesPrt;
    // TFLES (M4, C0): set = thickened; the sensor walks FvLowMachTables::walk.
    std::optional<TflesSetup> tfles{};
};

// The --bc file of the route: the incompressible patch table plus each inflow's
// mass fractions, by patch name. open: an `open` patch exists; every open patch
// draws the ONE ambient (ambT, ambY), and the outflow patches' backflow takes it.
struct FvLowMachBcs
{
    FvIncompBcs flow;
    std::map<std::string, std::array<double, kFvLmMaxY>> y;
    bool open = false;
    std::vector<std::string> openPatches;
    double ambT = 1;
    std::array<double, kFvLmMaxY> ambY{};
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
    // [nB] the sign-of-flux rule (gpu/lowmach's open faces): 2 on an `open` patch, 1
    // on an outflow while an open patch exists (its backflow takes the ambient), 0
    // elsewhere. Empty: no open face (the solver without it, bit for bit).
    std::vector<std::uint8_t> bOpen;
    double ambT = 1;
    std::array<double, kFvLmMaxY> ambY{};
    // M4 TFLES: the sensor's face walk and edge layers (dns/fvles fvTflesWalk; empty
    // next: not built, fvLowMachTfles fills it).
    FvTflesWalk walk;
};

// t.walk of fm (M4 TFLES). Error: fvFaceGeom's, fvTflesWalk's (a degenerate cell).
std::expected<void, std::string> fvLowMachTfles(const FaceMesh& fm, FvLowMachTables& t);

// spec (docs/fvcomp.md's file) on this route: inflow fixes m = rho_b u_b, T, y; a
// wall its T (isothermal) or no flux; outflow p = 0, scalars zero gradient both
// ways; open (t, y the ambient, default Tu and 0) the sign-of-flux face: outflow
// zero gradient, inflow the ambient at rest with p = -rho_a u_n^2/2, no diffusion.
// U, Tu: scales of the file's ratios. Error (exit 3), by name: `far` (no
// characteristic state), outflow `p`, `tback`, `burnt`, `y`, an inflow without `y`,
// two open patches with different ambients, no outflow or open face at all (a
// closed box needs P0(t), not built).
std::expected<FvLowMachBcs, std::string> fvLowMachBcs(const FvBcSpec& spec, double U, double Tu);

// Pre: s = buildFvIncompSetup(fm, b.flow), bindFvIncompBcs done, prGas the EOS's.
FvLowMachTables buildFvLowMachTables(const FaceMesh& fm, const FvLowMachBcs& b, const FvIncompSetup& s, double prGas);

// One inflow face of the burner (gpu/lowmach's per-row inflow on faces): m the mass
// flux along the patch's velocity direction, t = T, y the mass fractions.
struct FvLmRow
{
    double m = 0, t = 1;
    std::array<double, kFvLmMaxY> y{};
};

// Per-face inflow rows: every face q of Inflow patch `patch` takes row(q) (q the
// boundary index of s): bU = (m t/prGas) e, e the patch's unit velocity, rUb = m e,
// bT = t, bY = y. The kernels already read these tables per face: no device path.
// Pre: s, t of fm's bound table. Error (exit 3): no such inflow patch, a zero
// patch velocity (no direction).
std::expected<int, std::string> fvLowMachRows(const FaceMesh& fm, const std::string& patch, FvIncompSetup& s,
                                              FvLowMachTables& t, double prGas,
                                              const std::function<FvLmRow(std::size_t q)>& row);

// The flags' box: x-lo `xlo`, x-hi `xhi`, y cyclic (openY: patches `ylo`, `yhi`), z
// cyclic or `empty` at nz == 1, Morton order: the structured low-Mach grid's cells.
// solid non-empty (g.size(), nonzero = solid: lmTubeSolid's staircase): those cells
// removed, every fluid/solid face a boundary face of the `rim` patch, srcCell still
// Grid::idx. Error: the FaceMesh's; a solid map of the wrong size.
std::expected<FaceMesh, std::string> fvLowMachBox(const Grid& g, bool openY = false,
                                                  std::span<const std::uint8_t> solid = {});

// The box's --bc table: xlo inflow along +x, xhi outflow; openY: ylo, yhi open at
// the ambient (ambT, ambY ratios to Tu).
FvBcSpec fvLowMachBoxBc(bool openY = false, double ambT = 1, std::array<double, kFvLmMaxY> ambY = {});

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
