// --turb on a FaceMesh (M5; docs/fvcomp.md "Turbulence", docs/fvincomp.md "Low-Mach
// turbulence"): the turb_ic box at the cell centroids (IC) and swept past the inflow
// patch's face centres (the per-face rows) -- the ONE generator, evaluated as its
// Fourier sum at points, so any mesh reads it the same way. The box frame is the mesh
// frame: x along the flow, the inflow plane at x = 0, y and z the box's periodic axes.
// On the flags' uniform box the centroids are the structured cell centres, so the IC
// equals turbBox's field to rounding (direct FP64 sum vs FFTW c2r, test_turb_ic).
// Host, geometry layer, pure; the point sums are threaded and bit-identical at any
// thread count.
#pragma once

#include "dns/facemesh.hpp"
#include "dns/fvcomp_case.hpp"
#include "dns/fvflux.hpp"
#include "dns/fvlowmach.hpp"
#include "dns/fvref.hpp"
#include "dns/turb_ic.hpp"

#include <array>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace dns {

// The box and the IC window turbWindow(x, xw, ew).
struct FvTurb
{
    TurbSpec spec;
    std::vector<TurbMode> modes; // turbModes(spec)
    double xw = 0, ew = 1;
};

// The compressible IC: flame1d addTurbulence's arithmetic per cell (m += rho w u', E
// += the kinetic-energy change, p kept), w = turbWindow(C_x), u' the box at C. A cell
// whose window is exactly 0 skips the point sum (its u' is multiplied by 0).
// Pre: s has the mesh's cells, rows rho, m, E first.
void fvAddTurbulence(const FaceMesh& fm, const FvTurb& t, FvState& s, int threads = 0);

// The low-Mach IC: rho u += rho w u' per cell (no energy equation: rho T = P0/R).
// Pre: ic sized for fm.
void fvLowMachAddTurbulence(const FaceMesh& fm, const FvTurb& t, FvLowMachIC& ic, int threads = 0);

// The faces of patch `name`: boundary indices b (face nI + b), ascending, and centres.
struct FvPatchFaces
{
    std::vector<int> b;
    std::vector<std::array<double, 3>> cf;
};
// Error: no such patch.
std::expected<FvPatchFaces, std::string> fvPatchFaces(const FaceMesh& fm, std::string_view name);

// Per-face rows on patch `name` of a patch-row table (fvBcsOf: row p = patch p): bcRow
// [nB] holds every boundary face's patch row, each face of `name` its own copy of that
// patch's row appended to table in faces.b order (the rows a TurbSweep rewrites), first
// the first appended row. Error: no such patch.
struct FvFaceRows
{
    std::vector<int> bcRow;
    int first = 0;
    FvPatchFaces faces;
};
std::expected<FvFaceRows, std::string> fvFaceRows(const FaceMesh& fm, std::vector<FvBc>& table,
                                                  std::string_view name);

// The compressible inflow row of a face: the Far patch p's row (fvBcOf at frac 1)
// with the velocity U dir + du. du = 0 is fvBcOf's row bit for bit (test_turb_ic).
// Pre: p.kind == Far.
FvBc fvFarTurbRow(const FvPatchSpec& p, const FvFreestream& f, const std::array<double, 3>& du);

// The low-Mach inflow tables of a sweep: bU = bU0 + du, rUb = rUb0 + rho_b du on the
// faces b (rho_b = prGas / bT[b], the face's EOS density), every other face bU0, rUb0.
// du[c][i] belongs to face b[i]. du = 0 leaves the tables bit for bit.
// Pre: tables [3 nB], bT [nB], du[c].size() == b.size().
void fvLowMachTurbRows(const std::vector<int>& b, const std::array<std::vector<double>, 3>& du,
                       const std::vector<double>& bU0, const std::vector<double>& rUb0, const std::vector<double>& bT,
                       double prGas, std::vector<double>& bU, std::vector<double>& rUb);

} // namespace dns
