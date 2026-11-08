// From a resolved compressible FV case to what the solver reads: the freestream
// in box units and the FvBc table of a FaceMesh (docs/fvcomp.md). Host,
// geometry layer: one route for --bc files and for structured meshes, whose
// patches are named by a generated FvBcSpec.
#pragma once

#include "dns/facemesh.hpp"
#include "dns/fvcomp_config.hpp"
#include "dns/fvflux.hpp"
#include "dns/fvref.hpp"
#include "dns/polymesh.hpp"
#include "dns/rans.hpp"

#include <expected>
#include <span>
#include <string>
#include <vector>

namespace dns {

struct CompState;

// rho, U, p, T of the freestream, along dir. pBurnt, tBurnt: what a `burnt`
// Outflow reads (dns flame; unread otherwise). pRest: the pressure a nozzle's Total
// row ramps from (fvBcOf at frac < 1; unread otherwise).
struct FvFreestream
{
    double rho = 1, U = 1, p = 1, T = 1, gamma = 1.4;
    std::array<double, 3> dir{1, 0, 0};
    double pBurnt = 1, tBurnt = 1;
    double pRest = 0;
};

// A freestream of Mach M at unit rho and U: p = rho U^2 / (gamma M^2), T = p / (rho R).
FvFreestream fvFreestreamOf(double mach, double gamma, double Rgas, std::array<double, 3> dir);

// FvBc of one patch at ramp fraction frac in [0, 1] (the freestream velocity is
// frac U; a Total row's p0 runs pRest + frac (p0 - pRest); 1 = the converged table).
FvBc fvBcOf(const FvPatchSpec& p, const FvFreestream& f, double frac);

// One FvBc per fm.patches entry, one tally flag per patch.
// Pre: checkFvBc(spec, patches of fm) passed.
std::vector<FvBc> fvBcsOf(const FaceMesh& fm, const FvBcSpec& spec, const FvFreestream& f, double frac);
std::vector<std::uint8_t> fvTallyOf(const FaceMesh& fm, const FvBcSpec& spec);

// RANS: the exact distance to the nearest face of a Wall-kind patch, per cell of
// fm. --bc decides what a wall is, not the patch's type word. pm: the mesh fm was
// built from, by value (its patch types are rewritten).
// Errors (exit 3): no Wall patch; a cell count mismatch.
std::expected<std::vector<double>, std::string> fvRansWallDistance(PolyMesh pm, const FaceMesh& fm,
                                                                   const FvBcSpec& spec);

// s (5 rows) with the scalar rows of closure m appended: rho phiInf[e] per cell.
FvState fvWithScalars(const FvState& s, RansModel m, const std::array<double, 2>& phiInf);

// The compressible FV O-grid's patches for fromCurvMesh: node rows 1 and ny - 1 the
// `wall` and `far` lines, i cyclic (`ilo`, `ihi`), z cyclic (`empty` at nz 1).
StructuredSpec fvCompOgridSpec(const Grid& g);

// The compressible FV body-free box's patches: x and y `Far` patches (cyclic when periodic;
// y walls when wallY), z cyclic (`empty` at nz 1).
StructuredSpec fvCompBoxSpec(const Grid& g, bool periodic, bool wallY);

// The box's --bc: every non-empty patch of fm a Far, or an adiabatic tallied Wall on a
// `wall` patch.
FvBcSpec fvCompBoxBc(const FaceMesh& fm);

// The uniform start: rho, rho u0 dir, E = p/(gamma - 1) + rho u0^2/2 per cell (fs's gas).
FvState fvUniformState(const FaceMesh& fm, const FvFreestream& fs, double u0);

// A structured state (rows rho, m, E, then rhoY's) onto fm's numbering, cell n from
// srcCell[n] (fromCurvMesh: Grid::idx). Pre: q and each rhoY span srcCell's range.
FvState fvStateOf(const FaceMesh& fm, const CompState& q, std::span<const std::vector<double>> rhoY = {});

// The mesh's depth along emptyAxis (2D), else 1.
double fvMeshDepth(const FaceMesh& fm);

} // namespace dns
