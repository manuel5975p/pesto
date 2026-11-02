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
#include <string>
#include <vector>

namespace dns {

// rho, U, p, T of the freestream, along dir. pBurnt, tBurnt: what a `burnt`
// Outflow reads (dns flame; unread otherwise).
struct FvFreestream
{
    double rho = 1, U = 1, p = 1, T = 1, gamma = 1.4;
    std::array<double, 3> dir{1, 0, 0};
    double pBurnt = 1, tBurnt = 1;
};

// A freestream of Mach M at unit rho and U: p = rho U^2 / (gamma M^2), T = p / (rho R).
FvFreestream fvFreestreamOf(double mach, double gamma, double Rgas, std::array<double, 3> dir);

// FvBc of one patch at ramp fraction frac in [0, 1] (the freestream velocity is
// frac U; 1 = the converged table).
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

// The mesh's depth along emptyAxis (2D), else 1.
double fvMeshDepth(const FaceMesh& fm);

} // namespace dns
