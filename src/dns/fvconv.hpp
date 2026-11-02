// dns conv on a FaceMesh (docs/fvcomp.md, docs/fvincomp.md): the structured box
// as a mesh, the checks of a --bc table against what the route means. Host,
// geometry layer; pure, deterministic. The two physics share these and the
// statistics of gpu/runner_fvconv, no physics.
#pragma once

#include "dns/curvmesh.hpp"
#include "dns/facemesh.hpp"
#include "dns/fvcomp_config.hpp"
#include "dns/fvincomp.hpp"
#include "dns/grid.hpp"

#include <expected>
#include <span>
#include <string>
#include <vector>

namespace dns {

// conv's validated grid (ghost rows included) as a FaceMesh of its fluid cells,
// floor at y = 0, closed side wall faces at x, z = 0: patches `floor`, `lid`,
// `xlo`, `xhi` (closed), `back`, `front` (closed 3D; `empty` at nz == 1), the
// periodic sides cyclic. kind, amp move nodes, plates and closed sides stay flat:
// Sine y += amp H sin(2 pi x/Lx) sin(pi y/H), periodic x += amp H sin(2 pi y/H);
// Random interior nodes +- amp h per axis (fvIncomp's hash). Error (exit 3): the
// FaceMesh's.
CurvMesh fvConvCurv(const Grid& g, bool closed, FvPerturb kind = FvPerturb::None, double amp = 0,
                    std::uint64_t seed = 0);
StructuredSpec fvConvSpec(const Grid& g, bool closed);
std::expected<FaceMesh, std::string> fvConvBox(const Grid& g, bool closed, FvPerturb kind = FvPerturb::None,
                                               double amp = 0, int threads = 0);

// One mesh patch's thermal condition. fixed = false: adiabatic.
struct FvConvWall
{
    std::string name;
    bool fixed = false;
    double T = 0;
};

// fm.patches indices of the two plates.
struct FvConvPlates
{
    std::vector<int> floor, lid;
};

// Error (exit 3) by name: a patch T that is neither plate's, no floor or no lid,
// a plate face off y = 0 / y = H, a cell outside (0, H).
std::expected<FvConvPlates, std::string> fvConvPlates(const FaceMesh& fm, std::span<const FvConvWall> walls, double tBot,
                                                      double tTop, double H);

// The --bc table means something here: closed, resting patches, no reference or
// initial state. Errors (exit 3) by name.
std::expected<std::vector<FvConvWall>, std::string> fvConvWalls(const FvBcSpec& s);
std::expected<std::vector<FvConvWall>, std::string> fvConvWalls(const FvIncompBcs& b);

} // namespace dns
