// Case description of the compressible FV family (`dns flow --physics compressible --solver fv`,
// `dns flow --physics compressible --polymesh DIR --bc FILE.toml`,
// docs/fvcomp.md): the --bc TOML and its check against a mesh's patch list.
// Pure parse + validate: no mesh, no filesystem, no GPU.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dns {

// Open: the low-Mach route's sign-of-flux face (dns flame --physics lowmach only;
// every other route rejects it by name): outflow zero gradient, inflow the ambient
// tIn, y at rest with face p = -rho_a u_n^2/2.
enum class FvPatchKind : std::uint8_t { Wall, Slip, Inflow, Outflow, Far, Empty, Open };

// One patch's condition in freestream units: velocities / U, temperature and
// pressure ratios to T_inf and p_inf. Only the members of `kind` are read.
struct FvPatchSpec
{
    std::string name;
    FvPatchKind kind = FvPatchKind::Wall;
    bool isothermal = false; // Wall
    double tWall = 1;        // Wall: T_w / T_inf
    std::array<double, 3> uWall{}, omega{}, origin{}; // Wall motion
    bool tallied = false;    // Wall, Slip: enters the force tally (Wall default on)
    bool hasU = false;       // Inflow: u given, else the freestream
    std::array<double, 3> uIn{};
    double tIn = 1;          // Inflow; Open: the ambient T / T_inf
    double pOut = 1, tBack = 1; // Outflow
    // dns flame only (a `dns flow` file rejects them by name). Inflow, Outflow,
    // Far: y, the mass fractions an incoming face carries; burnt, an Outflow at
    // the flame's burnt pressure, T_back at T_b.
    bool hasY = false, burnt = false;
    std::array<double, 4> y{};
    // dns flame --mode diffusion only. Inflow: this patch is the slot jet, one FvBc
    // row per face (fvJetRows), a fixed face at the Burke-Schumann row.
    bool jet = false;
};

// The whole --bc file. length: L in Re = rho U L / mu. area: force reference
// area, 0 = L x depth on a 2D mesh, required for a tally on a 3D mesh.
// dir: unit freestream direction, lift: unit lift direction, normal to dir.
// origin (dns flame only, hasOrigin): where the flame coordinate s along dir is 0;
// --mode diffusion: the slot's centre on the inflow plane.
struct FvBcSpec
{
    std::vector<FvPatchSpec> patches;
    double length = 1, area = 0;
    std::array<double, 3> dir{1, 0, 0}, lift{0, 1, 0}, origin{0, 0, 0};
    bool hasOrigin = false;

    // The resolved table, one line per patch (--dry-run).
    std::string describe() const;
};

// Strict: an unknown table, key or kind and a wrong type are errors by name.
std::expected<FvBcSpec, std::string> parseFvBc(std::string_view toml);

// One mesh patch, as the polyMesh boundary file names it.
struct FvMeshPatch
{
    std::string name, type; // `empty`, `cyclic`, `wall`, `patch`, ...
};

// Every patch bound exactly once. Pre: spec parsed. Errors, by name: a mesh
// patch the file does not bind, a binding with no patch, cyclic patches bound
// (they are internal faces), `empty` vs Empty mismatch (an `empty` patch may be
// left out), `wedge` patches, a tally patch on no wall, a missing reference
// area on a 3D mesh with a tally.
std::expected<void, std::string> checkFvBc(const FvBcSpec& spec, std::span<const FvMeshPatch> mesh);

// The first key of this file that only `dns flame` reads, "" if none.
std::string fvBcFlameKey(const FvBcSpec& spec);

// True when a mesh with these patches is 2D (an `empty` patch exists).
bool fvMeshIs2D(std::span<const FvMeshPatch> mesh);

} // namespace dns
