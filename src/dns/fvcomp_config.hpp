// Case description of the compressible FV family (`dns flow --physics compressible --solver fv`,
// `dns flow --physics compressible --polymesh DIR --bc FILE.toml`,
// docs/fvcomp.md): the --bc TOML and its check against a mesh's patch list.
// Pure parse + validate: no mesh, no filesystem, no GPU.
#pragma once

#include "dns/fvflux.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dns {

// Open: the low-Mach route's sign-of-flux face (dns flame --physics lowmach only;
// every other route rejects it by name): outflow zero gradient, inflow the ambient
// tIn, y at rest with face p = -rho_a u_n^2/2.
// Total: a stagnation inlet (FvBcKind::Total), Wedge: an axisymmetric side (a Slip row
// the tables know): the --shock hll polyMesh route's, docs/nozzle.md.
enum class FvPatchKind : std::uint8_t { Wall, Slip, Inflow, Outflow, Far, Empty, Open, Total, Wedge };

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
    FvExit exit = FvExit::Fixed; // Outflow
    double p0 = 1, t0 = 1;       // Total: p0 / p_inf, t0 / T_inf
    // dns flame only (a `dns flow` file rejects them by name). Inflow, Outflow,
    // Far: y, the mass fractions an incoming face carries; burnt, an Outflow at
    // the flame's burnt pressure, T_back at T_b.
    bool hasY = false, burnt = false;
    std::array<double, 4> y{};
    // dns flame --mode diffusion only. Inflow: this patch is the slot jet, one FvBc
    // row per face (fvJetRows), a fixed face at the Burke-Schumann row.
    bool jet = false;
};

// [nozzle] of a --bc file (docs/nozzle.md): the rest state p, t (ratios), the throat area
// astar the mesh generator measured, the thrust axis (unit). hasRest: p and t given (dns flow
// requires them, dns flame refuses them: the flame starts from its composition IC).
// cd (combscale S5, dns flame only): the throat's discharge coefficient in (0, 1]; the reacting
// design runs on the effective chamber ratio A_ref / (cd A*). Absent: 1 (A_ref / A* / 1.0, the bytes).
struct FvNozzleSpec
{
    double p = 1, t = 1, astar = 0;
    std::array<double, 3> axis{1, 0, 0};
    bool hasP = false, hasT = false;
    double cd = 1;
    bool hasCd = false;
};

// [thrust] of a --bc file (docs/nozzle.md "Thrust", masterplan_combnozzle R2): the impulse ledger and
// _thrust.csv. pa: the ambient pressure (a ratio, as [nozzle] p), >= 0; station: the reacting nozzle's burnt
// stagnation plane along [nozzle] axis (dns flame only). Needs [nozzle].
struct FvThrustSpec
{
    double pa = 0;
    std::optional<double> station{};
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
    std::optional<FvNozzleSpec> nozzle{};
    std::optional<FvThrustSpec> thrust{};

    // The resolved table, one line per patch (--dry-run).
    std::string describe() const;
};

// Strict: an unknown table, key or kind and a wrong type are errors by name.
std::expected<FvBcSpec, std::string> parseFvBc(std::string_view toml);

// The [nozzle]/[thrust] lines a restart compares BY NAME before the --bc sha (combscale S1); shortest
// round trip ("{}"); empty without either section. Post: one '\n'-terminated line per present section.
std::string fvBcCheckpointSections(const FvBcSpec&);

// One mesh patch, as the polyMesh boundary file names it.
struct FvMeshPatch
{
    std::string name, type; // `empty`, `cyclic`, `wall`, `patch`, ...
};

// Every patch bound exactly once. Pre: spec parsed. Errors, by name: a mesh
// patch the file does not bind, a binding with no patch, cyclic patches bound
// (they are internal faces), `empty` vs Empty mismatch (an `empty` patch may be
// left out), `wedge` patches unless wedgeOk (then left out or bound type = "wedge"), a
// tally patch on no wall, a missing reference area on a 3D mesh with a tally.
std::expected<void, std::string> checkFvBc(const FvBcSpec& spec, std::span<const FvMeshPatch> mesh, bool wedgeOk = false);

// The first thing in this file only the nozzle route reads ("" if none): `total`,
// `wedge`, exit, [nozzle], [thrust]. Every other route refuses it by name.
std::string fvBcNozzleKey(const FvBcSpec& spec);

// The first key of this file that only `dns flame` reads, "" if none.
std::string fvBcFlameKey(const FvBcSpec& spec);

// True when a mesh with these patches is 2D (an `empty` patch exists).
bool fvMeshIs2D(std::span<const FvMeshPatch> mesh);

} // namespace dns
