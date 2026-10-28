// OpenFOAM polyMesh (constant/polyMesh) reader and writer: points, faces,
// owner, neighbour, boundary, cellZones. ASCII and binary (faceCompactList),
// label 32/64 (narrowed to int32, checked), scalar 64. Accepts exactly what
// blockMesh, snappyHexMesh, extrudeMesh, refineMesh, foamFormatConvert and
// plot3dToFoam write; everything else is rejected BY NAME (docs/mesh.md).
// Host only, no geometry: dns/facemesh turns a PolyMesh into the FV mesh.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace dns {

// OpenFOAM boundary types the reader accepts; any other type is rejected by name.
enum class FoamPatchType : std::uint8_t { Wall, Patch, Symmetry, SymmetryPlane, Empty, Cyclic, Wedge };

// The OpenFOAM spelling of a patch type ("wall", "symmetryPlane", ...).
const char* foamPatchTypeName(FoamPatchType);

struct FoamPatch
{
    std::string name;
    FoamPatchType type{};
    std::int64_t nFaces = 0, startFace = 0;
    std::string neighbourPatch;                        // Cyclic only
    std::optional<std::array<double, 3>> separation;   // Cyclic `transform translational`
                                                       // (separationVector: neighbour - this)
    bool operator==(const FoamPatch&) const = default;
};

// Header-level facts read without bulk data (`dns mesh --dry-run`).
struct PolyManifest
{
    std::int64_t nPoints = 0, nCells = 0, nFaces = 0, nInternalFaces = 0; // owner `note`
    std::vector<FoamPatch> patches;     // `boundary` order
    std::vector<std::string> cellZones; // names only
    bool binary = false;
    int labelBits = 32;
    bool operator==(const PolyManifest&) const = default;
};

// Reads boundary, the owner header note and the cellZones names; no bulk data.
// Pre: dir is a constant/polyMesh directory. Errors name the file and token (exit 3).
std::expected<PolyManifest, std::string> readPolyManifest(const std::filesystem::path& dir);

struct PolyMesh
{
    PolyManifest man;
    std::vector<double> px, py, pz;                    // points
    std::vector<std::int32_t> faceStart;               // nFaces + 1
    std::vector<std::int32_t> faceNodes;
    std::vector<std::int32_t> owner;                   // nFaces
    std::vector<std::int32_t> neighbour;               // nInternalFaces
    std::vector<std::vector<std::int32_t>> zoneCells;  // per man.cellZones, ascending
    bool operator==(const PolyMesh&) const = default;
};

// Full read. Rejects by name: *.gz, processor* patches ("decomposed: run
// reconstructParMesh"), cyclicAMI/ACMI/nonConformal*/mapped* types, cyclic
// `rotational`, #include/#calc, owner >= neighbour on an internal face, labels
// out of range, faces with < 3 nodes, patches not contiguous over
// [nInternalFaces, nFaces), unpaired cyclics, overlapping cellZones,
// truncated lists. Errors name the file (exit 3).
std::expected<PolyMesh, std::string> readPolyMesh(const std::filesystem::path& dir);

// Writes points/faces/owner/neighbour/boundary (+ cellZones when m has zones)
// in OpenFOAM's layout; labelBits 32 or 64. Creates dir.
// Post: readPolyMesh(dir) == m up to man.binary/man.labelBits, which take the
// written values.
std::expected<void, std::string> writePolyMesh(const std::filesystem::path& dir, const PolyMesh& m,
                                               bool binary, int labelBits = 32);

} // namespace dns
