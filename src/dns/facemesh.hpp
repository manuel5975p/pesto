// FaceMesh: the ONE finite-volume mesh type of the FV path (docs/mesh.md).
// Cells, owner/neighbour faces, area vectors, patches and zones in FP64 on the
// host, built from an OpenFOAM polyMesh (snappy, blockMesh, extrudeMesh, ...)
// or from any structured CurvMesh (uniform, --stretch, --focal, O-grid) --
// the structured builder writes a PolyMesh and routes through fromPolyMesh,
// so there is ONE geometry definition: OpenFOAM v2606's
// (primitiveMeshTools::makeFaceCentresAndAreas / makeCellCentresAndVols).
//
// Contract:
//  * faces [0, nI) internal, own < nei, sorted by (own, nei, srcFace);
//    [nI, nI + nB) boundary, contiguous per patch (patches[] order), each
//    patch sorted by (own, srcFace). S points own -> nei (boundary: outward).
//  * d: internal C_nei (+ cyclic offset) - C_own; boundary Cf - C_own.
//  * cyclic patch pairs are merged into internal faces (S, Cf of the srcFace
//    side, the side whose polyMesh owner is the lower source cell); Cf of a
//    cyclic face lies on the srcFace side's polyMesh owner's frame.
//  * cellStart/cellFace: CSR cell -> faces, ascending face index -- the
//    deterministic, atomics-free gather order of every FV kernel.
//  * 2D is a one-cell-thick mesh, not a code path: `empty` faces carry no
//    flux; every other face has its emptyAxis component of S and d EXACTLY 0
//    (verified |S . e| <= 1e-12 |S| first).
//  * geometry is computed once in SOURCE numbering; an ordering permutes
//    and re-orients (own/nei swap negates S and d exactly), so geometry is
//    bitwise invariant under CellOrder.
#pragma once

#include "dns/curvmesh.hpp"
#include "dns/polymesh.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace dns {

// Cell numbering: polyMesh/Grid order, Morton (Z-order of C, 21 bits per
// axis, ties by source index), or colour-major (greedy distance-1 colouring
// of the Morton graph, then stable by colour). Values are the `dns mesh
// --order` indices.
enum class CellOrder : std::uint8_t { Source, Morton, ColourMajor };

// The `--order` spellings, indexed by CellOrder.
std::span<const char* const> cellOrderNames();

// One boundary patch: faces [nI + first, nI + first + count).
struct MeshPatch
{
    std::string name;
    FoamPatchType type{};
    int first = 0, count = 0;
    bool operator==(const MeshPatch&) const = default;
};

// Structured -> FaceMesh: the real cell range and the six boundary patches
// (i-, i+, j-, j+, k-, k+). An axis whose two types are Cyclic is periodic:
// full range required; AxisWrap::Cyclic shares its nodes (internal faces, no
// patch), AxisWrap::Translate becomes a translational cyclic patch pair
// named name[2a], name[2a+1]. nz == 1 is 2D: types[4] == types[5] == Empty.
// Pre (else error): m.axisFaces == {-1,-1}, wrap[2] != MirrorZ (wedge: P8).
struct StructuredSpec
{
    std::array<int, 3> lo{}, hi{};      // inclusive real cells per axis
    std::array<std::string, 6> name;    // i-, i+, j-, j+, k-, k+
    std::array<FoamPatchType, 6> type{};
};

// Finite-volume mesh shared by every FV kernel, every builder, and the solver graph.
struct FaceMesh
{
    int nC = 0, nI = 0, nB = 0;
    int dim = 3, emptyAxis = -1;                 // 2D: S/d components on emptyAxis exactly 0
    std::vector<int> own, nei;                   // own[nI+nB], nei[nI]; own < nei
    std::array<std::vector<double>, 3> S, Cf, d; // per face
    std::array<std::vector<double>, 3> C;        // per cell
    std::vector<double> V;
    std::vector<int> cellStart, cellFace;        // CSR cell -> faces, ascending face index
    std::vector<MeshPatch> patches;              // `empty` kept (kernels skip), cyclic consumed
    std::vector<std::string> zoneNames;
    std::vector<std::vector<int>> zoneCells;     // ascending new indices
    std::vector<std::int64_t> srcCell, srcFace;  // new -> source (polyMesh index, or Grid::idx)
    std::vector<std::uint8_t> colour;            // ColourMajor only
    int nColours = 0;
    CellOrder order = CellOrder::Morton;

    bool operator==(const FaceMesh&) const = default;

    // Geometry = OpenFOAM v2606, FP64. Errors (exit 3) by name: open cell
    // (|sum S| > 1e-9 sum |S|), V <= 0, d.S <= 0, cyclic pairs unmatched
    // (> 1e-8 bbox diagonal against the separation), `empty` not a one-cell
    // extrusion along x|y|z, wedge patches (P8), a cell coupled to itself.
    // threads <= 0: hardware concurrency; byte-identical for any count.
    static std::expected<FaceMesh, std::string> fromPolyMesh(const PolyMesh&, CellOrder,
                                                             int threads = 0);
    // polyOf + fromPolyMesh (one geometry definition); srcCell = Grid::idx.
    static std::expected<FaceMesh, std::string> fromCurvMesh(const CurvMesh&, const StructuredSpec&,
                                                             CellOrder, int threads = 0);
    // The PolyMesh of the real cells: shared nodes on a Cyclic wrap, separated
    // cyclic patches on Translate, cells numbered i fastest over the range,
    // internal faces upper-triangular. Writable with writePolyMesh (export).
    static std::expected<PolyMesh, std::string> polyOf(const CurvMesh&, const StructuredSpec&);
};

// checkMesh-equivalent numbers (definitions of OpenFOAM v2606
// primitiveMeshTools / polyMeshTools): non-orthogonality over internal faces
// (cyclic pairs once), skewness over all faces, aspect over solution dims.
struct MeshQuality
{
    double vMin = 0, vMax = 0, vTotal = 0, nonOrthMax = 0, nonOrthAvg = 0, skewMax = 0, aspectMax = 0;
    // dim V / sum |S| over the non-empty faces (h/2 on a cube or square).
    double cflWidthMin = 0, cflWidthP1 = 0, cflWidthMedian = 0, cflWidthMax = 0;
    double growthMax = 1;                                       // max V_nb / V over internal faces
    std::array<int, 32> facesPerCell{}, nodesPerFace{};         // histograms (last bin = more)
};
// Pre: fm = FaceMesh::fromPolyMesh(pm, ...) (srcFace indexes pm). Threaded, deterministic.
MeshQuality meshQuality(const FaceMesh& fm, const PolyMesh& pm, int threads = 0);
// The `dns mesh` report: counts, patches, zones, quality, CFL-width contrast,
// growth (warns > 1.25), colours.
std::string meshReport(const FaceMesh&, const MeshQuality&);

} // namespace dns
