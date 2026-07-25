// Turning a Build into drawable geometry: the mesh, the cut cells, the solid
// mask, the face apertures and the body surface.
//
// Everything is emitted at PHYSICAL positions. Under --focal the node
// coordinates are the remapped ones (Focal1D::map applied per axis), so the
// clustering is visible as clustering rather than being silently undone -- the
// whole point of looking at a focal mesh is to see where the fine plateau
// actually lands relative to the body.
//
// One caveat is inherited from the solver and is not a bug here: sampleCut()
// evaluates the sdf at REFERENCE coordinates (dns/geometry.hpp), so the cut
// cells belong to the reference-space body drawn at mapped positions. The
// surface is built the same way, so surface and cut agree with each other and
// with the apertures the runner consumes.
#pragma once

#include "ui/doc.hpp"
#include "ui/draw.hpp"

#include <string>
#include <vector>

namespace ui {

struct ViewOptions
{
    bool showBox = true;
    bool showGrid = true;
    bool showSurface = true;
    bool showCut = true;
    bool showSolid = false;
    bool showApertures = false;

    // 3D only: which plane the cut/solid/aperture display is restricted to.
    // Drawing every cut cell of a 512^3 sphere is millions of boxes and reads
    // as a solid blob anyway; a slab is both faster and clearer.
    int sliceAxis = 2;        // 0 x, 1 y, 2 z
    int sliceIndex = 0;       // cell index on that axis
    int slabCells = 1;        // slab half-width in cells
    bool wholeBody = false;   // ignore the slab (subject to maxCells)

    int gridStride = 0;       // 0 = auto
    int maxCells = 400000;    // hard cap on drawn cells
};

// What the last build actually produced, for the UI to report honestly.
struct ViewStats
{
    std::size_t cutDrawn = 0, solidDrawn = 0;
    bool capped = false;
    int strideUsed = 1;
};

struct ViewMeshes
{
    GpuMesh lines;      // grid, domain box, slab outline
    GpuMesh surface;    // opaque body surface
    GpuMesh cells;      // translucent cut / solid / aperture geometry
    ViewStats stats;
};

// CPU side, no GPU: vertices only. Split out from buildView so the geometry
// (marching tets, cut-cell boxes, remapped grid lines) is exercisable without
// a window -- see dns-ui --check.
struct ViewGeometry
{
    std::vector<VtxL> lines;
    std::vector<Vtx3> surface;
    std::vector<Vtx3> cells;
    ViewStats stats;
};

void buildViewGeometry(const Build&, const ViewOptions&, ViewGeometry&);

// Rebuilds every mesh from `b`. Safe to call with an unsuccessful build (it
// clears the meshes). Runs on the main thread; the cost is bounded by
// opts.maxCells and by the grid-line stride.
void buildView(Renderer&, const Build&, const ViewOptions&, ViewMeshes&);

void destroyView(Renderer&, ViewMeshes&);

// World-space centre and radius of the domain, for framing the camera.
void domainBounds(const Build&, Vec3& centre, float& radius);

// Physical node coordinate along an axis (focal remap applied). i in [0, n].
double nodeCoord(const Build&, int axis, int i);

// The 12 edges of an axis-aligned box, appended as line vertices. Shared with
// app.cpp, which draws the live selection box outside the cached meshes.
void appendBoxLines(std::vector<VtxL>&, const Vec3& lo, const Vec3& hi, Rgba);

} // namespace ui
