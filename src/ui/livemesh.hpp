// Turning a live field into drawable geometry.
//
// Deliberately per-vertex colour on the EXISTING triangle pipeline rather than
// a textured quad: no new shader, no new descriptor set, no texture upload
// path, and the cut-cell renderer already proves the approach at these sizes.
// A 512^2 plane is 262k quads, which is why `stride` exists.
//
// Two products:
//   - a CUT PLANE: one coloured quad per cell, at the plane's world position;
//   - VOLUMETRIC FOG: the volume drawn as a stack of translucent slices,
//     back-to-front along the axis most nearly facing the camera. That is the
//     classic slice-based volume renderer, and it works on the blend pipeline
//     that already exists for cut cells.
//
// Pure CPU: no Vulkan here, so `dns-ui --check` exercises it headlessly.
#pragma once

#include "ui/draw.hpp"
#include "ui/field.hpp"

#include <vector>

namespace ui {

// Colouring a magnitude RELATIVE to a reference speed, so the picture shows
// where the flow is slower (blue) and faster (red) than the tunnel rather than
// just "how fast". Near the reference the colour sinks into the background, so
// the wake and the accelerated shoulders are what you see. Applies to the cut
// plane and the fog alike.
enum class RefMode : int
{
    Off = 0,        // the field's own ramp: sequential for a magnitude
    Mean = 1,       // the frame's own mean over the fluid cells
    Freestream = 2  // the case's uInf, supplied by the caller
};

struct LiveMeshOptions
{
    int plane = 2;          // which axis to cut: 0 yz, 1 xz, 2 xy
    // Where along that axis, as a fraction of the domain. 0.5 is the centre --
    // which is the only plane the solver publishes directly. Any other value is
    // sliced out of the VOLUME by the viewer, so moving the plane is instant
    // and needs nothing from the solver.
    float planeAt = 0.5f;
    bool showPlane = true;
    bool showFog = false;
    float planeAlpha = 1.0f;
    // Fog density per CELL of depth, not per slice drawn: the builder
    // compensates for the slice spacing, so changing `fogSlices` changes how
    // smooth the fog is and not how thick it looks.
    float fogAlpha = 0.06f;
    float fogFloor = 0.06f;    // |normalized - 0.5| below this contributes nothing
    // Softness of that threshold, in the same normalized units. A HARD cut on a
    // smooth field draws its own contour: the fog comes out as nested shells
    // with visible edges, which reads as layering rather than as fog.
    float fogSoft = 0.12f;
    int fogSlices = 192;       // cap on the number of slices drawn
    int maxQuads = 400000;     // hard cap, as the cut-cell view has
    bool hideSolid = true;     // drop cells the mask calls solid

    // Relative colouring. `refValue` is used by Freestream; Mean measures it
    // from the data. `contrast` scales the span: 1 fills the ramp at the
    // largest deviation present, higher saturates sooner.
    RefMode refMode = RefMode::Off;
    float refValue = 1.0f;
    float contrast = 1.0f;
};

struct LiveMeshStats
{
    std::size_t quads = 0;
    int stride = 1;
    int slices = 0;
    bool capped = false;
    bool slicedFromVolume = false;   // false = the solver's centre plane
    int sliceIndex = -1;
    // What the relative colouring actually used, so the panel can show it.
    float refUsed = 0;
    float spanUsed = 0;
    // The slice stack this mesh was built for. Alpha compositing is only
    // correct back-to-front, so the mesh is stale the moment the camera crosses
    // into another octant -- the app compares these against fogOrder() and
    // rebuilds. Without it, orbiting leaves the fog composited inside-out.
    int fogAxis = -1;
    bool fogAscending = false;
    int fogStep = 1;   // cells between slices; > 1 means the stack is thinned
};

// The slice axis and traversal order for a view direction: the axis most nearly
// facing the camera (slices seen edge-on vanish), traversed from the FAR end so
// the blend pipeline composites back to front. `viewDir` points from the eye
// into the scene.
int fogOrder(const Vec3& viewDir, bool& ascending);

// Appends the cut plane at its world position. `lo`/`hi` are the field's fixed
// display range (from the publisher, so the colours do not flicker frame to
// frame). World placement uses the published window origin and spacings.
void buildLivePlane(const LiveView&, const LiveMeshOptions&, bool diverging,
                    float lo, float hi, std::vector<Vtx3>& out, LiveMeshStats&);

// Appends the fog: translucent slices through the volume, ordered back-to-front
// for the given view direction so alpha compositing is correct.
void buildLiveFog(const LiveView&, const LiveMeshOptions&, bool diverging,
                  float lo, float hi, const Vec3& viewDir,
                  std::vector<Vtx3>& out, LiveMeshStats&);

} // namespace ui
