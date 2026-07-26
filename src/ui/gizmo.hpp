// The translate gizmo.
//
// Everything here is a pure function of plain inputs -- no Doc, no Camera, no
// Vulkan -- which is what lets `dns-ui --check` test the part that actually
// matters: that the point you grabbed stays under the cursor.
//
// Two design decisions worth stating, because both are load-bearing:
//
//   - The handles are TRIANGLES, not lines. `rs.lineWidth` is fixed at 1.0 and
//     VK_DYNAMIC_STATE_LINE_WIDTH is not enabled (draw.cpp), so a shaft drawn
//     as a line is one pixel wide and unusable as a handle.
//
//   - The drag is ABSOLUTE, not incremental. gizmoSolve() is a function of the
//     CURRENT ray and the state captured at grab time, so it cannot drift over
//     a long drag, it survives a dropped motion event or a frame hitch, and
//     zooming mid-drag is harmless because the anchor is in world space. The
//     old camera-plane nudge accumulated dx/dy and had none of that.
#pragma once

#include "ui/draw.hpp"
#include "ui/math.hpp"

#include <cstdint>
#include <vector>

namespace ui {

enum class GizmoPart : std::uint8_t
{
    None,
    AxisX, AxisY, AxisZ,
    PlaneX, PlaneY, PlaneZ,   // named by the plane's NORMAL: PlaneZ is the xy quad
    Screen                    // the centre handle: the camera-facing plane
};

struct GizmoStyle
{
    float shaftPx = 78;
    float shaftRadPx = 2.4f;
    float headLenPx = 20;
    float headRadPx = 6.5f;
    float planeOffPx = 26;
    float planeSizePx = 22;
    float screenRadPx = 12;
    float pickPadPx = 7;
    int coneSides = 12;
    float parallelCut = 0.985f;   // |dot(axis, fwd)| above: the arrow is a dot
    float edgeOnCut = 0.15f;      // |dot(planeN, fwd)| below: the quad is a line
};

// Everything positional, built once per frame and handed to BOTH the geometry
// builder and the hit test -- so the picture and the hit test cannot disagree,
// which is the most common way a gizmo goes subtly wrong.
struct GizmoFrame
{
    Vec3 origin;
    Vec3 axis[3];        // unit, already sign-flipped toward the camera
    bool enabled[3] = {true, true, true};
    Vec3 camFwd, camRight, camUp;
    float worldPerPx = 1;
    float uiScale = 1;
};

GizmoFrame makeGizmoFrame(const Vec3& origin, const Vec3& eye, const Vec3& camFwd,
                          const Vec3& camRight, const Vec3& camUp,
                          float worldPerPx, float uiScale, const bool enabled[3]);

// Appends the gizmo as triangles. `hot` is the part under the cursor, `active`
// the one being dragged; both are highlighted.
void gizmoBuild(std::vector<Vtx3>& out, const GizmoFrame&, const GizmoStyle&,
                GizmoPart hot, GizmoPart active);

struct GizmoHit
{
    GizmoPart part = GizmoPart::None;
    float pixelDist = 0;
};

// Hit test in PIXELS against the projected handles -- simpler than ray-vs-cone
// and exactly what the user sees. `mouse`, `vpW`, `vpH` are VIEWPORT-relative,
// the same coordinates Camera::ray is given; mixing window and viewport pixels
// here offsets every hit by the width of the left column.
GizmoHit gizmoPick(const GizmoFrame&, const GizmoStyle&, const Mat4& viewProj,
                   float vpW, float vpH, const Vec2& mouse);

struct GizmoDrag
{
    bool active = false;
    GizmoPart part = GizmoPart::None;
    Vec3 originAtGrab;
    Vec3 axis;          // unit, sign-resolved (axis drags)
    float grabT = 0;    // line parameter at grab (axis drags)
    Plane plane;        // frozen at grab (plane / screen drags)
    Vec3 grabPoint;     // the world point that was under the cursor
};

// False when the grab is geometrically degenerate (the ray lies in the handle's
// plane): reject the gesture rather than fling the object to infinity.
bool gizmoBeginDrag(GizmoDrag&, const GizmoFrame&, GizmoPart, const Ray&);

// THE solve. The new origin, such that the grabbed point stays under the cursor.
Vec3 gizmoSolve(const GizmoDrag&, const Ray&);

// --- snapping ----------------------------------------------------------------

struct SnapSettings
{
    enum class Mode : int { Free = 0, Grid = 1, Fraction = 2 };
    Mode mode = Mode::Free;
    double h[3] = {0, 0, 0};   // grid spacing per axis, world units
    double D = 1;              // body diameter, for fraction snapping
    double fraction = 0.25;    // of D
};

// Grid snapping is ABSOLUTE -- a position lands on a mesh node, which in a
// cut-cell editor is the point: it puts a body face exactly on a cell face.
// Fraction snapping is RELATIVE to `base`, because "half a diameter along" is a
// displacement, not a coordinate. An axis the handle already constrains is
// passed through untouched; snapping it would fight the constraint.
Vec3 snapOrigin(const SnapSettings&, const Vec3& base, const Vec3& candidate,
                const bool axisFree[3]);

} // namespace ui
