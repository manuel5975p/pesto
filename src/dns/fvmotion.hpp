// Solid-body mesh motion of the pimpleFoam path (docs/fvpimple.md "Moving
// mesh", stage 3b): OpenFOAM v2606's dynamicMotionSolverFvMesh /
// dynamicMultiMotionSolverFvMesh::update() with solidBodyMotionSolver +
// rotatingMotion, fvMesh::movePoints and what hangs on it, operand for operand,
// host, FP64, pure:
//  * points(t) = transformPoints(septernion(-origin) * quaternion(axis, omega t)
//    * septernion(origin), points0) for the points of the moving cells;
//  * the mesh flux meshPhi = face::sweptVol(oldPoints, newPoints) / deltaT
//    (fvGeometryScheme::setMeshPhi), the swept volume of the face's central
//    triangle fan (triangle::sweptVol), so the geometric conservation law holds
//    as it does in OpenFOAM;
//  * movingWallVelocityFvPatchVectorField::Uwall();
//  * the geometry itself is NOT restated here: the moved points go through
//    FaceMesh::fromPolyMesh and buildSimpleSetup / buildRasSetup (the one
//    geometry definition; the wall distance is recomputed every step, as
//    wallDist::movePoints does with its default updateInterval 1).
// face::centre is OpenFOAM's face-class centroid (Uwall, sweptVol): a formula
// of those two consumers, not mesh geometry.
#pragma once

#include "dns/fvcase.hpp"
#include "dns/fvsimple.hpp"
#include "dns/la/ell.hpp"
#include "dns/la/graph.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace dns {

// A point set as three coordinate arrays (PolyMesh::px, py, pz).
struct FoamPoints
{
    const double *x = nullptr, *y = nullptr, *z = nullptr;
};

// face::centre(points): the centroid of the fan about the node average
// (triangle: the node average itself). Pre: n.size() >= 3.
FvVec3 foamFaceCentre(std::span<const std::int32_t> n, FoamPoints p);
// face::sweptVol(oldPoints, newPoints): the fan's triangle::sweptVol sum.
double foamFaceSweptVol(std::span<const std::int32_t> n, FoamPoints oldP, FoamPoints newP);

// rotatingMotion::transformation() at time t as transformPoints applies it:
// p' = R & (p - trans), each part skipped by OpenFOAM's own tests
// (mag(trans) > VSMALL, mag(R - I) > SMALL).
struct FoamRigidTransform
{
    FvVec3 trans{};
    std::array<double, 9> R{1, 0, 0, 0, 1, 0, 0, 0, 1}; // row-major
    bool translate = false, rotate = false;
    FvVec3 apply(const FvVec3& p) const;
};
FoamRigidTransform foamRotatingMotion(const FvSolidBody& b, double t);

// One mesh.update(): what the device uploads beside the new geometry.
struct FvMeshMove
{
    std::vector<double> meshPhi;  // [nI] FaceMesh orientation (own -> nei)
    std::vector<double> meshPhiB; // [nB] the non-empty boundary faces of FvSimpleSetup
    std::vector<double> uWall;    // [3 nB] Uwall() on movingWallVelocity faces, 0 elsewhere
};

// The motion of one case: which points each body moves, the initial points.
class FvMeshMotion
{
public:
    // Pre: L = loadFvCase(...) of a case with c.motion.on, at time 0 (points0 =
    // the mesh points). Errors (exit 3) by name: a body's cellZone missing; a
    // 2D mesh rotating about another axis than its empty one; an internal face
    // between cells of different motion (a moving zone bordering other cells
    // through ordinary faces: it needs a cyclicAMI interface); a point shared
    // by cells of different motion; more than one motion group (a static
    // remainder or a second body) on a mesh without cyclicAMI patches.
    static std::expected<FvMeshMotion, std::string> make(const FvLoaded& L);
    // mesh.update() to time t (the new time) over deltaT: L.pm's points move,
    // L.fm = fromPolyMesh(L.pm) and s = buildSimpleSetup (+ buildRasSetup) are
    // rebuilt on them. G: buildFvGraph of the mesh before the update; without
    // cyclicAMI it stays (the topology is fixed), with it the AMI and the
    // pattern are rebuilt on the moved points (G.ami non-empty: the device
    // must rebuild its matrices). Error: the rebuilt mesh's (a FaceMesh
    // rejection), the AMI's, the wall distance's.
    // beside (optional): run ONCE on another thread with the rebuilt setup (its
    // ras part not yet written: do not read it) and G, while the wall distance
    // and meshPhi are computed; joined before update returns. It may only READ
    // its arguments (the runner prepares the device's matrix rebuild there).
    using Beside = std::function<void(const FvSimpleSetup&, const FvGraph&)>;
    std::expected<FvMeshMove, std::string> update(double t, double deltaT, FvLoaded& L, FvGraph& G, FvSimpleSetup& s,
                                                  const Beside& beside = {}) const;
    // Points moved by a body (the log's "Applying motion to N points").
    std::int64_t movedPoints() const;

private:
    std::vector<double> x0_, y0_, z0_; // points0
    std::vector<std::int32_t> body_;   // per point: -1 static, else the index of the LAST body moving it
};

} // namespace dns
