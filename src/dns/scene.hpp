// Scene: the generalized simulation environment. A scene is a set of solid
// objects placed relative to the standard single-body placement, plus an
// explicit model selection (DNS or LES). It is described by a small strict
// TOML file (--scene FILE.toml, docs/scene.md; parsed with the vendored
// toml++ in third_party/tomlplusplus) and compiled by buildSceneGeometry()
// -- the mesh builder -- into ONE union sdf + sampled cut geometry that the
// existing runners consume unchanged.
//
// Units: object offsets (x,y,z) and diameters (d) are in units of the
// reference diameter D that --dr already defines, measured from the standard
// placement. A lone `[[objects]] kind = "cylinder"` is the classic cylinder
// case (on the generic sampled cut, not the exact closed-form one).
#pragma once

#include "dns/runmeta.hpp"
#include "dns/sgs.hpp"
#include "dns/cutcell.hpp"
#include "dns/grid.hpp"
#include "dns/sdf.hpp"

#include <array>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace dns {

enum class SceneObjectKind { Cylinder, Sphere, Cube, Lens, Mesh };

// The ONE spelling of the kinds, indexed BY the enum value: the TOML parser,
// the writer, the `unknown kind` message and the editor's picker all read it.
std::span<const char* const> sceneObjectKindNames();
const char* name(SceneObjectKind);

struct SceneObject
{
    SceneObjectKind kind = SceneObjectKind::Cylinder;
    std::string label;          // force-column label; "" = kind + index
    double x = 0, y = 0, z = 0; // offset from the reference placement, in D
    double d = 1.0;             // diameter in units of D
    double aspect = 3.5;        // lens only: width / height
    std::string meshPath;       // mesh only (.stl/.gltf/.glb)
    double scale = 0;           // mesh only: world scale; 0 = fit extent to d*D
    double pitch = 0, yaw = 0;  // mesh only: RAW MODEL AXES (deprecated flags)
    bool robust = false;        // mesh only: winding-number sign
};

enum class SceneModel { Dns, Les };

struct Scene
{
    SceneModel model = SceneModel::Dns;
    double cs = 0;              // SGS model constant; > 0 only with Les
    SgsModel sgs = SgsModel::Smagorinsky;   // `sgs = "smag"|"sigma"`, Les only
    std::vector<SceneObject> objects;

    // True when the union sdf may be sampled on ONE x-y layer and replicated.
    // Excludes sphere, mesh AND cube. The cube exclusion USED to be merely
    // conservative -- a cube spanned the full z extent, so only its sdf VALUE
    // varied in z, not its sign. It is now load-bearing: a cube is z-finite
    // (sceneCube), so replicating one layer would extrude it back into the
    // pillar it is no longer. Do not relax it.
    bool zInvariant() const;
    // Any genuinely 3D object present (sphere or mesh)? Such a scene needs
    // nz > 1 or it has no solid cells.
    bool has3d() const;
    // One-line-per-object summary for --dry-run.
    std::string describe() const;
};

// --- rules shared with tools -------------------------------------------------
//
// These are the scene rules a tool (the editor) needs to ASK about rather than
// restate. Each is the ONE definition: parseScene() and buildSceneGeometry()
// below call them too, so a second, drifting opinion cannot exist. Adding a
// rule here and not routing the core's own caller through it defeats the point.

// The force-column name rule: non-empty and [A-Za-z0-9_-]. A label becomes a
// CSV column suffix (cd_<name>), which is why it is token-shaped.
bool validSceneName(std::string_view);

// The parser's default label for an unnamed object: kind + index.
std::string defaultSceneName(SceneObjectKind, std::size_t index);

// Non-empty, valid and pairwise unique. Same message the parser emits.
std::expected<void, std::string> checkSceneNames(const Scene&);

// True when this object alone needs nz > 1 (sphere or mesh). Scene::has3d() is
// any_of over this.
bool needs3d(const SceneObject&);

// The cube rule, in ONE place. A cube's cross-section is the L1 ball of
// half-diagonal r -- a square rotated 45 deg about z -- so `r` is the FRONTAL
// half-width and 2r = D is the reference length every drag coefficient divides
// by, exactly as for the cylinder and the sphere. The square's SIDE is
// therefore r*sqrt2, and a cube is as tall as its side: z spans
// cz +- r/sqrt2, NOT the box.
//
// At nz == 1 there is no third dimension to be tall in, so the cube degenerates
// to the z-invariant prism -- the 2D reading of a cube is its cross-section,
// the same reading that makes a 2D cylinder the cross-section of an infinite
// one. Only in that degenerate case does a cube span the box.
RotatedCube sceneCube(const Grid&, double cx, double cy, double cz, double r);

// The world box an object occupies: the same arithmetic buildSceneGeometry()
// uses. MESH objects report the NOMINAL d*D box -- the real one is only known
// once the file is loaded, so for a mesh this is advisory and the builder
// overwrites it.
std::array<double, 6> sceneObjectBox(const Grid&, const Placement& ref,
                                     const SceneObject&);

// The interior rule buildSceneGeometry() enforces before accepting an object:
// the solid must sit strictly inside, z exempt for objects that span the box
// by design.
bool sceneBoxInside(const Grid&, const std::array<double, 6>& bb);

// Strict parse of the scene TOML: unknown keys, wrong types and missing
// required fields are hard errors, never silent defaults (duplicate keys the
// TOML grammar already rejects). Does NOT touch the filesystem (mesh paths
// are recorded, not opened).
std::expected<Scene, std::string> parseScene(std::string_view toml);

// Reads and parses the file. Filesystem, so phase 1.5 of main -- after parse,
// before validate; a failure is an INPUT error (exit 3).
std::expected<Scene, std::string> loadScene(const std::string& path);

// (applySceneModel lives in config.cpp: merging a scene's model selection into
// the RawConfig is a CONFIG-layer act. Having the geometry layer mutate the
// parse layer inverted the documented order.)

// The compiled scene: what buildSceneGeometry() hands to the runners.
struct SceneGeometry
{
    Sdf sdf;                       // union of all objects (min of sdfs)
    CutGeometry cut;               // sampled once over the union
    std::array<double, 6> bbox{};  // union world box {xlo,ylo,zlo,xhi,yhi,zhi}

    struct ObjectBox
    {
        std::string label;
        std::array<double, 6> bbox;   // world box of this object's solid
    };
    std::vector<ObjectBox> objects;   // per-object force-tally boxes
};

// The mesh builder: resolves every object to world coordinates about the
// reference placement, loads mesh objects, composes the union sdf and samples
// the cut ({8,8}, z-invariant layer when the whole scene is). Errors are user
// input errors (bad mesh, 3D object at nz == 1, object outside the interior).
std::expected<SceneGeometry, std::string>
buildSceneGeometry(const Grid&, const Placement& ref, const Scene&);

} // namespace dns
