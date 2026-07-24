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

#include "dns/config.hpp"
#include "dns/cutcell.hpp"
#include "dns/grid.hpp"
#include "dns/sdf.hpp"

#include <array>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace dns {

enum class SceneObjectKind { Cylinder, Sphere, Cube, Lens, Mesh };

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

    // True when every object is z-invariant (no sphere, no mesh): the union
    // sdf can then be sampled on one x-y layer and replicated.
    bool zInvariant() const;
    // Any genuinely 3D object present (sphere or mesh)? Such a scene needs
    // nz > 1 or it has no solid cells.
    bool has3d() const;
    // One-line-per-object summary for --dry-run.
    std::string describe() const;
};

// Strict parse of the scene TOML: unknown keys, wrong types and missing
// required fields are hard errors, never silent defaults (duplicate keys the
// TOML grammar already rejects). Does NOT touch the filesystem (mesh paths
// are recorded, not opened).
std::expected<Scene, std::string> parseScene(std::string_view toml);

// Reads and parses the file. Filesystem, so phase 1.5 of main -- after parse,
// before validate; a failure is an INPUT error (exit 3).
std::expected<Scene, std::string> loadScene(const std::string& path);

// Merges the scene's model selection into the RawConfig before validate():
//   model les + no --les      -> raw.les = cs
//   model les + agreeing --les-> keep
//   any conflict              -> config error (strict, never silently resolved)
std::expected<void, ConfigError> applySceneModel(const Scene&, RawConfig&);

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
