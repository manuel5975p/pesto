// Triangle meshes for arbitrary embedded geometry: STL (binary/ASCII) and
// glTF/GLB loaders plus placement transforms; the input stage for MeshSdf.
#pragma once

#include <array>
#include <expected>
#include <string>
#include <vector>

namespace dns {

// Indexed triangle mesh in world units; CCW winding = outward normal.
struct TriMesh
{
    std::vector<std::array<double, 3>> verts;
    std::vector<std::array<int, 3>> tris;
};

// Axis-aligned bounding box; lo <= hi per axis for a non-empty mesh.
struct Aabb
{
    std::array<double, 3> lo, hi;
};

// Bounding box over all vertices. Pre: mesh non-empty.
Aabb bounds(const TriMesh& m);

// Load an STL file, sniffing binary vs ASCII; identical vertices are merged
// so shared edges/vertices exist (pseudonormals need them).
// Post: tris index valid verts; repeated-index (degenerate) tris dropped.
std::expected<TriMesh, std::string> loadStl(const std::string& path);

// Load glTF/GLB via vendored cgltf: every triangle primitive of every node,
// node world transforms applied, merged into one mesh. Post: as loadStl.
std::expected<TriMesh, std::string> loadGltf(const std::string& path);

// Dispatch on extension: .stl -> loadStl, .gltf/.glb -> loadGltf.
std::expected<TriMesh, std::string> loadMesh(const std::string& path);

// What a placement holds fixed.
//
// BoxCenter is what every static body has always used and stays the default.
// A ROTOR needs Origin: `center` is then the shaft, and a propeller's bounding
// box is not centred on its shaft unless the blade count happens to make it so
// -- three blades put it 10% of the tip radius off, which spins the model about
// a point that is not its axis. A rotor exported from CAD is built about its
// axis, so its own origin IS the shaft.
enum class MeshAnchor { BoxCenter, Origin };

// Scale about the anchor, rotate pitch about x then yaw about z (radians),
// then move the anchor to `center`. Pre: mesh non-empty.
void transform(TriMesh& m, double scale, double pitchRad, double yawRad,
               std::array<double, 3> center,
               MeshAnchor anchor = MeshAnchor::BoxCenter);

// Diameter of the circle the mesh sweeps when spun about the `axis` line
// through the ORIGIN: twice the largest distance from that line to any vertex.
// This -- not the bounding box -- is a rotor's diameter, and for an odd blade
// count the two differ by ~10%.
// Pre: mesh non-empty, axis in [0,2].
double sweptDiameter(const TriMesh& m, int axis);

} // namespace dns
