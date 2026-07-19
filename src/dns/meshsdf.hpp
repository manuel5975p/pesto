// Mesh-backed signed distance: AABB-BVH closest-triangle query, sign from
// angle-weighted pseudonormals (Baerentzen-Aanaes) or the generalized winding
// number (Jacobson) for leaky meshes; plus the supersampled cut geometry it
// induces. CPU, geometry build time only -- never in the stepping loop.
#pragma once

#include "dns/cutcell.hpp"
#include "dns/trimesh.hpp"

namespace dns {

class MeshSdf
{
public:
    // Builds the BVH and feature pseudonormals; robust switches the sign test
    // to the winding number (O(tris) per call, orientation/leak tolerant).
    // Pre: mesh non-empty with outward CCW winding; degenerate tris dropped.
    explicit MeshSdf(TriMesh mesh, bool robust = false);

    // Signed distance to the mesh surface, < 0 inside the solid.
    double operator()(double x, double y, double z) const;

    // Generalized winding number: ~1 inside, ~0 outside for a closed mesh.
    double winding(double x, double y, double z) const;

    // True iff every edge joins exactly two consistently oriented triangles;
    // false suggests the robust (winding) sign mode.
    bool watertight() const { return watertight_; }

    std::size_t triangleCount() const { return tris_.size(); }

private:
    struct Node
    {
        std::array<double, 3> lo, hi;
        int a, b;         // children (internal node)
        int start, count; // tri range in order_ (leaf iff count > 0)
    };
    struct Hit
    {
        double d2;
        int tri;
        std::array<double, 3> cp;
        int feat; // 0..2 vertex, 3..5 edge ab/bc/ca, 6 face
    };

    int build(int begin, int end, const std::vector<std::array<double, 3>>& cent);
    void nearest(int ni, const std::array<double, 3>& p, Hit& h) const;

    std::vector<std::array<double, 3>> verts_;
    std::vector<std::array<int, 3>> tris_;
    std::vector<std::array<double, 3>> faceN_, vertN_;
    std::vector<std::array<std::array<double, 3>, 3>> edgeN_; // per tri: ab,bc,ca
    std::vector<int> order_; // BVH leaf triangle permutation
    std::vector<Node> nodes_;
    bool robust_ = false;
    bool watertight_ = false;
};

} // namespace dns
