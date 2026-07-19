// Shared mesh generators/writers for the mesh tests (test_meshsdf,
// test_val_mesh_sphere): icosphere oracle geometry and a binary STL writer.
#pragma once

#include "dns/trimesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <utility>
#include <vector>

namespace meshtest {

// Icosphere of radius r about c: subdivided icosahedron, verts on the sphere.
inline dns::TriMesh icosphere(double r, std::array<double, 3> c, int subdiv)
{
    using A3 = std::array<double, 3>;
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    dns::TriMesh m;
    m.verts = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0},
               {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t},
               {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
    m.tris = {{0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
              {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
              {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
              {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};
    auto project = [&](A3& v) {
        const double s = 1.0 / std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        v = {v[0] * s, v[1] * s, v[2] * s};
    };
    for (auto& v : m.verts) project(v);
    for (int s = 0; s < subdiv; ++s)
    {
        std::map<std::pair<int, int>, int> mid;
        auto midpoint = [&](int a, int b) {
            const auto key = std::minmax(a, b);
            const auto [it, fresh] = mid.try_emplace(key, int(m.verts.size()));
            if (fresh)
            {
                A3 v{0.5 * (m.verts[a][0] + m.verts[b][0]),
                     0.5 * (m.verts[a][1] + m.verts[b][1]),
                     0.5 * (m.verts[a][2] + m.verts[b][2])};
                project(v);
                m.verts.push_back(v);
            }
            return it->second;
        };
        std::vector<std::array<int, 3>> next;
        next.reserve(4 * m.tris.size());
        for (const auto& tr : m.tris)
        {
            const int ab = midpoint(tr[0], tr[1]), bc = midpoint(tr[1], tr[2]),
                      ca = midpoint(tr[2], tr[0]);
            next.push_back({tr[0], ab, ca});
            next.push_back({tr[1], bc, ab});
            next.push_back({tr[2], ca, bc});
            next.push_back({ab, bc, ca});
        }
        m.tris = std::move(next);
    }
    for (auto& v : m.verts) v = {c[0] + r * v[0], c[1] + r * v[1], c[2] + r * v[2]};
    return m;
}

inline void writeBinaryStl(const std::filesystem::path& p, const dns::TriMesh& m)
{
    std::ofstream f(p, std::ios::binary);
    const char hdr[80]{};
    f.write(hdr, 80);
    const std::uint32_t n = std::uint32_t(m.tris.size());
    f.write(reinterpret_cast<const char*>(&n), 4);
    for (const auto& t : m.tris)
    {
        const float zero[3]{};
        f.write(reinterpret_cast<const char*>(zero), 12); // normal: ignored
        for (int v = 0; v < 3; ++v)
        {
            const float xyz[3]{float(m.verts[t[v]][0]), float(m.verts[t[v]][1]),
                               float(m.verts[t[v]][2])};
            f.write(reinterpret_cast<const char*>(xyz), 12);
        }
        const std::uint16_t attr = 0;
        f.write(reinterpret_cast<const char*>(&attr), 2);
    }
}

} // namespace meshtest
